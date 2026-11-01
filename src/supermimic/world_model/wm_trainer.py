"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.heads import ReferenceFrame
from supermimic.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        optimizer_steps = 0
        clip_hits = 0
        nse_first: float | None = None
        nse_last: float | None = None
        skipped = 0
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            if not torch.isfinite(loss):
                # A single poisoned window must not overwrite a good world model.
                self.opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            if not torch.isfinite(gn):
                self.opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            if float(gn) > self.cfg.grad_clip:
                clip_hits += 1
            self.opt.step()
            optimizer_steps += 1
            if "nse_1step" in m:
                nse = float(m["nse_1step"])
                if nse_first is None:
                    nse_first = nse
                nse_last = nse
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        out = {k: float(v) / max(optimizer_steps, 1) for k, v in agg.items()}
        out["optimizer_steps"] = float(optimizer_steps)
        out["clip_frac"] = float(clip_hits / max(optimizer_steps, 1))
        out["skipped"] = float(skipped)
        out["nse_first"] = float(nse_first) if nse_first is not None else float("nan")
        out["nse_last"] = float(nse_last) if nse_last is not None else float("nan")
        return out

    @torch.no_grad()
    def error_on(self, w: WindowBatch) -> torch.Tensor:
        """k-step open-loop error on a fixed window batch; (k,) tensor."""
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.error_on(w)
