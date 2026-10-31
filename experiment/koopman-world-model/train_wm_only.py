"""Stage-B experiment: fit MLP vs Koopman world models on identical data (plan P0/P1).

Recommended (frozen data): first run ``collect_wm_data.py`` once, then both
models train from the same ReplayBuffer snapshot and the same fixed evaluation
windows.  The legacy path (collect in-process) is kept for smoke tests only and
is *not* a held-out protocol.

Usage:
  .venv/bin/python experiment/koopman-world-model/train_wm_only.py \
      --wm-type koopman --protocol a \
      --data-in /root/autodl-tmp/data/wm/lafan1.pt \
      --eval-modes latent,reencode \
      --save-wm /root/autodl-tmp/logs/wm/k0r_latent.pt \
      --out experiment/koopman-world-model/wm_k0r_latent.json
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import ReplayBuffer, WindowBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _f(x) -> float | None:
    v = float(x)
    return v if math.isfinite(v) else None


def _to_device(batch: WindowBatch, device) -> WindowBatch:
    return WindowBatch(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in asdict(batch).items()})


class _MaskedBuffer:
    """ReplayBuffer view restricted to an (E,T) mask with a deterministic sampler.

    A dedicated generator (instead of the global RNG) makes MLP and Koopman see
    the same minibatches at the same seed, even though model construction
    consumes different amounts of global randomness.
    """

    def __init__(self, buffer: ReplayBuffer, mask: torch.Tensor | None, seed: int | None):
        self.buffer = buffer
        self.mask = mask
        self.gen = None if seed is None else torch.Generator(device=str(buffer.device)).manual_seed(seed)

    @property
    def size(self) -> int:
        if self.mask is None:
            return self.buffer.size
        return int((self.buffer.valid & self.mask).sum().item())

    def sample_windows(self, n: int, k: int, generator: torch.Generator | None = None) -> WindowBatch:
        return self.buffer.sample_windows(n, k, mask=self.mask, generator=generator or self.gen)


def _clip_ids(clip_names: list[str], clips: tuple[str, ...]) -> torch.Tensor:
    idx = {n: i for i, n in enumerate(clip_names)}
    missing = [c for c in clips if c not in idx]
    if missing:
        raise ValueError(f"clips not in snapshot: {missing}")
    return torch.tensor([idx[c] for c in clips], dtype=torch.long)


def main(
    wm_type: str = "koopman",
    collect_steps: int = 2000,
    action_std: float = 0.1,
    updates: int = 3000,
    # Upstream `lafan1_a800` WM trainer: batch 4096, 10 batches/update, lr 2e-4,
    # grad_clip 1.0, compile "default".  Keep these defaults so a plain run is
    # directly comparable with the original repo's world-model updates.
    batch_size: int = 4096,
    batches_per_update: int = 10,
    wm_lr: float = 2e-4,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    compile: str | None = "default",
    window: int = 8,
    dz: int = 128,
    rank: int = 8,
    koopman_use_xi_state: bool = False,
    koopman_state_in_latent: bool = False,
    koopman_rank_domain: int = 8,
    koopman_rank_action_domain: int = 4,
    koopman_mode: str = "latent",
    lambda_lift: float | None = None,  # None = KoopmanCfg default
    lambda_rec: float | None = None,  # None = KoopmanCfg default
    lambda_cons: float | None = None,  # P2-A: latent consistency loss
    a_rho_target: float | None = None,  # P2-B: spectral-normalise the A init
    stab_mode: str = "none",  # P2-B: none | frobenius | power
    lambda_stab: float = 0.0,  # P2-B: weight of the stability term
    stab_iters: int = 10,
    stab_batch: int = 256,
    # Upstream `lafan1_a800` config uses 8192 envs; the repo default/e2 uses 4096.
    num_envs: int = 8192,
    seed: int = 0,
    npz_names: list[str] | None = None,
    policy_ckpt: Path | None = None,
    data_in: Path | None = None,
    protocol: str = "a",
    train_clips: tuple[str, ...] | None = None,
    eval_windows_8: Path | None = None,
    eval_windows_32: Path | None = None,
    eval_modes: str = "latent,reencode",
    n_eval_windows: int = 1024,
    n_sigma: int = 64,
    save_wm: Path | None = None,
    replay_capacity: int = 1000,
    out: Path = Path("experiment/koopman-world-model/wm_only_result.json"),
    eval_interval: int = 250,
    device: str = "cuda:0",
    hidden: tuple[int, ...] = (2048, 2048, 2048),
    koopman_enc_hidden: tuple[int, ...] = (256, 256),
    koopman_dec_hidden: tuple[int, ...] = (256, 256),
) -> None:
    torch.manual_seed(seed)
    payload = None
    if data_in is not None:
        payload = torch.load(data_in, map_location="cpu", weights_only=False)
        meta_in = payload["meta"]
        files = list(meta_in["npz_files"])
        if train_clips is None:
            train_clips = tuple(meta_in.get("train_clips") or ("walk1_subject1", "dance1_subject1"))
        if protocol not in ("a", "b"):
            raise ValueError(f"protocol must be 'a' or 'b', got {protocol!r}")
        print(f"[data] frozen snapshot {data_in}", flush=True)
    else:
        names = npz_names or list(train_clips or ("walk1_subject1", "dance1_subject1"))
        files = [str(f) for f in discover_npz("data/npz/lafan1", names=list(names))]
        print("[data] legacy in-process collection (not a held-out protocol)", flush=True)

    # With a frozen snapshot the env is only used for FK/library/heads; creating
    # it at the collection scale would allocate an unused 8192x1000 replay buffer.
    env_num_envs = 1 if payload is not None else num_envs
    env_replay_capacity = 1 if payload is not None else replay_capacity
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            num_envs=env_num_envs,
            replay_capacity=env_replay_capacity,
            seed=seed,
        ),
        device,
    )
    try:
        obs, _ = env.reset()
        # ---- data: frozen snapshot or in-process collection -------------------
        if payload is not None:
            buf = ReplayBuffer.from_snapshot(payload["buffer"], device)
            print(f"[data] replay buffer size {buf.size}", flush=True)
        else:
            if policy_ckpt is not None:
                from supermimic.eval.benchmark import load_policy_from_checkpoint

                policy, _ = load_policy_from_checkpoint(policy_ckpt, device)
                print(f"[data] collecting with policy from {policy_ckpt}", flush=True)
                for _ in range(collect_steps):
                    with torch.no_grad():
                        a = policy(obs["policy"], obs["goal"])
                    obs, _, _, _, _ = env.step(a)
            else:
                print(f"[data] collecting with random actions (std={action_std:g})", flush=True)
                for _ in range(collect_steps):
                    obs, _, _, _, _ = env.step(action_std * torch.randn(num_envs, 29, device=device))
            rec = env.recorder_manager.get_term("replay")
            buf = rec.buffer
            print(f"[data] replay buffer size {buf.size}", flush=True)

        # ---- world model -----------------------------------------------------
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm_cfg = WorldModelCfg(type=wm_type, hidden=tuple(hidden))
        if wm_type == "koopman":
            wm_cfg.koopman.dz = dz
            wm_cfg.koopman.rank = rank
            wm_cfg.koopman.use_xi_state = koopman_use_xi_state
            wm_cfg.koopman.state_in_latent = koopman_state_in_latent
            wm_cfg.koopman.rank_domain = koopman_rank_domain
            wm_cfg.koopman.rank_action_domain = koopman_rank_action_domain
            wm_cfg.koopman.enc_hidden = tuple(koopman_enc_hidden)
            wm_cfg.koopman.dec_hidden = tuple(koopman_dec_hidden)
            wm_cfg.koopman_mode = koopman_mode
            if lambda_lift is not None:
                wm_cfg.koopman.lambda_lift = lambda_lift
            if lambda_rec is not None:
                wm_cfg.koopman.lambda_rec = lambda_rec
            if lambda_cons is not None:
                wm_cfg.koopman.lambda_cons = lambda_cons
            wm_cfg.koopman.a_rho_target = a_rho_target
            wm_cfg.koopman.stab_mode = stab_mode
            wm_cfg.koopman.lambda_stab = lambda_stab
            wm_cfg.koopman.stab_iters = stab_iters
            wm_cfg.koopman.stab_batch = stab_batch
        wm = WorldModel(wm_cfg, heads, device)
        lib = env.command_manager.get_term("motion").library
        clip_names = [c.name for c in lib.clips]
        tr = WorldModelTrainer(
            wm,
            lib,
            WMTrainerCfg(
                batch_size=batch_size,
                batches_per_update=batches_per_update,
                lr=wm_lr,
                grad_clip=grad_clip,
                weight_decay=weight_decay,
                compile=compile,
                window=window,
            ),
        )
        n_params = sum(p.numel() for p in wm.parameters())
        print(
            f"[wm] type={wm_type} params={n_params/1e6:.2f}M window={tr.cfg.window} "
            f"batch={batch_size}x{batches_per_update} lr={wm_lr:g} compile={compile} "
            f"xi_state={koopman_use_xi_state if wm_type == 'koopman' else False} "
            f"state_z={koopman_state_in_latent if wm_type == 'koopman' else False}",
            flush=True,
        )

        # ---- training buffer view --------------------------------------------
        train_mask = None
        if payload is not None and train_clips:
            ids = _clip_ids(clip_names, tuple(train_clips)).to(buf.device)
            train_mask = torch.isin(buf.clip, ids)
        train_buffer = _MaskedBuffer(buf, train_mask, seed + 1000 if payload is not None else None)

        # ---- fixed evaluation windows ----------------------------------------
        modes = [m.strip() for m in eval_modes.split(",") if m.strip()]
        if wm_type != "koopman":
            modes = ["latent"]
        eval_sets: dict[str, WindowBatch] = {}
        if payload is not None and eval_windows_8 is None and eval_windows_32 is None:
            prefix = protocol.strip().lower() + "_"
            for name, w in payload["eval"].items():
                if name.startswith(prefix):
                    eval_sets[name] = _to_device(WindowBatch(**w), device)
        if eval_windows_8 is not None or eval_windows_32 is not None:
            eval_sets = {}
            if eval_windows_8 is not None:
                eval_sets["ext_8"] = _to_device(WindowBatch(**torch.load(eval_windows_8, map_location="cpu", weights_only=False)), device)
            if eval_windows_32 is not None:
                eval_sets["ext_32"] = _to_device(WindowBatch(**torch.load(eval_windows_32, map_location="cpu", weights_only=False)), device)
        if not eval_sets:
            gen = torch.Generator(device=str(buf.device)).manual_seed(seed + 99)
            for k in (8, 32):
                if buf.size > k:
                    try:
                        eval_sets[f"train_{k}"] = buf.sample_windows(n_eval_windows, k, generator=gen)
                    except RuntimeError as exc:
                        print(f"[eval] skip k={k}: {exc}", flush=True)
            if not eval_sets:
                raise RuntimeError("no evaluation windows: provide --data-in or enough collected data")
        print(f"[eval] sets={[(k, tuple(v.states.shape)) for k, v in eval_sets.items()]} modes={modes}", flush=True)

        def eval_row(update: int, name: str, mode: str, diag: dict) -> dict:
            nse = diag["nse"]
            row = {
                "update": update,
                "set": name,
                "mode": mode,
                "E1": _f(nse[0]),
                "E8": _f(nse[7]) if len(nse) >= 8 else None,
                "E32": _f(nse[31]) if len(nse) >= 32 else None,
                "cons_abs": _f(diag["cons_abs"].mean()),
                "cons_mse": _f(diag["cons_mse"].mean()),
                "cons_l2": _f(diag["cons_l2"].mean()),
                "cons_rel": _f(diag["cons_rel"].mean()),
                "phi_cons_mse": _f(diag["phi_cons_mse"].mean()),
                "phi_cons_l2": _f(diag["phi_cons_l2"].mean()),
                "phi_cons_rel": _f(diag["phi_cons_rel"].mean()),
                "z_norm": _f(diag["z_norm"].mean()),
                "latent_var": _f(diag["latent_var"].mean()),
                "latent_erank": _f(diag["latent_erank"].mean()),
                "sigma_max_M": _f(diag["sigma_max_M"].max()),
                "prod_norm": _f(diag["prod_norm"].max()),
                "invalid": int(diag["invalid"].sum().item()),
                "nse_curve": [_f(x) for x in nse.tolist()],
            }
            return row

        # ---- training --------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_train = time.perf_counter()
        history = []
        for i in range(updates):
            m = tr.update(train_buffer)
            if (i + 1) % eval_interval == 0 or i == updates - 1:
                evals = []
                for name, w in eval_sets.items():
                    for mode in modes:
                        diag = wm.rollout_diagnostics(
                            w.states, w.a_exec, w.xi, mode=mode, n_sigma=n_sigma
                        )
                        row = eval_row(i + 1, name, mode, diag)
                        evals.append(row)
                        print(
                            f"[{wm_type}] u{i+1} {name}/{mode} E1={row['E1']} E8={row['E8']} E32={row['E32']} "
                            f"cons_rel={row['cons_rel']} erank={row['latent_erank']}",
                            flush=True,
                        )
                history.append({"update": i + 1, "train": {k: float(v) for k, v in m.items()}, "evals": evals})

        train_seconds = time.perf_counter() - t_train
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta_out = {
            "wm_type": wm_type,
            "params": n_params,
            "window": window,
            "batch_size": batch_size,
            "batches_per_update": batches_per_update,
            "koopman_use_xi_state": koopman_use_xi_state,
            "koopman_state_in_latent": koopman_state_in_latent,
            "koopman_rank_domain": koopman_rank_domain,
            "koopman_rank_action_domain": koopman_rank_action_domain,
            "lambda_cons": lambda_cons,
            "n_sigma": n_sigma,
            "wm_lr": wm_lr,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "compile": compile,
            "a_rho_target": a_rho_target,
            "stab_mode": stab_mode,
            "lambda_stab": lambda_stab,
            "updates": updates,
            "train_seconds": round(train_seconds, 3),
            "updates_per_s": round(updates / max(train_seconds, 1e-9), 4),
            "peak_mem_gb": round(peak_mem_gb, 3),
            "seed": seed,
            "device": device,
            "protocol": protocol,
            "train_clips": list(train_clips or []),
            "eval_modes": modes,
            "data_in": str(data_in) if data_in is not None else None,
            "data_sha256": _sha256(data_in) if data_in is not None else None,
            "eval_windows_8": str(eval_windows_8) if eval_windows_8 is not None else None,
            "eval_windows_32": str(eval_windows_32) if eval_windows_32 is not None else None,
            "eval_sha256": {
                "8": _sha256(eval_windows_8) if eval_windows_8 is not None else None,
                "32": _sha256(eval_windows_32) if eval_windows_32 is not None else None,
            },
            "eval_sets": {k: list(v.states.shape) for k, v in eval_sets.items()},
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "argv": sys.argv,
        }
        if wm_type == "koopman":
            meta_out["spectral"] = wm.dynamics.spectral_stats()
        result = {"wm_type": wm_type, "params": n_params, "config": asdict(wm_cfg), "meta": meta_out, "history": history}
        out.write_text(json.dumps(result, indent=2))
        print(f"saved {out}", flush=True)
        if save_wm is not None:
            save_wm = Path(save_wm)
            save_wm.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in wm.state_dict().items()},
                    "config": asdict(wm_cfg),
                    "meta": meta_out,
                },
                save_wm,
            )
            print(f"saved wm {save_wm}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
