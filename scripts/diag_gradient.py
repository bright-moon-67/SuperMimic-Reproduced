"""Compare the Koopman action Jacobian against the MuJoCo/warp simulator.

This is the first gradient-fidelity diagnostic from plan6/plan8.  For the same
state/action pairs it computes the directional derivative of the one-step state
map through the world model and through the real simulator:

    J_wm v  ~= (F_wm(s, a+eps v) - F_wm(s, a-eps v)) / (2 eps)
    J_sim v ~= (F_sim(s, a+eps v) - F_sim(s, a-eps v)) / (2 eps)

and reports cosine similarity, relative L2 error and magnitude ratio.  It is a
single-step diagnostic; multi-step BPTT and policy-gradient tests are separate.

Example:
  .venv/bin/python scripts/diag_gradient.py \
      --wm-state /root/autodl-tmp/logs/wm_plan5/state_koopman.pt \
      --data-in /root/autodl-tmp/data/wm/lafan1_paper_quick.pt \
      --n-samples 4 --n-dirs 8 --out experiment/koopman-world-model/diag_gradient.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.data.replay_buffer import WindowBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.world_model.koopman import KoopmanCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


class _FakeHeads:
    @staticmethod
    def termination_features(root_pos, root_quat, ref):
        return torch.zeros(*root_pos.shape[:-1], 2, dtype=root_pos.dtype, device=root_pos.device)


def _cfg_from_dict(d: dict) -> WorldModelCfg:
    d = dict(d)
    if isinstance(d.get("koopman"), dict):
        d["koopman"] = KoopmanCfg(**d["koopman"])
    if isinstance(d.get("state_spec"), dict):
        d["state_spec"] = StateSpecCfg(**d["state_spec"])
    if "hidden" in d:
        d["hidden"] = tuple(d["hidden"])
    return WorldModelCfg(**d)


def _load_wm(path: Path, device: str) -> tuple[WorldModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ck:
        cfg_dict = ck["config"]
        state = ck["state_dict"]
    elif "wm" in ck:
        cfg_dict = ck["config"]["world_model"]
        state = ck["wm"]
    else:
        raise KeyError(f"unsupported checkpoint format: {list(ck)}")
    cfg = _cfg_from_dict(cfg_dict)
    wm = WorldModel(cfg, _FakeHeads(), device)
    wm.load_state_dict(state, strict=True)
    wm.eval()
    return wm, ck


def _load_windows(path: Path, device: str, split: str) -> WindowBatch:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "eval" in d:
        if split not in d["eval"]:
            raise KeyError(f"split {split!r} not in {list(d['eval'])}")
        d = d["eval"][split]
    return WindowBatch(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()})


def _finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"non-finite {name}")


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return (a * b).sum() / denom


def _make_env(npz_dir: Path, device: str):
    files = [str(f) for f in discover_npz(npz_dir)]
    if not files:
        raise FileNotFoundError(f"no npz files under {npz_dir}")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[files[0]],
            num_envs=1,
            future_frames=10,
            episode_length_s=10.0,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            replay_capacity=2,
            record_transitions=False,
            seed=0,
            compact_library=False,
            anchor_obs=False,
        ),
        device,
    )
    return env


def main(
    wm_state: Path,
    data_in: Path = Path("/root/autodl-tmp/data/wm/lafan1_paper_quick.pt"),
    split: str = "a_32",
    npz_dir: Path = Path("/root/autodl-tmp/data/npz/lafan1"),
    n_samples: int = 4,
    n_dirs: int = 8,
    eps: float = 1e-3,
    out: Path = Path("experiment/koopman-world-model/diag_gradient.json"),
    device: str = "cuda:0",
) -> None:
    wm, ck = _load_wm(wm_state, device)
    w = _load_windows(data_in, device, split)
    n_samples = min(n_samples, int(w.states.shape[1]))
    states = w.states[0, :n_samples].clone()
    actions = w.a_exec[0, :n_samples].clone()
    xi = w.xi[:n_samples].clone()
    env = _make_env(npz_dir, device)
    state_io = StateIO(env)
    sim_context = SimContext(env, FOOT_GEOM_REGEX)
    env_ids = torch.zeros(1, dtype=torch.long, device=device)

    def wm_step(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if wm.is_koopman:
            z = wm.encode_state(s, xi[: s.shape[0]])
            s_next, _ = wm.step_z(z, s, a, xi[: s.shape[0]])
            return s_next
        return wm.step(s, a, xi[: s.shape[0]])

    def sim_step(s: torch.Tensor, a: torch.Tensor, xi_sample: torch.Tensor) -> torch.Tensor:
        state_io.write(env_ids, s)
        sim_context.write(xi_sample)
        _, _, term, tout, _ = env.step(a)
        if bool((term | tout).any()):
            raise RuntimeError("episode terminated during one-step simulator probe")
        return state_io.read()

    gens = torch.Generator(device=device)
    gens.manual_seed(0)
    cos_all: list[float] = []
    rel_all: list[float] = []
    mag_all: list[float] = []
    valid = 0
    skipped = 0
    with torch.no_grad():
        for i in range(n_samples):
            s0 = states[i : i + 1]
            a0 = actions[i : i + 1]
            xi0 = xi[i : i + 1]
            try:
                sim0 = sim_step(s0, a0, xi0)
                wm0 = wm_step(s0, a0)
            except RuntimeError:
                skipped += 1
                continue
            _finite("sim0", sim0)
            _finite("wm0", wm0)
            for _ in range(n_dirs):
                v = torch.randn_like(a0, generator=gens)
                v = v / v.norm().clamp_min(1e-12)
                ap, am = a0 + eps * v, a0 - eps * v
                try:
                    sp = sim_step(s0, ap, xi0)
                    sm = sim_step(s0, am, xi0)
                except RuntimeError:
                    skipped += 1
                    continue
                wp = wm_step(s0, ap)
                wm_m = wm_step(s0, am)
                ds_sim = (sp - sm) / (2.0 * eps)
                ds_wm = (wp - wm_m) / (2.0 * eps)
                if not (torch.isfinite(ds_sim).all() and torch.isfinite(ds_wm).all()):
                    skipped += 1
                    continue
                cos_all.append(float(_cos(ds_wm.flatten(), ds_sim.flatten())))
                rel_all.append(float((ds_wm - ds_sim).norm() / ds_sim.norm().clamp_min(1e-12)))
                mag_all.append(float(ds_wm.norm() / ds_sim.norm().clamp_min(1e-12)))
                valid += 1

    payload = {
        "wm_state": str(wm_state),
        "data_in": str(data_in),
        "config": asdict(wm.cfg),
        "checkpoint": {k: ck.get(k) for k in ("iteration", "meta")},
        "n_samples": n_samples,
        "n_dirs": n_dirs,
        "eps": eps,
        "valid": valid,
        "skipped": skipped,
        "one_step_action_jacobian": {
            "cosine_mean": _mean(cos_all),
            "cosine_min": min(cos_all) if cos_all else None,
            "relative_l2_mean": _mean(rel_all),
            "magnitude_ratio_mean": _mean(mag_all),
            "cosine_values": cos_all,
            "relative_l2_values": rel_all,
            "magnitude_ratio_values": mag_all,
        },
        "argv": sys.argv,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    env.close()
    print(json.dumps(payload["one_step_action_jacobian"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    tyro.cli(main)
