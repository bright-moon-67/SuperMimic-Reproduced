"""Koopman diagnostics: same-weight latent vs reencode, spectra, norms, channels.

This is the P0/P1 tool from ``docs/KOOPMAN_FIX_PLAN.md``.  It answers:

1. Does the same checkpoint behave differently in ``latent`` and ``reencode``?
2. Is the propagation matrix amplifying the latent state (rho(A), sigma_max(M),
   ||M_{t-1}...M_0||)?
3. Is the latent state drifting away from the encoding of the predicted physical
   state (cons_abs / cons_rel), and is the representation collapsing
   (latent_var / latent_erank)?
4. Which physical channels dominate the error (six error groups)?
5. Are the 8-step training windows and 32-step evaluation windows distributed
   differently (E8@8step vs E8@32step)?

The script never trains.  It loads a WM checkpoint and writes a JSON report.

Usage:
  .venv/bin/python scripts/diag_koopman.py \
      --wm-state /root/autodl-tmp/logs/wm/k0r_latent.pt \
      --eval-windows-a8 /root/autodl-tmp/data/wm/eval_a_k8.pt \
      --eval-windows-a32 /root/autodl-tmp/data/wm/eval_a_k32.pt \
      --out experiment/koopman-world-model/diag_p1.json

  # or a full Trainer checkpoint
  .venv/bin/python scripts/diag_koopman.py --ckpt logs/run/ckpt_latest.pt ...
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.data.replay_buffer import WindowBatch
from supermimic.world_model.koopman import KoopmanCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


class _FakeHeads:
    """rollout_diagnostics only needs the dynamics; the WM ctor needs a heads object."""

    @staticmethod
    def termination_features(root_pos, root_quat, ref):
        return torch.zeros(*root_pos.shape[:-1], 2, dtype=root_pos.dtype, device=root_pos.device)


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


def _cfg_from_dict(d: dict) -> WorldModelCfg:
    d = dict(d)
    if isinstance(d.get("koopman"), dict):
        d["koopman"] = KoopmanCfg(**d["koopman"])
    if isinstance(d.get("state_spec"), dict):
        d["state_spec"] = StateSpecCfg(**d["state_spec"])
    if "hidden" in d:
        d["hidden"] = tuple(d["hidden"])
    return WorldModelCfg(**d)


def _load_wm(ckpt: Path | None, wm_state: Path | None, device: str) -> tuple[WorldModel, dict]:
    if ckpt is None and wm_state is None:
        raise ValueError("provide --ckpt or --wm-state")
    path = ckpt or wm_state
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt is not None:
        state = ck["wm"]
        cfg_dict = ck["config"]["world_model"]
        meta = {"source": "trainer_ckpt", "path": str(path), "iteration": ck.get("iteration")}
    else:
        state = ck["state_dict"]
        cfg_dict = ck["config"]
        meta = {"source": "wm_state", "path": str(path), "meta": ck.get("meta", {})}
    cfg = _cfg_from_dict(cfg_dict)
    wm = WorldModel(cfg, _FakeHeads(), device)
    wm.load_state_dict(state, strict=True)
    wm.eval()
    return wm, meta


def _load_windows(path: Path | None, device: str) -> WindowBatch | None:
    if path is None:
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    return WindowBatch(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()})


def _closed_loop_fd(wm: WorldModel, w: WindowBatch, steps: int) -> dict:
    """Finite-difference norm of the full (s, z) closed-loop map over H steps.

    sigma_max(M) only measures the linear latent matrix at one operating point;
    it misses the decoder/integrator and the state feedback.  This diagnostic
    perturbs both s and z for one sample and reports ||d(s_H, z_H)|| / ||d(s_0, z_0)||.
    """
    steps = min(steps, w.a_exec.shape[0])
    s0 = w.states[0, :1].clone()
    z0 = wm.encode_state(s0, w.xi[:1])
    actions = w.a_exec[:steps, :1]
    xi = w.xi[:1]

    def rollout(s: torch.Tensor, z: torch.Tensor):
        for t in range(steps):
            s, z = wm.step_z(z, s, actions[t], xi)
        return s, z

    with torch.no_grad():
        s1, z1 = rollout(s0, z0)
        torch.manual_seed(0)
        vs = torch.randn_like(s0)
        vz = torch.randn_like(z0)
        eps = 1e-4
        s2, z2 = rollout(s0 + eps * vs, z0 + eps * vz)
        ds = (s2 - s1) / eps
        dz = (z2 - z1) / eps
        num = torch.sqrt(ds.flatten(1).square().sum(-1) + dz.flatten(1).square().sum(-1)).mean()
        den = torch.sqrt(vs.flatten(1).square().sum(-1) + vz.flatten(1).square().sum(-1)).mean()
    return {"steps": steps, "norm": _f(num / den)}


def _row(wm: WorldModel, w: WindowBatch, mode: str, n_sigma: int) -> dict:
    d = wm.rollout_diagnostics(w.states, w.a_exec, w.xi, mode=mode, n_sigma=n_sigma)
    nse = d["nse"]
    groups = d["nse_groups"]
    return {
        "n": int(w.states.shape[1]),
        "k": int(w.states.shape[0] - 1),
        "E1": _f(nse[0]),
        "E8": _f(nse[7]) if len(nse) >= 8 else None,
        "E32": _f(nse[31]) if len(nse) >= 32 else None,
        "nse_curve": [_f(x) for x in nse.tolist()],
        "error_groups": [
            [_f(x) for x in groups[:, i].tolist()] for i in range(groups.shape[1])
        ],
        "group_names": ["root_pos", "root_rot", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel"],
        "z_norm": [_f(x) for x in d["z_norm"].tolist()],
        "cons_abs": [_f(x) for x in d["cons_abs"].tolist()],  # legacy MSE alias
        "cons_mse": [_f(x) for x in d["cons_mse"].tolist()],
        "cons_l2": [_f(x) for x in d["cons_l2"].tolist()],
        "cons_rel": [_f(x) for x in d["cons_rel"].tolist()],
        "cons_rel_mean": _f(d["cons_rel"].mean()),
        "phi_cons_mse": [_f(x) for x in d["phi_cons_mse"].tolist()],
        "phi_cons_l2": [_f(x) for x in d["phi_cons_l2"].tolist()],
        "phi_cons_rel": [_f(x) for x in d["phi_cons_rel"].tolist()],
        "phi_cons_rel_mean": _f(d["phi_cons_rel"].mean()),
        "latent_var_mean": _f(d["latent_var"].mean()),
        "latent_erank_mean": _f(d["latent_erank"].mean()),
        "sigma_max_M": [_f(x) for x in d["sigma_max_M"].tolist()],
        "sigma_max_M_max": _f(d["sigma_max_M"].max()),
        "prod_norm": [_f(x) for x in d["prod_norm"].tolist()],
        "prod_norm_max": _f(d["prod_norm"].max()),
        "invalid": int(d["invalid"].sum().item()),
    }


def main(
    ckpt: Path | None = None,
    wm_state: Path | None = None,
    eval_windows_a8: Path | None = None,
    eval_windows_a32: Path | None = None,
    eval_windows_b8: Path | None = None,
    eval_windows_b32: Path | None = None,
    n_sigma: int = 64,
    out: Path = Path("experiment/koopman-world-model/diag_koopman.json"),
    device: str = "cuda:0",
) -> None:
    wm, src = _load_wm(ckpt, wm_state, device)
    specs = {
        "a_8": eval_windows_a8,
        "a_32": eval_windows_a32,
        "b_8": eval_windows_b8,
        "b_32": eval_windows_b32,
    }
    modes = ["latent", "reencode"] if wm.is_koopman else ["latent"]
    spectral = wm.dynamics.spectral_stats() if wm.is_koopman else {}
    e1_diff = None
    closed_loop: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for name, path in specs.items():
        w = _load_windows(path, device)
        if w is None:
            continue
        results[name] = {}
        for mode in modes:
            results[name][mode] = _row(wm, w, mode, n_sigma)
        if wm.is_koopman:
            closed_loop[name] = {
                "fd_1step": _closed_loop_fd(wm, w, 1),
                "fd_all": _closed_loop_fd(wm, w, w.a_exec.shape[0]),
            }
        if wm.is_koopman and "latent" in modes and "reencode" in modes:
            w0 = w
            z0 = wm.encode_state(w0.states[0], w0.xi)
            s_lat, _ = wm.step_z(z0, w0.states[0], w0.a_exec[0], w0.xi)
            s_re = wm.step(w0.states[0], w0.a_exec[0], w0.xi)
            e1_diff = float((s_lat - s_re).abs().max().item())

    payload = {
        "source": src,
        "config": asdict(wm.cfg),
        "spectral": spectral,
        "e1_latent_vs_reencode_max_abs": e1_diff,
        "g0b_pass": e1_diff is not None and e1_diff <= 1e-6,
        "closed_loop_fd": closed_loop,
        "eval_sha256": {k: (_sha256(p) if p is not None else None) for k, p in specs.items()},
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "argv": sys.argv,
        "results": results,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"spectral": spectral, "e1_max_abs": e1_diff, "g0b_pass": payload["g0b_pass"], "sets": {k: list(v) for k, v in results.items()}}, indent=2))
    for name, res in results.items():
        for mode, r in res.items():
            print(
                f"{name:5s} {mode:8s} n={r['n']:4d} k={r['k']:2d} "
                f"E1={r['E1']} E8={r['E8']} E32={r['E32']} "
                f"cons_rel={r['cons_rel_mean']} erank={r['latent_erank_mean']} "
                f"sigma_max={r['sigma_max_M_max']} prod_norm={r['prod_norm_max']} invalid={r['invalid']}"
            )
    for name, cl in closed_loop.items():
        print(f"{name:5s} closed_loop_fd_1={cl['fd_1step']['norm']:.4f} fd_all={cl['fd_all']['norm']:.4f}")
    print(f"saved {out}")


if __name__ == "__main__":
    tyro.cli(main)
