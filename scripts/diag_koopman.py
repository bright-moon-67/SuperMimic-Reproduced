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
from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul, quat_normalize
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


def _perturb_state(s: torch.Tensor, direction: torch.Tensor, eps: float) -> torch.Tensor:
    """Perturb a state in a representation-aware way.

    Position/joint/velocity blocks use additive directions.  The root quaternion
    (wxyz) is perturbed on the unit sphere via a body-frame tangent vector:
    q' = normalize(q * exp(eps * omega)).  This keeps ||q'||=1 and avoids
    treating the four quaternion components as independent Euclidean axes.
    """
    p, q, qj, v, w, qd = (
        s[..., 0:3],
        s[..., 3:7],
        s[..., 7:36],
        s[..., 36:39],
        s[..., 39:42],
        s[..., 42:71],
    )
    dq = quat_from_rotvec(eps * direction[..., 3:6])
    q_new = quat_normalize(quat_mul(q, dq))
    return torch.cat(
        [
            p + eps * direction[..., 0:3],
            q_new,
            qj + eps * direction[..., 7:36],
            v + eps * direction[..., 36:39],
            w + eps * direction[..., 39:42],
            qd + eps * direction[..., 42:71],
        ],
        dim=-1,
    )


def _closed_loop_sensitivity(
    wm: WorldModel,
    w: WindowBatch,
    steps: int,
    n_dirs: int = 8,
    n_samples: int = 4,
    eps: float = 1e-4,
) -> dict:
    """Directional sensitivity of the closed-loop map, not a stability proof.

    Physical perturbations keep s and z consistent by re-encoding z' = E(s').
    Latent perturbations keep s fixed and perturb only the learned phi block for
    state-augmented Koopman (the full z for classic Koopman).  Both use central
    differences and report mean/max over random directions and samples.
    """
    steps = min(steps, w.a_exec.shape[0])
    n_samples = min(n_samples, int(w.states.shape[1]))
    s0 = w.states[0, :n_samples].clone()
    xi = w.xi[:n_samples]
    actions = w.a_exec[:steps, :n_samples]
    state_in_latent = bool(getattr(wm.dynamics, "state_in_latent", False))
    z0 = wm.encode_state(s0, xi)
    latent_dim = z0.shape[-1] - (71 if state_in_latent else 0)

    def rollout(s: torch.Tensor, z: torch.Tensor):
        for t in range(steps):
            s, z = wm.step_z(z, s, actions[t], xi)
        return s, z

    def latent_block(z: torch.Tensor) -> torch.Tensor:
        return z[..., 71:] if state_in_latent else z

    def stats(values: list[torch.Tensor]) -> dict:
        v = torch.stack(values)
        return {
            "mean": _f(v.mean()),
            "max": _f(v.max()),
            "values": [_f(x) for x in v.flatten().tolist()],
        }

    gen = torch.Generator(device=w.states.device)
    gen.manual_seed(0)
    phys_out: list[torch.Tensor] = []
    phys_to_latent: list[torch.Tensor] = []
    latent_only: list[torch.Tensor] = []
    lat_to_phys: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_dirs):
            # ---- physical-consistent perturbation: s' and z'=E(s') ----------------
            u = torch.randn(s0.shape, generator=gen, device=s0.device)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            sp, sm = _perturb_state(s0, u, eps), _perturb_state(s0, u, -eps)
            zp, zm = wm.encode_state(sp, xi), wm.encode_state(sm, xi)
            sp_h, zp_h = rollout(sp, zp)
            sm_h, zm_h = rollout(sm, zm)
            ds = (sp_h - sm_h) / (2.0 * eps)
            dz = (latent_block(zp_h) - latent_block(zm_h)) / (2.0 * eps)
            phys_out.append(ds.flatten(1).norm(dim=-1).mean())
            phys_to_latent.append(dz.flatten(1).norm(dim=-1).mean())

            # ---- latent-only perturbation: fix s, perturb phi/z -------------------
            uphi = torch.randn(n_samples, latent_dim, generator=gen, device=s0.device)
            uphi = uphi / uphi.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if state_in_latent:
                zbase = z0.clone()
                zp = torch.cat([zbase[..., :71], zbase[..., 71:] + eps * uphi], dim=-1)
                zm = torch.cat([zbase[..., :71], zbase[..., 71:] - eps * uphi], dim=-1)
            else:
                zp = z0 + eps * uphi
                zm = z0 - eps * uphi
            sp_h, zp_h = rollout(s0, zp)
            sm_h, zm_h = rollout(s0, zm)
            ds = (sp_h - sm_h) / (2.0 * eps)
            dz = (latent_block(zp_h) - latent_block(zm_h)) / (2.0 * eps)
            lat_to_phys.append(ds.flatten(1).norm(dim=-1).mean())
            latent_only.append(dz.flatten(1).norm(dim=-1).mean())

    return {
        "steps": steps,
        "n_dirs": n_dirs,
        "n_samples": n_samples,
        "direction_kind": "central_difference",
        "physical": stats(phys_out),
        "physical_to_latent": stats(phys_to_latent),
        "latent_only": stats(latent_only),
        "latent_to_physical": stats(lat_to_phys),
    }


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
    n_dirs: int = 8,
    n_samples: int = 4,
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
                "fd_1step": _closed_loop_sensitivity(wm, w, 1, n_dirs, n_samples),
                "fd_all": _closed_loop_sensitivity(wm, w, w.a_exec.shape[0], n_dirs, n_samples),
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
        "closed_loop_sensitivity": closed_loop,
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
        for horizon, r in (("1", cl["fd_1step"]), ("all", cl["fd_all"])):
            print(
                f"{name:5s} closed_loop[{horizon}] physical={r['physical']['mean']:.4f} "
                f"physical_to_latent={r['physical_to_latent']['mean']:.4f} "
                f"latent_only={r['latent_only']['mean']:.4f} "
                f"latent_to_physical={r['latent_to_physical']['mean']:.4f}"
            )
    print(f"saved {out}")


if __name__ == "__main__":
    tyro.cli(main)
