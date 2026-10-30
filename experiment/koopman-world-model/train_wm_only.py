"""Stage-B experiment (plan_one.md §5): fit MLP vs Koopman world models on the
SAME collected transitions and compare open-loop E1 / E8 / E32 (E32 is a held-out
validation metric, never part of the training loss).

Usage:
  # random-policy data
  .venv/bin/python experiment/koopman-world-model/train_wm_only.py --wm-type koopman --out out_koopman.json
  .venv/bin/python experiment/koopman-world-model/train_wm_only.py --wm-type mlp --out out_mlp.json

  # data collected with a trained actor (recommended for policy-relevant states)
  .venv/bin/python experiment/koopman-world-model/train_wm_only.py --policy-ckpt /root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def main(
    wm_type: str = "koopman",
    collect_steps: int = 2000,
    updates: int = 3000,
    batch_size: int = 2048,
    window: int = 8,
    dz: int = 128,
    rank: int = 8,
    koopman_mode: str = "latent",
    lambda_lift: float | None = None,  # None = KoopmanCfg default
    num_envs: int = 2048,
    seed: int = 0,
    npz_names: list[str] | None = None,
    policy_ckpt: Path | None = None,
    out: Path = Path("experiment/koopman-world-model/wm_only_result.json"),
    eval_interval: int = 250,
) -> None:
    device = "cuda:0"
    torch.manual_seed(seed)
    names = npz_names or ["walk1_subject1", "dance1_subject1"]
    files = [str(f) for f in discover_npz("data/npz/lafan1", names=names)]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, num_envs=num_envs, replay_capacity=400, seed=seed), device)
    try:
        obs, _ = env.reset()
        # ---- data collection (random actions or a trained actor) -------------
        if policy_ckpt is not None:
            from supermimic.eval.benchmark import load_policy_from_checkpoint

            policy, _ = load_policy_from_checkpoint(policy_ckpt, device)
            print(f"[data] collecting with policy from {policy_ckpt}", flush=True)
            for _ in range(collect_steps):
                with torch.no_grad():
                    a = policy(obs["policy"], obs["goal"])
                obs, _, _, _, _ = env.step(a)
        else:
            print("[data] collecting with random actions", flush=True)
            for _ in range(collect_steps):
                obs, _, _, _, _ = env.step(0.3 * torch.randn(num_envs, 29, device=device))
        rec = env.recorder_manager.get_term("replay")
        print(f"[data] replay buffer size {rec.buffer.size}", flush=True)

        # ---- world model -------------------------------------------------------
        robot = env.scene["robot"]
        fk = G1Kinematics(load_g1_model(), device=device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm_cfg = WorldModelCfg(type=wm_type)
        if wm_type == "koopman":
            wm_cfg.koopman.dz = dz
            wm_cfg.koopman.rank = rank
            wm_cfg.koopman_mode = koopman_mode
            if lambda_lift is not None:
                wm_cfg.koopman.lambda_lift = lambda_lift
        wm = WorldModel(wm_cfg, heads, device)
        lib = env.command_manager.get_term("motion").library
        tr = WorldModelTrainer(wm, lib, WMTrainerCfg(batch_size=batch_size, batches_per_update=1, lr=2e-4, compile=None))
        n_params = sum(p.numel() for p in wm.parameters())
        print(f"[wm] type={wm_type} params={n_params/1e6:.2f}M")

        history = []
        for i in range(updates):
            m = tr.update(rec.buffer)
            if (i + 1) % eval_interval == 0 or i == updates - 1:
                w = rec.buffer.sample_windows(1024, 32)
                e32 = wm.rollout_error(w.states, w.a_exec, w.xi)
                row = {
                    "update": i + 1,
                    "E1": float(e32[0]),
                    "E8": float(e32[7]),
                    "E32": float(e32[-1]),
                    "state_loss": float(m["state_loss"]),
                    "nse_kstep": float(m["nse_kstep"]),
                }
                if wm_type == "koopman":
                    row["lift_loss"] = float(m["lift_loss"])
                    row["rec_loss"] = float(m["rec_loss"])
                    row["consistency"] = float(m["consistency"])
                history.append(row)
                print(f"[{wm_type}] {row}", flush=True)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"wm_type": wm_type, "params": n_params, "history": history}, indent=2))
        print(f"saved {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
