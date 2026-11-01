"""Freeze a replay-buffer snapshot for Koopman world-model experiments (plan P0).

Collect once, then let MLP and Koopman train from *exactly* the same data.  The
snapshot contains the full ring buffer (states / executed actions / xi / done /
frame / clip / ep_id / ring bookkeeping) and fixed evaluation window sets for
the two protocols:

* protocol A (in-distribution): train on walk1_subject1 + dance1_subject1,
  evaluate on unseen subjects of the same motions;
* protocol B (cross-motion): same training split, evaluate on run/sprint clips.

Both protocols save k=8 and k=32 windows so ``E8@8step`` and ``E8@32step`` are
not confused with each other.

Usage:
  # diagnostic A/B split (8 clips, random or policy actions)
  .venv/bin/python experiment/koopman-world-model/collect_wm_data.py \
      --out /root/autodl-tmp/data/wm/lafan1.pt

  # paper-aligned LAFAN1 setup: all 40 clips, pretrained policy, original env
  .venv/bin/python experiment/koopman-world-model/collect_wm_data.py \
      --paper \
      --policy-ckpt /root/autodl-tmp/logs/lafan1_a800/ckpt_latest.pt \
      --num-envs 8192 --sampling-mode return \
      --out /root/autodl-tmp/data/wm/lafan1_paper.pt
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz

DEFAULT_TRAIN = ("walk1_subject1", "dance1_subject1")
DEFAULT_VAL = ("walk1_subject2", "dance1_subject2")
DEFAULT_TEST_A = ("walk1_subject5", "dance1_subject3")
DEFAULT_TEST_B = ("run2_subject1", "sprint1_subject2")


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


def _to_cpu(batch) -> dict:
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in asdict(batch).items()}


def main(
    npz_dir: Path = Path("data/npz/lafan1"),
    train_clips: tuple[str, ...] = DEFAULT_TRAIN,
    val_clips: tuple[str, ...] = DEFAULT_VAL,
    test_a_clips: tuple[str, ...] = DEFAULT_TEST_A,
    test_b_clips: tuple[str, ...] = DEFAULT_TEST_B,
    collect_steps: int = 2000,
    # Upstream `lafan1_a800` config uses 8192 envs; the repo default/e2 uses 4096.
    num_envs: int = 8192,
    replay_capacity: int = 1000,
    # Upstream env action noise is 0.1. Only used for random-action collection;
    # policy collection ignores it (the env's own action noise still applies).
    action_std: float = 0.1,
    sampling_mode: str = "return",
    paper: bool = False,
    window: int = 8,
    eval_horizons: tuple[int, ...] = (8, 32),
    n_train_windows: int = 8192,
    n_eval_windows: int = 2048,
    policy_ckpt: Path | None = None,
    seed: int = 0,
    eval_seed: int = 1234,
    out: Path = Path("/root/autodl-tmp/data/wm/lafan1.pt"),
    device: str = "cuda:0",
) -> None:
    if paper:
        # Paper/upstream LAFAN1 protocol: train and evaluate on all 40 clips.
        # The resulting a_* / b_* eval sets are both in-distribution; use protocol a.
        all_names = tuple(sorted(Path(f).stem for f in discover_npz(npz_dir)))
        train_clips = val_clips = test_a_clips = test_b_clips = all_names
        print(f"[data] paper mode: all {len(all_names)} LAFAN1 clips", flush=True)
    all_clips = tuple(dict.fromkeys((*train_clips, *val_clips, *test_a_clips, *test_b_clips)))
    files = [str(f) for f in discover_npz(npz_dir, names=list(all_clips))]
    found = {Path(f).stem for f in files}
    missing = [c for c in all_clips if c not in found]
    if missing:
        raise FileNotFoundError(f"clips not found under {npz_dir}: {missing}")

    torch.manual_seed(seed)
    t0 = time.time()
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            num_envs=num_envs,
            replay_capacity=replay_capacity,
            sampling_mode=sampling_mode,
            seed=seed,
        ),
        device,
    )
    try:
        obs, _ = env.reset()
        print(f"[data] clips={list(all_clips)} envs={num_envs} steps={collect_steps}", flush=True)
        if policy_ckpt is not None:
            from supermimic.eval.benchmark import load_policy_from_checkpoint

            policy, _ = load_policy_from_checkpoint(policy_ckpt, device)
            print(f"[data] collecting with policy {policy_ckpt}", flush=True)
            for _ in range(collect_steps):
                with torch.no_grad():
                    a = policy(obs["policy"], obs["goal"])
                obs, _, _, _, _ = env.step(a)
        else:
            print("[data] collecting with random actions", flush=True)
            for _ in range(collect_steps):
                obs, _, _, _, _ = env.step(action_std * torch.randn(num_envs, 29, device=device))

        rec = env.recorder_manager.get_term("replay")
        buf = rec.buffer
        lib = env.command_manager.get_term("motion").library
        clip_names = [c.name for c in lib.clips]
        name_to_id = {n: i for i, n in enumerate(clip_names)}
        missing_ids = [c for c in all_clips if c not in name_to_id]
        if missing_ids:
            raise RuntimeError(f"clips missing from MotionLibrary: {missing_ids}")
        valid = buf.valid
        if not valid.any():
            raise RuntimeError("replay buffer is empty")
        for field in ("s", "a", "a_exec", "xi", "r"):
            if not torch.isfinite(getattr(buf, field)[valid]).all():
                raise RuntimeError(f"non-finite values in buffer.{field}")
        if (buf.frame[valid] < 0).any() or (buf.ep_id[valid] < 0).any():
            raise RuntimeError("negative frame or ep_id in replay buffer")
        if (buf.clip[valid] < 0).any() or (buf.clip[valid] >= len(clip_names)).any():
            raise RuntimeError("clip id out of range in replay buffer")
        print(f"[data] replay size={buf.size} coverage={sorted(set(buf.clip[buf.valid].tolist()))}", flush=True)

        def ids(clips: tuple[str, ...]) -> torch.Tensor:
            return torch.tensor([name_to_id[c] for c in clips], dtype=torch.long, device=buf.device)

        def mask_of(clips: tuple[str, ...]) -> torch.Tensor:
            return torch.isin(buf.clip, ids(clips))

        def sample(n: int, k: int, clips: tuple[str, ...], gen_seed: int):
            gen = torch.Generator(device=str(buf.device)).manual_seed(gen_seed)
            return buf.sample_windows(n, k, mask=mask_of(clips), generator=gen)

        eval_sets: dict[str, dict] = {}
        for horizon in eval_horizons:
            # Keep a/b as combined compatibility sets, but also save clean splits:
            # val is for model selection; test_a/test_b are final held-out sets.
            split_sets = (
                ("val", val_clips),
                ("test_a", test_a_clips),
                ("test_b", test_b_clips),
                ("a", (*val_clips, *test_a_clips)),
                ("b", (*val_clips, *test_b_clips)),
            )
            for offset, (name, clips) in enumerate(split_sets):
                w = sample(n_eval_windows, horizon, clips, eval_seed + 17 * horizon + offset)
                eval_sets[f"{name}_{horizon}"] = _to_cpu(w)

        train_sample = None
        if n_train_windows > 0:
            train_sample = _to_cpu(sample(n_train_windows, window, train_clips, seed + 7))

        meta = {
            "npz_dir": str(npz_dir),
            "npz_files": files,
            "npz_sha256": {Path(f).name: _sha256(Path(f)) for f in files},
            "clips": clip_names,
            "train_clips": list(train_clips),
            "val_clips": list(val_clips),
            "test_a_clips": list(test_a_clips),
            "test_b_clips": list(test_b_clips),
            "collect_steps": int(collect_steps),
            "num_envs": int(num_envs),
            "replay_capacity": int(replay_capacity),
            "action_std": float(action_std),
            "sampling_mode": sampling_mode,
            "paper": bool(paper),
            "window": int(window),
            "eval_horizons": list(eval_horizons),
            "n_train_windows": int(n_train_windows),
            "n_eval_windows": int(n_eval_windows),
            "policy_ckpt": str(policy_ckpt) if policy_ckpt is not None else None,
            "policy_sha256": _sha256(policy_ckpt) if policy_ckpt is not None else None,
            "seed": int(seed),
            "eval_seed": int(eval_seed),
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "argv": sys.argv,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "collect_seconds": round(time.time() - t0, 2),
        }
        payload = {
            "buffer": buf.snapshot(),
            "clips": clip_names,
            "meta": meta,
            "train_sample": train_sample,
            "eval": eval_sets,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        for key, w in eval_sets.items():
            name, horizon = key.split("_")
            path = out.parent / f"eval_{name}_k{horizon}.pt"
            torch.save(w, path)
            print(f"[data]   eval {key} -> {path}", flush=True)
        print(f"[data] saved {out} ({out.stat().st_size / 1e9:.2f} GB) + {meta_path}", flush=True)
        for key, w in eval_sets.items():
            print(f"[data]   eval {key}: n={w['states'].shape[1]} k={w['states'].shape[0] - 1}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
