"""P0 protocol tests: frozen replay snapshots, deterministic sampling, clip masks."""

from __future__ import annotations

import torch

from supermimic.data.replay_buffer import ReplayBuffer


def _fill_buffer(envs: int = 4, capacity: int = 32, rows: int = 40) -> ReplayBuffer:
    buf = ReplayBuffer(envs, capacity, 71, 29, 33, "cpu")
    ids = torch.arange(envs)
    for step in range(rows):
        buf.add(
            ids,
            torch.randn(envs, 71),
            torch.randn(envs, 29),
            torch.randn(envs, 29),
            torch.randn(envs, 33),
            torch.randn(envs),
            torch.zeros(envs),
            torch.full((envs,), step, dtype=torch.long),
            torch.full((envs,), step % 3, dtype=torch.long),
            torch.zeros(envs, dtype=torch.bool),
            torch.full((envs,), step == 0, dtype=torch.bool),
        )
        if step % 10 == 9:
            buf.end_episode(ids)
    return buf


def test_snapshot_roundtrip_and_generator_determinism():
    buf = _fill_buffer()
    snap = buf.snapshot()
    buf2 = ReplayBuffer.from_snapshot(snap, "cpu")
    assert buf.size == buf2.size

    w1 = buf.sample_windows(16, 4, generator=torch.Generator().manual_seed(7))
    w2 = buf2.sample_windows(16, 4, generator=torch.Generator().manual_seed(7))
    for name in ("states", "a_exec", "a_clean", "xi", "reward", "done", "frame", "clip", "ep_id"):
        torch.testing.assert_close(getattr(w1, name), getattr(w2, name), rtol=0, atol=0)


def test_sample_windows_mask_and_horizon():
    buf = _fill_buffer()
    k = 4
    mask = buf.window_mask(k) & (buf.clip == 1)
    w = buf.sample_windows(12, k, mask=mask, generator=torch.Generator().manual_seed(3))
    assert w.states.shape == (k + 1, 12, 71)
    assert bool((w.clip == 1).all())
    # Windows stay inside one episode (no cross-episode concatenation).
    assert w.ep_id is not None
    e, t = buf._sample_from_mask(mask, 12, generator=torch.Generator().manual_seed(3))
    offs = torch.arange(k + 1)
    tt = (t[:, None] + offs[None, :]) % buf.T
    ep_ids = buf.ep_id[e[:, None], tt]
    assert bool((ep_ids == ep_ids[:, :1]).all()), "window crosses an episode boundary"
    assert buf.window_mask(k).shape == buf.valid.shape
