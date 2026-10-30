"""Koopman world-model tests (plan_one.md §6): interface, identity prior, latent
linearity, finite-difference Jacobian, window loss / BPTT gradients, and the
partial-resume path for policy reuse (experiment B)."""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def _koopman_cfg(dz=64, rank=4):
    from supermimic.world_model.koopman import KoopmanCfg

    c = KoopmanCfg(dz=dz, rank=rank, enc_hidden=(64, 64), dec_hidden=(64, 64))
    return c


class _FakeHeads:
    """Stand-in for AnalyticHeads: window_loss only needs termination_features."""

    @staticmethod
    def termination_features(root_pos, root_quat, ref):
        return torch.zeros(*root_pos.shape[:-1], 2, dtype=root_pos.dtype, device=root_pos.device)


def _fake_refs(k, n, device):
    from supermimic.world_model.heads import ReferenceFrame

    return [ReferenceFrame(None, None, None, None, torch.zeros(n, 3, device=device), torch.zeros(n, 4, device=device)) for _ in range(k)]


def test_koopman_interface_and_identity_prior():
    """forward matches DynamicsMLP's (feat, a, xi) -> delta; zero z gives zero delta."""
    from supermimic.world_model.koopman import KoopmanDynamics

    m = KoopmanDynamics(_koopman_cfg())
    feat = torch.randn(8, 71)
    a = torch.randn(8, 29)
    xi = torch.randn(8, 33)
    delta = m(feat, a, xi)
    assert tuple(delta.shape) == (8, 70)
    z0 = torch.zeros(8, 64)
    assert m.decode(z0).abs().max() == 0.0  # zero-init decoder: identity dynamics at init
    z = m.encode(feat, xi)
    assert tuple(z.shape) == (8, 64)
    assert tuple(m.reconstruct(z).shape) == (8, 71)


def test_koopman_propagate_linear_in_z():
    """L(z) = A z + sum_i a_i N_i z must be linear: L(z1+z2) + L(0) = L(z1) + L(z2)."""
    from supermimic.world_model.koopman import KoopmanDynamics

    m = KoopmanDynamics(_koopman_cfg()).double()
    z1 = torch.randn(4, 64, dtype=torch.float64)
    z2 = torch.randn(4, 64, dtype=torch.float64)
    a = torch.randn(4, 29, dtype=torch.float64)
    lhs = m.propagate(z1 + z2, a) + m.propagate(torch.zeros_like(z1), a)
    rhs = m.propagate(z1, a) + m.propagate(z2, a)
    torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-6)


def test_koopman_jacobian_finite_difference():
    """d(sum delta)/da via autograd must match central differences (plan test 1)."""
    from supermimic.world_model.koopman import KoopmanDynamics

    m = KoopmanDynamics(_koopman_cfg()).double().eval()
    n = 4
    feat = torch.randn(n, 71, dtype=torch.float64)
    xi = torch.randn(n, 33, dtype=torch.float64)
    a = torch.randn(n, 29, dtype=torch.float64, requires_grad=True)
    f = lambda aa: m(feat, aa, xi).sum()  # noqa: E731
    j_auto = torch.autograd.grad(f(a), a)[0]

    v = torch.randn_like(a)
    eps = 1e-4
    fd = (f(a + eps * v) - f(a - eps * v)) / (2 * eps)
    dir_auto = (j_auto * v).sum(-1)
    rel = ((dir_auto - fd).abs() / fd.abs().clamp_min(1e-8)).max()
    assert rel.item() < 1e-3, f"Jacobian mismatch rel={rel.item()}"


def test_koopman_window_loss_and_bptt_gradients():
    """8-step window loss runs on CPU, metrics populated, gradients reach the action."""
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent", koopman=_koopman_cfg())
    wm = WorldModel(cfg, _FakeHeads(), "cpu")
    k, n = 8, 16
    states = torch.randn(k + 1, n, 71)
    a = torch.randn(k, n, 29)
    xi = torch.rand(n, 33)
    done = torch.zeros(k, n)
    refs = _fake_refs(k, n, "cpu")

    loss, m = wm.window_loss(states, a, xi, done, refs)
    assert torch.isfinite(loss)
    for key in ("state_loss", "term_bce", "lift_loss", "rec_loss", "consistency", "nse_1step", "nse_kstep"):
        assert key in m, key
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)

    # step / step_z / rollout_error interfaces
    s1 = wm.step(states[0], a[0], xi)
    assert tuple(s1.shape) == (n, 71)
    z = wm.encode_state(states[0], xi)
    s2, z2 = wm.step_z(z, states[0], a[0], xi)
    assert tuple(s2.shape) == (n, 71) and tuple(z2.shape) == (n, 64)
    with torch.no_grad():
        err = wm.rollout_error(states, a, xi)
    assert tuple(err.shape) == (k,) and torch.isfinite(err).all()

    # BPTT: gradient through latent propagation reaches the action input.
    # Note: at init the zero-initialised decoder makes d s / d a = 0 (identity
    # prior), so first fit the window for a few steps, mirroring real usage.
    opt = torch.optim.Adam(wm.parameters(), lr=1e-2)
    for _ in range(10):
        loss, _ = wm.window_loss(states, a, xi, done, refs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    a_req = a[:4].clone().requires_grad_(True)
    s = states[0]
    z = wm.encode_state(s, xi)
    for j in range(4):
        s, z = wm.step_z(z, s, a_req[j], xi)
    s.square().sum().backward()
    assert torch.isfinite(a_req.grad).all() and a_req.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_partial_resume_skips_wm(tmp_path):
    """Experiment B: resume a trained policy/critic under a different world model."""
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    base = dict(run_name="k_resume", log_root=tmp_path, seed=1)
    cfg = TrainConfig(**base)
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 64
    cfg.world_model.type = "koopman"
    cfg.world_model.koopman.dz = 64
    cfg.world_model.koopman.rank = 4
    cfg.world_model.koopman.enc_hidden = (64, 64)
    cfg.world_model.koopman.dec_hidden = (64, 64)
    tr = Trainer(cfg)
    try:
        policy_ref = {k: v.clone() for k, v in tr.policy.state_dict().items()}
        critic_ref = {k: v.clone() for k, v in tr.critic.state_dict().items()}
        tr.save("ckpt_a.pt")
    finally:
        tr.env.close()
    ckpt = tmp_path / "k_resume" / "ckpt_a.pt"
    assert ckpt.exists()

    cfg2 = TrainConfig(**base)
    cfg2.run_name = "k_resume2"
    cfg2.data.clips = ["walk1_subject1"]
    cfg2.env.num_envs = 64
    cfg2.world_model.type = "mlp"  # different WM structure than the checkpoint
    cfg2.resume = ckpt
    cfg2.resume_skip = "wm"
    tr2 = Trainer(cfg2)
    try:
        for k, v in policy_ref.items():
            torch.testing.assert_close(tr2.policy.state_dict()[k], v)
        for k, v in critic_ref.items():
            torch.testing.assert_close(tr2.critic.state_dict()[k], v)
        # the MLP world model was NOT overwritten by the koopman checkpoint
        assert "dynamics.mlp" in next(iter(tr2.wm.state_dict()))
        assert tr2.iteration == 0  # never trained, just constructed
    finally:
        tr2.env.close()
