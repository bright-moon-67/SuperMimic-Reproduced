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


def _randomize_decoder(m):
    """Break the zero-init identity prior so Jacobian tests are not trivially 0 == 0."""
    with torch.no_grad():
        last = m.decoder.net[-1]
        last.weight.normal_(0, 0.1)
        last.bias.normal_(0, 0.1)


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


def test_koopman_transition_matrix_matches_propagate():
    """M_t z + B a must equal propagate(z, a)."""
    from supermimic.world_model.koopman import KoopmanDynamics

    m = KoopmanDynamics(_koopman_cfg()).double()
    z = torch.randn(5, 64, dtype=torch.float64)
    a = torch.randn(5, 29, dtype=torch.float64)
    M = m.transition_matrix(a)
    lhs = torch.bmm(M, z.unsqueeze(-1)).squeeze(-1) + torch.nn.functional.linear(a, m.B)
    rhs = m.propagate(z, a)
    torch.testing.assert_close(lhs, rhs)


def test_koopman_xi_state_zero_init_matches_base():
    """use_xi_state=True with zero A_xi/V_xi must be exactly the old model."""
    from supermimic.world_model.koopman import KoopmanDynamics

    base = KoopmanDynamics(_koopman_cfg()).double()
    cfg = _koopman_cfg()
    cfg.use_xi_state = True
    aug = KoopmanDynamics(cfg).double()
    aug.load_state_dict(base.state_dict(), strict=False)
    assert aug.A_xi.abs().max().item() == 0.0
    assert aug.V_xi.abs().max().item() == 0.0

    torch.manual_seed(0)
    feat = torch.randn(8, 71, dtype=torch.float64)
    a = torch.randn(8, 29, dtype=torch.float64)
    xi = torch.randn(8, 33, dtype=torch.float64)
    z = torch.randn(8, 64, dtype=torch.float64)
    torch.testing.assert_close(base(feat, a, xi), aug(feat, a, xi))
    torch.testing.assert_close(base.propagate(z, a), aug.propagate(z, a, xi))


def test_koopman_xi_state_affects_z_and_block_matrix():
    """xi must change z; the augmented block matrix must match propagate."""
    from supermimic.world_model.koopman import KoopmanDynamics

    cfg = _koopman_cfg()
    cfg.use_xi_state = True
    m = KoopmanDynamics(cfg).double()
    with torch.no_grad():
        m.A_xi.normal_(0, 0.1)
        m.V_xi.normal_(0, 0.1)

    z = torch.randn(5, 64, dtype=torch.float64)
    a = torch.randn(5, 29, dtype=torch.float64)
    xi1 = torch.randn(5, 33, dtype=torch.float64)
    xi2 = torch.randn(5, 33, dtype=torch.float64)
    z1 = m.propagate(z, a, xi1)
    z2 = m.propagate(z, a, xi2)
    assert not torch.allclose(z1, z2)

    M = m.augmented_transition_matrix(a)
    assert M.shape == (5, 64 + 33, 64 + 33)
    b = torch.cat([torch.nn.functional.linear(a, m.B), torch.zeros(5, 33, dtype=torch.float64)], dim=-1)
    z_aug = torch.cat([z, xi1], dim=-1)
    want = torch.bmm(M, z_aug.unsqueeze(-1)).squeeze(-1) + b
    got = torch.cat([z1, xi1], dim=-1)
    torch.testing.assert_close(want, got)
    eye = torch.eye(33, dtype=torch.float64).expand(5, 33, 33)
    torch.testing.assert_close(M[..., 64:, 64:], eye)
    assert M[..., 64:, :64].abs().max().item() == 0.0


def test_koopman_xi_state_stability_uses_z_block():
    """A_xi/V_xi must not enter the P2-B z-block sigma_max hinge."""
    from supermimic.world_model.koopman import KoopmanCfg, KoopmanDynamics

    m = KoopmanDynamics(
        KoopmanCfg(dz=32, rank=4, enc_hidden=(16, 16), dec_hidden=(16, 16), use_xi_state=True, stab_mode="power", stab_iters=10)
    )
    a = torch.randn(8, 29)
    with torch.no_grad():
        m.A.zero_()
        m.A.add_(1.5 * torch.eye(32))
        m.U.zero_()
        m.V.zero_()
        m.A_xi.normal_(0, 10.0)
        m.V_xi.normal_(0, 10.0)
    assert m.stability_penalty(a).item() > 0.1
    with torch.no_grad():
        m.A.mul_(0.3)
    assert m.stability_penalty(a).item() == 0.0


def test_koopman_xi_state_step_z_and_window_gradients():
    """WorldModel.step_z must use xi; window_loss must backprop to A_xi/V_xi."""
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent", koopman=_koopman_cfg())
    cfg.koopman.use_xi_state = True
    wm = WorldModel(cfg, _FakeHeads(), "cpu")
    _randomize_decoder(wm.dynamics)
    with torch.no_grad():
        wm.dynamics.A_xi.normal_(0, 0.1)
        wm.dynamics.V_xi.normal_(0, 0.1)

    s = torch.randn(16, 71)
    a = torch.randn(16, 29)
    xi1 = torch.randn(16, 33)
    xi2 = torch.randn(16, 33)
    z = wm.encode_state(s, xi1)
    s1, _ = wm.step_z(z, s, a, xi1)
    s2, _ = wm.step_z(z, s, a, xi2)
    assert not torch.allclose(s1, s2)

    k = 8
    states = torch.randn(k + 1, 16, 71)
    a_seq = torch.randn(k, 16, 29)
    done = torch.zeros(k, 16)
    refs = _fake_refs(k, 16, "cpu")
    loss, _ = wm.window_loss(states, a_seq, xi1, done, refs)
    loss.backward()
    for p in (wm.dynamics.A_xi, wm.dynamics.V_xi):
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_state_augmented_koopman_shapes_and_zero_init():
    from supermimic.world_model.koopman import KoopmanCfg, StateAugmentedKoopman
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    cfg = KoopmanCfg(
        dz=16,
        rank=4,
        rank_domain=2,
        rank_action_domain=2,
        state_in_latent=True,
        enc_hidden=(32, 32),
        dec_hidden=(32, 32),
    )
    m = StateAugmentedKoopman(cfg).double()
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64)
    feat = torch.randn(4, 71, dtype=torch.float64)
    a = torch.randn(4, 29, dtype=torch.float64)
    xi = torch.randn(4, 33, dtype=torch.float64)
    xi_norm = spec.normalize_xi(xi)

    phi = m.encode(feat, xi_norm)
    z = m.initial_state(feat, xi_norm)
    assert phi.shape == (4, 16)
    assert z.shape == (4, 71 + 16)
    delta = m(feat, a, xi_norm)
    assert delta.shape == (4, 70)
    assert delta.abs().max().item() == 0.0

    s = torch.randn(4, 71, dtype=torch.float64)
    s2, z2, d = m.step_latent(z, s, a, xi_norm, spec)
    assert s2.shape == (4, 71)
    assert z2.shape == (4, 71 + 16)
    assert d.shape == (4, 70)


def test_state_augmented_world_model_g0b_and_loss():
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent", koopman=_koopman_cfg())
    cfg.koopman.state_in_latent = True
    cfg.koopman.rank_domain = 2
    cfg.koopman.rank_action_domain = 2
    wm = WorldModel(cfg, _FakeHeads(), "cpu")
    with torch.no_grad():
        for p in (
            wm.dynamics.A,
            wm.dynamics.B,
            wm.dynamics.U,
            wm.dynamics.V,
            wm.dynamics.P,
            wm.dynamics.Q,
        ):
            p.normal_(0, 0.02)

    s = torch.randn(16, 71)
    a = torch.randn(16, 29)
    xi = torch.rand(16, 33)
    z = wm.encode_state(s, xi)
    assert z.shape == (16, 71 + 64)
    s_lat, _ = wm.step_z(z, s, a, xi)
    s_re = wm.step(s, a, xi)
    torch.testing.assert_close(s_lat, s_re, atol=1e-5, rtol=1e-5)

    k = 8
    states = torch.randn(k + 1, 16, 71)
    a_seq = torch.randn(k, 16, 29)
    done = torch.zeros(k, 16)
    refs = _fake_refs(k, 16, "cpu")
    loss, metrics = wm.window_loss(states, a_seq, xi, done, refs)
    assert torch.isfinite(loss)
    assert metrics["nse_1step"] >= 0
    loss.backward()
    assert wm.dynamics.A.grad is not None and torch.isfinite(wm.dynamics.A.grad).all()


def test_state_augmented_domain_coupling_and_stability():
    from supermimic.world_model.koopman import KoopmanCfg, StateAugmentedKoopman

    cfg = KoopmanCfg(
        dz=8,
        rank=2,
        rank_domain=2,
        rank_action_domain=2,
        state_in_latent=True,
        enc_hidden=(8, 8),
        dec_hidden=(8, 8),
        stab_mode="power",
        stab_iters=10,
    )
    m = StateAugmentedKoopman(cfg)
    a = torch.randn(4, 29)
    xi = torch.zeros(4, 33)
    xi[:, 0] = 1.0
    with torch.no_grad():
        m.A.zero_()
        m.U.zero_()
        m.V.zero_()
        m.P.zero_()
        m.Q.zero_()
        m.P[0, 0, 0] = 1.5
        m.Q[0, 0, 0] = 1.0

    M = m.transition_matrix(a, xi)
    assert M.shape == (4, 71 + 8, 71 + 8)
    assert M[0, 0, 0].item() == 1.5
    assert m.transition_matrix(a, torch.zeros_like(xi)).abs().max().item() == 0.0
    assert m.stability_penalty(a, xi).item() > 0.1


def test_koopman_sigma_max_power_vs_svd():
    """The training-time estimate must stay close to exact SVD (P0 calibration)."""
    from supermimic.world_model.koopman import KoopmanDynamics

    torch.manual_seed(0)
    M = torch.randn(8, 32, 32, dtype=torch.float64)
    exact = torch.linalg.svdvals(M).max(dim=-1).values
    est = KoopmanDynamics.sigma_max_power(M, iters=10)
    rel = ((est - exact).abs() / exact).max()
    assert rel.item() < 0.05, f"power iteration rel={rel.item()}"

    # Near-degenerate top singular values (the case plan3 warned about).
    q1, _ = torch.linalg.qr(torch.randn(4, 32, 32, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(4, 32, 32, dtype=torch.float64))
    s = torch.ones(4, 32, dtype=torch.float64)
    s[:, 1] = 0.99
    s[:, 2:] = 0.1
    M2 = q1 @ torch.diag_embed(s) @ q2.transpose(-1, -2)
    exact2 = torch.linalg.svdvals(M2).max(dim=-1).values
    est2 = KoopmanDynamics.sigma_max_power(M2, iters=10)
    rel2 = ((est2 - exact2).abs() / exact2).max()
    assert rel2.item() < 0.05, f"near-degenerate power iteration rel={rel2.item()}"


def test_koopman_prod_norm_matches_exact():
    from supermimic.world_model.koopman import KoopmanDynamics

    torch.manual_seed(1)
    mats = torch.randn(4, 16, 16, dtype=torch.float64)
    got = KoopmanDynamics.prod_norm(mats)
    P = mats[0]
    for M in mats[1:]:
        P = M @ P
    want = torch.linalg.svdvals(P).max(dim=-1).values
    torch.testing.assert_close(got, want)
    # Single 2-D matrix returns a scalar.
    assert KoopmanDynamics.prod_norm(mats[0]).ndim == 0


def test_koopman_e1_modes_equal():
    """Same weights: the first latent/reencode step must be identical (G0b)."""
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent", koopman=_koopman_cfg())
    wm = WorldModel(cfg, _FakeHeads(), "cpu")
    _randomize_decoder(wm.dynamics)
    s = torch.randn(16, 71)
    a = torch.randn(16, 29)
    xi = torch.rand(16, 33)
    z0 = wm.encode_state(s, xi)
    s_lat, _ = wm.step_z(z0, s, a, xi)
    s_re = wm.step(s, a, xi)
    torch.testing.assert_close(s_lat, s_re, atol=1e-6, rtol=1e-5)


def test_koopman_rollout_diagnostics():
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent", koopman=_koopman_cfg())
    wm = WorldModel(cfg, _FakeHeads(), "cpu")
    _randomize_decoder(wm.dynamics)
    k, n = 8, 16
    states = torch.randn(k + 1, n, 71)
    a = torch.randn(k, n, 29)
    xi = torch.rand(n, 33)
    d_lat = wm.rollout_diagnostics(states, a, xi, mode="latent", n_sigma=8)
    d_re = wm.rollout_diagnostics(states, a, xi, mode="reencode", n_sigma=8)
    for key in ("nse", "nse_groups", "z_norm", "cons_abs", "cons_rel", "latent_var", "latent_erank", "sigma_max_M", "prod_norm", "invalid"):
        assert key in d_lat, key
    assert d_lat["nse"].shape == (k,)
    assert d_lat["nse_groups"].shape == (k, 6)
    assert torch.isfinite(d_lat["nse"]).all()
    torch.testing.assert_close(d_lat["nse"][0], d_re["nse"][0], atol=1e-6, rtol=1e-5)


def test_koopman_init_spectral_radius():
    """a_rho_target must bound rho(A) at initialisation (P2-B)."""
    from supermimic.world_model.koopman import KoopmanCfg, KoopmanDynamics

    for seed in range(3):
        torch.manual_seed(seed)
        m = KoopmanDynamics(KoopmanCfg(dz=64, rank=4, enc_hidden=(32, 32), dec_hidden=(32, 32), a_rho_target=0.99))
        assert m.spectral_stats()["rho_A"] <= 0.99 + 1e-4


def test_koopman_stability_penalty_hinge():
    """sigma_max hinge is 0 below 1 and positive above 1; Frobenius is a baseline."""
    from supermimic.world_model.koopman import KoopmanCfg, KoopmanDynamics

    torch.manual_seed(0)
    m = KoopmanDynamics(KoopmanCfg(dz=32, rank=4, enc_hidden=(16, 16), dec_hidden=(16, 16), stab_mode="power", stab_iters=10))
    a = torch.randn(8, 29)
    assert m.stability_penalty(a).item() >= 0.0

    with torch.no_grad():
        m.A.zero_()
        m.A.add_(1.5 * torch.eye(32))
        m.U.zero_()
        m.V.zero_()
    assert m.stability_penalty(a).item() > 0.1  # (1.5 - 1)^2 = 0.25

    with torch.no_grad():
        m.A.mul_(0.3)
    assert m.stability_penalty(a).item() == 0.0

    pf = m.stability_penalty(a, mode="frobenius")
    assert torch.isfinite(pf) and pf.item() > 0.0


def test_koopman_window_loss_stab_metric():
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(type="koopman", koopman_mode="latent", koopman=_koopman_cfg(dz=32, rank=4))
    cfg.koopman.stab_mode = "power"
    cfg.koopman.stab_batch = 8
    wm = WorldModel(cfg, _FakeHeads(), "cpu")
    k, n = 8, 16
    states = torch.randn(k + 1, n, 71)
    a = torch.randn(k, n, 29)
    xi = torch.rand(n, 33)
    done = torch.zeros(k, n)
    refs = _fake_refs(k, n, "cpu")
    loss0, _ = wm.window_loss(states, a, xi, done, refs)
    wm.cfg.koopman.lambda_stab = 1.0
    loss1, m1 = wm.window_loss(states, a, xi, done, refs)
    assert "stab_loss" in m1
    assert m1["stab_loss"].item() >= 0.0
    assert loss1.item() >= loss0.item() - 1e-6


def test_koopman_jacobian_finite_difference():
    """d(sum delta)/da via autograd must match central differences (plan test 1).

    The old test compared per-sample directional derivatives with a batch-sum
    scalar finite difference.  It also ran on the zero-initialised decoder, so
    both sides were zero and the test passed vacuously.  We now check both the
    batch-sum and the per-sample versions on a non-zero decoder.
    """
    from supermimic.world_model.koopman import KoopmanDynamics

    m = KoopmanDynamics(_koopman_cfg()).double().eval()
    _randomize_decoder(m)
    n = 4
    feat = torch.randn(n, 71, dtype=torch.float64)
    xi = torch.randn(n, 33, dtype=torch.float64)
    a = torch.randn(n, 29, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(a)
    eps = 1e-4

    # Batch-sum scalar: f(a) = sum over batch and output dims.
    f = lambda aa: m(feat, aa, xi).sum()
    j_auto = torch.autograd.grad(f(a), a, retain_graph=True)[0]
    fd = (f(a + eps * v) - f(a - eps * v)) / (2 * eps)
    dir_auto = (j_auto * v).sum()
    rel = (dir_auto - fd).abs() / fd.abs().clamp_min(1e-8)
    assert rel.item() < 1e-3, f"batch-sum Jacobian mismatch rel={rel.item()}"

    # Per-sample directional derivative: f_b(a) = sum over output dims of sample b.
    f_per = lambda aa: m(feat, aa, xi).sum(-1)
    fd_per = (f_per(a + eps * v) - f_per(a - eps * v)) / (2 * eps)
    dir_auto_per = (j_auto * v).sum(-1)
    rel_per = ((dir_auto_per - fd_per).abs() / fd_per.abs().clamp_min(1e-8)).max()
    assert rel_per.item() < 1e-3, f"per-sample Jacobian mismatch rel={rel_per.item()}"


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

    base = {"run_name": "k_resume", "log_root": tmp_path, "seed": 1}
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
