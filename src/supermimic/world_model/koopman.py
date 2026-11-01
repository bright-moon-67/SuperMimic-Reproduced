"""Low-rank bilinear Deep Koopman dynamics (plan_one.md §2–§3).

Model (dz latent, r rank, 29 action dims):

    z_t     = E_phi([canonical(s_t), xi_norm])        encoder
    z_{t+1} = A z_t + B a_t + sum_i a_{t,i} (U_i V_i^T) z_t
    delta   = D_psi(z_{t+1})                           zero-init last layer (identity prior)
    feat_hat= R_eta(z_t)                               auxiliary reconstruction head

The module exposes the same ``forward(feat, action, xi_norm) -> delta`` signature as
``DynamicsMLP`` so ``WorldModel.step`` can swap the two, plus the split primitives
(encode / propagate / decode / reconstruct) needed by the latent-state rollouts.

Initialisation notes:
  * D_psi's last layer is zero-initialised: zero z => zero delta => the model
    extrapolates at constant velocity, matching the MLP baseline's identity prior.
  * E_phi's last layer uses a small orthogonal gain so z starts tiny but nonzero
    (a fully zero encoder would freeze A, B, U, V and the lift loss at zero).
  * A starts near identity for spectral stability of long rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from supermimic.world_model.dynamics import MLP

ACT_DIM = 29
FEAT_DIM = 71
XI_DIM = 33
DELTA_DIM = 70


@dataclass
class KoopmanCfg:
    dz: int = 128
    rank: int = 8
    enc_hidden: tuple[int, ...] = (256, 256)
    dec_hidden: tuple[int, ...] = (256, 256)
    # Loss weights of the latent-consistency and reconstruction terms (plan §3).
    lambda_lift: float = 0.1
    lambda_rec: float = 0.1
    # plan6 P0-2: reconstruct the TRUE canonical state from E(x_true, xi), in
    # addition to the predicted-state consistency loss.  Default off so old
    # checkpoints/configs keep their behaviour; enable for the ablation.
    lambda_rec_true: float = 0.0
    # P2-A: propagated latent vs re-encoding of the predicted state.  For
    # state-augmented Koopman this is applied to the phi block only, because the
    # canonical-state block is already tied to the integration path.
    lambda_cons: float = 0.0
    # plan5 P0-1: stop gradient through the consistency target encoder.
    cons_target_detach: bool = False
    # plan5 P1: split lift_loss into physical and learned-latent blocks.
    # None means "use lambda_lift for that block" (backward compatible).
    lambda_lift_phys: float | None = None
    lambda_lift_phi: float | None = None
    # A init: A = a_eye * I + a_noise * N(0, 1).
    a_eye: float = 0.99
    a_noise: float = 0.01
    enc_last_gain: float = 0.01
    # Bilinear factor init scale.
    bilinear_init: float = 0.01
    # P3b: augment the latent state with normalized xi, z~ = [z, xi_norm].
    # A_xi and V_xi are zero-initialized, so enabling this reproduces the old
    # (xi-only-in-encoder) model exactly at initialization.
    use_xi_state: bool = False
    # P3c: z = [canonical(s); phi(s, xi)].  Dynamics predicts u=(delta, phi'),
    # a linear decoder maps u[:71] to delta, and StateSpec.integrate produces
    # s_next.  Rollouts never re-encode the predicted state.
    state_in_latent: bool = False
    rank_domain: int = 8
    rank_action_domain: int = 4
    # P2-B stability (plan §5). Defaults reproduce K0 exactly.
    a_rho_target: float | None = None  # None = legacy init; 0.99 scales rho(A)
    stab_mode: str = "none"  # "none" | "frobenius" | "power"
    lambda_stab: float = 0.0
    stab_iters: int = 10
    stab_batch: int = 256
    _xi: int = XI_DIM  # kept for tyro completeness / documentation

    def __post_init__(self):
        object.__setattr__(self, "_xi", XI_DIM)


class KoopmanDynamics(nn.Module):
    """delta = D_psi(K(E_phi([feat, xi_norm]), a)); interfaces match DynamicsMLP."""

    def __init__(
        self,
        cfg: KoopmanCfg | None = None,
        feat_dim: int = FEAT_DIM,
        act_dim: int = ACT_DIM,
        xi_dim: int = XI_DIM,
        out_dim: int = DELTA_DIM,
    ):
        super().__init__()
        self.cfg = cfg or KoopmanCfg()
        dz, r = self.cfg.dz, self.cfg.rank

        # Encoder: heading-invariant canonical features + normalised sim context.
        self.encoder = MLP(feat_dim + xi_dim, tuple(self.cfg.enc_hidden), dz, last_gain=self.cfg.enc_last_gain)
        # Auxiliary reconstruction head (plan §3.3), separate from D_psi.
        self.recon = MLP(dz, tuple(self.cfg.dec_hidden), feat_dim)
        # Prediction head: z -> delta. Zero-init last layer => identity dynamics at init.
        self.decoder = MLP(dz, tuple(self.cfg.dec_hidden), out_dim, zero_last=True)

        # Linear + low-rank bilinear operators.  a_rho_target only changes the
        # initialisation; the action-dependent M_t is controlled by the hinge below.
        A_init = self.cfg.a_eye * torch.eye(dz) + self.cfg.a_noise * torch.randn(dz, dz)
        if self.cfg.a_rho_target is not None:
            rho = torch.linalg.eigvals(A_init).abs().max()
            if rho > self.cfg.a_rho_target:
                A_init = A_init * (self.cfg.a_rho_target / rho)
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(torch.zeros(dz, act_dim))
        self.U = nn.Parameter(self.cfg.bilinear_init * torch.randn(act_dim, dz, r))
        self.V = nn.Parameter(self.cfg.bilinear_init * torch.randn(act_dim, dz, r))
        if self.cfg.use_xi_state:
            # Augmented state [z; xi_norm].  xi stays constant; A_xi gives it a
            # direct bias on z, and V_xi lets xi modulate the action-dependent
            # low-rank term U_i (V_i^T z + V_xi_i^T xi).
            self.A_xi = nn.Parameter(torch.zeros(dz, XI_DIM))
            self.V_xi = nn.Parameter(torch.zeros(act_dim, XI_DIM, r))

    # ------------------------------------------------------------------ primitives

    def encode(self, feat: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """x_t -> z_t."""
        return self.encoder(torch.cat([feat, xi_norm], dim=-1))

    def propagate(self, z: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor | None = None) -> torch.Tensor:
        """z_t, a_t -> z_{t+1}.

        Base model: A z + B a + sum_i a_i (U_i V_i^T) z.
        With ``use_xi_state``: additionally + A_xi xi and the rank-r term
        U_i (V_i^T z + V_xi_i^T xi), i.e. the z-block of the augmented
        transition [[M_z(a), A_xi + sum_i a_i U_i V_xi_i^T], [0, I]].
        """
        z_next = F.linear(z, self.A) + F.linear(action, self.B)
        if self.cfg.use_xi_state:
            if xi_norm is None:
                raise ValueError("use_xi_state=True requires xi_norm")
            z_next = z_next + F.linear(xi_norm, self.A_xi)
            # (..., 29, r) = z^T V_i + xi^T V_xi_i ; rank-r sum against U_i.
            zv = torch.einsum("...z,azr->...ar", z, self.V)
            zv = zv + torch.einsum("...x,axr->...ar", xi_norm, self.V_xi)
        else:
            zv = torch.einsum("...z,azr->...ar", z, self.V)
        za = torch.einsum("...a,...ar->...ar", action, zv)
        z_next = z_next + torch.einsum("...ar,azr->...z", za, self.U)
        return z_next

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z_{t+1} -> delta (…, 70)."""
        return self.decoder(z)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """z_t -> reconstructed canonical features x_t."""
        return self.recon(z)

    # ------------------------------------------------------------------ diagnostics

    def transition_matrix(self, action: torch.Tensor) -> torch.Tensor:
        """Z-block transition matrix M_t = A + sum_i a_{t,i} U_i V_i^T.

        ``action`` has shape (..., act_dim); returns (..., dz, dz).  This is the
        same operator used by :meth:`propagate` on z.  The augmented xi block is
        excluded on purpose: it is an integrator with eigenvalue 1, so putting it
        into the P2-B sigma_max hinge would make the penalty useless.
        """
        uv = torch.einsum("ipr,iqr->ipq", self.U, self.V)  # (act, dz, dz)
        return self.A + torch.einsum("...i,ipq->...pq", action, uv)

    def augmented_transition_matrix(self, action: torch.Tensor) -> torch.Tensor:
        """Full block matrix for z~ = [z, xi_norm] on action (diagnostics only).

        [[M_z(a), A_xi + sum_i a_i U_i V_xi_i^T],
         [0,      I                             ]]

        The xi block is an identity integrator; stability/diagnostics must use
        the z-block ``transition_matrix`` instead.
        """
        if not self.cfg.use_xi_state:
            raise ValueError("use_xi_state=False has no augmented transition matrix")
        Mz = self.transition_matrix(action)
        uv_xi = torch.einsum("ipr,iqr->ipq", self.U, self.V_xi)  # (act, dz, XI)
        top_right = self.A_xi + torch.einsum("...i,ipq->...pq", action, uv_xi)
        lead = Mz.shape[:-2]
        dz = self.cfg.dz
        M = Mz.new_zeros(*lead, dz + XI_DIM, dz + XI_DIM)
        M[..., :dz, :dz] = Mz
        M[..., :dz, dz:] = top_right
        M[..., dz:, dz:] = torch.eye(XI_DIM, dtype=Mz.dtype, device=Mz.device)
        return M

    @torch.no_grad()
    def spectral_stats(self) -> dict[str, float]:
        """Exact rho(A) and sigma_max(A) for diagnostics (not used in training)."""
        rho = torch.linalg.eigvals(self.A).abs().max().item()
        smax = torch.linalg.svdvals(self.A).max().item()
        return {"rho_A": float(rho), "sigma_max_A": float(smax)}

    @staticmethod
    def sigma_max_power(M: torch.Tensor, iters: int = 10, detach_v: bool = True, v0: torch.Tensor | None = None) -> torch.Tensor:
        """Approximate sigma_max for a batch of matrices (..., dz, dz).

        Power iteration on M^T M.  ``v`` is detached after each step so the
        returned estimate is differentiable w.r.t. M but has low gradient
        variance.  For diagnostics always calibrate against ``torch.linalg.svdvals``.
        """
        shape = M.shape[:-2]
        Mm = M.reshape(-1, M.shape[-2], M.shape[-1])
        if v0 is None:
            v = torch.randn(Mm.shape[0], Mm.shape[-1], device=M.device, dtype=M.dtype)
        else:
            v = v0.reshape(Mm.shape[0], -1).to(dtype=M.dtype)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        for _ in range(iters):
            v = torch.bmm(Mm.transpose(1, 2), torch.bmm(Mm, v.unsqueeze(-1))).squeeze(-1)
            if detach_v:
                v = v.detach()
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sigma = torch.bmm(Mm, v.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
        return sigma.reshape(shape)

    @staticmethod
    @torch.no_grad()
    def prod_norm(mats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Exact ||M_{k-1} ... M_0||_2 for (T, B, dz, dz) or a list of (B, dz, dz).

        Returns (B,) for batched input; a scalar tensor for a single 2-D matrix.
        Computed in float64 (diagnostic only) to avoid float32 overflow over 32 steps.
        """
        seq = mats if isinstance(mats, (list, tuple)) else ([mats] if mats.dim() == 2 else list(mats.unbind(0)))
        # float64 avoids overflow: 32 steps with sigma_max ~1.8 can exceed float32.
        P = seq[0].double()
        for M in seq[1:]:
            P = M.double() @ P if M.dim() == 2 else torch.bmm(M.double(), P)
        return torch.linalg.svdvals(P).amax(dim=-1)

    def stability_penalty(self, action: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """P2-B stability regularizer on the z-block M_t = A + sum_i a_i U_i V_i^T.

        With ``use_xi_state`` the augmented matrix also has an identity xi block;
        that block is excluded here, otherwise sigma_max >= 1 always and the hinge
        would be meaningless.  The xi coupling is an affine input to z, not part
        of the z-block spectral radius.
        "power" is the sigma_max hinge (recommended); "frobenius" is the cheaper
        matrix-weight-decay baseline and is *not* equivalent to the hinge.
        """
        mode = mode or self.cfg.stab_mode
        if mode == "none":
            return torch.zeros((), device=action.device, dtype=self.A.dtype)
        M = self.transition_matrix(action)
        if mode == "power":
            sigma = self.sigma_max_power(M, iters=self.cfg.stab_iters)
            return torch.relu(sigma - 1.0).pow(2).mean()
        if mode == "frobenius":
            return M.pow(2).sum(dim=(-2, -1)).mean() / self.cfg.dz
        raise ValueError(f"unknown stability mode {mode!r}")

    # ------------------------------------------------------------------ DynamicsMLP-compatible

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """One re-encode step: delta = D_psi(K(E_phi([feat, xi]), a, xi))."""
        z = self.encode(feat, xi_norm)
        return self.decode(self.propagate(z, action, xi_norm))


class StateAugmentedKoopman(nn.Module):
    """State-augmented Koopman: z = [canonical(s); phi(s, xi)].

    Dynamics predicts u = A z + B a + C xi + b
        + sum_i a_i U_i V_i^T z
        + sum_j xi_j P_j Q_j^T z
        + sum_i a_i W_i xi,
    where the first feat_dim entries of u go through a linear decoder to the
    70-d delta; the remaining dz entries are phi_next.  A rollout integrates
    delta with StateSpec and forms z_next = [canonical(s_next); phi_next].
    The encoder is called only to initialise z_0 (and for the detached
    consistency target), never inside the rollout.
    """

    state_in_latent = True

    def __init__(
        self,
        cfg: KoopmanCfg | None = None,
        feat_dim: int = FEAT_DIM,
        act_dim: int = ACT_DIM,
        xi_dim: int = XI_DIM,
        out_dim: int = DELTA_DIM,
    ):
        super().__init__()
        self.cfg = cfg or KoopmanCfg(state_in_latent=True)
        self.feat_dim = feat_dim
        self.act_dim = act_dim
        self.xi_dim = xi_dim
        self.out_dim = out_dim
        dz, r = self.cfg.dz, self.cfg.rank
        rd, rw = self.cfg.rank_domain, self.cfg.rank_action_domain
        D = feat_dim + dz
        self.D = D

        self.encoder = MLP(
            feat_dim + xi_dim, tuple(self.cfg.enc_hidden), dz, last_gain=self.cfg.enc_last_gain
        )
        # plan5 P0-2: reconstruct canonical state from the learned phi block, not
        # from the explicit state block (which would make rec_loss identically 0).
        self.recon = MLP(dz, tuple(self.cfg.dec_hidden), feat_dim)
        # Zero-init dynamics -> zero delta / constant-velocity prior at start.
        self.A = nn.Parameter(torch.zeros(D, D))
        self.B = nn.Parameter(torch.zeros(D, act_dim))
        self.C = nn.Parameter(torch.zeros(D, xi_dim))
        self.b = nn.Parameter(torch.zeros(D))
        # Break the U=V=0 / P=Q=0 / Wr=Ws=0 symmetry: one factor is small
        # random, the other is zero.  The initial low-rank output is still zero,
        # but the zero factor receives a non-zero gradient immediately.
        self.U = nn.Parameter(self.cfg.bilinear_init * torch.randn(act_dim, D, r))
        self.V = nn.Parameter(torch.zeros(act_dim, D, r))
        self.P = nn.Parameter(self.cfg.bilinear_init * torch.randn(xi_dim, D, rd))
        self.Q = nn.Parameter(torch.zeros(xi_dim, D, rd))
        self.Wr = nn.Parameter(self.cfg.bilinear_init * torch.randn(act_dim, D, rw))
        self.Ws = nn.Parameter(torch.zeros(act_dim, xi_dim, rw))

        # Linear state projection: delta = W_d u[:feat_dim].  Start as the
        # first 70 coordinates so the initial delta is exactly the dynamics
        # output (and zero under the zero-init dynamics).
        self.decoder = nn.Linear(feat_dim, out_dim)
        with torch.no_grad():
            self.decoder.weight.copy_(torch.eye(out_dim, feat_dim))
            self.decoder.bias.zero_()

    # ------------------------------------------------------------------ primitives

    def encode(self, feat: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """Canonical state + xi -> phi (dz)."""
        return self.encoder(torch.cat([feat, xi_norm], dim=-1))

    def initial_state(self, feat: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """z_0 = [canonical(s_0); phi(s_0, xi)]."""
        return torch.cat([feat, self.encode(feat, xi_norm)], dim=-1)

    def advance(self, z: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """One latent advance z -> u (D), before integration."""
        u = F.linear(z, self.A) + F.linear(action, self.B) + F.linear(xi_norm, self.C) + self.b
        if self.cfg.rank:
            zv = torch.einsum("...d,adr->...ar", z, self.V)
            za = action.unsqueeze(-1) * zv
            u = u + torch.einsum("...ar,adr->...d", za, self.U)
        if self.cfg.rank_domain:
            zq = torch.einsum("...d,jdr->...jr", z, self.Q)
            zq = zq * xi_norm.unsqueeze(-1)
            u = u + torch.einsum("...jr,jdr->...d", zq, self.P)
        if self.cfg.rank_action_domain:
            wx = torch.einsum("axr,...x->...ar", self.Ws, xi_norm)
            za = action.unsqueeze(-1) * wx
            u = u + torch.einsum("...ar,adr->...d", za, self.Wr)
        return u

    def step_latent(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        action: torch.Tensor,
        xi_norm: torch.Tensor,
        spec,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z_t -> (s_{t+1}, z_{t+1}, delta_t) without re-encoding."""
        u = self.advance(z, action, xi_norm)
        delta = self.decoder(u[..., : self.feat_dim])
        s_next = spec.integrate(s, delta)
        x_next = spec.canonical(s_next)
        phi_next = u[..., self.feat_dim :]
        z_next = torch.cat([x_next, phi_next], dim=-1)
        return s_next, z_next, delta

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """One re-encode step (used by WorldModel.step): returns delta."""
        z = self.initial_state(feat, xi_norm)
        u = self.advance(z, action, xi_norm)
        return self.decoder(u[..., : self.feat_dim])

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct canonical state from phi (plan5 P0-2).

        The explicit state block z[..., :feat_dim] is not used here: returning it
        would make the reconstruction loss identically zero.
        """
        return self.recon(z[..., self.feat_dim :])

    # ------------------------------------------------------------------ diagnostics

    def transition_matrix(self, action: torch.Tensor, xi_norm: torch.Tensor | None = None) -> torch.Tensor:
        """M(a, xi) = A + sum_i a_i U_i V_i^T + sum_j xi_j P_j Q_j^T."""
        if xi_norm is None:
            xi_norm = action.new_zeros(*action.shape[:-1], self.xi_dim)
        M = self.A
        if self.cfg.rank:
            uv = torch.einsum("ipr,iqr->ipq", self.U, self.V)
            M = M + torch.einsum("...i,ipq->...pq", action, uv)
        if self.cfg.rank_domain:
            pq = torch.einsum("jpr,jqr->jpq", self.P, self.Q)
            M = M + torch.einsum("...j,jpq->...pq", xi_norm, pq)
        return M

    @torch.no_grad()
    def spectral_stats(self) -> dict[str, float]:
        rho = torch.linalg.eigvals(self.A).abs().max().item()
        smax = torch.linalg.svdvals(self.A).max().item()
        return {"rho_A": float(rho), "sigma_max_A": float(smax)}

    sigma_max_power = staticmethod(KoopmanDynamics.sigma_max_power)
    prod_norm = staticmethod(KoopmanDynamics.prod_norm)

    def stability_penalty(
        self,
        action: torch.Tensor,
        xi_norm: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> torch.Tensor:
        mode = mode or self.cfg.stab_mode
        if mode == "none":
            return torch.zeros((), device=action.device, dtype=self.A.dtype)
        M = self.transition_matrix(action, xi_norm)
        if mode == "power":
            sigma = self.sigma_max_power(M, iters=self.cfg.stab_iters)
            return torch.relu(sigma - 1.0).pow(2).mean()
        if mode == "frobenius":
            return M.pow(2).sum(dim=(-2, -1)).mean() / self.D
        raise ValueError(f"unknown stability mode {mode!r}")
