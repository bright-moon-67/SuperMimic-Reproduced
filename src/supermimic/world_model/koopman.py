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

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # A init: A = a_eye * I + a_noise * N(0, 1).
    a_eye: float = 0.99
    a_noise: float = 0.01
    enc_last_gain: float = 0.01
    # Bilinear factor init scale.
    bilinear_init: float = 0.01
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

        # Linear + low-rank bilinear operators.
        self.A = nn.Parameter(self.cfg.a_eye * torch.eye(dz) + self.cfg.a_noise * torch.randn(dz, dz))
        self.B = nn.Parameter(torch.zeros(dz, act_dim))
        self.U = nn.Parameter(self.cfg.bilinear_init * torch.randn(act_dim, dz, r))
        self.V = nn.Parameter(self.cfg.bilinear_init * torch.randn(act_dim, dz, r))

    # ------------------------------------------------------------------ primitives

    def encode(self, feat: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """x_t -> z_t."""
        return self.encoder(torch.cat([feat, xi_norm], dim=-1))

    def propagate(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z_t, a_t -> z_{t+1} = A z + B a + sum_i a_i (U_i V_i^T) z (Eq. 4 of plan_one)."""
        z_next = F.linear(z, self.A) + F.linear(action, self.B)
        # (..., 29, r) = z^T V_i ; then rank-r sum against U_i.
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

    # ------------------------------------------------------------------ DynamicsMLP-compatible

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """One re-encode step: delta = D_psi(K(E_phi([feat, xi]), a))."""
        z = self.encode(feat, xi_norm)
        return self.decode(self.propagate(z, action))
