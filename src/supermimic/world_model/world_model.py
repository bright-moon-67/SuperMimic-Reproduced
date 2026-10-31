"""World model: learned dynamics + learned termination head + analytic observation/reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.koopman import KoopmanCfg, KoopmanDynamics, StateAugmentedKoopman
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

# Slice boundaries of StateSpec.error_vector: [dp, rotvec, dv, dw, dq, dqd].
_ERROR_GROUPS = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 41), (41, 70))
_ERROR_GROUP_NAMES = ("root_pos", "root_rot", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")


@dataclass
class WorldModelCfg:
    # "mlp" (baseline) or "koopman" (low-rank bilinear Deep Koopman, plan_one.md).
    type: str = "mlp"
    # Koopman latent rollout mode: "latent" (continuous z propagation, the formal experiment)
    # or "reencode" (re-encode the predicted state every step, the control).
    koopman_mode: str = "latent"
    koopman: KoopmanCfg = field(default_factory=KoopmanCfg)
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr: float = 0.25, tilt_thr: float = 0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) = [height error, tilt error] -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        if cfg.type == "koopman":
            if cfg.koopman.state_in_latent:
                self.dynamics = StateAugmentedKoopman(cfg.koopman)
            else:
                self.dynamics = KoopmanDynamics(cfg.koopman)
        else:
            self.dynamics = DynamicsMLP(hidden=cfg.hidden)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    @property
    def is_koopman(self) -> bool:
        return self.cfg.type == "koopman"

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        feat = self.spec.canonical(s)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def encode_state(self, s: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Koopman latent state.

        Classic Koopman: z = E_phi([canonical(s), xi_norm]).
        State-augmented: z = [canonical(s), E_phi([canonical(s), xi_norm])].
        """
        feat = self.spec.canonical(s)
        phi = self.dynamics.encode(feat, self.spec.normalize_xi(xi))
        if getattr(self.dynamics, "state_in_latent", False):
            return torch.cat([feat, phi], dim=-1)
        return phi

    def step_z(self, z: torch.Tensor, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent-propagation step: z -> K(z, a, xi) -> delta -> integrate.

        With ``koopman.use_xi_state`` the normalized xi is an explicit input to
        the z update (augmented state [z; xi]); otherwise it is ignored here and
        only entered the encoder initially.  With ``state_in_latent`` the
        dynamics returns both s_next and z_next, so the encoder is never called
        inside the rollout.
        """
        xi_norm = self.spec.normalize_xi(xi)
        if getattr(self.dynamics, "state_in_latent", False):
            s_next, z_next, _ = self.dynamics.step_latent(z, s, action, xi_norm, self.spec)
            return s_next, z_next
        z_next = self.dynamics.propagate(z, action, xi_norm)
        s_next = self.spec.integrate(s, self.dynamics.decode(z_next))
        return s_next, z_next

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)
        return self.term_head(feats)

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,70), term_bce (n,), logits (n,))."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat))
        logits = self.term_head(feats)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def _loss_step_z(self, z, s, a, xi_norm, s_true, done, ref_root_pos, ref_root_quat):
        """Latent-propagation autoregressive step: returns (s_next, z_next, sq_err, term_bce, logits)."""
        if getattr(self.dynamics, "state_in_latent", False):
            s_next, z_next, _ = self.dynamics.step_latent(z, s, a, xi_norm, self.spec)
        else:
            z_next = self.dynamics.propagate(z, a, xi_norm)
            s_next = self.spec.integrate(s, self.dynamics.decode(z_next))
        e = self.spec.error_vector(s_next, s_true)
        feats = self.heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat))
        logits = self.term_head(feats)
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, z_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step
        if self.is_koopman:
            self._loss_step_z_fn = torch.compile(self._loss_step_z, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step_z

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame]):
        """k-step autoregressive loss.

        states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1).
        """
        if self.is_koopman:
            return self._window_loss_koopman(states, a_exec, xi, done, refs)
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ koopman loss (plan §3)

    def _window_loss_koopman(self, states, a_exec, xi, done, refs):
        """k-step loss with latent-state propagation plus lift / reconstruction terms.

        z_0 = E(x_0); z_{j+1} = K(z_j, a_j, xi); s_{j+1} = Integrate(s_j, D(z_{j+1})).
        L_lift_j = ||K(z_j, a_j, xi) - E(x_{j+1}^true)||^2   (target detached, plan Eq. 5)
        L_rec_j  = ||R(z_j) - x_j||^2                     (plan Eq. 6)
        L_stab_j = hinge(sigma_max(M_t)) or Frobenius baseline (P2-B).
        With ``use_xi_state``, xi also enters every K step via A_xi/V_xi.
        In "reencode" mode z_j = E(x_j) is re-encoded from the predicted state every step.
        """
        k = a_exec.shape[0]
        s = states[0]
        xi_norm = self.spec.normalize_xi(xi)
        z = self.encode_state(s, xi)
        klift, krec = self.cfg.koopman.lambda_lift, self.cfg.koopman.lambda_rec
        kstab = self.cfg.koopman.lambda_stab
        total_state = 0.0
        total_term = 0.0
        total_lift = 0.0
        total_rec = 0.0
        total_stab = torch.zeros((), device=states.device)
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        cons_num = 0.0
        step_fn = getattr(self, "_loss_step_z_fn", None) or self._loss_step_z
        for j in range(k):
            if self.cfg.koopman_mode == "reencode":
                z = self.encode_state(s, xi)
            # reconstruction of the current canonical features from the current z
            x_hat = self.dynamics.reconstruct(z)
            x_cur = self.spec.canonical(s)
            total_rec = total_rec + (x_hat - x_cur).pow(2).mean(-1).mean()
            if kstab > 0:
                a_stab = a_exec[j][: self.cfg.koopman.stab_batch]
                if getattr(self.dynamics, "state_in_latent", False):
                    total_stab = total_stab + self.dynamics.stability_penalty(
                        a_stab, xi_norm[: a_stab.shape[0]]
                    )
                else:
                    total_stab = total_stab + self.dynamics.stability_penalty(a_stab)
            s, z, sq, bce, logits = step_fn(z, s, a_exec[j], xi_norm, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w)
            # lift: predicted z vs encoding of the TRUE next state (target side detached)
            z_target = self.encode_state(states[j + 1], xi).detach()
            total_lift = total_lift + (z - z_target).pow(2).mean(-1).mean()
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
                # plan §9 consistency diagnostic: propagated z vs re-encoding the predicted state
                cons_num += (z - self.encode_state(s, xi)).pow(2).mean(-1).mean()
        loss = (
            total_state / k
            + self.cfg.term_loss_weight * total_term / k
            + klift * total_lift / k
            + krec * total_rec / k
            + kstab * total_stab / k
        )
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            lift_loss=(total_lift / k).detach(),
            rec_loss=(total_rec / k).detach(),
            stab_loss=(total_stab / k).detach(),
            consistency=(cons_num / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch.

        ``mode`` overrides ``cfg.koopman_mode`` for the same weights ("latent" or
        "reencode"); it is ignored for the MLP dynamics.
        """
        if self.is_koopman and mode is not None and mode not in ("latent", "reencode"):
            raise ValueError(f"unknown koopman rollout mode {mode!r}")
        mode = mode or self.cfg.koopman_mode
        k = a_exec.shape[0]
        s = states[0]
        out = []
        if self.is_koopman and mode != "reencode":
            z = self.encode_state(s, xi)
            for j in range(k):
                s, z = self.step_z(z, s, a_exec[j], xi)
                out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        else:
            # re-encode every step (both the reencode Koopman mode and the MLP baseline).
            for j in range(k):
                s = self.step(s, a_exec[j], xi)
                out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)

    @torch.no_grad()
    def rollout_diagnostics(
        self,
        states: torch.Tensor,
        a_exec: torch.Tensor,
        xi: torch.Tensor,
        mode: str | None = None,
        n_sigma: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Per-step open-loop diagnostics.

        Returns float tensors of shape (k,) for E-curves and latent metrics, and
        (k, 6) for the six physical error groups.  For the MLP, latent metrics are NaN.
        ``sigma_max`` / ``prod_norm`` are computed on the first ``n_sigma`` windows
        only (exact SVD is O(dz^3) per matrix).
        """
        if self.is_koopman and mode is not None and mode not in ("latent", "reencode"):
            raise ValueError(f"unknown koopman rollout mode {mode!r}")
        mode = mode or self.cfg.koopman_mode
        k = a_exec.shape[0]
        n = states.shape[1]
        ns = min(int(n_sigma), int(n))
        s = states[0]
        nse: list[torch.Tensor] = []
        groups: list[torch.Tensor] = []
        invalid: list[torch.Tensor] = []
        z_norm: list[torch.Tensor] = []
        cons_abs: list[torch.Tensor] = []
        cons_rel: list[torch.Tensor] = []
        latent_var: list[torch.Tensor] = []
        latent_erank: list[torch.Tensor] = []
        sigma_max_M: list[torch.Tensor] = []
        prod_norm: list[torch.Tensor] = []

        if self.is_koopman:
            xi_norm = self.spec.normalize_xi(xi)
            state_in_latent = getattr(self.dynamics, "state_in_latent", False)
            z = self.encode_state(s, xi) if mode != "reencode" else None
            P = None
            for j in range(k):
                a = a_exec[j]
                if mode == "reencode":
                    z = self.encode_state(s, xi)
                if state_in_latent:
                    s_next, z_next, _ = self.dynamics.step_latent(z, s, a, xi_norm, self.spec)
                else:
                    z_next = self.dynamics.propagate(z, a, xi_norm)
                    s_next = self.spec.integrate(s, self.dynamics.decode(z_next))
                z_re = self.encode_state(s_next, xi)

                e = self.spec.error_vector(s_next, states[j + 1])
                sq = e * e
                nse.append(sq.mean(-1).mean())
                groups.append(torch.stack([sq[..., lo:hi].mean(-1).mean() for lo, hi in _ERROR_GROUPS]))
                invalid.append((~torch.isfinite(s_next).all(-1)).sum())

                z_norm.append(z_next.norm(dim=-1).mean())
                cons = (z_next - z_re).pow(2).mean(-1)  # (n,)
                cons_abs.append(cons.mean())
                cons_rel.append((cons.sqrt() / (z_next.norm(dim=-1) + 1e-8)).mean())

                zc = z_next - z_next.mean(0, keepdim=True)
                latent_var.append(zc.pow(2).mean(0).mean())
                if n > 1:
                    cov = zc.T @ zc / (n - 1)
                    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
                    p = ev / ev.sum().clamp_min(1e-12)
                    latent_erank.append(torch.exp(-(p * (p + 1e-12).log()).sum()))
                else:
                    latent_erank.append(torch.full((), float("nan"), device=s.device))

                if state_in_latent:
                    M = self.dynamics.transition_matrix(a[:ns], xi_norm[:ns])
                else:
                    M = self.dynamics.transition_matrix(a[:ns])
                # Diagnostics use exact SVD (deterministic); the P2-B training
                # regularizer will use the cheaper differentiable power iteration.
                sigma_max_M.append(torch.linalg.svdvals(M).amax(dim=-1).mean())
                P = M.double() if P is None else torch.bmm(M.double(), P)
                prod_norm.append(torch.linalg.svdvals(P).amax(dim=-1).mean())

                s = s_next
                z = z_next if mode != "reencode" else None
        else:
            for j in range(k):
                s = self.step(s, a_exec[j], xi)
                e = self.spec.error_vector(s, states[j + 1])
                sq = e * e
                nse.append(sq.mean(-1).mean())
                groups.append(torch.stack([sq[..., lo:hi].mean(-1).mean() for lo, hi in _ERROR_GROUPS]))
                invalid.append((~torch.isfinite(s).all(-1)).sum())

        nan = torch.full((k,), float("nan"), device=states.device)
        out = {
            "nse": torch.stack(nse),
            "nse_groups": torch.stack(groups),
            "invalid": torch.stack(invalid),
            "z_norm": torch.stack(z_norm) if z_norm else nan,
            "cons_abs": torch.stack(cons_abs) if cons_abs else nan,
            "cons_rel": torch.stack(cons_rel) if cons_rel else nan,
            "latent_var": torch.stack(latent_var) if latent_var else nan,
            "latent_erank": torch.stack(latent_erank) if latent_erank else nan,
            "sigma_max_M": torch.stack(sigma_max_M) if sigma_max_M else nan,
            "prod_norm": torch.stack(prod_norm) if prod_norm else nan,
        }
        return out
