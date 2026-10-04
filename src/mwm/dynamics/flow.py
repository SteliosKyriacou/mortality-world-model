"""Model B: latent conditional flow matching (PLAN.md §4.3), B' stochastic variant, and the
OT-CFM snapshot variant for cross-sectional data.

Paired CFM: z_s = (1-s) z0 + s z1, a_s = a0 + s*dt, target (z1 - z0)/dt, loss on the FULL
velocity v = -grad V + g (so V is actually trained -- the bug in the chat's code).

B' (StochasticFlow): probability-flow velocity (CFM) + score (DSM) + fitted diffusion,
recombined into an SDE drift f = v + 1/2 g^2 grad log p (see class docstring).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import LatentDynamics
from .potential import DiagDiffusion, DriftField


class FlowModel(LatentDynamics):
    stochastic = False

    def __init__(self, d, n_u=1, hidden=128, use_potential=True, bridge_sigma=0.0, use_age=True,
                 jac_penalty=0.0):
        super().__init__()
        self.field = DriftField(d, n_u, hidden, use_potential, use_age)
        self.bridge_sigma = bridge_sigma
        self.jac_penalty = jac_penalty
        self.cfg = dict(d=d, n_u=n_u, hidden=hidden, use_potential=use_potential,
                        bridge_sigma=bridge_sigma, use_age=use_age, jac_penalty=jac_penalty)

    def drift(self, z, a, u):
        return self.field(z, a, u)

    def cfm_loss(self, z0, z1, a0, a1, u, j_penalty=1e-3, jac_penalty=None):
        s = torch.rand(z0.shape[0], 1, device=z0.device)
        dt = (a1 - a0).unsqueeze(-1)
        zs = (1 - s) * z0 + s * z1
        if self.bridge_sigma > 0:
            zs = zs + self.bridge_sigma * torch.sqrt(s * (1 - s) * dt) * torch.randn_like(zs)
        target = (z1 - z0) / dt
        jac_penalty = self.jac_penalty if jac_penalty is None else jac_penalty
        if jac_penalty > 0:
            zs = zs.detach().requires_grad_(True)
        v = self.field(zs, a0 + s.squeeze(-1) * dt.squeeze(-1), u)  # velocity only
        loss = ((v - target) ** 2).sum(-1).mean()
        if jac_penalty > 0:
            # Hutchinson estimate of ||dv/dz||_F^2 (RNODE-style): curbs spurious expansion /
            # stiffness that otherwise makes long deterministic rollouts diverge
            e = torch.randn_like(zs)
            jv = torch.autograd.grad((v * e).sum(), zs, create_graph=True)[0]
            loss = loss + jac_penalty * (jv ** 2).sum(-1).mean()
        if j_penalty > 0 and self.field.use_potential:
            j = self.field.nonconservative(zs, a0 + s.squeeze(-1) * dt.squeeze(-1), u)
            loss = loss + j_penalty * (j ** 2).sum(-1).mean()
        return loss

    def loss(self, batch, **kw):
        z0, z1, a0, a1, u = batch
        l = self.cfm_loss(z0, z1, a0, a1, u, kw.get("j_penalty", 1e-3))
        return l, {"cfm": l.item()}

    @torch.no_grad()
    def val_loss(self, batch, **kw):
        z0, z1, a0, a1, u = batch
        torch.manual_seed(0)
        return self.cfm_loss(z0, z1, a0, a1, u, 0.0, 0.0).item()

    @torch.no_grad()
    def odeint(self, z0, a0, a1, u, method="rk4", n_steps=20, rtol=1e-4, atol=1e-5):
        """torchdiffeq integration (fixed-step rk4 or adaptive dopri5); returns z(a1)."""
        from torchdiffeq import odeint
        dt = (a1 - a0).unsqueeze(-1)

        def f(s, z):
            self.nfe += 1
            return self.drift(z, a0 + s * dt.squeeze(-1), u) * dt
        ts = torch.tensor([0.0, 1.0], device=z0.device)
        opts = {"step_size": 1.0 / n_steps} if method in ("rk4", "euler", "midpoint") else {}
        return odeint(f, z0, ts, method=method, options=opts, rtol=rtol, atol=atol)[-1]


class ScoreNet(nn.Module):
    """s(z, a, u) ~ grad_z log p_a(z | u): denoising score matching on visit states."""

    def __init__(self, d, n_u=1, hidden=128, sigma=0.1):
        super().__init__()
        from .potential import mlp
        self.net = mlp(d + 1 + n_u, d, hidden)
        self.sigma = sigma

    def forward(self, z, a, u):
        from .potential import age_norm
        return self.net(torch.cat([z, age_norm(a).unsqueeze(-1), u], -1))

    def dsm_loss(self, z, a, u):
        eps = torch.randn_like(z)
        s = self(z + self.sigma * eps, a, u)
        return ((s * self.sigma + eps) ** 2).sum(-1).mean()


class StochasticFlow(FlowModel):
    """B': stochastic flow matching.

    Paired CFM learns the *probability-flow* velocity v of the population marginals, not the
    SDE drift: for an SDE with diffusion D = diag(g^2), f = v + 1/2 D grad log p_a. So B' adds
      (1) a score network trained by denoising score matching on visit latents, and
      (2) a diagonal age-dependent diffusion g fitted (velocity + score frozen) by the same
          simulated-pair Gaussian NLL as Model A, with drift f = v + 1/2 g^2 * s.
    score_correction=False gives the naive variant "CFM velocity + additive noise".
    Training of v and s is simulation-free; only the small diffusion net needs simulation.
    """
    stochastic = True

    def __init__(self, d, n_u=1, hidden=128, use_potential=True, bridge_sigma=0.0,
                 diffusion_mode="state_age", use_age=True, score_correction=True, dsm_sigma=0.2,
                 max_corr=1.0, jac_penalty=0.0):
        super().__init__(d, n_u, hidden, use_potential, bridge_sigma, use_age, jac_penalty)
        self.diff = DiagDiffusion(d, 64, diffusion_mode)
        self.score = ScoreNet(d, n_u, hidden, dsm_sigma)
        self.score_correction = score_correction
        self.max_corr = max_corr
        self.log_r = nn.Parameter(torch.full((d,), -3.0))
        self.cfg.update(diffusion_mode=diffusion_mode, score_correction=score_correction,
                        dsm_sigma=dsm_sigma)

    def velocity(self, z, a, u):
        return self.field(z, a, u)

    def drift(self, z, a, u):
        v = self.field(z, a, u)
        if self.score_correction:
            g = self.diff(z, a)
            corr = 0.5 * g ** 2 * self.score(z, a, u)
            # soft clamp (|corr| <= max_corr latent units / year) for stability off-manifold
            v = v + self.max_corr * torch.tanh(corr / self.max_corr)
        return v

    def diffusion(self, z, a):
        return self.diff(z, a)

    def diffusion_loss(self, batch, K=32, n_steps=32):
        from .base import gaussian_sample_nll
        z0, z1, a0, a1, u = batch
        S = self.simulate(z0, a0, a1, u, K, n_steps)
        return gaussian_sample_nll(S, z1, self.obs_var)


class SnapshotOTFlow(FlowModel):
    """OT-CFM between adjacent age-bin marginals (unpaired / cross-sectional data)."""

    @staticmethod
    def ot_pairs(zA, zB, rng):
        import ot
        n = min(len(zA), len(zB))
        ia = rng.choice(len(zA), n, replace=False)
        ib = rng.choice(len(zB), n, replace=False)
        A, Bm = zA[ia], zB[ib]
        M = ot.dist(A, Bm)
        P = ot.emd(np.full(n, 1.0 / n), np.full(n, 1.0 / n), M)
        j = P.argmax(1)
        return ia, ib[j]
