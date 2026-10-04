"""Model A: latent Neural SDE  dz = F(z, a, u) dt + Sigma(z, a) dW  (PLAN.md §4.2).

Training: for each visit pair (z0, a0) -> (z1, a1) simulate K paths with a fixed-step
solver (torchsde Euler-Maruyama / reversible Heun, or a native EM loop), all pairs
integrated jointly in normalised time s in [0,1] (dz = dt*F ds + sqrt(dt)*Sigma dW_s).
Loss: Gaussian NLL of z1 under N(mean_K, var_K + r^2) where r is a learned
observation-noise scale (encoder error at both visits) -- without r, encoder noise would
be absorbed into Sigma -- optionally plus the energy score.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchsde

from .base import LatentDynamics, gaussian_sample_nll
from .potential import DiagDiffusion, DriftField


class _ScaledSDE(nn.Module):
    noise_type = "diagonal"

    def __init__(self, model: "NeuralSDE", a0, dt, u, sde_type="ito"):
        super().__init__()
        self.m, self.a0, self.dt, self.u = model, a0, dt, u
        self.sde_type = sde_type
        self.sq = torch.sqrt(dt.clamp_min(0)).unsqueeze(-1)

    def f(self, s, z):
        self.m.nfe += 1
        return self.m.drift(z, self.a0 + s * self.dt, self.u) * self.dt.unsqueeze(-1)

    def g(self, s, z):
        return self.m.diffusion(z, self.a0 + s * self.dt) * self.sq


class NeuralSDE(LatentDynamics):
    stochastic = True

    def __init__(self, d, n_u=1, hidden=128, use_potential=True, diffusion_mode="state_age",
                 solver="euler", n_steps=32, learn_obs_noise=True):
        super().__init__()
        self.field = DriftField(d, n_u, hidden, use_potential)
        self.diff = DiagDiffusion(d, 64, diffusion_mode)
        self.solver, self.n_steps = solver, n_steps
        self.log_r = nn.Parameter(torch.full((d,), -3.0)) if learn_obs_noise else None
        self.cfg = dict(d=d, n_u=n_u, hidden=hidden, use_potential=use_potential,
                        diffusion_mode=diffusion_mode, solver=solver, n_steps=n_steps,
                        learn_obs_noise=learn_obs_noise)

    def drift(self, z, a, u):
        return self.field(z, a, u)

    def diffusion(self, z, a):
        return self.diff(z, a)

    def simulate(self, z0, a0, a1, u, K, n_steps=None, grad=True):
        """Differentiable K-sample simulation, returns (K, B, d)."""
        n_steps = n_steps or self.n_steps
        B, d = z0.shape
        z = self._start(z0, K)
        a0r, dtr, ur = a0.repeat(K), (a1 - a0).repeat(K), u.repeat(K, 1)
        if self.solver == "native":
            return super().simulate(z0, a0, a1, u, K, n_steps, grad)
        sde_type = "stratonovich" if self.solver == "reversible_heun" else "ito"
        sde = _ScaledSDE(self, a0r, dtr, ur, sde_type)
        ts = torch.tensor([0.0, 1.0], device=z0.device)
        kw = {}
        if self.solver == "reversible_heun":
            kw["bm"] = torchsde.BrownianInterval(0.0, 1.0, size=z.shape, device=z.device)
        zs = torchsde.sdeint(sde, z, ts, method=self.solver, dt=1.0 / n_steps, **kw)
        return zs[-1].view(K, B, d)

    def loss(self, batch, K=32, w_energy=0.0, j_penalty=1e-3, sigma_penalty=1e-3):
        z0, z1, a0, a1, u = batch
        S = self.simulate(z0, a0, a1, u, K)
        nll = gaussian_sample_nll(S, z1, self.obs_var)
        loss = nll
        if w_energy > 0:
            Sn = S + torch.randn_like(S) * torch.as_tensor(self.obs_var) ** 0.5
            loss = loss + w_energy * energy_score(Sn, z1).mean()
        # regularisers on random states along the batch
        if j_penalty > 0 and self.field.use_potential:
            j = self.field.nonconservative(z0, a0, u)
            loss = loss + j_penalty * (j ** 2).sum(-1).mean()
        if sigma_penalty > 0:
            loss = loss + sigma_penalty * (self.diffusion(z0, a0) ** 2).sum(-1).mean()
        return loss, {"nll": nll.item()}

    @torch.no_grad()
    def val_loss(self, batch, K=32):
        z0, z1, a0, a1, u = batch
        was = self.training
        self.train()  # include start (obs) noise, as in training
        try:
            S = self.simulate(z0, a0, a1, u, K, grad=False)
        finally:
            self.train(was)
        return gaussian_sample_nll(S, z1, self.obs_var).item()


def energy_score(S, y):
    """S: (K, B, d) samples, y: (B, d). ES = E|X-y| - 0.5 E|X-X'| (lower is better)."""
    K = S.shape[0]
    t1 = (S - y.unsqueeze(0)).norm(dim=-1).mean(0)
    diff = (S.unsqueeze(0) - S.unsqueeze(1)).norm(dim=-1)
    t2 = diff.sum((0, 1)) / (K * (K - 1))
    return t1 - 0.5 * t2
