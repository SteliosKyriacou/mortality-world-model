"""Common interface for latent dynamics models (A, B, B', baselines-in-latent)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    """Subclasses implement drift(z, a, u) and optionally diffusion(z, a).

    sample(): integrates from (z0, a0) to each a1 (per-sample horizons supported by
    integrating in normalised time s in [0, 1] with dz/ds = dt * f). Returns (K, B, d).
    """
    stochastic = False

    def __init__(self):
        super().__init__()
        self.nfe = 0

    # ---------------- observation noise of the latent (encoder error) ----------------
    # Either learned (parameter log_r, set by subclasses) or fixed from an external estimate
    # via set_obs_noise(); with a fixed estimate the start state is also perturbed
    # (errors-in-variables), so both visits' encoder noise is accounted for.
    def set_obs_noise(self, var):
        v = torch.as_tensor(var, dtype=torch.float32)
        self.register_buffer("obs_r2_fixed", v.to(next(self.parameters()).device))
        if getattr(self, "log_r", None) is not None:
            self.log_r.requires_grad_(False)

    @property
    def obs_var(self):
        if getattr(self, "obs_r2_fixed", None) is not None:
            return self.obs_r2_fixed
        if getattr(self, "log_r", None) is not None:
            return torch.exp(2 * self.log_r)
        return 0.0

    def _start(self, z0, K):
        B, d = z0.shape
        z = z0.unsqueeze(0).expand(K, B, d).reshape(K * B, d)
        if self.training and getattr(self, "obs_r2_fixed", None) is not None:
            z = z + torch.randn_like(z) * self.obs_r2_fixed.sqrt()
        return z

    def drift(self, z, a, u):
        raise NotImplementedError

    def diffusion(self, z, a):
        return None

    @torch.no_grad()
    def sample(self, z0, a0, a1, u, n_samples=32, n_steps=40, return_path=False):
        """Native fixed-step Euler(-Maruyama) / RK4 (deterministic models) integrator."""
        B, d = z0.shape
        K = n_samples if self.stochastic else 1
        z = z0.unsqueeze(0).expand(K, B, d).reshape(K * B, d).clone()
        a0r = a0.repeat(K)
        dtr = (a1 - a0).repeat(K)
        ur = u.repeat(K, 1)
        h = 1.0 / n_steps
        path = [z.view(K, B, d)] if return_path else None
        for i in range(n_steps):
            s = i * h
            if self.stochastic:
                a = a0r + s * dtr
                f = self.drift(z, a, ur)
                g = self.diffusion(z, a)
                self.nfe += 1
                z = z + f * dtr[:, None] * h + g * torch.sqrt(dtr.clamp_min(0)[:, None] * h) * torch.randn_like(z)
            else:
                z = self._rk4_step(z, a0r, dtr, ur, s, h)
            if return_path:
                path.append(z.view(K, B, d))
        out = z.view(K, B, d)
        if return_path:
            return out, torch.stack(path, 0)  # (T+1, K, B, d)
        return out

    def _rk4_step(self, z, a0, dt, u, s, h):
        def f(zz, ss):
            self.nfe += 1
            return self.drift(zz, a0 + ss * dt, u) * dt[:, None]
        k1 = f(z, s)
        k2 = f(z + 0.5 * h * k1, s + 0.5 * h)
        k3 = f(z + 0.5 * h * k2, s + 0.5 * h)
        k4 = f(z + h * k3, s + h)
        return z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, z0, a0, a1, u, K, n_steps=32, grad=True):
        """Differentiable native Euler-Maruyama, K samples per start, returns (K, B, d)."""
        B, d = z0.shape
        z = self._start(z0, K)
        a0r, dtr, ur = a0.repeat(K), (a1 - a0).repeat(K), u.repeat(K, 1)
        h = 1.0 / n_steps
        for i in range(n_steps):
            a = a0r + i * h * dtr
            self.nfe += 1
            z = z + self.drift(z, a, ur) * dtr[:, None] * h + \
                self.diffusion(z, a) * torch.sqrt(dtr[:, None] * h) * torch.randn_like(z)
        return z.view(K, B, d)

    @torch.no_grad()
    def predict(self, z0, a0, a1, u, n_samples=32, n_steps=40):
        """Returns (mean, samples)."""
        S = self.sample(z0, a0, a1, u, n_samples, n_steps)
        return S.mean(0), S


def gaussian_sample_nll(S, y, obs_var=0.0):
    """NLL of y under N(mean_K(S), var_K(S) + obs_var), summed over dims, mean over batch."""
    mu = S.mean(0)
    var = S.var(0) + obs_var + 1e-5
    return (0.5 * (y - mu) ** 2 / var + 0.5 * torch.log(var)).sum(-1).mean()
