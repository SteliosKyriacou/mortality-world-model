"""Decoder and hazard heads on the latent z."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureDecoder(nn.Module):
    """z -> per-feature Gaussian mean; per-feature learned log-sigma (homoscedastic)."""

    def __init__(self, d_latent: int, n_features: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, n_features))
        self.log_sigma = nn.Parameter(torch.full((n_features,), -0.5))

    def forward(self, z):
        return self.net(z)

    def nll(self, z, x, m):
        mu = self(z)
        ls = self.log_sigma.clamp(-4, 3)
        nll = 0.5 * ((x - mu) / ls.exp()) ** 2 + ls
        return (nll * m).sum() / m.sum().clamp_min(1)


class HazardHead(nn.Module):
    """log lambda(z[, age]); trained with a piecewise-exponential likelihood with z held
    constant within each visit interval."""

    def __init__(self, d_latent: int, hidden: int = 64, use_age: bool = False):
        super().__init__()
        self.use_age = use_age
        self.net = nn.Sequential(nn.Linear(d_latent, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.age_coef = nn.Parameter(torch.zeros(1)) if use_age else None
        self.bias = nn.Parameter(torch.tensor([-5.0]))

    def forward(self, z, age=None):
        lh = self.net(z).squeeze(-1) + self.bias
        if self.use_age and age is not None:
            lh = lh + self.age_coef * (age - 60.0) / 10.0
        return lh

    def nll(self, z, t0, t1, died, age=None):
        lh = self(z, age)
        expo = (t1 - t0).clamp_min(1e-3)
        return (torch.exp(lh) * expo - died * lh).mean()
