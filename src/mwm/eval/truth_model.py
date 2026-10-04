"""The synthetic ground truth wrapped in the model interfaces, so every hallmark test can be run
on the *true* dynamics in the *true* latent (sanity check of the tests themselves: they must
detect planted hallmarks on hallmarks-ON truth and not on hallmarks-OFF truth)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data import synthetic as syn
from ..dynamics.base import LatentDynamics


class TrueDynamics(LatentDynamics):
    stochastic = True
    steps_per_year = 20  # the true double well is stiff away from its minima

    def __init__(self, cfg: syn.SynthConfig):
        super().__init__()
        self.c = cfg
        self.dummy = nn.Parameter(torch.zeros(1))

    def drift(self, z, a, u):
        c = self.c
        u = u.reshape(len(z), -1)[:, 0]
        f = torch.zeros_like(z)
        if c.on("irreversibility"):
            f[:, 0] = c.beta0
        else:
            f[:, 0] = -c.k_rev * (z[:, 0] - c.c_rev)
        if c.on("inflammaging"):
            tgt = c.gamma * c.w * nn.functional.softplus((z[:, 0] - c.theta) / c.w)
        else:
            tgt = c.c_lin * (z[:, 0] - c.z0_ref)
        f[:, 1] = -c.k1 * (z[:, 1] - tgt)
        kap = c.kappa if c.on("nutrient") else 0.0
        f[:, 2] = c.beta2 * (1 - kap * u)
        if not c.on("irreversibility"):
            f[:, 2] = f[:, 2] - c.k_rev * z[:, 2]
        f[:, 3] = -c.k_rot * z[:, 3] - c.omega * z[:, 4]
        f[:, 4] = -c.k_rot * z[:, 4] + c.omega * z[:, 3]
        if c.on("frailty_basin"):
            f[:, 5] = -4 * c.h_frail * z[:, 5] * (z[:, 5] ** 2 - 1) + c.tilt_frail
        else:
            f[:, 5] = -c.k5_single * (z[:, 5] + 1)
        f[:, 6] = -c.k_ou[0] * z[:, 6]
        f[:, 7] = -c.k_ou[1] * z[:, 7]
        return f

    def diffusion(self, z, a):
        s = torch.as_tensor(syn.diffusion_scale(a.detach().cpu().numpy(), self.c), dtype=z.dtype, device=z.device)
        return s[:, None] * torch.as_tensor(self.c.sigma_base, dtype=z.dtype, device=z.device)[None]


class TrueBundle(nn.Module):
    """decode(z*) -> [clinical (standardised scale of the generator), omics]; log_hazard(z*)."""

    def __init__(self, cfg, L):
        super().__init__()
        self.c = cfg
        f = lambda k: torch.tensor(L[k], dtype=torch.float32)
        self.register_buffer("Wc", f("Wc"))
        self.register_buffer("quad", f("quad"))
        self.register_buffer("Wg", f("Wg"))
        self.register_buffer("rm", f("ref_mean"))
        self.register_buffer("rs", f("ref_scale"))
        self.prim = torch.argmax(self.Wc.abs(), 1)

    def decode(self, z):
        zt = (z - self.rm) / self.rs
        xc = zt @ self.Wc.T + self.quad * zt[:, self.prim.to(z.device)] ** 2
        return torch.cat([xc, zt @ self.Wg.T], -1)

    def log_hazard(self, z, age=None):
        c = self.c
        lh = c.alpha + c.b0 * z[:, 0] + c.b2 * z[:, 2]
        if c.on("inflammaging"):
            lh = lh + c.b1 * z[:, 1]
        if c.on("frailty_basin"):
            lh = lh + c.b5 * torch.sigmoid(3 * z[:, 5])
        return lh
