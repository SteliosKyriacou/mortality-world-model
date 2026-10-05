"""Drift field with potential decomposition: F(z, a, u) = -grad V(z) + J(z, a, u)."""
from __future__ import annotations

import torch
import torch.nn as nn


def age_norm(a):
    return (a - 60.0) / 10.0


def mlp(i, o, h=128, n=2, act=nn.SiLU):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class DriftField(nn.Module):
    """-grad V_psi(z) + J_theta(z, age, u).

    use_potential=False -> plain MLP drift (ablation / null for potential-based tests).
    V depends on z only (a landscape); J carries age, intervention and the
    non-conservative (rotational) part. `j_penalty` (applied by trainers) keeps J small so
    that V explains as much of the drift as it can.
    """

    def __init__(self, d: int, n_u: int = 1, hidden: int = 128, use_potential: bool = True,
                 use_age: bool = True):
        super().__init__()
        self.d, self.n_u, self.use_potential, self.use_age = d, n_u, use_potential, use_age
        self.J = mlp(d + 1 + n_u, d, hidden)
        if use_potential:
            self.V = mlp(d, 1, hidden)

    def potential(self, z):
        if not self.use_potential:
            return torch.zeros(z.shape[:-1], device=z.device)
        return self.V(z).squeeze(-1)

    def grad_V(self, z):
        create = torch.is_grad_enabled() and self.training
        with torch.enable_grad():
            zz = z if z.requires_grad else z.detach().requires_grad_(True)
            V = self.V(zz).sum()
            g = torch.autograd.grad(V, zz, create_graph=create or z.requires_grad)[0]
        return g

    def nonconservative(self, z, a, u):
        a_in = age_norm(a) if self.use_age else torch.zeros_like(a)
        return self.J(torch.cat([z, a_in.unsqueeze(-1), u], -1))

    def forward(self, z, a, u):
        j = self.nonconservative(z, a, u)
        if self.use_potential:
            return -self.grad_V(z) + j
        return j


class DiagDiffusion(nn.Module):
    """sigma(z, a) diagonal, positive. mode: 'state_age' | 'state' | 'age' | 'constant'.
    'state' ignores age entirely (autonomous dynamics)."""

    def __init__(self, d: int, hidden: int = 64, mode: str = "state_age", init: float = 0.2):
        super().__init__()
        self.mode, self.d = mode, d
        inv = torch.log(torch.expm1(torch.tensor(init)))
        if mode == "constant":
            self.p = nn.Parameter(torch.full((d,), float(inv)))
        else:
            i = {"state_age": d + 1, "state": d, "age": 1}[mode]
            self.net = mlp(i, d, hidden)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, float(inv))

    def forward(self, z, a):
        if self.mode == "constant":
            return nn.functional.softplus(self.p).expand(z.shape[0], -1) + 1e-4
        if self.mode == "state":
            return nn.functional.softplus(self.net(z)) + 1e-4
        x = age_norm(a).unsqueeze(-1)
        if self.mode == "state_age":
            x = torch.cat([z, x], -1)
        return nn.functional.softplus(self.net(x)) + 1e-4
