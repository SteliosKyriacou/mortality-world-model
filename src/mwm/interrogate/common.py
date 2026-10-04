"""Shared helpers for the hallmark interrogation toolkit (PLAN.md §7).

Every test returns a dict with at least: `statistic`, `null` (summary of the null / control
distribution), `p_value` or CI, `detected` (bool by the pre-registered decision rule) and
`rule` (text of the rule)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV


def T(x, device="cuda"):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def bootstrap_ci(fn, n: int, reps: int = 200, seed: int = 0, alpha: float = 0.05):
    """fn(idx) -> scalar; resample person indices."""
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(reps)])
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


@torch.no_grad()
def drift_at(model, Z, A, U, device="cuda", batch=4096):
    out = []
    for s in range(0, len(Z), batch):
        u = T(U[s:s + batch], device).reshape(len(Z[s:s + batch]), -1)
        out.append(model.drift(T(Z[s:s + batch], device), T(A[s:s + batch], device), u).cpu().numpy())
    return np.concatenate(out)


def decoder_jacobian(bundle, Z, device="cuda", batch=256):
    """d decode(z) / dz at each state -> (N, F, d)."""
    out = []
    for s in range(0, len(Z), batch):
        z = T(Z[s:s + batch], device).requires_grad_(True)
        x = bundle.decode(z)
        J = torch.stack([torch.autograd.grad(x[:, j].sum(), z, retain_graph=True)[0]
                         for j in range(x.shape[1])], 1)
        out.append(J.detach().cpu().numpy())
    return np.concatenate(out)


def mean_decoder_jacobian(bundle, Z, device="cuda", max_states=512, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(max_states, len(Z)), replace=False)
    return decoder_jacobian(bundle, Z[idx], device).mean(0)  # (F, d)


def drift_jacobian(model, Z, A, U, device="cuda"):
    """d drift / dz at each state -> (N, d, d) (works with -grad V terms: double backprop)."""
    z = T(Z, device).requires_grad_(True)
    a, u = T(A, device), T(U, device).reshape(len(Z), -1)
    model.eval()
    with torch.enable_grad():
        f = model.drift(z, a, u)
        J = torch.stack([torch.autograd.grad(f[:, i].sum(), z, retain_graph=True)[0]
                         for i in range(f.shape[1])], 1)
    return J.detach().cpu().numpy()


class LatentAgeProbe:
    """Linear 'latent age' readout fit on train visits: age ~ w.z + b."""

    def fit(self, Z, age):
        self.r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z, age)
        self.w = self.r.coef_
        return self

    def __call__(self, Z):
        return self.r.predict(Z)


def rollout_mean_path(model, Z0, A0, horizon, U, n_samples=32, steps_per_year=4, device="cuda"):
    """Returns ages grid (n,T+1) relative offsets and latent paths mean over samples (T+1,N,d)."""
    steps_per_year = getattr(model, "steps_per_year", steps_per_year)
    n_steps = int(np.ceil(horizon * steps_per_year))
    z0, a0 = T(Z0, device), T(A0, device)
    u = T(U, device).reshape(len(Z0), -1)
    _, path = model.sample(z0, a0, a0 + horizon, u, n_samples=n_samples, n_steps=n_steps,
                           return_path=True)
    grid = np.linspace(0, horizon, n_steps + 1)
    return grid, path.mean(1).cpu().numpy(), path.cpu().numpy()


@torch.no_grad()
def decode_np(bundle, Z, device="cuda", batch=8192):
    sh = Z.shape
    Zf = Z.reshape(-1, sh[-1])
    out = [bundle.decode(T(Zf[s:s + batch], device)).cpu().numpy() for s in range(0, len(Zf), batch)]
    return np.concatenate(out).reshape(sh[:-1] + (-1,))


@torch.no_grad()
def log_hazard_np(bundle, Z, A=None, device="cuda", batch=8192):
    sh = Z.shape
    Zf = Z.reshape(-1, sh[-1])
    Af = None if A is None else np.asarray(A).reshape(-1)
    out = []
    for s in range(0, len(Zf), batch):
        a = None if Af is None else T(Af[s:s + batch], device)
        out.append(bundle.log_hazard(T(Zf[s:s + batch], device), a).cpu().numpy())
    return np.concatenate(out).reshape(sh[:-1])
