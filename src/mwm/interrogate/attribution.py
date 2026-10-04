"""Altered intercellular communication: integrated-gradients attribution of the hazard (through
the encoder) to the input clinical features, and how the inflammatory share changes with age.

Statistic: slope (per decade) of the inflammatory share of |attribution| vs age, bootstrap CI.
Control: the same on the baseline Cox model (beta_j * (x_j - mean_j)).
"""
from __future__ import annotations

import numpy as np
import torch

from .common import T, bootstrap_ci


def integrated_gradients(bundle, X, M, steps=32, device="cuda", batch=512):
    """X: standardised (NaN-free), M: observed mask. Baseline = 0 (train mean), only observed
    entries are interpolated; returns (N, F) attributions of log-hazard."""
    out = []
    for s in range(0, len(X), batch):
        x, m = T(X[s:s + batch], device), torch.as_tensor(M[s:s + batch], device=device)
        tot = torch.zeros_like(x)
        for k in range(1, steps + 1):
            xi = (x * k / steps).requires_grad_(True)
            lh = bundle.log_hazard(bundle.encode(xi * m, m))
            g, = torch.autograd.grad(lh.sum(), xi)
            tot += g
        out.append((x * tot / steps * m).detach().cpu().numpy())
    return np.concatenate(out)


def share_slope(attr, ages, infl_idx):
    a = np.abs(attr)
    share = a[:, infl_idx].sum(1) / np.maximum(a.sum(1), 1e-12)
    return np.polyfit(ages, share, 1)[0] * 10, share


def test_attribution(bundle, Xs, ages, infl_idx, n_clinical, cox_attr=None, device="cuda",
                     max_n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(max_n, len(Xs)), replace=False)
    X = Xs[idx]
    M = ~np.isnan(X)
    M[:, n_clinical:] = False  # attribute to clinical inputs only (omics off -> comparable)
    attr = integrated_gradients(bundle, np.nan_to_num(X), M, device=device)[:, :n_clinical]
    a = ages[idx]
    slope, share = share_slope(attr, a, infl_idx)
    ci = bootstrap_ci(lambda i: share_slope(attr[i], a[i], infl_idx)[0], len(a), seed=seed)
    out = {"statistic": float(slope), "ci95": ci, "mean_share": float(share.mean()),
           "share_by_age": {f"{lo}-{lo+10}": float(share[(a >= lo) & (a < lo + 10)].mean())
                            for lo in (40, 50, 60, 70, 80) if ((a >= lo) & (a < lo + 10)).sum() > 20},
           "detected": bool(ci[0] > 0),
           "rule": "inflammatory share of |IG attribution| increases with age (slope CI lower > 0)"}
    if cox_attr is not None:
        cs, _ = share_slope(cox_attr[idx][:, :n_clinical], a, infl_idx)
        out["cox_control_slope"] = float(cs)
        out["exceeds_cox"] = bool(slope > cs)
    return out
