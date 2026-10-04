"""Attractor / basin structure transverse to the aging direction.

An aging field has no global fixed points (it keeps drifting along the aging direction), so
we look for attractors of the field restricted to the hyperplane orthogonal to the aging
("arrow") direction d (max-margin direction of the drift, see irreversibility.py): states are placed on a common slice (their projection on d set to the
held-out median) at a reference age, then flowed with dz/dt = P_perp v(z, a_ref, 0) until
convergence. End points are clustered; each cluster is an attractor with a hazard (hazard
head at its centroid). Every held-out person is assigned to the basin their own baseline
state flows into; death rates are compared between the highest- and lowest-hazard basins.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import fisher_exact
from sklearn.cluster import AgglomerativeClustering

from .common import T, drift_at, log_hazard_np


@torch.no_grad()
def _flow_perp(model, Z, a_ref, d, steps=400, dt=0.25, device="cuda"):
    z = T(Z, device)
    dd = T(d, device)
    a = torch.full((len(Z),), float(a_ref), device=device)
    u = torch.zeros(len(Z), 1, device=device)
    for _ in range(steps):
        v = model.drift(z, a, u)
        v = v - (v @ dd)[:, None] * dd[None]
        z = z + dt * v
    return z.cpu().numpy()


def test_attractors(model, bundle, Z, A, event, a_ref=70.0, device="cuda", dist_threshold=0.5,
                    min_frac=0.03, seed=0, max_states=2000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(max_states, len(Z)), replace=False)
    f = drift_at(model, Z[idx], A[idx], np.zeros(len(idx)), device)
    # aging direction = max-margin "arrow" direction (robust to sign-changing drift components,
    # e.g. within-basin restoring forces, which the plain mean drift direction picks up)
    from .irreversibility import arrow_direction
    d = arrow_direction(f)
    proj = Z[idx] @ d
    c = np.median(proj)
    Zs = Z[idx] + (c - proj)[:, None] * d[None]
    E = _flow_perp(model, Zs, a_ref, d, device=device)
    resid = np.linalg.norm(drift_at(model, E, np.full(len(E), a_ref), np.zeros(len(E)), device) -
                           (drift_at(model, E, np.full(len(E), a_ref), np.zeros(len(E)), device) @ d)[:, None] * d, axis=1)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_threshold,
                                  linkage="single").fit_predict(E)
    labs, counts = np.unique(lab, return_counts=True)
    keep = labs[counts >= min_frac * len(E)]
    cents = np.stack([E[lab == k].mean(0) for k in keep]) if len(keep) else np.zeros((0, Z.shape[1]))
    haz = log_hazard_np(bundle, cents) if len(keep) else np.array([])
    out = {"n_attractors": int(len(keep)), "attractor_sizes": [int((lab == k).sum()) for k in keep],
           "attractor_log_hazard": haz.round(3).tolist(), "median_residual_speed": float(np.median(resid))}
    if len(keep) >= 2:
        # basin of each held-out person: nearest attractor of its flowed state
        bas = np.argmin(((E[:, None, :] - cents[None]) ** 2).sum(-1), 1)
        # basin hazard = mean model log-hazard of its members' own (unsliced) states; the
        # hazard head evaluated at slice centroids extrapolates and proved unreliable
        lh_members = log_hazard_np(bundle, Z[idx])
        haz_b = np.array([lh_members[bas == k].mean() if (bas == k).any() else -np.inf
                          for k in range(len(keep))])
        out["basin_mean_member_log_hazard"] = haz_b.round(3).tolist()
        hi, lo = int(np.argmax(haz_b)), int(np.argmin(haz_b))
        ev = event[idx]
        a_hi, a_lo = ev[bas == hi], ev[bas == lo]
        tab = [[a_hi.sum(), len(a_hi) - a_hi.sum()], [a_lo.sum(), len(a_lo) - a_lo.sum()]]
        orr, p = fisher_exact(tab, alternative="greater")
        out.update({"death_rate_high_basin": float(a_hi.mean()) if len(a_hi) else float("nan"),
                    "death_rate_low_basin": float(a_lo.mean()) if len(a_lo) else float("nan"),
                    "odds_ratio": float(orr), "p_fisher": float(p)})
        out["detected"] = bool(p < 0.05)
    else:
        out["detected"] = False
    out["statistic"] = out["n_attractors"]
    out["rule"] = ">=2 attractors (>=3% of states each) transverse to the aging direction AND " \
                  "held-out deaths over-represented in the basin with the highest mean model " \
                  "hazard of its members (one-sided Fisher p<0.05)"
    return out
