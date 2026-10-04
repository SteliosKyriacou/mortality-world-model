"""Deregulated nutrient sensing: what does the intervention u do to the learned field?

* Delta v = drift(z, a, u=1) - drift(z, a, u=0) at held-out states, decoded to clinical
  feature space through the decoder Jacobian. Statistic 1: metabolic-feature share of
  |decoded Delta v|; null: random latent directions of equal norm (decoded the same way).
* Statistic 2: 10-year rollout difference (u=1 minus u=0) in the decoded metabolic composite,
  bootstrap CI over people; slowing => negative.
"""
from __future__ import annotations

import numpy as np

from .common import bootstrap_ci, decode_np, drift_at, mean_decoder_jacobian, rollout_mean_path


def test_intervention(model, bundle, Z, A, metab_idx, clinical_slice, device="cuda", n_null=1000,
                      horizon=10.0, max_states=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(max_states, len(Z)), replace=False)
    Zs, As = Z[idx], A[idx]
    dv = drift_at(model, Zs, As, np.ones(len(idx)), device) - drift_at(model, Zs, As, np.zeros(len(idx)), device)
    Jdec = mean_decoder_jacobian(bundle, Z, device)[clinical_slice]   # (Fc, d)
    dx = np.abs(dv.mean(0) @ Jdec.T)       # decode the systematic (state-averaged) effect, (Fc,)
    share = dx[metab_idx].sum() / dx.sum()
    mean_norm = np.linalg.norm(dv.mean(0))
    null = []
    for _ in range(n_null):
        r = rng.normal(size=Z.shape[1])
        r = r / np.linalg.norm(r) * mean_norm
        x = np.abs(Jdec @ r)
        null.append(x[metab_idx].sum() / x.sum())
    null = np.array(null)
    p_share = float((np.sum(null >= share) + 1) / (n_null + 1)) if np.isfinite(share) else 1.0
    # rollouts
    import torch
    torch.manual_seed(seed)   # common random numbers for the u=1 vs u=0 comparison
    grid, m1, _ = rollout_mean_path(model, Zs, As, horizon, np.ones(len(idx)), 16, 4, device)
    torch.manual_seed(seed)
    _, m0, _ = rollout_mean_path(model, Zs, As, horizon, np.zeros(len(idx)), 16, 4, device)
    x1 = decode_np(bundle, m1[-1])[:, clinical_slice][:, metab_idx]
    x0 = decode_np(bundle, m0[-1])[:, clinical_slice][:, metab_idx]
    # sign-align: metabolic composite in the direction of "higher = worse" uses feature signs of
    # the population drift (features that increase with age under u=0 count +)
    start = decode_np(bundle, Zs)[:, clinical_slice][:, metab_idx]
    sgn = np.sign((x0 - start).mean(0))
    eff = ((x1 - x0) * sgn).mean(1)
    ci = bootstrap_ci(lambda i: eff[i].mean(), len(eff), seed=seed)
    return {"statistic": float(share), "null_share_mean": float(null.mean()),
            "null_share_q95": float(np.quantile(null, 0.95)), "p_share": p_share,
            "rollout_effect_10y": float(eff.mean()), "rollout_ci95": ci,
            "delta_v_norm": float(mean_norm),
            "detected": bool(p_share < 0.05 and ci[1] < 0),
            "rule": "metabolic share of decoded Delta v above random-direction null (p<0.05) AND "
                    "10y rollout slows age-wise metabolic change (CI upper < 0)"}
