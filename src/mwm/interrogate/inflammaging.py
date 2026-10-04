"""Chronic inflammation (inflammaging): late-life non-linear rise of decoded inflammatory
markers in model rollouts from middle age, vs the raw held-out cross-section.

Statistic: acceleration ratio = slope(late window) / slope(early window) of the decoded
inflammatory composite (mean of standardised inflammatory features) along rollouts started
from held-out people aged 40-50, run to age 90 with u = 0.
"""
from __future__ import annotations

import numpy as np

from .common import bootstrap_ci, decode_np, rollout_mean_path

EARLY, LATE = (45.0, 60.0), (72.0, 87.0)


def _slope(ages, y, lo_hi):
    m = (ages >= lo_hi[0]) & (ages <= lo_hi[1])
    return np.polyfit(ages[m], y[m], 1)[0]


def accel_ratio(ages, y):
    e, l = _slope(ages, y, EARLY), _slope(ages, y, LATE)
    return l / e if abs(e) > 1e-6 else (np.sign(l) * np.inf if l != 0 else 1.0), e, l


def composite(Xdec, feat_idx):
    return Xdec[..., feat_idx].mean(-1)


def raw_cross_section_curve(Xs, ages, feat_idx, bins=np.arange(40, 92, 2.5)):
    """Observed (standardised) composite by age bin (NaN-mean over available markers)."""
    c = np.nanmean(Xs[:, feat_idx], 1)
    centers, vals = [], []
    for lo in bins[:-1]:
        m = (ages >= lo) & (ages < lo + 2.5) & np.isfinite(c)
        if m.sum() >= 20:
            centers.append(lo + 1.25)
            vals.append(c[m].mean())
    return np.array(centers), np.array(vals)


def test_inflammaging(model, bundle, Z, A, feat_idx, Xs_obs=None, A_obs=None, device="cuda",
                      start_window=(40.0, 50.0), end_age=90.0, n_samples=16, max_starts=400,
                      seed=0, threshold=1.5):
    rng = np.random.default_rng(seed)
    sel = np.nonzero((A >= start_window[0]) & (A < start_window[1]))[0]
    sel = rng.choice(sel, min(max_starts, len(sel)), replace=False)
    a0 = A[sel]
    horizon = end_age - a0.min()
    # integrate everyone from their own age for the same horizon, then read on an age grid
    grid, mean_path, _ = rollout_mean_path(model, Z[sel], a0, horizon,
                                           np.zeros(len(sel)), n_samples, 4, device)
    Xdec = decode_np(bundle, mean_path)                       # (T+1, N, F)
    comp = composite(Xdec, feat_idx)                          # (T+1, N)
    ages_abs = a0[None, :] + grid[:, None]                    # (T+1, N)
    age_grid = np.arange(45.0, end_age + 0.01, 0.5)

    def curve(idx):
        vals = np.zeros(len(age_grid))
        cnt = np.zeros(len(age_grid))
        for i in idx:
            v = np.interp(age_grid, ages_abs[:, i], comp[:, i], left=np.nan, right=np.nan)
            ok = np.isfinite(v)
            vals[ok] += v[ok]
            cnt[ok] += 1
        return vals / np.maximum(cnt, 1)

    full = curve(np.arange(len(sel)))
    ratio, e, l = accel_ratio(age_grid, full)
    ci = bootstrap_ci(lambda idx: accel_ratio(age_grid, curve(idx))[0], len(sel), reps=100, seed=seed)
    out = {"statistic": float(ratio), "slope_early": float(e), "slope_late": float(l), "ci95": ci,
           "age_grid": age_grid.tolist(), "curve": full.tolist(),
           "detected": bool(ci[0] > 1.0 and ratio > threshold),
           "rule": f"accel ratio late{LATE}/early{EARLY} > {threshold} and bootstrap 95% CI lower > 1"}
    if Xs_obs is not None:
        c_age, c_val = raw_cross_section_curve(Xs_obs, A_obs, feat_idx)
        r_raw = accel_ratio(c_age, c_val)[0]
        model_at = np.interp(c_age, age_grid, full)
        out.update({"raw_ratio": float(r_raw), "raw_age": c_age.tolist(), "raw_curve": c_val.tolist(),
                    # agreement with held-out older people: correlation of curve shapes (>=60)
                    "match_corr_60plus": float(np.corrcoef(model_at[c_age >= 60], c_val[c_age >= 60])[0, 1])
                    if (c_age >= 60).sum() > 3 else float("nan")})
    return out
