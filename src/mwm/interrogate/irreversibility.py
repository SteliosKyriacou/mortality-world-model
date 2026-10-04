"""Irreversibility: with u = 0, is there an "arrow of aging" -- a latent direction w along which
the learned field never points backwards -- and do people never move back along it?

* w is fitted as a max-margin direction (linear SVM through the origin separating {f_i} from
  {-f_i}) on half of the held-out states; statistic 1 = fraction of the OTHER half with
  w . f <= 0 (states whose drift points "younger").
* statistic 2 = fraction of 10-year mean-path rollouts whose displacement along w is negative.
* Descriptive: the same with a linear latent-age probe (age ~ z), and for potential models
  the fraction of observed held-out visit pairs where V decreases (downhill = older).

Ground truth: hallmarks-ON has a monotone axis (age core, constant positive drift) -> both
fractions ~0; hallmarks-OFF has only mean-reverting axes -> large fractions.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.svm import LinearSVC

from .common import T, drift_at, rollout_mean_path


def arrow_direction(F: np.ndarray, C: float = 1.0) -> np.ndarray:
    s = np.median(np.linalg.norm(F, axis=1)) + 1e-12
    X = np.concatenate([F, -F]) / s
    y = np.r_[np.ones(len(F)), -np.ones(len(F))]
    w = LinearSVC(fit_intercept=False, C=C, max_iter=20000).fit(X, y).coef_[0]
    return w / np.linalg.norm(w)


def test_irreversibility(model, probe, Z, A, pairs=None, device="cuda", horizon=10.0,
                         max_states=1000, seed=0, state_threshold=0.10, rollout_threshold=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(max_states, len(Z)), replace=False)
    f = drift_at(model, Z[idx], A[idx], np.zeros(len(idx)), device)
    half = len(idx) // 2
    w = arrow_direction(f[:half])
    frac_states = float((f[half:] @ w <= 0).mean())
    grid, mp, _ = rollout_mean_path(model, Z[idx], A[idx], horizon, np.zeros(len(idx)), 16, 4, device)
    disp = mp[-1] - mp[0]
    frac_roll = float((disp[half:] @ w < 0).mean())
    vel_probe = f @ probe.w
    la0, la1 = probe(mp[0]), probe(mp[-1])
    out = {"statistic": frac_states, "frac_rollout_backwards": frac_roll,
           "arrow_dir_cos_mean_drift": float(w @ f.mean(0) / np.linalg.norm(f.mean(0))),
           "probe_frac_neg_velocity": float((vel_probe < 0).mean()),
           "probe_frac_rollout_younger": float((la1 < la0).mean()),
           "latent_age_velocity_mean": float(vel_probe.mean())}
    if pairs is not None and getattr(model, "field", None) is not None and model.field.use_potential:
        with torch.no_grad():
            V0 = model.field.potential(T(pairs["z0"], device)).cpu().numpy()
            V1 = model.field.potential(T(pairs["z1"], device)).cpu().numpy()
        out["frac_pairs_V_downhill"] = float((V1 < V0).mean())
        out["corr_V_age"] = float(np.corrcoef(V0, pairs["a0"])[0, 1])
    out["detected"] = bool(frac_states < state_threshold and frac_roll < rollout_threshold)
    out["rule"] = (f"max-margin arrow direction w (fit on half the states): fraction of other-half "
                   f"states with w.f<=0 < {state_threshold} AND fraction of 10y mean rollouts moving "
                   f"backwards along w < {rollout_threshold}")
    return out
