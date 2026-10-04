"""Loss of homeostasis / entropy: does the model's noise grow with age?

* A / B' (stochastic): instantaneous diffusion trace tr(Sigma Sigma^T)(z, a) at held-out
  states. Statistic = ratio of mean trace when the *same states* are evaluated at age 80 vs
  age 45 (pure age effect), plus the slope of log-trace on age over states at their own age.
  Null: age-permutation (states paired with shuffled ages -> slope distribution).
* Ensemble dispersion: total variance of 5-year-ahead samples for starts aged 65-72 vs 43-50.
* Observed: between-person variance growth of the latent in held-out data (descriptive).
"""
from __future__ import annotations

import numpy as np
import torch

from .common import T


@torch.no_grad()
def _trace(model, Z, A, device):
    g = model.diffusion(T(Z, device), T(A, device))
    return (g ** 2).sum(-1).cpu().numpy()


def test_dispersion(model, Z, A, U, device="cuda", n_perm=200, seed=0, ratio_threshold=1.2,
                    n_samples=64):
    rng = np.random.default_rng(seed)
    if not getattr(model, "stochastic", False):
        return {"detected": False, "statistic": float("nan"),
                "note": "deterministic model: cannot express dispersion",
                "rule": "n/a"}
    tr_own = _trace(model, Z, A, device)
    lt = np.log(tr_own)
    slope = np.polyfit(A, lt, 1)[0] * 10  # per decade
    perm = np.array([np.polyfit(rng.permutation(A), np.log(_trace(model, Z, rng.permutation(A), device)), 1)[0] * 10
                     for _ in range(n_perm // 10)])
    # age-only effect at fixed states
    r_age = float(_trace(model, Z, np.full(len(Z), 80.0), device).mean() /
                  _trace(model, Z, np.full(len(Z), 45.0), device).mean())
    # bootstrap CI of the fixed-state ratio
    t80, t45 = _trace(model, Z, np.full(len(Z), 80.0), device), _trace(model, Z, np.full(len(Z), 45.0), device)
    boots = [t80[i].mean() / t45[i].mean() for i in (rng.integers(0, len(Z), len(Z)) for _ in range(200))]
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    # ensemble dispersion over 5 years
    ens = {}
    for name, (lo, hi) in {"young": (43, 50), "old": (65, 72)}.items():
        idx = np.nonzero((A >= lo) & (A < hi))[0][:300]
        if len(idx) < 10:
            ens[name] = float("nan")
            continue
        z0, a0 = T(Z[idx], device), T(A[idx], device)
        u = T(np.zeros((len(idx), 1)), device)
        S = model.sample(z0, a0, a0 + 5.0, u, n_samples=n_samples, n_steps=int(5 * getattr(model, 'steps_per_year', 4))).cpu().numpy()
        ens[name] = float(S.var(0).sum(-1).mean())
    p_perm = float((np.abs(perm) >= abs(slope)).mean()) if slope > 0 else 1.0
    return {"statistic": r_age, "ci95": ci, "log_trace_slope_per_decade": float(slope),
            "perm_slopes_sd": float(perm.std()), "p_perm": p_perm,
            "ensemble_var_5y": ens,
            "ensemble_ratio": ens["old"] / ens["young"] if ens.get("young") else float("nan"),
            "detected": bool(r_age > ratio_threshold and ci[0] > 1.0 and slope > 0 and p_perm < 0.05),
            "rule": f"trace ratio (age 80 vs 45, fixed states) > {ratio_threshold}, CI lower > 1, "
                    f"slope>0 with age-permutation p<0.05"}


def observed_dispersion(Z, A, bins=((40, 50), (50, 60), (60, 70), (70, 80))):
    """Between-person total latent variance by age band in held-out data (descriptive)."""
    return {f"{lo}-{hi}": float(Z[(A >= lo) & (A < hi)].var(0).sum()) for lo, hi in bins
            if ((A >= lo) & (A < hi)).sum() > 20}
