"""Forecast, uncertainty and survival metrics (PLAN.md §5.1)."""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ point forecasts
def masked_mae_rmse(pred: np.ndarray, y: np.ndarray):
    """pred, y: (N, F); NaN in y = unobserved. Returns per-feature MAE/RMSE and overall."""
    m = ~np.isnan(y)
    e = np.where(m, pred - np.nan_to_num(y), np.nan)
    mae = np.nanmean(np.abs(e), 0)
    rmse = np.sqrt(np.nanmean(e ** 2, 0))
    return {"mae": float(np.nanmean(np.abs(e))), "rmse": float(np.sqrt(np.nanmean(e ** 2))),
            "mae_per_feature": mae, "rmse_per_feature": rmse}


def latent_mse(pred, z1):
    return float(((pred - z1) ** 2).sum(-1).mean())


# ------------------------------------------------------------------ probabilistic
def crps_samples(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S: (K, N, F) samples; y: (N, F) with NaN -> returns (N, F) CRPS (NaN where y missing).
    Uses the sorted-sample identity: E|X-y| - 0.5 E|X-X'|."""
    K = S.shape[0]
    t1 = np.abs(S - y[None]).mean(0)
    Ss = np.sort(S, axis=0)
    w = (2 * np.arange(1, K + 1) - K - 1).reshape(-1, *([1] * (S.ndim - 1)))
    t2 = (w * Ss).sum(0) / (K * K)
    return t1 - t2


def energy_score_np(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S: (K, N, d), y: (N, d) -> (N,)"""
    K = S.shape[0]
    t1 = np.linalg.norm(S - y[None], axis=-1).mean(0)
    t2 = np.zeros(S.shape[1])
    for i in range(K):
        t2 += np.linalg.norm(S[i][None] - S, axis=-1).sum(0)
    return t1 - 0.5 * t2 / (K * (K - 1))


def interval_coverage(S: np.ndarray, y: np.ndarray, levels=(0.5, 0.9)) -> dict:
    out = {}
    m = ~np.isnan(y)
    for lv in levels:
        lo = np.quantile(S, (1 - lv) / 2, axis=0)
        hi = np.quantile(S, 1 - (1 - lv) / 2, axis=0)
        inside = (y >= lo) & (y <= hi)
        out[f"cov{int(lv*100)}"] = float(inside[m].mean())
    return out


def gaussian_samples(mu: np.ndarray, sd: np.ndarray, K: int, rng) -> np.ndarray:
    return mu[None] + sd[None] * rng.normal(size=(K,) + mu.shape)


# ------------------------------------------------------------------ survival
def survival_metrics(time: np.ndarray, event: np.ndarray, risk: np.ndarray,
                     train_time: np.ndarray, train_event: np.ndarray,
                     surv_fn: np.ndarray | None = None, grid: np.ndarray | None = None,
                     horizons=(5.0, 10.0)) -> dict:
    """risk: higher = worse. surv_fn: (N, len(grid)) survival probabilities (for IBS)."""
    from lifelines.utils import concordance_index
    from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
    from sksurv.util import Surv
    out = {"c_harrell": float(concordance_index(time, -risk, event))}
    ytr = Surv.from_arrays(train_event.astype(bool), train_time)
    yte = Surv.from_arrays(event.astype(bool), time)
    tau = min(np.quantile(time[event == 1], 0.95) if event.sum() else time.max(),
              train_time.max() - 1e-3)
    try:
        out["c_uno"] = float(concordance_index_ipcw(ytr, yte, risk, tau=tau)[0])
    except Exception as e:  # pragma: no cover
        out["c_uno"] = float("nan")
    hz = [h for h in horizons if h < min(time.max(), train_time.max())]
    if hz:
        try:
            auc, _ = cumulative_dynamic_auc(ytr, yte, risk, np.asarray(hz))
            for h, a in zip(hz, auc):
                out[f"auc_{int(h)}y"] = float(a)
        except Exception:
            pass
    if surv_fn is not None and grid is not None:
        lo, hi = max(grid.min(), 0.5), min(time[event == 1].max(), train_time.max(), grid.max()) - 0.5
        g = grid[(grid >= lo) & (grid <= hi)]
        sel = (grid >= lo) & (grid <= hi)
        try:
            out["ibs"] = float(integrated_brier_score(ytr, yte, surv_fn[:, sel], g))
        except Exception:
            out["ibs"] = float("nan")
    return out
