"""Compare learned dynamics with the synthetic ground truth.

Two settings:
* oracle latent: model trained directly on the true z* -> compare in the same coordinates.
* encoder latent: an affine map z* ~ z W + b is fit on train visits; the learned drift is
  pushed forward (f* = f W, exact for affine maps) and diffusion as W^T diag(s^2) W.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from ..data import synthetic as syn


def fit_affine(Z_learned: np.ndarray, Z_true: np.ndarray):
    reg = LinearRegression().fit(Z_learned, Z_true)
    return reg.coef_.T, reg.intercept_  # W: (d_learned, d_true)


def affine_r2(Z_learned, Z_true, W, b):
    P = Z_learned @ W + b
    return 1 - ((P - Z_true) ** 2).mean(0) / Z_true.var(0)


@torch.no_grad()
def model_drift_diffusion(model, Z, A, U, device="cuda", batch=4096):
    F, S = [], []
    for s in range(0, len(Z), batch):
        z = torch.tensor(Z[s:s + batch], dtype=torch.float32, device=device)
        a = torch.tensor(A[s:s + batch], dtype=torch.float32, device=device)
        u = torch.tensor(U[s:s + batch], dtype=torch.float32, device=device).reshape(len(z), -1)
        F.append(model.drift(z, a, u).cpu().numpy())
        g = model.diffusion(z, a)
        S.append(g.cpu().numpy() if g is not None else np.zeros_like(F[-1]))
    return np.concatenate(F), np.concatenate(S)


def drift_recovery(model, Z, A, U, Z_true, cfg, W=None, device="cuda") -> dict:
    """Z: states in model coordinates; Z_true: the corresponding true latents."""
    F, S = model_drift_diffusion(model, Z, A, U, device)
    Ftrue = syn.true_drift(Z_true, A, U.reshape(-1), cfg)
    Strue = syn.true_diffusion(Z_true, A, cfg)
    if W is not None:
        F = F @ W
        Dvar = np.einsum("ki,nk,kj->nij", W, S ** 2, W)  # W^T diag(s^2) W per state
        Svar = np.stack([np.diag(d) for d in Dvar])
    else:
        Svar = S ** 2
    err = F - Ftrue
    out = {
        "drift_rel_mse": float((err ** 2).sum(1).mean() / (Ftrue ** 2).sum(1).mean()),
        # per-dim normalised RMSE (err RMS / true-drift RMS); robust for near-constant dims
        "drift_nrmse_per_dim": (np.sqrt((err ** 2).mean(0)) / np.sqrt((Ftrue ** 2).mean(0) + 1e-12)).round(3).tolist(),
        "drift_r2_per_dim": [round(float(1 - (err[:, k] ** 2).mean() / Ftrue[:, k].var()), 3)
                             if Ftrue[:, k].std() > 1e-3 else None for k in range(F.shape[1])],
        "drift_cos": float(np.mean((F * Ftrue).sum(1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(Ftrue, axis=1) + 1e-12))),
        "drift_mean_true": Ftrue.mean(0).round(4).tolist(),
        "drift_mean_model": F.mean(0).round(4).tolist(),
    }
    if model.stochastic:
        st = Strue ** 2
        out["diff_var_rel_err_per_dim"] = (np.abs(Svar.mean(0) - st.mean(0)) / st.mean(0)).round(3).tolist()
        # age profile of total variance rate: ratio of mean at age>=70 vs <=50
        old, young = A >= 70, A <= 50
        out["diff_trace_ratio_old_young_model"] = float(Svar[old].sum(1).mean() / Svar[young].sum(1).mean())
        out["diff_trace_ratio_old_young_true"] = float(st[old].sum(1).mean() / st[young].sum(1).mean())
        out["diff_trace_rel_err"] = float(abs(Svar.sum(1).mean() - st.sum(1).mean()) / st.sum(1).mean())
    return out
