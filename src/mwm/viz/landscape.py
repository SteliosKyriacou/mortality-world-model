"""2D landscape slices of a learned latent drift field (used by the report figures and the web app).

A plane through the latent space is spanned by the max-margin "arrow of aging" w and a second
direction e2 (the latent direction that best predicts a chosen ground-truth axis, orthogonalised
against w). The height is the least-squares (Helmholtz) potential U of the in-plane drift:
-grad U is the curl-free part of the flow; `curl_frac` is the share of the flow it cannot show.
Zhou et al. 2012 (J R Soc Interface 9:3539) discuss such quasi-potentials for non-gradient systems.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import LinearRegression

from ..interrogate.common import drift_at, log_hazard_np


def ls_potential(fx: np.ndarray, fy: np.ndarray, hx: float, hy: float):
    """Least-squares U with dU/dx ~ -fx, dU/dy ~ -fy on an (nx, ny) grid (edge midpoints)."""
    nx, ny = fx.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals, b = [], [], [], []
    r = 0
    for i in range(nx - 1):
        for j in range(ny):
            rows += [r, r]; cols += [idx[i + 1, j], idx[i, j]]; vals += [1 / hx, -1 / hx]
            b.append(-0.5 * (fx[i, j] + fx[i + 1, j])); r += 1
    for i in range(nx):
        for j in range(ny - 1):
            rows += [r, r]; cols += [idx[i, j + 1], idx[i, j]]; vals += [1 / hy, -1 / hy]
            b.append(-0.5 * (fy[i, j] + fy[i, j + 1])); r += 1
    rows.append(r); cols.append(idx[0, 0]); vals.append(1.0); b.append(0.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r + 1, nx * ny))
    U = lsqr(A, np.array(b), atol=1e-10, btol=1e-10, iter_lim=20000)[0].reshape(nx, ny)
    gx, gy = np.gradient(U, hx, axis=0), np.gradient(U, hy, axis=1)
    curl_frac = float(np.sqrt(((fx + gx) ** 2 + (fy + gy) ** 2).sum() / (fx ** 2 + fy ** 2).sum()))
    return U - U.min(), curl_frac


def plane_direction(Z_train: np.ndarray, target: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Latent direction that best linearly predicts `target`, orthogonalised against w."""
    c = LinearRegression().fit(Z_train, target).coef_
    c = c - (c @ w) * w
    return c / np.linalg.norm(c)


def learned_surface(model, bundle, w, e2, xr, yr, origin, age_of_x, device="cpu", n=61):
    """Helmholtz potential + hazard of the learned drift (u = 0) on the plane origin + x w + y e2."""
    xs, ys = np.linspace(*xr, n), np.linspace(*yr, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = origin[None] + X.reshape(-1, 1) * w[None] + Y.reshape(-1, 1) * e2[None]
    A = age_of_x(X.reshape(-1))
    f = drift_at(model, P, A, np.zeros(len(P)), device)
    fx, fy = (f @ w).reshape(n, n), (f @ e2).reshape(n, n)
    U, curl = ls_potential(fx, fy, xs[1] - xs[0], ys[1] - ys[0])
    lh = log_hazard_np(bundle, P, device=device).reshape(n, n)
    return {"x": xs, "y": ys, "U": U, "curl_frac": curl, "log_hazard": lh, "age_x": age_of_x(xs)}
