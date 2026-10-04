"""Baselines (PLAN.md §5.2). All forecast baselines work on (x0 or z0, a0, dt, u) -> target,
return a point prediction and a Gaussian predictive sd so CRPS/coverage are comparable."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _resid_var_model(dt, r2):
    """Fit per-output residual variance a + b*dt by least squares (masked), clip positive."""
    F = r2.shape[1]
    ab = np.zeros((F, 2))
    X = np.stack([np.ones_like(dt), dt], 1)
    for j in range(F):
        m = ~np.isnan(r2[:, j])
        if m.sum() < 10:
            ab[j] = [np.nanmean(r2[:, j]) if m.any() else 1.0, 0.0]
            continue
        coef, *_ = np.linalg.lstsq(X[m], r2[m, j], rcond=None)
        ab[j] = coef
    ab[:, 0] = np.maximum(ab[:, 0], 1e-4)
    ab[:, 1] = np.maximum(ab[:, 1], 0.0)
    return ab


class LOCF:
    """Last observation carried forward; fallback values (e.g. decoded z0) fill missing x0."""
    name = "locf"

    def fit(self, X0, X1, a0, dt, u):
        r = X1 - X0
        self.ab = _resid_var_model(dt, r ** 2)
        return self

    def predict(self, X0, a0, dt, u):
        sd = np.sqrt(self.ab[:, 0] + self.ab[:, 1] * dt[:, None])
        return X0.copy(), sd


class LinearDrift:
    """Per-output linear trend in age with person-level random intercept (LMM fixed slope
    estimated from within-person changes: slope = sum(dx*dt)/sum(dt^2)). Prediction
    x1 = x0 + slope*dt (the BLUP when the random-intercept variance dominates)."""
    name = "linear_drift"

    def fit(self, X0, X1, a0, dt, u):
        d = X1 - X0
        m = ~np.isnan(d)
        self.slope = np.array([(np.nan_to_num(d[:, j]) * dt * m[:, j]).sum() /
                               max((dt ** 2 * m[:, j]).sum(), 1e-9) for j in range(d.shape[1])])
        r = d - self.slope * dt[:, None]
        self.ab = _resid_var_model(dt, r ** 2)
        return self

    def predict(self, X0, a0, dt, u):
        sd = np.sqrt(self.ab[:, 0] + self.ab[:, 1] * dt[:, None])
        return X0 + self.slope * dt[:, None], sd


class DirectMLP:
    """MLP from (x0 with mask, a0, dt, u) -> x1 (masked MSE); heteroscedastic Gaussian head."""
    name = "direct_mlp"

    def __init__(self, hidden=256, epochs=150, lr=2e-3, device="cuda", seed=0):
        self.hidden, self.epochs, self.lr, self.device, self.seed = hidden, epochs, lr, device, seed

    def _inp(self, X0, a0, dt, u):
        m = ~np.isnan(X0)
        return np.concatenate([np.nan_to_num(X0), m, ((a0 - 60) / 10)[:, None], (dt / 5)[:, None],
                               u.reshape(len(a0), -1)], 1).astype(np.float32)

    def fit(self, X0, X1, a0, dt, u):
        torch.manual_seed(self.seed)
        I = torch.tensor(self._inp(X0, a0, dt, u), device=self.device)
        Y = torch.tensor(np.nan_to_num(X1), dtype=torch.float32, device=self.device)
        M = torch.tensor(~np.isnan(X1), dtype=torch.float32, device=self.device)
        F = X1.shape[1]
        self.net = nn.Sequential(nn.Linear(I.shape[1], self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 2 * F)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        n = len(I)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, 512):
                b = perm[s:s + 512]
                out = self.net(I[b])
                mu, ls = out[:, :F], out[:, F:].clamp(-5, 3)
                nll = (0.5 * ((Y[b] - mu) / ls.exp()) ** 2 + ls) * M[b]
                loss = nll.sum() / M[b].sum().clamp_min(1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X0, a0, dt, u):
        I = torch.tensor(self._inp(X0, a0, dt, u), device=self.device)
        out = self.net(I).cpu().numpy()
        F = out.shape[1] // 2
        return out[:, :F], np.exp(np.clip(out[:, F:], -5, 3))


class DirectGBM:
    """HistGradientBoosting per output from (x0 incl. NaN, a0, dt, u) -> x1."""
    name = "direct_gbm"

    def __init__(self, max_iter=200, seed=0):
        self.max_iter, self.seed = max_iter, seed

    def fit(self, X0, X1, a0, dt, u):
        from sklearn.ensemble import HistGradientBoostingRegressor
        I = np.concatenate([X0, a0[:, None], dt[:, None], u.reshape(len(a0), -1)], 1)
        self.models, res = [], np.full_like(X1, np.nan)
        for j in range(X1.shape[1]):
            m = ~np.isnan(X1[:, j])
            g = HistGradientBoostingRegressor(max_iter=self.max_iter, random_state=self.seed,
                                              early_stopping=True)
            g.fit(I[m], X1[m, j])
            self.models.append(g)
            res[m, j] = (g.predict(I[m]) - X1[m, j]) ** 2
        self.ab = _resid_var_model(dt, res)
        return self

    def predict(self, X0, a0, dt, u):
        I = np.concatenate([X0, a0[:, None], dt[:, None], u.reshape(len(a0), -1)], 1)
        mu = np.stack([g.predict(I) for g in self.models], 1)
        return mu, np.sqrt(self.ab[:, 0] + self.ab[:, 1] * dt[:, None])


# ------------------------------------------------------------------ survival baselines
class CoxBaseline:
    """Cox PH on baseline features (+ age), mean-imputed, small ridge penalty."""

    def __init__(self, penalizer=0.05, use_age=True):
        self.penalizer, self.use_age = penalizer, use_age

    def _df(self, X, age):
        import pandas as pd
        Xi = np.where(np.isnan(X), self.mean, X)
        df = pd.DataFrame(Xi, columns=[f"x{j}" for j in range(X.shape[1])])
        if self.use_age:
            df["age"] = (age - 60) / 10
        return df

    def fit(self, X, age, time, event):
        from lifelines import CoxPHFitter
        self.mean = np.nan_to_num(np.nanmean(X, 0))
        df = self._df(X, age)
        df["T"], df["E"] = np.maximum(time, 1e-3), event
        self.cph = CoxPHFitter(penalizer=self.penalizer).fit(df, "T", "E")
        return self

    def risk(self, X, age):
        return np.asarray(self.cph.predict_partial_hazard(self._df(X, age))).ravel()

    def survival(self, X, age, grid):
        sf = self.cph.predict_survival_function(self._df(X, age), times=grid)
        return np.asarray(sf).T

    def attributions(self, X, age):
        """beta_j * (x_j - mean_j) per feature (log-hazard contributions)."""
        beta = self.cph.params_.to_numpy()[:X.shape[1]]
        return beta * (np.where(np.isnan(X), self.mean, X) - self.mean)
