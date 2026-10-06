"""Model A2: second-order ("momentum") latent SDE, age-free.

    dz = v dt + Σ_z(z) dW₁                         (position noise: fast fluctuations; optional)
    dv = [F(z, u) − Γ(z) ⊙ v] dt + Σ_v(z) dW₂,     F = −∇V(z) + J(z, u)

Without position noise all randomness must pass through v, so one friction Γ has to serve both fast
fluctuations and a slow, persistent pace, and the learned persistence collapses to months. With
position noise (pos_noise=True, the default) Γ is free to be small.

Each person carries an aging velocity v (their current pace and direction of aging) that persists and
relaxes at rate Γ(z) (friction). With strong friction the model reduces to the first-order Model A with
drift F/Γ (the overdamped limit). Nothing depends on age: only the state, the elapsed time and the
forcing u.

The starting velocity at a visit is inferred:
  * from the previous visit when there is one: q(v | z_prev, z_now, Δt)   (amortised, Gaussian)
  * from the current state alone otherwise:    p(v | z_now)              (Gaussian prior)
Training: for consecutive visit pairs (with the earlier visit as history when available), sample K
starting velocities, simulate to the next visit with Euler–Maruyama, Gaussian likelihood of the next
latent (same objective as Model A, errors-in-variables observation noise).

For the hallmark tools (which expect a first-order drift f(z, a, u) and noise Σ(z, a)), `drift` returns the
terminal velocity F/Γ and `diffusion` returns Σ/Γ: the overdamped limit of this model.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .base import LatentDynamics, gaussian_sample_nll
from .potential import DiagDiffusion, DriftField, mlp


class SecondOrderSDE(LatentDynamics):
    stochastic = True

    def __init__(self, d, n_u=1, hidden=128, use_potential=True, n_steps=32, friction_init=0.3, pos_noise=True):
        super().__init__()
        self.d, self.n_steps = d, n_steps
        self.force = DriftField(d, n_u, hidden, use_potential, use_age=False)
        self.fric = mlp(d, d, 64)
        nn.init.zeros_(self.fric[-1].weight)
        nn.init.constant_(self.fric[-1].bias, float(torch.log(torch.expm1(torch.tensor(friction_init)))))
        self.noise = DiagDiffusion(d, 64, "state", init=0.05)
        self.pos_noise = pos_noise
        self.znoise = DiagDiffusion(d, 64, "state", init=0.1) if pos_noise else None
        self.vprior = mlp(d, 2 * d, hidden)            # mean, log-sd of v given z
        self.vpost = mlp(2 * d + 1, 2 * d, hidden)     # mean, log-sd of v given (z_prev, z_now, Δt)
        self.log_r = nn.Parameter(torch.full((d,), -3.0))
        self.cfg = dict(d=d, n_u=n_u, hidden=hidden, use_potential=use_potential, n_steps=n_steps,
                        friction_init=friction_init, pos_noise=pos_noise)

    # ---------------- components ----------------
    def friction(self, z):
        return nn.functional.softplus(self.fric(z)) + 1e-3

    def F(self, z, u):
        return self.force(z, torch.zeros(z.shape[0], device=z.device), u)

    def v_start(self, z_now, z_prev=None, dt_prev=None, has_prev=None):
        """Mean and sd of the starting velocity (posterior with history, prior without)."""
        mp, lsp = self.vprior(z_now).chunk(2, -1)
        if z_prev is None:
            return mp, lsp.clamp(-6, 2).exp()
        x = torch.cat([z_prev, z_now, dt_prev.unsqueeze(-1) / 5.0], -1)
        mq, lsq = self.vpost(x).chunk(2, -1)
        # finite-difference velocity is the natural starting point; the network learns the correction
        fd = (z_now - z_prev) / dt_prev.clamp_min(0.25).unsqueeze(-1)
        mq = mq + fd
        h = has_prev.unsqueeze(-1)
        return torch.where(h, mq, mp), torch.where(h, lsq, lsp).clamp(-6, 2).exp()

    # first-order (overdamped) view for the hallmark/interpretation tools
    def drift(self, z, a, u):
        return self.F(z, u) / self.friction(z)

    def diffusion(self, z, a):
        # overdamped-limit noise in z: velocity noise passed through friction, plus position noise
        s = self.noise(z, a) / self.friction(z)
        if self.pos_noise:
            s = torch.sqrt(s ** 2 + self.znoise(z, a) ** 2)
        return s

    @property
    def uses_age(self):
        return False

    # ---------------- simulation ----------------
    def _step(self, z, v, u, h, eps, eps_z=None):
        f = self.F(z, u)
        g = self.friction(z)
        s = self.noise(z, None)
        z_new = z + v * h
        if self.pos_noise:
            z_new = z_new + self.znoise(z, None) * torch.sqrt(h) * (eps_z if eps_z is not None else torch.randn_like(z))
        v_new = v + (f - g * v) * h + s * torch.sqrt(h) * eps
        return z_new, v_new

    def simulate2(self, z0, v0, dt, u, n_steps=None, return_path=False):
        """z0, v0: (N, d) (N may already include the K replicas), dt: (N,), u: (N, n_u)."""
        n = n_steps or self.n_steps
        h = (dt / n).unsqueeze(-1)
        z, v = z0, v0
        path = [z] if return_path else None
        for _ in range(n):
            self.nfe += 1
            z, v = self._step(z, v, u, h, torch.randn_like(v))
            if return_path:
                path.append(z)
        return (z, v, torch.stack(path)) if return_path else (z, v)

    def loss(self, batch, K=32, j_penalty=1e-3, sigma_penalty=1e-3):
        z0, z1, a0, a1, u, zp, dtp, hp = batch
        B, d = z0.shape
        mu, sd = self.v_start(z0, zp, dtp, hp)
        rep = lambda x: x.unsqueeze(0).expand(K, *x.shape).reshape(K * B, *x.shape[1:])
        zs = rep(z0)
        if self.training and getattr(self, "obs_r2_fixed", None) is not None:
            zs = zs + torch.randn_like(zs) * self.obs_r2_fixed.sqrt()
        v0 = rep(mu) + rep(sd) * torch.randn(K * B, d, device=z0.device)
        zT, _ = self.simulate2(zs, v0, rep(a1 - a0), rep(u))
        nll = gaussian_sample_nll(zT.view(K, B, d), z1, self.obs_var)
        loss = nll
        if j_penalty > 0:
            j = self.force.nonconservative(z0, torch.zeros(B, device=z0.device), u)
            loss = loss + j_penalty * (j ** 2).sum(-1).mean()
        if sigma_penalty > 0:
            loss = loss + sigma_penalty * (self.noise(z0, None) ** 2).sum(-1).mean()
            if self.pos_noise:
                loss = loss + sigma_penalty * (self.znoise(z0, None) ** 2).sum(-1).mean()
        return loss, {"nll": nll.item()}

    @torch.no_grad()
    def val_loss(self, batch, K=32):
        was = self.training
        self.train()
        try:
            return self.loss(batch, K, 0.0, 0.0)[1]["nll"]
        finally:
            self.train(was)

    @torch.no_grad()
    def sample(self, z0, a0, a1, u, n_samples=32, n_steps=40, return_path=False, z_prev=None, dt_prev=None):
        """Forecast from a visit (optionally with the previous visit as history). Returns (K, B, d)."""
        B, d = z0.shape
        K = n_samples
        hp = torch.ones(B, dtype=torch.bool, device=z0.device) if z_prev is not None else None
        mu, sd = self.v_start(z0, z_prev, dt_prev, hp) if z_prev is not None else self.v_start(z0)
        rep = lambda x: x.unsqueeze(0).expand(K, *x.shape).reshape(K * B, *x.shape[1:])
        v0 = rep(mu) + rep(sd) * torch.randn(K * B, d, device=z0.device)
        out = self.simulate2(rep(z0), v0, rep(a1 - a0), rep(u), n_steps, return_path)
        if return_path:
            P = out[2].view(-1, K, B, d)
            return P[-1], P
        return out[0].view(K, B, d)

    # ---------------- persistence ----------------
    def save_config(self, run_dir):
        (Path(run_dir) / "model_A2_config.json").write_text(json.dumps(self.cfg, indent=1))

    @classmethod
    def from_run_dir(cls, run_dir, d, device="cpu"):
        f = Path(run_dir) / "model_A2_config.json"
        cfg = dict(d=d)
        if f.exists():
            cfg.update(json.loads(f.read_text()))
        return cls(**cfg).to(device)


class SeqSecondOrderSDE(SecondOrderSDE):
    """Model A2 with a FILTER over the whole visit history.

    A GRU reads the visits in order (latent z_k, the finite-difference velocity since the previous
    visit, and the gap) and keeps a summary h_k of everything seen so far; the starting velocity at
    visit k is q(v | h_k, z_k). Trained on whole sequences: at every visit, sample K velocities from the
    filter, simulate to the next visit, Gaussian likelihood of the next latent. Evidence about a person's
    pace therefore accumulates across visits instead of coming from a single difference.
    """

    def __init__(self, d, n_u=1, hidden=128, use_potential=True, n_steps=8, friction_init=0.3, pos_noise=True,
                 filter_hidden=64, w_dim=0, horizons=1):
        """w_dim: extra per-visit filter inputs (e.g. wearable summaries for the interval before the visit).
        horizons: train on predictions 1..horizons visits ahead along continued simulated paths, so that
        persistence must be carried by the dynamics, not only by the filter."""
        super().__init__(d, n_u, hidden, use_potential, n_steps, friction_init, pos_noise)
        self.w_dim, self.horizons = w_dim, horizons
        self.filt = nn.GRUCell(2 * d + 2 + w_dim, filter_hidden)
        self.vhead = mlp(filter_hidden + d, 2 * d, hidden)
        self.filter_hidden = filter_hidden
        self.cfg.update(filter_hidden=filter_hidden, w_dim=w_dim, horizons=horizons)

    def filter_step(self, h, z, z_prev, dt_prev, has_prev, w=None):
        """Update the filter with a new visit. has_prev = False at a person's first visit.
        w: (B, w_dim) extra inputs for the interval before this visit (zeros when absent)."""
        hp = has_prev.float().unsqueeze(-1)
        fd = (z - z_prev) / dt_prev.clamp_min(0.25).unsqueeze(-1) * hp
        x = torch.cat([z, fd, torch.log(dt_prev.clamp_min(0.25)).unsqueeze(-1) * hp, hp], -1)
        if self.w_dim:
            x = torch.cat([x, w if w is not None else torch.zeros(len(z), self.w_dim, device=z.device)], -1)
        return self.filt(x, h)

    def v_from_filter(self, h, z):
        m, ls = self.vhead(torch.cat([h, z], -1)).chunk(2, -1)
        return m, ls.clamp(-6, 2).exp()

    def seq_loss(self, Zs, As, Us, L, K=16, j_penalty=1e-3, sigma_penalty=1e-3, Ws=None):
        """Zs (B, T, d), As (B, T), Us (B, T, n_u), L (B,) number of visits, Ws (B, T, w_dim) optional.
        Mean NLL over all (visit, horizon) predictions."""
        B, Tn, d = Zs.shape
        h = torch.zeros(B, self.filter_hidden, device=Zs.device)
        tot, cnt = 0.0, 0.0
        for k in range(Tn - 1):
            has_prev = torch.full((B,), k > 0, dtype=torch.bool, device=Zs.device)
            zp = Zs[:, k - 1] if k > 0 else Zs[:, 0]
            dtp = (As[:, k] - As[:, k - 1]) if k > 0 else torch.ones(B, device=Zs.device)
            h = self.filter_step(h, Zs[:, k], zp, dtp, has_prev, Ws[:, k] if Ws is not None else None)
            valid = (k + 1) < L
            if not valid.any():
                break
            idx = valid.nonzero(as_tuple=True)[0]
            z0, u = Zs[idx, k], Us[idx, k]
            mu, sd = self.v_from_filter(h[idx], z0)
            n = len(idx)
            rep = lambda x: x.unsqueeze(0).expand(K, *x.shape).reshape(K * n, *x.shape[1:])
            zs = rep(z0)
            if self.training and getattr(self, "obs_r2_fixed", None) is not None:
                zs = zs + torch.randn_like(zs) * self.obs_r2_fixed.sqrt()
            v = rep(mu) + rep(sd) * torch.randn(K * n, d, device=Zs.device)
            a_prev = As[idx, k]
            for j in range(1, self.horizons + 1):          # continue the same simulated paths
                if k + j >= Tn:
                    break
                ok = (k + j) < L[idx]
                if not ok.any():
                    break
                dt = (As[idx, k + j] - a_prev).clamp_min(1e-3)
                zs, v = self.simulate2(zs, v, rep(dt), rep(u))
                nll_each = self._nll_per(zs.view(K, n, d), Zs[idx, k + j])
                tot = tot + (nll_each * ok).sum()
                cnt += float(ok.sum())
                a_prev = As[idx, k + j]
        loss = tot / max(cnt, 1)
        zf = Zs[:, 0]
        if j_penalty > 0:
            j = self.force.nonconservative(zf, torch.zeros(B, device=Zs.device), Us[:, 0])
            loss = loss + j_penalty * (j ** 2).sum(-1).mean()
        if sigma_penalty > 0:
            loss = loss + sigma_penalty * (self.noise(zf, None) ** 2).sum(-1).mean()
            if self.pos_noise:
                loss = loss + sigma_penalty * (self.znoise(zf, None) ** 2).sum(-1).mean()
        return loss, float(loss.detach()) if not torch.is_tensor(tot) else float((tot / max(cnt, 1)).detach())

    def _nll_per(self, S, y):
        mu = S.mean(0)
        var = S.var(0) + self.obs_var + 1e-5
        return (0.5 * (y - mu) ** 2 / var + 0.5 * torch.log(var)).sum(-1)

    @torch.no_grad()
    def filter_history(self, Zs, As, upto, Ws=None):
        """Run the filter over visits 0..upto (inclusive) for every sequence; returns h, z at `upto`."""
        B = Zs.shape[0]
        h = torch.zeros(B, self.filter_hidden, device=Zs.device)
        for k in range(upto + 1):
            has_prev = torch.full((B,), k > 0, dtype=torch.bool, device=Zs.device)
            zp = Zs[:, k - 1] if k > 0 else Zs[:, 0]
            dtp = (As[:, k] - As[:, k - 1]) if k > 0 else torch.ones(B, device=Zs.device)
            h = self.filter_step(h, Zs[:, k], zp, dtp, has_prev, Ws[:, k] if Ws is not None else None)
        return h

    @torch.no_grad()
    def forecast_from_history(self, Zs, As, Us, k, a_target, n_samples=64, Ws=None):
        """Samples of z at age a_target (B,) for people observed at visits 0..k. Returns (K, B, d)."""
        h = self.filter_history(Zs, As, k, Ws)
        z0 = Zs[:, k]
        mu, sd = self.v_from_filter(h, z0)
        B, d = z0.shape
        K = n_samples
        rep = lambda x: x.unsqueeze(0).expand(K, *x.shape).reshape(K * B, *x.shape[1:])
        v0 = rep(mu) + rep(sd) * torch.randn(K * B, d, device=z0.device)
        n = max(4, int(round(float((a_target - As[:, k]).max()) * 4)))
        zT, _ = self.simulate2(rep(z0), v0, rep(a_target - As[:, k]), rep(Us[:, k]), n_steps=n)
        return zT.view(K, B, d)
