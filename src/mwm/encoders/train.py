"""Train the shared encoder (masked autoencoder + hazard), freeze, export per-visit latents.

Exported latents are whitened (per-dim z-score fit on train visits); the bundle's
decoder/hazard heads take whitened z so downstream code never sees the raw scale.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.dataset import CohortArrays
from ..heads.heads import FeatureDecoder, HazardHead
from .tabular import MaskedTabularEncoder, MLPEncoder


class EncoderBundle(nn.Module):
    def __init__(self, n_clinical, n_omics, d_latent=16, kind="transformer", hazard_use_age=False,
                 **enc_kw):
        super().__init__()
        F = n_clinical + n_omics
        self.cfg = dict(n_clinical=n_clinical, n_omics=n_omics, d_latent=d_latent, kind=kind,
                        hazard_use_age=hazard_use_age, **enc_kw)
        if kind == "transformer":
            self.encoder = MaskedTabularEncoder(n_clinical, n_omics, d_latent, **enc_kw)
        elif kind == "hetgnn":  # v2; needs enc_kw["pathways"] = list of omics index lists
            from .hetgnn import HetGNNEncoder
            self.encoder = HetGNNEncoder(n_clinical, n_omics, d_latent=d_latent, **enc_kw)
        else:
            self.encoder = MLPEncoder(F, d_latent, **enc_kw)
        self.decoder = FeatureDecoder(d_latent, F)
        self.hazard = HazardHead(d_latent, use_age=hazard_use_age)
        self.register_buffer("z_mean", torch.zeros(d_latent))
        self.register_buffer("z_std", torch.ones(d_latent))

    # raw-latent API (training)
    def encode_raw(self, x, m):
        return self.encoder(torch.nan_to_num(x) * m, m)

    # whitened API (everything downstream)
    def encode(self, x, m):
        return (self.encode_raw(x, m) - self.z_mean) / self.z_std

    def unwhiten(self, zw):
        return zw * self.z_std + self.z_mean

    def decode(self, zw):
        return self.decoder(self.unwhiten(zw))

    def log_hazard(self, zw, age=None):
        return self.hazard(self.unwhiten(zw), age)

    @property
    def feature_sigma(self):
        return self.decoder.log_sigma.clamp(-4, 3).exp()

    def save(self, path):
        torch.save({"cfg": self.cfg, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        d = torch.load(path, map_location=map_location, weights_only=False)
        b = cls(**d["cfg"])
        b.load_state_dict(d["state"])
        return b.eval()


def _tensors(ca: CohortArrays, idx, device):
    X = torch.tensor(ca.Xs[idx], device=device)
    M = ~torch.isnan(X)
    return (torch.nan_to_num(X), M, torch.tensor(ca.t_start[idx], dtype=torch.float32, device=device),
            torch.tensor(ca.t_stop[idx], dtype=torch.float32, device=device),
            torch.tensor(ca.died_in_interval[idx], device=device))


def train_encoder(ca: CohortArrays, d_latent=16, kind="transformer", epochs=60, batch=512, lr=2e-3,
                  hide_frac=0.3, w_visible=0.5, w_hazard=0.5, w_consist=0.5, seed=0, device="cuda", log=None,
                  **enc_kw) -> EncoderBundle:
    torch.manual_seed(seed)
    np.random.seed(seed)
    nc, no = len(ca.clinical), len(ca.omics)
    model = EncoderBundle(nc, no, d_latent, kind, **enc_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr = ca.idx("train")
    va = ca.idx("val")
    Xtr = _tensors(ca, tr, device)
    Xva = _tensors(ca, va, device)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        tot = 0.0
        for s in range(0, len(tr), batch):
            b = perm[s:s + batch]
            x, m, ts, te, d = (t[b] for t in Xtr)
            hide = (torch.rand_like(x) < hide_frac) & m
            m_in = m & ~hide
            z = model.encode_raw(x, m_in)
            # reconstruction mostly/only of HIDDEN entries: a latent that reconstructs its own
            # visible inputs also encodes their idiosyncratic measurement noise, which then looks
            # like (fast) dynamics to A/B. w_visible=0 -> pure masked modelling (denoising).
            rec = model.decoder.nll(z, x, hide.float() + w_visible * (m & ~hide).float())
            haz = model.hazard.nll(z, ts, te, d, ts)
            loss = rec + w_hazard * haz
            if no > 0 and w_consist > 0:
                has_om = m[:, nc:].any(-1)
                if has_om.any():
                    m_noom = m_in.clone()
                    m_noom[:, nc:] = False
                    z2 = model.encode_raw(x[has_om], m_noom[has_om])
                    loss = loss + w_consist * ((z2 - z[has_om].detach()) ** 2).mean() \
                        + model.decoder.nll(z2, x[has_om], m[has_om].float())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            x, m, ts, te, d = Xva
            z = model.encode_raw(x, m)
            vrec = model.decoder.nll(z, x, m.float()).item()
            vhaz = model.hazard.nll(z, ts, te, d, ts).item()
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_rec": vrec, "val_haz": vhaz})
        if log and (ep % 10 == 0 or ep == epochs - 1):
            log(f"[enc] ep {ep} train {tot/len(tr):.4f} val_rec {vrec:.4f} val_haz {vhaz:.4f}")
    # whitening on train
    with torch.no_grad():
        Z = encode_all(model, ca, raw=True, device=device)[tr]
        model.z_mean.copy_(torch.tensor(Z.mean(0), device=device))
        model.z_std.copy_(torch.tensor(Z.std(0) + 1e-6, device=device))
    model.history = hist
    model.train_seconds = time.time() - t0
    return model.eval()


@torch.no_grad()
def encode_all(model: EncoderBundle, ca: CohortArrays, raw=False, device="cuda", batch=4096,
               drop_omics=False) -> np.ndarray:
    out = []
    nc = len(ca.clinical)
    for s in range(0, len(ca.Xs), batch):
        X = torch.tensor(ca.Xs[s:s + batch], device=device)
        M = ~torch.isnan(X)
        if drop_omics:
            M[:, nc:] = False
        X = torch.nan_to_num(X)
        z = model.encode_raw(X, M) if raw else model.encode(X, M)
        out.append(z.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def estimate_latent_noise(model: EncoderBundle, ca: CohortArrays, n=4000, K=16, device="cuda", seed=0):
    """Per-dim variance of the (whitened) latent when observed inputs are perturbed by the
    decoder's per-feature noise sigma_x: an estimate of encoder/measurement error in z,
    used as the fixed observation noise of the dynamics models (errors-in-variables)."""
    g = torch.Generator(device=device).manual_seed(seed)
    idx = ca.idx("train")[:n]
    X = torch.tensor(ca.Xs[idx], device=device)
    M = ~torch.isnan(X)
    X = torch.nan_to_num(X)
    sx = model.feature_sigma
    Z = torch.stack([model.encode(X + sx * torch.randn(X.shape, device=device, generator=g) * M, M)
                     for _ in range(K)])
    return Z.var(0).mean(0).cpu().numpy()


def export_latents(model: EncoderBundle, ca: CohortArrays, out_dir: str | Path, device="cuda"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Z = encode_all(model, ca, device=device)
    df = pd.DataFrame(Z, columns=[f"z_{k}" for k in range(Z.shape[1])])
    df.insert(0, "split", ca.split)
    df.insert(0, "visit_age", ca.visits.visit_age)
    df.insert(0, "person_id", ca.visits.person_id)
    df["u"] = ca.u[:, 0]
    df.to_parquet(out / "latents.parquet", index=False)
    model.save(out / "encoder.pt")
    (out / "encoder_meta.json").write_text(json.dumps({
        "features": ca.features, "clinical": ca.clinical, "omics": ca.omics,
        "standardizer": ca.standardizer.to_dict(), "history": getattr(model, "history", []),
        "train_seconds": getattr(model, "train_seconds", None),
        "latent_obs_noise_var": estimate_latent_noise(model, ca, device=device).tolist()}, indent=0))
    return df
