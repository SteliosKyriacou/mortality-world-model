"""v1 shared encoder: masked tabular transformer (PLAN.md §4.1).

Each clinical feature is a token: value * w_j + e_j (missing -> learned mask token, and
excluded from attention). The (high-dimensional, often absent) omics block is projected
into a few tokens from [x*m, m]; if no omics were measured at a visit these tokens are
padded out. A CLS token is pooled into z.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedTabularEncoder(nn.Module):
    def __init__(self, n_clinical: int, n_omics: int = 0, d_latent: int = 16, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4, n_omics_tokens: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_clinical, self.n_omics, self.d_latent = n_clinical, n_omics, d_latent
        self.val_w = nn.Parameter(torch.randn(n_clinical, d_model) * 0.1)
        self.feat_e = nn.Parameter(torch.randn(n_clinical, d_model) * 0.1)
        self.mask_e = nn.Parameter(torch.randn(n_clinical, d_model) * 0.1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.n_omics_tokens = n_omics_tokens if n_omics > 0 else 0
        if n_omics > 0:
            self.omics_proj = nn.Sequential(nn.Linear(2 * n_omics, 256), nn.GELU(),
                                            nn.Linear(256, n_omics_tokens * d_model))
            self.omics_pos = nn.Parameter(torch.randn(n_omics_tokens, d_model) * 0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_latent))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x: (B, F) standardised with zeros at missing; m: (B, F) bool observed."""
        B = x.shape[0]
        xc, mc = x[:, :self.n_clinical], m[:, :self.n_clinical]
        tok = xc.unsqueeze(-1) * self.val_w + self.feat_e
        tok = torch.where(mc.unsqueeze(-1), tok, self.mask_e.expand(B, -1, -1))
        toks = [self.cls.expand(B, -1, -1), tok]
        pad = [torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mc]
        if self.n_omics_tokens:
            xo, mo = x[:, self.n_clinical:], m[:, self.n_clinical:].float()
            ot = self.omics_proj(torch.cat([xo * mo, mo], -1)).view(B, self.n_omics_tokens, -1)
            toks.append(ot + self.omics_pos)
            has = mo.sum(-1, keepdim=True) > 0
            pad.append((~has).expand(B, self.n_omics_tokens))
        h = self.tf(torch.cat(toks, 1), src_key_padding_mask=torch.cat(pad, 1))
        return self.out(h[:, 0])


class MLPEncoder(nn.Module):
    """Cheap alternative: MLP on [x*m, m]."""

    def __init__(self, n_features: int, d_latent: int = 16, hidden: int = 256, **_):
        super().__init__()
        self.d_latent = d_latent
        self.net = nn.Sequential(nn.Linear(2 * n_features, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, d_latent))

    def forward(self, x, m):
        return self.net(torch.cat([x * m, m.float()], -1))
