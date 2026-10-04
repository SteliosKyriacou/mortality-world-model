"""v2 encoder: heterogeneous GNN over (visit, clinical-feature, protein, pathway) nodes.

For each visit a small graph is built on the fly from the observed entries only:

    clin  --(obs)-->  visit          one 'clin' node per observed clinical value
    prot  --(member)--> path          one 'prot' node per observed protein value
    prot  --(obs)-->  visit          (proteins not in any pathway still reach the visit)
    path  --(agg)-->  visit          one 'path' node per (visit, pathway) with >=1 observed member

Missing modalities are therefore handled structurally: a visit without proteomics simply
has no prot/path nodes, and its embedding comes from the clinical nodes alone. Node inputs
are learned feature-identity embeddings plus value * learned per-feature vector (same
tokenisation as the v1 masked tabular encoder). Output: z in R^d via an MLP on the visit node.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv


class HetGNNEncoder(nn.Module):
    def __init__(self, n_clinical: int, n_omics: int, pathways: list[list[int]], d_latent: int = 16,
                 hidden: int = 64, n_layers: int = 2):
        """pathways: list of lists of omics column indices (0-based within the omics block)."""
        super().__init__()
        self.nc, self.no, self.d_latent, self.h = n_clinical, n_omics, d_latent, hidden
        self.n_path = len(pathways)
        gi, pi = [], []
        for p, genes in enumerate(pathways):
            gi += list(genes)
            pi += [p] * len(genes)
        self.register_buffer("mem_gene", torch.tensor(gi, dtype=torch.long))
        self.register_buffer("mem_path", torch.tensor(pi, dtype=torch.long))
        self.clin_e = nn.Parameter(torch.randn(n_clinical, hidden) * 0.1)
        self.clin_w = nn.Parameter(torch.randn(n_clinical, hidden) * 0.1)
        self.prot_e = nn.Parameter(torch.randn(max(n_omics, 1), hidden) * 0.1)
        self.prot_w = nn.Parameter(torch.randn(max(n_omics, 1), hidden) * 0.1)
        self.path_e = nn.Parameter(torch.randn(max(self.n_path, 1), hidden) * 0.1)
        self.visit0 = nn.Parameter(torch.zeros(hidden))
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(HeteroConv({
                ("clin", "obs", "visit"): SAGEConv((hidden, hidden), hidden),
                ("prot", "member", "path"): SAGEConv((hidden, hidden), hidden),
                ("prot", "obs", "visit"): SAGEConv((hidden, hidden), hidden),
                ("path", "agg", "visit"): SAGEConv((hidden, hidden), hidden),
                ("visit", "rev", "clin"): SAGEConv((hidden, hidden), hidden),
            }, aggr="sum"))
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_latent))

    def build_graph(self, x: torch.Tensor, m: torch.Tensor):
        B, dev = x.shape[0], x.device
        xc, mc = x[:, :self.nc], m[:, :self.nc]
        vb, fc = torch.nonzero(mc, as_tuple=True)
        xd = {"visit": self.visit0.expand(B, -1).clone(),
              "clin": self.clin_e[fc] + xc[vb, fc].unsqueeze(-1) * self.clin_w[fc]}
        ei = {("clin", "obs", "visit"): torch.stack([torch.arange(len(vb), device=dev), vb]),
              ("visit", "rev", "clin"): torch.stack([vb, torch.arange(len(vb), device=dev)])}
        if self.no > 0:
            xo, mo = x[:, self.nc:], m[:, self.nc:]
            vp, fp = torch.nonzero(mo, as_tuple=True)
            xd["prot"] = self.prot_e[fp] + xo[vp, fp].unsqueeze(-1) * self.prot_w[fp]
            ei[("prot", "obs", "visit")] = torch.stack([torch.arange(len(vp), device=dev), vp])
            # pathway nodes per (visit, pathway) with >=1 observed member
            node_id = torch.full((B, self.no), -1, dtype=torch.long, device=dev)
            node_id[vp, fp] = torch.arange(len(vp), device=dev)
            # all (visit, membership) combos where the member gene is observed
            src = node_id[:, self.mem_gene]                       # (B, n_mem)
            ok = src >= 0
            bv, k = torch.nonzero(ok, as_tuple=True)
            pkey = bv * self.n_path + self.mem_path[k]
            uniq, inv = torch.unique(pkey, return_inverse=True)
            xd["path"] = self.path_e[uniq % self.n_path]
            ei[("prot", "member", "path")] = torch.stack([src[bv, k], inv])
            ei[("path", "agg", "visit")] = torch.stack([torch.arange(len(uniq), device=dev), uniq // self.n_path])
        else:
            xd["prot"] = torch.zeros(0, self.h, device=dev)
            xd["path"] = torch.zeros(0, self.h, device=dev)
            for k in (("prot", "obs", "visit"), ("prot", "member", "path"), ("path", "agg", "visit")):
                ei[k] = torch.zeros(2, 0, dtype=torch.long, device=dev)
        return xd, ei

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x)
        xd, ei = self.build_graph(x, m.bool())
        for conv in self.convs:
            new = conv(xd, ei)
            xd = {k: (xd[k] + torch.relu(new[k])) if k in new else xd[k] for k in xd}
        return self.out(self.norm(xd["visit"]))
