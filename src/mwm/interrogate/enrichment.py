"""Pathway enrichment of decoded latent directions.

Gene score = |decoded loading| of a latent direction/subspace onto each omics feature.
Pathway score = mean gene score within the set. Two nulls:
  * competitive: random gene sets of equal size (p_gene_perm)
  * direction null: the same score for random latent directions/subspaces decoded the same
    way (p_dir) -- protects against "any direction looks enriched because genes co-load".
Detected = BH-adjusted p_dir < alpha AND p_gene < alpha.
Optional `gsea_prerank` wraps gseapy for a classical preranked GSEA on the same scores.
"""
from __future__ import annotations

import numpy as np


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def subspace_gene_scores(Jdec_omics: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Jdec_omics: (G, d) mean decoder Jacobian on omics; V: (d, m) orthonormal-ish basis.
    Returns rotation-invariant score per gene: norm of decoded basis loadings."""
    L = Jdec_omics @ V
    return np.sqrt((L ** 2).sum(1))


def pathway_scores(scores: np.ndarray, gene_idx: dict) -> dict:
    return {p: float(scores[ix].mean()) for p, ix in gene_idx.items()}


def enrichment_test(Jdec_omics, V, genes, pathways: dict, n_dir=500, n_gene_perm=2000, seed=0,
                    alpha=0.05):
    rng = np.random.default_rng(seed)
    gpos = {g: i for i, g in enumerate(genes)}
    gene_idx = {p: np.array([gpos[g] for g in gs if g in gpos]) for p, gs in pathways.items()}
    gene_idx = {p: ix for p, ix in gene_idx.items() if len(ix) >= 5}
    s = subspace_gene_scores(Jdec_omics, V)
    obs = pathway_scores(s, gene_idx)
    # direction null
    d, m = V.shape
    null = {p: [] for p in gene_idx}
    for _ in range(n_dir):
        R, _ = np.linalg.qr(rng.normal(size=(d, m)))
        sr = subspace_gene_scores(Jdec_omics, R)
        for p, v in pathway_scores(sr, gene_idx).items():
            null[p].append(v)
    p_dir = {p: float((np.sum(np.array(null[p]) >= obs[p]) + 1) / (n_dir + 1)) for p in gene_idx}
    # competitive gene-set null (same direction, random genes)
    G = len(s)
    p_gene = {}
    for p, ix in gene_idx.items():
        k = len(ix)
        rnd = np.array([s[rng.choice(G, k, replace=False)].mean() for _ in range(n_gene_perm // 4)])
        p_gene[p] = float((np.sum(rnd >= obs[p]) + 1) / (len(rnd) + 1))
    names = list(gene_idx)
    q_dir = dict(zip(names, bh([p_dir[n] for n in names])))
    det = [p for p in names if q_dir[p] < alpha and p_gene[p] < alpha]
    return {"score": obs, "p_dir": p_dir, "q_dir": q_dir, "p_gene": p_gene, "detected": det}


def gsea_prerank(scores: np.ndarray, genes: list[str], pathways: dict, permutations=200, seed=0):
    """Classical preranked GSEA via gseapy (optional; slower)."""
    import gseapy as gp
    import pandas as pd
    rnk = pd.Series(scores, index=genes).sort_values(ascending=False)
    res = gp.prerank(rnk=rnk, gene_sets=pathways, permutation_num=permutations, seed=seed,
                     min_size=5, max_size=1000, outdir=None, verbose=False, threads=4)
    return res.res2d[["Term", "NES", "NOM p-val", "FDR q-val"]]
