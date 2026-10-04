"""Jacobian eigen-analysis of the drift, decoded to feature / pathway space.

At held-out states (u = 0) compute J = d drift / dz, take the eigenvectors with the largest
real parts (the least stable / "aging" modes: deviations along them do not decay), average
the (real, sign-aligned) leading eigen-subspace over states, decode it through the decoder
Jacobian into omics space and run pathway enrichment against random-direction nulls.
"""
from __future__ import annotations

import numpy as np

from .common import drift_jacobian, mean_decoder_jacobian
from .enrichment import enrichment_test


def leading_subspace(J: np.ndarray, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """J: (N, d, d). Returns basis (d, m) of the averaged leading eigen-subspace and the mean
    sorted real eigenvalues."""
    d = J.shape[1]
    P = np.zeros((d, d))
    eigs = []
    for Ji in J:
        w, V = np.linalg.eig(Ji)
        o = np.argsort(-w.real)
        eigs.append(w.real[o])
        # real basis of the top-m eigenvectors (complex pairs -> real & imag parts)
        B = []
        for k in o[:m]:
            B.append(V[:, k].real)
            if abs(w[k].imag) > 1e-8:
                B.append(V[:, k].imag)
        Q, _ = np.linalg.qr(np.stack(B, 1))
        P += Q[:, :m] @ Q[:, :m].T
    P /= len(J)
    w, V = np.linalg.eigh(P)
    return V[:, ::-1][:, :m], np.mean(eigs, 0)


def test_jacobian_enrichment(model, bundle, Z, A, genes, omics_slice, pathways, planted,
                             m=2, n_states=300, device="cuda", seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n_states, len(Z)), replace=False)
    J = drift_jacobian(model, Z[idx], A[idx], np.zeros(len(idx)), device)
    V, eigs = leading_subspace(J, m)
    Jdec = mean_decoder_jacobian(bundle, Z, device)
    res = enrichment_test(Jdec[omics_slice], V, genes, pathways, n_dir=2000, seed=seed)
    det = set(res["detected"])
    planted = set(planted)
    tp = len(det & planted)
    res.update({
        "mean_eigs": eigs.round(4).tolist(), "basis": V.tolist(),
        "n_detected": len(det), "true_pos": tp, "false_pos": len(det - planted),
        "precision": tp / len(det) if det else float("nan"),
        "recall": tp / len(planted) if planted else float("nan"),
        "detected_flag": bool(tp >= 1 and len(det - planted) <= 1),
        "rule": "leading-2 eigen-subspace: >=1 planted aging pathway with BH q_dir<0.05 and "
                "p_gene<0.05, and <=1 non-aging pathway detected",
    })
    # feature-space share (clinical) of the leading subspace
    return res
