"""GTEx v10 open-access bulk RNA-seq (gene TPM) + public sample/subject attributes.

Human post-mortem, cross-sectional: ~19.6k samples, ~980 donors, ~54 tissues. Open-access
subject phenotypes contain only SEX, AGE (10-year bins 20-29 ... 70-79) and DTHHRDY
(Hardy death-circumstance scale 0-4). Exact age, cause of death, medical history are in the
protected dbGaP phs000424 files (not used). Fetch with ``python scripts/fetch_gtex.py``;
see ``data_cards/gtex.md``.

Outputs (``data/interim/gtex/``)
--------------------------------
* ``expression.parquet``  samples x genes, float32 log1p(TPM) rounded to 0.01; genes with mean TPM >= 1 over
                          all samples (columns = Ensembl gene id without version)
* ``genes.parquet``       gene_id, gene_version_id, symbol, mean_tpm
* ``metadata.parquet``    sample_id, subject_id, tissue, tissue_detail, sex (1 male / 0 female),
                          age_bin, visit_age (bin midpoint, years), age_frac_lifespan
                          (visit_age / 80 y, common.SPECIES_MEDIAN_LIFESPAN_YEARS), hardy
                          (DTHHRDY), rin, ischemic_time_min, analysis_freeze

Usage
-----
>>> from mwm.data import gtex
>>> gtex.build()                      # ~10 min, ~8 GB peak RAM; output ~0.5 GB
>>> X, meta = gtex.load(tissue_detail="Whole Blood")   # or tissue="Blood" (SMTS)
>>> X_sym = gtex.to_symbols(X)        # columns -> HGNC symbols (duplicates averaged)
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .common import lifespan_fraction

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "gtex"
INTERIM = ROOT / "data" / "interim" / "gtex"
TPM = "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
SAMPLE_ATTR = "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
SUBJECT_ATTR = "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt"
MIN_MEAN_TPM = 1.0


def build_metadata() -> pd.DataFrame:
    sa = pd.read_csv(RAW / SAMPLE_ATTR, sep="\t", low_memory=False,
                     usecols=["SAMPID", "SMTS", "SMTSD", "SMRIN", "SMTSISCH", "SMAFRZE"])
    su = pd.read_csv(RAW / SUBJECT_ATTR, sep="\t")
    sa["SUBJID"] = sa["SAMPID"].str.split("-").str[:2].str.join("-")
    m = sa.merge(su, on="SUBJID", how="left")
    lo = m["AGE"].str.split("-").str[0].astype(float)
    meta = pd.DataFrame({
        "sample_id": m["SAMPID"],
        "subject_id": m["SUBJID"],
        "tissue": m["SMTS"],
        "tissue_detail": m["SMTSD"],
        "sex": (m["SEX"] == 1).astype(float).where(m["SEX"].notna()),
        "age_bin": m["AGE"],
        "visit_age": lo + 5.0,
        "hardy": m["DTHHRDY"],
        "rin": m["SMRIN"],
        "ischemic_time_min": m["SMTSISCH"],
        "analysis_freeze": m["SMAFRZE"],
    })
    meta["age_frac_lifespan"] = lifespan_fraction(meta["visit_age"].to_numpy(), "human")
    return meta


def build(chunksize: int = 2000) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    meta = build_metadata()
    kept, genes = [], []
    with gzip.open(RAW / TPM, "rt") as fh:
        fh.readline(), fh.readline()
        reader = pd.read_csv(fh, sep="\t", chunksize=chunksize, index_col=0)
        for chunk in reader:
            desc = chunk.pop("Description")
            vals = chunk.to_numpy(np.float32)
            mean = vals.mean(axis=1)
            keep = mean >= MIN_MEAN_TPM
            if keep.any():
                kept.append(np.round(np.log1p(vals[keep]), 2))  # 0.01 log-units: ample precision, ~2x smaller
                genes.append(pd.DataFrame({"gene_version_id": chunk.index[keep],
                                           "symbol": desc.to_numpy()[keep], "mean_tpm": mean[keep]}))
            samples = chunk.columns
    X = np.vstack(kept).T  # samples x genes
    g = pd.concat(genes, ignore_index=True)
    g.insert(0, "gene_id", g["gene_version_id"].str.split(".").str[0])
    expr = pd.DataFrame(X, index=pd.Index(samples, name="sample_id"), columns=g["gene_id"].to_numpy())
    expr.reset_index().to_parquet(INTERIM / "expression.parquet", index=False, compression="zstd")
    g.to_parquet(INTERIM / "genes.parquet", index=False)
    meta = meta.set_index("sample_id").reindex(pd.Index(samples, name="sample_id")).reset_index()
    meta.to_parquet(INTERIM / "metadata.parquet", index=False)
    print("expression", expr.shape, "metadata", meta.shape)


def load(tissue: str | None = None, tissue_detail: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X samples x genes log1p TPM, metadata) aligned on sample_id."""
    meta = pd.read_parquet(INTERIM / "metadata.parquet")
    if tissue is not None:
        meta = meta[meta["tissue"] == tissue]
    if tissue_detail is not None:
        meta = meta[meta["tissue_detail"] == tissue_detail]
    X = pd.read_parquet(INTERIM / "expression.parquet").set_index("sample_id")
    X = X.loc[meta["sample_id"]]
    return X, meta.reset_index(drop=True)


def to_symbols(X: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(INTERIM / "genes.parquet").set_index("gene_id")["symbol"]
    out = X.copy()
    out.columns = g.reindex(X.columns).to_numpy()
    return out.T.groupby(level=0).mean().T


if __name__ == "__main__":
    build()
