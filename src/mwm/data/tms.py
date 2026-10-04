"""Tabula Muris Senis (TMS) loaders: FACS single-cell pseudobulk + bulk RNA-seq atlas.

Sources (fetch with ``python scripts/fetch_tms.py``; see ``data_cards/tabula_muris_senis.md``):

* FACS / Smart-seq2 single-cell atlas: Tabula Muris Consortium, Nature 583:590 (2020),
  figshare 10.6084/m9.figshare.8273102 file ``tabula-muris-senis-facs-official-raw-obj.h5ad``
  (raw counts; MIT licence). Reduced here to pseudobulk = sum of raw counts per
  mouse x tissue x cell_ontology_class (groups with >= ``min_cells`` cells).
* Bulk RNA-seq organ atlas: Schaum et al., Nature 583:596 (2020), GEO GSE132040
  (raw HTSeq counts, 17 organs, 1-27 months).

Outputs (``data/interim/tms/``)
-------------------------------
* ``facs_pseudobulk_counts.parquet``  groups x mouse gene symbols, summed raw counts (float32)
* ``facs_pseudobulk_meta.parquet``    group_id, mouse_id, tissue, cell_type, cell_ontology_id,
                                      sex, age_months, visit_age, age_frac_lifespan, n_cells,
                                      total_counts, cohort, dataset_source
* ``bulk_counts.parquet``             samples x mouse gene symbols, raw counts (float32)
* ``bulk_meta.parquet``               sample_id, mouse_id, tissue, sex, age_months, ...

``load_pseudobulk()`` / ``load_bulk()`` return log1p-CPM expression mapped to human
one-to-one orthologs (MGI; ``data/raw/orthologs/mouse_human_1to1.tsv``), samples x
human gene symbols, plus metadata.

Usage
-----
>>> from mwm.data import tms
>>> X, meta = tms.load_pseudobulk()          # pseudobulk log1p CPM, human genes
>>> Xb, mb = tms.load_bulk()                 # GSE132040 bulk log1p CPM, human genes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import lifespan_fraction

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "tms"
ORTHO = ROOT / "data" / "raw" / "orthologs"
INTERIM = ROOT / "data" / "interim" / "tms"

FACS_PSEUDOBULK_COUNTS = "facs_pseudobulk_counts.parquet"
FACS_PSEUDOBULK_META = "facs_pseudobulk_meta.parquet"
BULK_COUNTS = "bulk_counts.parquet"
BULK_META = "bulk_meta.parquet"
BULK_COUNTS_RAW = "GSE132040_counts.csv.gz"
BULK_META_RAW = "GSE132040_MACA_Bulk_metadata.csv.gz"

# GSE132040 "source name" tissue prefixes -> readable tissue names
BULK_TISSUES = {
    "BAT": "Brown adipose", "GAT": "Gonadal adipose", "MAT": "Mesenteric adipose",
    "SCAT": "Subcutaneous adipose", "Bone": "Bone", "Brain": "Brain", "Heart": "Heart",
    "Kidney": "Kidney", "Limb_Muscle": "Limb muscle", "Liver": "Liver", "Lung": "Lung",
    "Marrow": "Bone marrow", "Pancreas": "Pancreas", "Skin": "Skin",
    "Small_Intestine": "Small intestine", "Spleen": "Spleen", "WBC": "White blood cells",
}


# --------------------------------------------------------------------------------------
# minimal h5ad reader (h5py only; no anndata dependency)
# --------------------------------------------------------------------------------------
def _as_str(v: np.ndarray) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v], dtype=object)


def _read_h5_str(ds) -> np.ndarray:
    v = ds[()]
    return _as_str(v) if v.dtype.kind in ("S", "O") else v


def _read_obs_column(obs, name: str) -> np.ndarray:
    """Read one obs column, supporting anndata's group encodings (>=0.7) incl. categoricals
    stored either as {categories, codes} groups or as codes + obs/__categories/<name>."""
    import h5py

    node = obs[name]
    if isinstance(node, h5py.Group):  # anndata >= 0.8 categorical
        cats = _read_h5_str(node["categories"])
        codes = node["codes"][()]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
    vals = node[()]
    if "__categories" in obs and name in obs["__categories"]:
        cats = _read_h5_str(obs["__categories"][name])
        return np.where(vals >= 0, cats[np.clip(vals, 0, None)], None)
    if "categories" in node.attrs:  # anndata 0.7: attr holds a reference
        cats = _read_h5_str(obs.file[node.attrs["categories"]])
        return np.where(vals >= 0, cats[np.clip(vals, 0, None)], None)
    if vals.dtype.kind in ("S", "O"):
        return _read_h5_str(node)
    return vals


def _read_obs(f, columns: list[str]) -> pd.DataFrame:
    obs = f["obs"]
    if hasattr(obs, "dtype") and obs.dtype.names:  # legacy anndata (<0.7): compound dataset
        rec = obs[()]
        out = {}
        for c in columns:
            v = rec[c]
            uns = f["uns"]
            cat_ds = (uns[f"{c}_categories"] if f"{c}_categories" in uns  # anndata 0.6.x
                      else uns["__categories"][c] if "__categories" in uns and c in uns["__categories"]
                      else None)
            if cat_ds is not None:
                cats = _read_h5_str(cat_ds)
                v = np.where(v >= 0, cats[np.clip(v, 0, None)], None)
            elif v.dtype.kind in ("S", "O"):
                v = _as_str(v)
            out[c] = v
        return pd.DataFrame(out)
    return pd.DataFrame({c: _read_obs_column(obs, c) for c in columns})


def _read_var_names(f) -> np.ndarray:
    var = f["var"]
    if hasattr(var, "dtype") and var.dtype.names:
        return _as_str(var[()]["index"])
    key = var.attrs.get("_index", "index")
    if isinstance(key, bytes):
        key = key.decode()
    return _read_h5_str(var[key])


# --------------------------------------------------------------------------------------
# build: FACS h5ad -> pseudobulk
# --------------------------------------------------------------------------------------
def build_facs_pseudobulk(h5ad: Path, out_dir: Path = INTERIM, min_cells: int = 20,
                          chunk: int = 20_000) -> Path:
    """Sum raw counts per (mouse.id, tissue, cell_ontology_class); keep groups with
    >= min_cells cells. Streams the CSR matrix in row chunks (low memory)."""
    import h5py
    import scipy.sparse as sp

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5ad, "r") as f:
        obs_cols = ["mouse.id", "tissue", "cell_ontology_class", "cell_ontology_id", "sex", "age"]
        obs = _read_obs(f, obs_cols)
        genes = _read_var_names(f)
        X = f["X"]
        if not isinstance(X, h5py.Group):
            raise ValueError("expected sparse X (CSR group) in the raw FACS object")
        fmt = X.attrs.get("encoding-type", X.attrs.get("h5sparse_format", "csr_matrix"))
        fmt = fmt.decode() if isinstance(fmt, bytes) else fmt
        if "csr" not in fmt:
            raise ValueError(f"expected CSR, got {fmt}")
        indptr = X["indptr"][()]
        n_cells, n_genes = len(indptr) - 1, len(genes)

        obs = obs.astype({c: str for c in obs_cols})
        key = obs["mouse.id"] + "|" + obs["tissue"] + "|" + obs["cell_ontology_class"]
        grp, uniq = pd.factorize(key)
        sizes = np.bincount(grp, minlength=len(uniq))
        keep_g = np.flatnonzero(sizes >= min_cells)
        remap = np.full(len(uniq), -1)
        remap[keep_g] = np.arange(len(keep_g))
        g_of_cell = remap[grp]
        acc = np.zeros((len(keep_g), n_genes), dtype=np.float64)
        for s in range(0, n_cells, chunk):
            e = min(s + chunk, n_cells)
            lo, hi = indptr[s], indptr[e]
            sub = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], indptr[s:e + 1] - lo),
                                shape=(e - s, n_genes))
            gc = g_of_cell[s:e]
            m = gc >= 0
            if m.any():
                G = sp.csr_matrix((np.ones(m.sum()), (gc[m], np.flatnonzero(m))),
                                  shape=(len(keep_g), e - s))
                acc += (G @ sub).toarray()
            print(f"[tms] pseudobulk {e:,}/{n_cells:,} cells", end="\r")
        print()

    first = obs.groupby(grp).first()
    meta = first.iloc[keep_g].reset_index(drop=True)
    age_m = meta["age"].str.extract(r"(\d+)")[0].astype(float)
    counts = pd.DataFrame(acc.astype(np.float32), columns=pd.Index(genes, name="mouse_gene"))
    gid = ("tms_facs:" + meta["mouse.id"] + "|" + meta["tissue"] + "|" + meta["cell_ontology_class"])
    counts.index = pd.Index(gid, name="group_id")
    sex = meta["sex"].str.lower().map({"male": "male", "female": "female", "m": "male", "f": "female"})
    pb_meta = pd.DataFrame({
        "group_id": gid.to_numpy(),
        "cohort": "tms_facs",
        "mouse_id": meta["mouse.id"].to_numpy(),
        "tissue": meta["tissue"].to_numpy(),
        "cell_type": meta["cell_ontology_class"].to_numpy(),
        "cell_ontology_id": meta["cell_ontology_id"].to_numpy(),
        "sex": sex.fillna("unknown").to_numpy(),
        "age_months": age_m.to_numpy(),
        "visit_age": (age_m / 12).to_numpy(),
        "age_frac_lifespan": lifespan_fraction((age_m / 12).to_numpy(), "mouse"),
        "n_cells": sizes[keep_g],
        "total_counts": acc.sum(axis=1),
        "dataset_source": "Tabula Muris Senis FACS (figshare 8273102, raw obj)",
    })
    counts.to_parquet(out_dir / FACS_PSEUDOBULK_COUNTS)
    pb_meta.to_parquet(out_dir / FACS_PSEUDOBULK_META, index=False)
    print(f"[tms] pseudobulk: {len(keep_g)} groups (>= {min_cells} cells) of {len(uniq)}; "
          f"{n_genes} genes; {n_cells:,} cells")
    return out_dir / FACS_PSEUDOBULK_COUNTS


# --------------------------------------------------------------------------------------
# build: GSE132040 bulk
# --------------------------------------------------------------------------------------
def build_bulk(force: bool = False) -> Path:
    INTERIM.mkdir(parents=True, exist_ok=True)
    fc, fm = INTERIM / BULK_COUNTS, INTERIM / BULK_META
    if fc.exists() and fm.exists() and not force:
        return fc
    c = pd.read_csv(RAW / BULK_COUNTS_RAW, index_col=0)
    c = c[~c.index.str.startswith("__")]  # drop HTSeq summary rows
    c.columns = c.columns.str.replace(r"\.gencode\.vM19$", "", regex=True)
    m = pd.read_csv(RAW / BULK_META_RAW)
    m.columns = [x.strip() for x in m.columns]
    m = m[m["Sample name"].isin(c.columns)].copy()
    src = m["source name"].astype(str)
    tissue_code = src.str.rsplit("_", n=1).str[0]
    mouse = src.str.rsplit("_", n=1).str[1]
    age_m = pd.to_numeric(m["characteristics: age"], errors="coerce")
    sex = m["characteristics: sex"].str.lower().map({"m": "male", "f": "female"}).fillna("unknown")
    meta = pd.DataFrame({
        "sample_id": m["Sample name"].to_numpy(),
        "cohort": "tms_bulk",
        "mouse_id": ("bulk_" + mouse).to_numpy(),
        "tissue_code": tissue_code.to_numpy(),
        "tissue": tissue_code.map(BULK_TISSUES).fillna(tissue_code).to_numpy(),
        "sex": sex.to_numpy(),
        "age_months": age_m.to_numpy(),
        "visit_age": (age_m / 12).to_numpy(),
        "age_frac_lifespan": lifespan_fraction((age_m / 12).to_numpy(), "mouse"),
        "srr": m["raw file"].to_numpy(),
        "dataset_source": "Tabula Muris Senis bulk RNA-seq, GEO GSE132040 (Schaum et al. 2020)",
    })
    counts = c[meta["sample_id"]].T.astype(np.float32)
    counts.index.name, counts.columns.name = "sample_id", "mouse_gene"
    meta["total_counts"] = counts.sum(axis=1).to_numpy()
    counts.to_parquet(fc)
    meta.to_parquet(fm, index=False)
    print(f"[tms] bulk: {counts.shape[0]} samples x {counts.shape[1]} genes")
    return fc


# --------------------------------------------------------------------------------------
# normalisation + ortholog mapping
# --------------------------------------------------------------------------------------
def log1p_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=1).to_numpy()[:, None]
    return np.log1p(counts / np.where(lib > 0, lib, 1) * 1e6).astype(np.float32)


def to_human(expr: pd.DataFrame) -> pd.DataFrame:
    """samples x mouse symbols -> samples x human symbols via MGI 1:1 orthologs.
    CPM is computed on all mouse genes *before* subsetting (call log1p_cpm first)."""
    o = pd.read_csv(ORTHO / "mouse_human_1to1.tsv", sep="\t")
    o = o.drop_duplicates("mouse_symbol").set_index("mouse_symbol")["human_symbol"]
    keep = [g for g in expr.columns if g in o.index]
    out = expr[keep].copy()
    out.columns = pd.Index(o.loc[keep].to_numpy(), name="human_gene")
    return out.loc[:, ~out.columns.duplicated()]


def load_pseudobulk(min_cells: int = 20, min_counts: float = 5e4,
                    human: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FACS pseudobulk as log1p CPM (rows aligned with returned metadata)."""
    counts = pd.read_parquet(INTERIM / FACS_PSEUDOBULK_COUNTS)
    meta = pd.read_parquet(INTERIM / FACS_PSEUDOBULK_META)
    keep = ((meta["n_cells"] >= min_cells) & (meta["total_counts"] >= min_counts)).to_numpy()
    counts, meta = counts[keep], meta[keep].reset_index(drop=True)
    X = log1p_cpm(counts)
    return (to_human(X) if human else X), meta


def load_bulk(min_counts: float = 1e5, human: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GSE132040 bulk as log1p CPM (rows aligned with returned metadata)."""
    if not (INTERIM / BULK_COUNTS).exists():
        build_bulk()
    counts = pd.read_parquet(INTERIM / BULK_COUNTS)
    meta = pd.read_parquet(INTERIM / BULK_META)
    keep = (meta["total_counts"] >= min_counts).to_numpy()
    counts, meta = counts[keep], meta[keep].reset_index(drop=True)
    X = log1p_cpm(counts)
    return (to_human(X) if human else X), meta


if __name__ == "__main__":  # python -m mwm.data.tms
    build_bulk()
    Xb, mb = load_bulk()
    print("bulk", Xb.shape, mb.shape)
    if (INTERIM / FACS_PSEUDOBULK_COUNTS).exists():
        X, m = load_pseudobulk()
        print("pseudobulk", X.shape, m.shape)
