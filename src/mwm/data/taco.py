"""TACO rodent transcriptomic meta-dataset loader (molecular arm, PLAN.md §3.3).

Source: Tyshkovskiy et al., "Universal transcriptomic hallmarks of mammalian ageing and
mortality", Nature 654:173-188 (2026), doi:10.1038/s41586-026-10542-3; preprocessed
meta-dataset on Zenodo doi:10.5281/zenodo.18763485 (MGB Open Access License 1.0,
non-commercial academic use). Fetch with ``python scripts/fetch_taco.py`` and
``python scripts/fetch_orthologs.py``; see ``data_cards/taco.md``.

What the raw matrix is
----------------------
``Expression_data_absolute_rodents_Scaled.csv``: rows = mouse Entrez gene IDs (rat genes
were already mapped to mouse orthologs by the authors), columns = 4,539 samples (3,876
mouse + 663 rat; 96 sources; RNA-seq *and* microarray). Values are the authors' "Scaled"
preprocessing: log expression *standardised per sample across genes* (each column has
mean ~0, sd ~1-1.5 over its measured genes; e.g. Alb in liver ~ +3.7). Gene-level
differences are thus preserved, but study/platform batch effects are NOT removed.
NaN = gene not measured / filtered in that dataset. It is NOT log1p CPM; true counts
would have to be re-derived per study from GEO (not done here).
The "relative" matrix holds the same samples centred per gene on each study's matched
control group (authors' definition; captures intervention/genotype/age effects); optional.

Outputs (``data/interim/taco/``)
--------------------------------
* ``expression.parquet``           samples x human genes (HGNC symbols, MGI 1:1 orthologs), float32
* ``expression_relative.parquet``  same for the relative matrix (only if the raw file exists)
* ``metadata.parquet``             one row per sample (see ``METADATA_COLUMNS``)
* ``genes.parquet``                mouse_entrez, mouse_symbol, human_entrez, human_symbol,
                                   frac_samples_measured

Usage
-----
>>> from mwm.data import taco
>>> taco.build()                                  # raw -> interim (idempotent)
>>> X, meta = taco.load()                         # X: DataFrame samples x human genes
>>> X, meta = taco.load(min_coverage=0.9)         # genes measured in >= 90% of samples
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import SPECIES_MEDIAN_LIFESPAN_YEARS, lifespan_fraction

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "taco"
ORTHO = ROOT / "data" / "raw" / "orthologs"
INTERIM = ROOT / "data" / "interim" / "taco"

ABS_CSV = "Expression_data_absolute_rodents_Scaled.csv"
REL_CSV = "Expression_data_relative_rodents_Scaled.csv"
ANN_ABS = "Data_annotation_absolute_rodents.xlsx"
ANN_REL = "Data_annotation_relative_rodents.xlsx"
DAYS_PER_MONTH = 365.25 / 12

METADATA_COLUMNS = [
    "sample_id",                 # TACO sample name, e.g. Mouse_GSE3150_Liver_Male.GSM68912
    "cohort",                    # "taco"
    "study",                     # source accession (GSE..., E-MTAB..., "Current" = ITP data of the paper)
    "species",                   # "mouse" | "rat"
    "strain", "tissue", "tissue_detail",
    "sex",                       # "male" | "female" | "unknown"
    "age_days", "age_months", "visit_age",   # visit_age = age in years (common contract)
    "age_frac_lifespan",         # age / species median lifespan (mwm.data.common: mouse 26 mo, rat 30 mo)
    "age_frac_maxlifespan_taco", # paper: age / species max lifespan (mouse 1460 d, rat 1387 d)
    "age_norm_taco",             # paper: age / strain-intervention-specific 99.9th pct lifespan
    "intervention", "intervention_type", "is_control",
    "expected_max_lifespan_days",   # paper's 99.9th-percentile lifespan estimate for the group
    "expected_lifespan_p90_days",   # paper's 90th-percentile lifespan estimate for the group
    "expected_log10_hazard",        # paper's expected log10 mortality hazard at sample age
    "lifespan_effect_log_ratio",    # log(p90 lifespan of group / p90 of matched study controls)
    "diff_log10_hazard_vs_control", # paper (relative annotation): log10 hazard minus matched control
    "diff_norm_age_vs_control",     # paper (relative annotation): normalised age minus matched control
    "platform", "platform_type",
    "dataset_source",
]


# --------------------------------------------------------------------------------------
# orthologs
# --------------------------------------------------------------------------------------
def load_orthologs(species: str = "mouse") -> pd.DataFrame:
    """MGI one-to-one rodent->human orthologs (built by scripts/fetch_orthologs.py)."""
    f = ORTHO / f"{species}_human_1to1.tsv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run python scripts/fetch_orthologs.py")
    return pd.read_csv(f, sep="\t")


# --------------------------------------------------------------------------------------
# raw readers
# --------------------------------------------------------------------------------------
def _read_matrix(path: Path) -> pd.DataFrame:
    """Read a TACO genes x samples CSV -> float32 DataFrame (index = mouse Entrez int)."""
    import pyarrow.csv as pacsv

    tbl = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        null_values=["NA", ""], strings_can_be_null=True))
    first = tbl.column_names[0]
    idx = pd.Index(tbl.column(first).to_pandas().astype("int64"), name="mouse_entrez")
    tbl = tbl.drop([first])
    arr = np.column_stack([c.to_numpy(zero_copy_only=False).astype(np.float32)
                           for c in tbl.columns])
    return pd.DataFrame(arr, index=idx, columns=tbl.column_names)


def _read_annotation() -> pd.DataFrame:
    a = pd.read_excel(RAW / ANN_ABS)
    rel_f = RAW / ANN_REL
    if rel_f.exists():
        r = pd.read_excel(rel_f)
        keep = ["Sample", "Difference.Hazard.log10", "Difference.Normalized_age.by_99.9th_percentile"]
        a = a.merge(r[keep], on="Sample", how="left")
    return a


def _lifespan_effect(a: pd.DataFrame) -> pd.Series:
    """log(p90 lifespan estimate of the group / median p90 estimate of controls in same
    study x species x sex). NaN when no control or no estimate; 0 for controls."""
    p90 = a["Estimate.Lifespan.90th_percentile"].astype(float)
    ctrl = a["Intervention"].eq("Control")
    keys = ["Source", "Species", "Sex"]
    ref = (a[ctrl].assign(p90=p90[ctrl]).groupby(keys)["p90"].median().rename("ref").reset_index())
    ref_al = a[keys].merge(ref, on=keys, how="left")["ref"].to_numpy()
    out = np.log(p90.to_numpy() / ref_al)
    out[ctrl.to_numpy()] = 0.0
    return pd.Series(out, index=a.index)


def build_metadata(a: pd.DataFrame | None = None) -> pd.DataFrame:
    a = _read_annotation() if a is None else a
    species = a["Species"].str.lower()
    age_days = a["Age.days"].astype(float)
    age_years = age_days / 365.25
    frac = np.full(len(a), np.nan)
    for sp in species.unique():
        m = (species == sp).to_numpy()
        frac[m] = lifespan_fraction(age_years[m].to_numpy(), sp)
    sex = a["Sex"].str.lower().where(a["Sex"].str.lower().isin(["male", "female"]), "unknown")
    meta = pd.DataFrame({
        "sample_id": a["Sample"].astype(str),
        "cohort": "taco",
        "study": a["Source"].astype(str),
        "species": species,
        "strain": a["Strain"].astype(str),
        "tissue": a["Tissue"].astype(str),
        "tissue_detail": a["Tissue.details"].astype(str),
        "sex": sex,
        "age_days": age_days,
        "age_months": age_days / DAYS_PER_MONTH,
        "visit_age": age_years,
        "age_frac_lifespan": frac,
        "age_frac_maxlifespan_taco": a["Age.Chronological.Normalized_by_species_max_lifespan"].astype(float),
        "age_norm_taco": a["Age.Normalized_by_99.9th_percentile"].astype(float),
        "intervention": a["Intervention"].astype(str),
        "intervention_type": a["Intervention.type"].astype(str),
        "is_control": a["Intervention"].eq("Control"),
        "expected_max_lifespan_days": a["Estimate.Lifespan.99.9th_percentile"].astype(float),
        "expected_lifespan_p90_days": a["Estimate.Lifespan.90th_percentile"].astype(float),
        "expected_log10_hazard": a["Expected_Hazard.log10"].astype(float),
        "lifespan_effect_log_ratio": _lifespan_effect(a),
        "diff_log10_hazard_vs_control": a.get("Difference.Hazard.log10", np.nan),
        "diff_norm_age_vs_control": a.get("Difference.Normalized_age.by_99.9th_percentile", np.nan),
        "platform": a["Platform"].astype(str),
        "platform_type": a["Platform.type"].astype(str),
        "dataset_source": "TACO meta-dataset, Zenodo 10.5281/zenodo.18763485 / " + a["Source"].astype(str),
    })
    return meta[METADATA_COLUMNS]


def _to_human(mat: pd.DataFrame, ortho: pd.DataFrame) -> pd.DataFrame:
    """genes(mouse Entrez) x samples -> samples x human symbols (1:1 orthologs only)."""
    o = ortho.set_index("mouse_entrez")
    common = mat.index.intersection(o.index)
    sub = mat.loc[common]
    sub.index = o.loc[common, "human_symbol"].to_numpy()
    out = sub.T
    out.index.name = "sample_id"
    out.columns.name = "human_gene"
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def build(force: bool = False, relative: bool = True) -> Path:
    """raw -> data/interim/taco/*.parquet. Skips work if outputs exist (unless force)."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    f_expr, f_meta, f_genes = INTERIM / "expression.parquet", INTERIM / "metadata.parquet", INTERIM / "genes.parquet"
    ortho = load_orthologs("mouse")
    meta = build_metadata()
    if force or not f_meta.exists():
        meta.to_parquet(f_meta, index=False)
    if force or not f_expr.exists():
        mat = _read_matrix(RAW / ABS_CSV)
        n_mouse = len(mat)
        X = _to_human(mat, ortho).reindex(meta["sample_id"])
        X.to_parquet(f_expr)
        cov = X.notna().mean(axis=0)
        g = ortho[ortho["human_symbol"].isin(X.columns)].copy()
        g["frac_samples_measured"] = g["human_symbol"].map(cov).astype(float)
        g.to_parquet(f_genes, index=False)
        print(f"[taco] {n_mouse} mouse genes -> {X.shape[1]} human 1:1 orthologs; "
              f"expression {X.shape}")
        del mat, X
    f_rel = INTERIM / "expression_relative.parquet"
    if relative and (RAW / REL_CSV).exists() and (force or not f_rel.exists()):
        R = _to_human(_read_matrix(RAW / REL_CSV), ortho).reindex(meta["sample_id"])
        R.to_parquet(f_rel)
        print(f"[taco] relative expression {R.shape}")
    return INTERIM


def load(min_coverage: float = 0.0, relative: bool = False,
         species: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, meta). X: samples x human genes (float32, NaN = not measured),
    rows aligned with meta. ``min_coverage`` keeps genes measured in at least that
    fraction of (the selected) samples."""
    f = INTERIM / ("expression_relative.parquet" if relative else "expression.parquet")
    if not f.exists():
        build()
    meta = pd.read_parquet(INTERIM / "metadata.parquet")
    X = pd.read_parquet(f)
    if species is not None:
        keep = meta["species"].eq(species).to_numpy()
        meta, X = meta[keep].reset_index(drop=True), X[keep]
    if min_coverage > 0:
        X = X.loc[:, X.notna().mean(axis=0) >= min_coverage]
    assert (X.index.to_numpy() == meta["sample_id"].to_numpy()).all()
    return X, meta


if __name__ == "__main__":  # python -m mwm.data.taco
    build()
    X, meta = load()
    print("expression", X.shape, "metadata", meta.shape)
    print(meta.groupby(["species", "intervention_type"]).size())
    print("species median lifespan (years) used:", SPECIES_MEDIAN_LIFESPAN_YEARS)
