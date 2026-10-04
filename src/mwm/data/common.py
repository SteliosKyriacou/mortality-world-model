"""Shared data contract for every cohort loader (PLAN.md §3.1).

Every cohort loader (`src/mwm/data/<cohort>.py`) must emit two tables:

1. **Long table** -- one row per (person, visit, feature):

   ============  ========  ==========================================================
   column        dtype     meaning
   ============  ========  ==========================================================
   person_id     str       cohort-unique id; we prefix with "<cohort>:" when pooling
   cohort        str       short lowercase cohort name, e.g. "nhanes", "ukb", "synth"
   visit_age     float64   age in (fractional) years at the visit/sample
   feature       str       canonical feature name (see CANONICAL_FEATURES), else free
   value         float64   numeric value in *raw* units (no z-scoring, no log)
   unit          str       unit string of `value` (e.g. "mg/dL", "mmHg", "NPX")
   ============  ========  ==========================================================

   Rules: no NaN in `value` (missing = row absent; we never impute in stored data),
   binary flags stored as 0.0/1.0 with unit "flag", categorical codes as float with
   unit "code". Static covariates (sex) are repeated at every visit (or at least at
   baseline). Rodent arm: `visit_age` is in species-native units; use
   `lifespan_fraction()` to put it on the shared clock and record it in `unit`-free
   column `visit_age` only after conversion (cohort name tells which).

2. **Outcomes table** -- one row per person:

   =======================  ========  =================================================
   column                   dtype     meaning
   =======================  ========  =================================================
   person_id                str
   age_at_baseline          float64   age at first visit in the long table
   age_at_death_or_censor   float64   age at death (event=1) or last known alive (0)
   event                    int8      1 = died, 0 = censored
   cause                    str       one of CAUSE_VALUES ("none" when event == 0)
   =======================  ========  =================================================

Files: `data/interim/<cohort>/long.parquet` and `data/interim/<cohort>/outcomes.parquet`
(use `save_cohort` / `load_cohort`). Splits: `data/processed/splits/<cohort>.csv`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LONG_COLUMNS = ["person_id", "cohort", "visit_age", "feature", "value", "unit"]
LONG_DTYPES = {
    "person_id": "string",
    "cohort": "string",
    "visit_age": "float64",
    "feature": "string",
    "value": "float64",
    "unit": "string",
}
OUTCOME_COLUMNS = ["person_id", "age_at_baseline", "age_at_death_or_censor", "event", "cause"]
OUTCOME_DTYPES = {
    "person_id": "string",
    "age_at_baseline": "float64",
    "age_at_death_or_censor": "float64",
    "event": "int8",
    "cause": "string",
}
CAUSE_VALUES = ("none", "cvd", "cancer", "other", "unknown")

# Canonical names for the v1 human core panel (PLAN.md §3.2) -> preferred raw unit.
# Loaders should map their native variables onto these names (converting units)
# whenever the concept matches; anything else may use a free-form snake_case name.
CANONICAL_FEATURES: dict[str, str] = {
    "sex": "code",               # 0 = female, 1 = male
    "bmi": "kg/m2",
    "waist": "cm",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "total_chol": "mmol/L",
    "hdl": "mmol/L",
    "ldl": "mmol/L",
    "triglycerides": "mmol/L",
    "hba1c": "mmol/mol",
    "glucose": "mmol/L",
    "crp": "mg/L",
    "fibrinogen": "g/L",
    "creatinine": "umol/L",
    "egfr": "mL/min/1.73m2",
    "haemoglobin": "g/dL",
    "albumin": "g/L",
    "grip": "kg",
    "smoking": "code",           # 0 never, 1 former, 2 current
    "diabetes": "flag",
    "bp_med": "flag",
    "lipid_med": "flag",
    "heart_rate": "bpm",
    "wbc": "10^9/L",
    "lymph_pct": "%",
    "mcv": "fL",
    "rdw": "%",
    "alp": "U/L",
}
# Markers that are log-transformed before z-scoring (skewed, positive).
LOG_FEATURES = {"crp", "triglycerides", "creatinine", "glucose", "alp", "wbc", "fibrinogen"}

SPECIES_MEDIAN_LIFESPAN_YEARS = {"human": 80.0, "mouse": 26.0 / 12.0, "rat": 30.0 / 12.0}


# --------------------------------------------------------------------------------------
# validation / IO
# --------------------------------------------------------------------------------------
def validate_long(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Check and coerce a long table to the contract. Returns a new DataFrame."""
    missing = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"long table missing columns: {missing}")
    out = df[LONG_COLUMNS].astype(LONG_DTYPES).reset_index(drop=True)
    if out["value"].isna().any():
        raise ValueError("long table has NaN values; drop missing rows instead")
    if out["visit_age"].isna().any():
        raise ValueError("long table has NaN visit_age")
    if out["person_id"].isna().any() or out["feature"].isna().any():
        raise ValueError("long table has null person_id/feature")
    if strict:
        dup = out.duplicated(["person_id", "visit_age", "feature"])
        if dup.any():
            raise ValueError(f"{int(dup.sum())} duplicate (person_id, visit_age, feature) rows")
    return out


def validate_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OUTCOME_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"outcomes table missing columns: {missing}")
    out = df[OUTCOME_COLUMNS].astype(OUTCOME_DTYPES).reset_index(drop=True)
    if out["person_id"].duplicated().any():
        raise ValueError("outcomes table has duplicate person_id")
    if not set(out["event"].unique()) <= {0, 1}:
        raise ValueError("event must be 0/1")
    bad = ~out["cause"].isin(CAUSE_VALUES)
    if bad.any():
        raise ValueError(f"unknown cause values: {sorted(out.loc[bad, 'cause'].unique())}")
    if (out.loc[out["event"] == 0, "cause"] != "none").any():
        raise ValueError("censored rows (event=0) must have cause='none'")
    if (out["age_at_death_or_censor"] < out["age_at_baseline"] - 1e-6).any():
        raise ValueError("age_at_death_or_censor < age_at_baseline for some rows")
    return out


def save_cohort(long: pd.DataFrame, outcomes: pd.DataFrame, cohort: str,
                root: str | Path = "data/interim") -> Path:
    d = Path(root) / cohort
    d.mkdir(parents=True, exist_ok=True)
    validate_long(long).to_parquet(d / "long.parquet", index=False)
    validate_outcomes(outcomes).to_parquet(d / "outcomes.parquet", index=False)
    return d


def load_cohort(cohort: str, root: str | Path = "data/interim") -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(root) / cohort
    return (validate_long(pd.read_parquet(d / "long.parquet")),
            validate_outcomes(pd.read_parquet(d / "outcomes.parquet")))


def lifespan_fraction(age: np.ndarray | float, species: str) -> np.ndarray | float:
    """Shared clock for cross-species work (PLAN.md §3.3)."""
    return np.asarray(age) / SPECIES_MEDIAN_LIFESPAN_YEARS[species]


# --------------------------------------------------------------------------------------
# long -> visit-level wide arrays
# --------------------------------------------------------------------------------------
@dataclass
class VisitTable:
    """Wide visit-level view: one row per (person, visit)."""
    person_id: np.ndarray            # (N_visits,) str
    visit_age: np.ndarray            # (N_visits,) float
    features: list[str]
    X: np.ndarray                    # (N_visits, F) float32, NaN where missing
    visit_index: np.ndarray          # (N_visits,) int, 0 = baseline, per person

    @property
    def mask(self) -> np.ndarray:
        return ~np.isnan(self.X)

    def subset(self, persons: Iterable[str]) -> "VisitTable":
        keep = np.isin(self.person_id, np.asarray(list(persons)))
        return VisitTable(self.person_id[keep], self.visit_age[keep], self.features,
                          self.X[keep], self.visit_index[keep])


def long_to_visits(long: pd.DataFrame, features: list[str] | None = None,
                   age_round: int = 3) -> VisitTable:
    """Pivot a long table into a VisitTable. Visits are keyed by (person, rounded age)."""
    df = long.copy()
    df["visit_age"] = df["visit_age"].round(age_round)
    if features is None:
        features = sorted(df["feature"].unique().tolist())
    df = df[df["feature"].isin(features)]
    wide = df.pivot_table(index=["person_id", "visit_age"], columns="feature",
                          values="value", aggfunc="mean")
    wide = wide.reindex(columns=features)
    wide = wide.sort_index()
    pid = wide.index.get_level_values(0).to_numpy().astype(str)
    age = wide.index.get_level_values(1).to_numpy().astype(np.float64)
    vidx = pd.Series(pid).groupby(pid).cumcount().to_numpy()
    return VisitTable(pid, age, list(features), wide.to_numpy(np.float32), vidx)


# --------------------------------------------------------------------------------------
# normalisation (fit on TRAIN persons only)
# --------------------------------------------------------------------------------------
@dataclass
class Standardizer:
    features: list[str]
    log_features: set[str] = field(default_factory=set)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64).copy()
        for j, f in enumerate(self.features):
            if f in self.log_features:
                X[:, j] = np.log(np.clip(X[:, j], 1e-6, None))
        return X

    def fit(self, X: np.ndarray) -> "Standardizer":
        Z = self._pre(X)
        self.mean = np.nanmean(Z, axis=0)
        self.std = np.nanstd(Z, axis=0)
        self.mean = np.nan_to_num(self.mean)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-8), self.std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((self._pre(X) - self.mean) / self.std).astype(np.float32)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(Z, dtype=np.float64) * self.std + self.mean
        for j, f in enumerate(self.features):
            if f in self.log_features:
                X[..., j] = np.exp(X[..., j])
        return X

    def to_dict(self) -> dict:
        return {"features": self.features, "log_features": sorted(self.log_features),
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(d["features"], set(d["log_features"]), np.array(d["mean"]), np.array(d["std"]))


# --------------------------------------------------------------------------------------
# person-level splits (PLAN.md §3.4): fixed once, saved, never re-drawn
# --------------------------------------------------------------------------------------
def make_splits(outcomes: pd.DataFrame, strata: pd.DataFrame | None = None,
                fractions=(0.75, 0.10, 0.15), seed: int = 20261003) -> pd.DataFrame:
    """Stratified person-level split. `strata` is an optional DataFrame indexed like
    outcomes with extra stratification columns (e.g. sex). Age band (10y) and event
    are always used."""
    rng = np.random.default_rng(seed)
    o = outcomes.reset_index(drop=True)
    key = (np.floor(o["age_at_baseline"] / 10).astype(int).astype(str) + "_" + o["event"].astype(str))
    if strata is not None:
        for c in strata.columns:
            key = key + "_" + strata[c].astype(str).reset_index(drop=True)
    split = np.empty(len(o), dtype=object)
    for _, idx in pd.Series(np.arange(len(o))).groupby(key.to_numpy()):
        idx = rng.permutation(idx.to_numpy())
        n = len(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    return pd.DataFrame({"person_id": o["person_id"].astype(str), "split": split})


def save_splits(splits: pd.DataFrame, cohort: str, root: str | Path = "data/processed/splits") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{cohort}.csv"
    if f.exists():
        old = pd.read_csv(f, dtype={"person_id": str})
        if not old.sort_values("person_id").reset_index(drop=True).equals(
                splits.sort_values("person_id").reset_index(drop=True)):
            raise FileExistsError(f"{f} exists with different content; splits are never re-drawn")
        return f
    splits.to_csv(f, index=False)
    return f


def load_splits(cohort: str, root: str | Path = "data/processed/splits") -> pd.DataFrame:
    return pd.read_csv(Path(root) / f"{cohort}.csv", dtype={"person_id": str})
