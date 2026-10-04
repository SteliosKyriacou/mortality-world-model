"""Turn a cohort directory (long + outcomes + splits) into model-ready arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (LOG_FEATURES, Standardizer, VisitTable, load_splits, long_to_visits,
                     validate_long, validate_outcomes)


@dataclass
class CohortArrays:
    visits: VisitTable                 # raw values
    Xs: np.ndarray                     # standardised, NaN where missing
    standardizer: Standardizer
    clinical: list[str]
    omics: list[str]
    split: np.ndarray                  # per visit
    u: np.ndarray                      # per visit intervention flag(s), (N, n_u)
    outcomes: pd.DataFrame             # indexed by person_id
    # per-visit survival interval for piecewise-exponential hazard training
    t_start: np.ndarray
    t_stop: np.ndarray
    died_in_interval: np.ndarray

    @property
    def features(self):
        return self.visits.features

    def idx(self, split: str) -> np.ndarray:
        return np.nonzero(self.split == split)[0]


def load_cohort_dir(path: str | Path, intervention_features=("u_nutrient",),
                    omics_unit: str = "NPX", cohort: str = "synth") -> CohortArrays:
    p = Path(path)
    long = validate_long(pd.read_parquet(p / "long.parquet"))
    outcomes = validate_outcomes(pd.read_parquet(p / "outcomes.parquet")).set_index("person_id")
    splits = load_splits(cohort, root=p).set_index("person_id")["split"]

    is_u = long["feature"].isin(intervention_features)
    ulong = long[is_u]
    long = long[~is_u]
    units = long.drop_duplicates("feature").set_index("feature")["unit"]
    omics = sorted(units.index[units == omics_unit].tolist())
    clinical = sorted(units.index[units != omics_unit].tolist())
    vt = long_to_visits(long, clinical + omics)

    # intervention per visit (0 if absent)
    u = np.zeros((len(vt.person_id), len(intervention_features)), np.float32)
    if len(ulong):
        uw = long_to_visits(ulong, list(intervention_features))
        key = pd.MultiIndex.from_arrays([uw.person_id, uw.visit_age])
        ser = pd.DataFrame(uw.X, index=key)
        look = ser.reindex(pd.MultiIndex.from_arrays([vt.person_id, vt.visit_age])).to_numpy()
        u = np.nan_to_num(look).astype(np.float32)

    split = splits.reindex(vt.person_id).to_numpy().astype(str)
    tr = split == "train"
    log_feats = {f for f in clinical if f in LOG_FEATURES}
    # synthetic markers that are log-normal by construction are also log-transformed
    log_feats |= {f for f in clinical if f in {"il6", "insulin", "nt_probnp", "cortisol", "vitamin_d"}}
    st = Standardizer(clinical + omics, log_feats).fit(vt.X[tr])
    Xs = st.transform(vt.X)

    # survival intervals: from visit age to next visit (or end of follow-up)
    o = outcomes.reindex(vt.person_id)
    end = o["age_at_death_or_censor"].to_numpy()
    ev = o["event"].to_numpy()
    nxt = pd.Series(vt.visit_age).groupby(vt.person_id).shift(-1).to_numpy()
    last = np.isnan(nxt)
    t_stop = np.where(last, np.maximum(end, vt.visit_age + 1e-3), nxt)
    died = (last & (ev == 1)).astype(np.float32)
    return CohortArrays(vt, Xs, st, clinical, omics, split, u, outcomes,
                        vt.visit_age.astype(np.float64), t_stop, died)


def make_pairs(lat: pd.DataFrame, zcols: list[str], mode: str = "consecutive") -> dict:
    """Visit pairs from a per-visit latent table (sorted by person, age).

    mode="consecutive": every adjacent pair; mode="first_last": baseline -> last visit;
    mode="all": every ordered pair i<j."""
    lat = lat.sort_values(["person_id", "visit_age"]).reset_index(drop=True)
    g = lat.groupby("person_id").cumcount().to_numpy()
    n = lat.groupby("person_id")["visit_age"].transform("size").to_numpy()
    if mode == "consecutive":
        i0 = np.nonzero(g < n - 1)[0]
        i1 = i0 + 1
    elif mode == "first_last":
        i0 = np.nonzero((g == 0) & (n >= 2))[0]
        i1 = i0 + n[i0] - 1
    elif mode == "all":
        a, b = [], []
        start = np.nonzero(g == 0)[0]
        for s, k in zip(start, n[start]):
            for x in range(k):
                for y in range(x + 1, k):
                    a.append(s + x)
                    b.append(s + y)
        i0, i1 = np.array(a, int), np.array(b, int)
    else:
        raise ValueError(mode)
    Z = lat[zcols].to_numpy(np.float32)
    A = lat["visit_age"].to_numpy(np.float32)
    U = lat[[c for c in lat.columns if c.startswith("u")][:1]].to_numpy(np.float32) \
        if any(c.startswith("u") for c in lat.columns) else np.zeros((len(lat), 1), np.float32)
    return {"z0": Z[i0], "z1": Z[i1], "a0": A[i0], "a1": A[i1], "u": U[i0],
            "person_id": lat["person_id"].to_numpy()[i0], "i0": i0, "i1": i1}
