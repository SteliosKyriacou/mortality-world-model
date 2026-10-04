"""UK Biobank loader -> shared long format (validation ladder step 5, "UKB readiness").

Written against UKB conventions *before* any UKB data has been seen; tested on a synthetic
file in the same format (tests/ukb_fake.py). Supports both column naming styles:

* ukbconv / legacy:   ``f.<field>.<instance>.<array>``   e.g. ``f.4080.0.1``; id column ``f.eid``/``eid``
* RAP (DNAnexus) dataset export: ``p<field>[_i<instance>][_a<array>]`` e.g. ``p4080_i0_a1``,
  ``p31`` (no instance), id column ``eid``.

Instances 0-3 = assessment visits (0: 2006-10 baseline, 1: 2012-13 repeat, 2: 2014+ imaging,
3: 2019+ repeat imaging). Visit age is computed from year/month of birth (34/52) and the
assessment date (53); falls back to field 21003 (integer age) if dates are missing.

Outcomes: death date 40000, primary cause 40001 (ICD-10), HES diagnoses 41270 with first
dates 41280 (array-aligned), lost to follow-up 191. Proteomics: Olink NPX long table
(``eid, ins_index, protein_id, result``) with coding 143 (``coding, meaning`` where meaning
is ``"GENE;Protein name"``) -> long rows with unit "NPX".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import validate_long, validate_outcomes

# field -> (canonical feature, unit, how to combine arrays)
UKB_FIELDS: dict[int, tuple[str, str, str]] = {
    31: ("sex", "code", "first"),
    21001: ("bmi", "kg/m2", "first"),
    48: ("waist", "cm", "first"),
    4080: ("sbp", "mmHg", "mean"),
    4079: ("dbp", "mmHg", "mean"),
    102: ("heart_rate", "bpm", "mean"),
    46: ("grip_left", "kg", "first"),
    47: ("grip_right", "kg", "first"),
    30690: ("total_chol", "mmol/L", "first"),
    30760: ("hdl", "mmol/L", "first"),
    30780: ("ldl", "mmol/L", "first"),
    30870: ("triglycerides", "mmol/L", "first"),
    30750: ("hba1c", "mmol/mol", "first"),
    30740: ("glucose", "mmol/L", "first"),
    30710: ("crp", "mg/L", "first"),
    30700: ("creatinine", "umol/L", "first"),
    30720: ("cystatin_c", "mg/L", "first"),
    30020: ("haemoglobin", "g/dL", "first"),
    30600: ("albumin", "g/L", "first"),
    30000: ("wbc", "10^9/L", "first"),
    30180: ("lymph_pct", "%", "first"),
    30040: ("mcv", "fL", "first"),
    30070: ("rdw", "%", "first"),
    30080: ("platelets", "10^9/L", "first"),
    30610: ("alp", "U/L", "first"),
    30620: ("alt", "U/L", "first"),
    30890: ("vitamin_d", "nmol/L", "first"),
    20116: ("smoking", "code", "first"),
    2443: ("diabetes", "flag", "first"),
}
MED_FIELDS = (6177, 6153)          # male / female medication for chol, BP, diabetes (1/2/3)
F_AGE, F_DATE, F_YOB, F_MOB = 21003, 53, 34, 52
F_DEATH_DATE, F_DEATH_CAUSE, F_AGE_DEATH = 40000, 40001, 40007
F_HES_CODE, F_HES_DATE, F_LOST = 41270, 41280, 191
NEGATIVE_CODES = {-1, -3, -7, -818, -121}   # do not know / prefer not / none of the above ...

_F_RE = re.compile(r"^f\.(\d+)\.(\d+)\.(\d+)$")
_P_RE = re.compile(r"^p(\d+)(?:_i(\d+))?(?:_a(\d+))?$")


def parse_column(col: str):
    """-> (field, instance, array) or None. Missing instance/array -> 0."""
    m = _F_RE.match(col)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = _P_RE.match(col)
    if m:
        return int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)
    return None


def id_column(df: pd.DataFrame) -> str:
    for c in ("eid", "f.eid", "p_eid", "participant.eid"):
        if c in df.columns:
            return c
    raise KeyError("no eid column found")


def melt_fields(df: pd.DataFrame, fields) -> pd.DataFrame:
    """Wide UKB table -> (eid, field, instance, array, value) for the requested fields."""
    eid = id_column(df)
    parsed = {c: parse_column(c) for c in df.columns}
    cols = [c for c, p in parsed.items() if p is not None and p[0] in set(fields)]
    if not cols:
        return pd.DataFrame(columns=["eid", "field", "instance", "array", "value"])
    m = df[[eid] + cols].melt(id_vars=eid, var_name="col", value_name="value").dropna(subset=["value"])
    p = np.array([parsed[c] for c in m["col"]])
    return pd.DataFrame({"eid": m[eid].astype(str).to_numpy(), "field": p[:, 0], "instance": p[:, 1],
                         "array": p[:, 2], "value": m["value"].to_numpy()})


def _to_date(s):
    return pd.to_datetime(s, errors="coerce")


@dataclass
class UKBResult:
    long: pd.DataFrame
    outcomes: pd.DataFrame
    hes: pd.DataFrame       # person_id, icd10, age_at_event
    visits: pd.DataFrame    # person_id, instance, visit_age, date


def visit_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Per (eid, instance): fractional age at assessment."""
    eid = id_column(df)
    base = melt_fields(df, [F_YOB, F_MOB])
    birth = base.pivot_table(index="eid", columns="field", values="value", aggfunc="first")
    dates = melt_fields(df, [F_DATE])
    dates["date"] = _to_date(dates["value"])
    ages = melt_fields(df, [F_AGE]).rename(columns={"value": "age_int"})
    v = dates[["eid", "instance", "date"]].merge(ages[["eid", "instance", "age_int"]], on=["eid", "instance"], how="outer")
    if F_YOB in birth.columns:
        yob = birth[F_YOB].astype(float)
        mob = birth[F_MOB].astype(float) if F_MOB in birth.columns else pd.Series(7.0, index=birth.index)
        bdate = pd.to_datetime(dict(year=yob.astype("Int64"), month=mob.fillna(7).astype("Int64"), day=15),
                               errors="coerce")
        v["birth"] = v["eid"].map(bdate)
        frac = (v["date"] - v["birth"]).dt.days / 365.25
    else:
        frac = pd.Series(np.nan, index=v.index)
    v["visit_age"] = frac.where(frac.notna(), v["age_int"].astype(float) + 0.5)
    return v.dropna(subset=["visit_age"])[["eid", "instance", "visit_age", "date"]]


def load_ukb(main: pd.DataFrame, olink: pd.DataFrame | None = None, olink_coding: pd.DataFrame | None = None,
             admin_censor_date: str = "2022-10-31", cohort: str = "ukb") -> UKBResult:
    eid = id_column(main)
    v = visit_ages(main)
    vkey = v.set_index(["eid", "instance"])["visit_age"]
    sex = melt_fields(main, [31]).groupby("eid")["value"].first()

    # ----- clinical features
    m = melt_fields(main, [f for f in UKB_FIELDS if f != 31])
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    m = m.dropna(subset=["value"])
    m = m[~m["value"].isin(NEGATIVE_CODES) | ~m["field"].isin([20116, 2443])]
    rows = []
    for f, g in m.groupby("field"):
        name, unit, how = UKB_FIELDS[f]
        agg = g.groupby(["eid", "instance"])["value"].mean() if how == "mean" else \
            g.sort_values("array").groupby(["eid", "instance"])["value"].first()
        rows.append(pd.DataFrame({"eid": agg.index.get_level_values(0), "instance": agg.index.get_level_values(1),
                                  "feature": name, "value": agg.to_numpy(), "unit": unit}))
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["eid", "instance", "feature", "value", "unit"])
    # smoking/diabetes negative codes -> drop
    long = long[~((long.feature.isin(["smoking", "diabetes"])) & (long.value < 0))]
    # grip = max(left, right)
    grip = long[long.feature.isin(["grip_left", "grip_right"])].groupby(["eid", "instance"])["value"].max()
    long = long[~long.feature.isin(["grip_left", "grip_right"])]
    long = pd.concat([long, pd.DataFrame({"eid": grip.index.get_level_values(0), "instance": grip.index.get_level_values(1),
                                          "feature": "grip", "value": grip.to_numpy(), "unit": "kg"})])
    # medication flags (6177 men / 6153 women): 1 chol-lowering, 2 BP, 3 insulin
    med = melt_fields(main, list(MED_FIELDS))
    if len(med):
        med["value"] = pd.to_numeric(med["value"], errors="coerce")
        anyrow = med.groupby(["eid", "instance"])["value"]
        for code, name in ((1, "lipid_med"), (2, "bp_med")):
            flag = anyrow.apply(lambda s: float((s == code).any()) if (s >= 0).any() or (s == -7).any() else np.nan).dropna()
            long = pd.concat([long, pd.DataFrame({"eid": flag.index.get_level_values(0),
                                                  "instance": flag.index.get_level_values(1),
                                                  "feature": name, "value": flag.to_numpy(), "unit": "flag"})])
    # sex repeated at each visit
    sv = v[["eid", "instance"]].copy()
    sv["value"] = sv["eid"].map(sex).astype(float)
    sv = sv.dropna()
    long = pd.concat([long, sv.assign(feature="sex", unit="code")])

    # ----- olink (long NPX)
    if olink is not None:
        o = olink.rename(columns={"ins_index": "instance", "result": "value"}).copy()
        o["eid"] = o["eid"].astype(str)
        if olink_coding is not None:
            cmap = olink_coding.set_index("coding")["meaning"].astype(str).str.split(";").str[0].str.lower()
            o["feature"] = "olink_" + o["protein_id"].map(cmap).fillna(o["protein_id"].astype(str))
        else:
            o["feature"] = "olink_" + o["protein_id"].astype(str)
        o["unit"] = "NPX"
        long = pd.concat([long, o[["eid", "instance", "feature", "value", "unit"]]])

    long["visit_age"] = [vkey.get((e, i), np.nan) for e, i in zip(long["eid"], long["instance"])]
    long = long.dropna(subset=["visit_age", "value"])
    long = long.assign(person_id=long["eid"].astype(str), cohort=cohort)
    long = validate_long(long.drop_duplicates(["person_id", "visit_age", "feature"]))

    # ----- outcomes
    base = v.sort_values("visit_age").groupby("eid").first()
    births = base["date"] - pd.to_timedelta(base["visit_age"] * 365.25, unit="D")
    dd = melt_fields(main, [F_DEATH_DATE])
    dd["date"] = _to_date(dd["value"])
    death = dd.groupby("eid")["date"].min()
    cause = melt_fields(main, [F_DEATH_CAUSE]).sort_values(["instance", "array"]).groupby("eid")["value"].first()
    lost = melt_fields(main, [F_LOST])
    lost = pd.Series(_to_date(lost["value"]).to_numpy(), index=lost["eid"]).groupby(level=0).min() if len(lost) else pd.Series(dtype="datetime64[ns]")
    censor = pd.Timestamp(admin_censor_date)
    out = pd.DataFrame(index=base.index)
    out["age_at_baseline"] = base["visit_age"]
    dth = death.reindex(out.index)
    lst = lost.reindex(out.index)
    end_date = dth.where(dth.notna(), lst.where(lst.notna() & (lst < censor), censor))
    out["event"] = dth.notna().astype(int)
    out["age_at_death_or_censor"] = ((end_date - births.reindex(out.index)).dt.days / 365.25).fillna(out["age_at_baseline"])
    out["age_at_death_or_censor"] = np.maximum(out["age_at_death_or_censor"], out["age_at_baseline"])

    def cause_cat(code):
        if not isinstance(code, str) or not code:
            return "unknown"
        c = code.upper()
        if c.startswith("I"):
            return "cvd"
        if c.startswith("C") or (c.startswith("D") and c[1:3].isdigit() and int(c[1:3]) <= 48):
            return "cancer"
        return "other"
    out["cause"] = [cause_cat(cause.get(e)) if ev else "none" for e, ev in zip(out.index, out["event"])]
    out = validate_outcomes(out.reset_index().rename(columns={"eid": "person_id"}).assign(
        person_id=lambda d: d["person_id"].astype(str)))

    # ----- HES (array-aligned code/date)
    hc = melt_fields(main, [F_HES_CODE]).rename(columns={"value": "icd10"})
    hd = melt_fields(main, [F_HES_DATE])
    hd["date"] = _to_date(hd["value"])
    hes = hc.merge(hd[["eid", "array", "date"]], on=["eid", "array"], how="left")
    hes["age_at_event"] = (hes["date"] - hes["eid"].map(births)).dt.days / 365.25
    hes = hes.rename(columns={"eid": "person_id"})[["person_id", "icd10", "age_at_event"]]
    visits = v.rename(columns={"eid": "person_id"})
    return UKBResult(long, out, hes.reset_index(drop=True), visits.reset_index(drop=True))


def read_main(path: str, fields=None, **kw) -> pd.DataFrame:
    """Read a UKB main table (csv/tsv/parquet), optionally only the columns of given fields."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if str(path).endswith((".tsv", ".tab", ".txt")) else ","
    head = pd.read_csv(path, sep=sep, nrows=0).columns
    use = None
    if fields is not None:
        fs = set(fields)
        use = [c for c in head if c in ("eid", "f.eid") or ((p := parse_column(c)) is not None and p[0] in fs)]
    return pd.read_csv(path, sep=sep, usecols=use, low_memory=False, **kw)


ALL_FIELDS = sorted(set(UKB_FIELDS) | set(MED_FIELDS) | {F_AGE, F_DATE, F_YOB, F_MOB, F_DEATH_DATE,
                                                          F_DEATH_CAUSE, F_AGE_DEATH, F_HES_CODE,
                                                          F_HES_DATE, F_LOST})
