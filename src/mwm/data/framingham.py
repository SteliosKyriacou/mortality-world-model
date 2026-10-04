"""Framingham Heart Study teaching dataset (frmgham2) -> common long format + outcomes.

TEACHING DATA ONLY: anonymised/perturbed by NHLBI, "inappropriate for publication purposes".
Use for end-to-end pipeline smoke tests (real-shaped longitudinal data: 4,434 people,
up to 3 exams ~6 years apart, 24-year mortality/CVD follow-up). Fetch with
``python scripts/fetch_framingham.py``; see ``data_cards/framingham.md``.

Timing: ``TIME`` = days since the period-1 exam; ``AGE`` = integer age at each exam.
visit_age = AGE(period 1) + 0.5 + TIME/365.25 (the +0.5 centres the integer baseline age).
Outcomes from the period-1 row: age_at_death_or_censor = AGE1 + 0.5 + TIMEDTH/365.25,
event = DEATH. Cause of death is not coded in the teaching file; we label a death "cvd"
when a CVD/CHD/MI-or-fatal-CHD/stroke event is dated on the death date (event time ==
TIMEDTH, i.e. fatal event), otherwise "unknown". This is a heuristic lower bound on CVD deaths. Incident CVD (fatal or not) is saved separately in ``events.parquet``.

Features (canonical units): sex, total_chol, hdl, ldl (period 3 only), sbp, dbp, bmi,
heart_rate, glucose_nonfasting (casual serum glucose, mmol/L), diabetes, bp_med,
current_smoker (flag), cigs_per_day, prevalent_cvd (PREVCHD or PREVSTRK), educ (code).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_cohort, save_cohort

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "framingham" / "frmgham2.csv"
INTERIM_ROOT = ROOT / "data" / "interim"
COHORT = "framingham"

# feature -> (raw column, scale, unit)
FEATURES = {
    "sex": ("SEX", None, "code"),
    "total_chol": ("TOTCHOL", 1 / 38.67, "mmol/L"),
    "hdl": ("HDLC", 1 / 38.67, "mmol/L"),
    "ldl": ("LDLC", 1 / 38.67, "mmol/L"),
    "sbp": ("SYSBP", 1.0, "mmHg"),
    "dbp": ("DIABP", 1.0, "mmHg"),
    "bmi": ("BMI", 1.0, "kg/m2"),
    "heart_rate": ("HEARTRTE", 1.0, "bpm"),
    "glucose_nonfasting": ("GLUCOSE", 1 / 18.016, "mmol/L"),
    "diabetes": ("DIABETES", 1.0, "flag"),
    "bp_med": ("BPMEDS", 1.0, "flag"),
    "current_smoker": ("CURSMOKE", 1.0, "flag"),
    "cigs_per_day": ("CIGPDAY", 1.0, "count"),
    "educ": ("educ", 1.0, "code"),
}


def build(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW)
    base = df[df["PERIOD"] == 1].set_index("RANDID")
    age1 = df["RANDID"].map(base["AGE"])
    df["visit_age"] = age1 + 0.5 + df["TIME"] / 365.25
    df["person_id"] = df["RANDID"].astype(str)

    wide = pd.DataFrame({"person_id": df["person_id"], "visit_age": df["visit_age"]})
    for feat, (col, scale, _) in FEATURES.items():
        if feat == "sex":
            wide[feat] = (df[col] == 1).astype(float)  # 1 = male, 0 = female
        else:
            wide[feat] = df[col] * scale
    wide["prevalent_cvd"] = ((df["PREVCHD"] == 1) | (df["PREVSTRK"] == 1)).astype(float)
    units = {f: u for f, (_, _, u) in FEATURES.items()} | {"prevalent_cvd": "flag"}

    long = wide.melt(id_vars=["person_id", "visit_age"], var_name="feature",
                     value_name="value").dropna(subset=["value"])
    long["cohort"] = COHORT
    long["unit"] = long["feature"].map(units)

    b = base.reset_index()
    a0 = b["AGE"] + 0.5
    dead = b["DEATH"] == 1
    fatal_cvd = dead & (
        ((b["CVD"] == 1) & (b["TIMECVD"] == b["TIMEDTH"]))
        | ((b["MI_FCHD"] == 1) & (b["TIMEMIFC"] == b["TIMEDTH"]))
        | ((b["ANYCHD"] == 1) & (b["TIMECHD"] == b["TIMEDTH"]))
        | ((b["STROKE"] == 1) & (b["TIMESTRK"] == b["TIMEDTH"])))
    cause = np.where(~dead, "none", np.where(fatal_cvd, "cvd", "unknown"))
    outcomes = pd.DataFrame({
        "person_id": b["RANDID"].astype(str),
        "age_at_baseline": a0,
        "age_at_death_or_censor": a0 + b["TIMEDTH"] / 365.25,
        "event": dead.astype(int),
        "cause": cause,
    })
    if save:
        d = save_cohort(long, outcomes, COHORT, INTERIM_ROOT)
        ev = pd.DataFrame({"person_id": b["RANDID"].astype(str)})
        for e in ("ANGINA", "HOSPMI", "MI_FCHD", "ANYCHD", "STROKE", "CVD", "HYPERTEN"):
            tcol = {"ANGINA": "TIMEAP", "HOSPMI": "TIMEMI", "MI_FCHD": "TIMEMIFC", "ANYCHD": "TIMECHD",
                    "STROKE": "TIMESTRK", "CVD": "TIMECVD", "HYPERTEN": "TIMEHYP"}[e]
            ev[e.lower()] = b[e].astype(int)
            ev[f"age_{e.lower()}"] = a0 + b[tcol] / 365.25
        ev.to_parquet(d / "events.parquet", index=False)
    return long, outcomes


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_cohort(COHORT, INTERIM_ROOT)


if __name__ == "__main__":
    lg, oc = build()
    print(lg.shape, oc.shape, oc["event"].mean(), oc["cause"].value_counts().to_dict())
