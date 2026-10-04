"""NHANES 1999-2018 + public-use Linked Mortality Files (LMF, follow-up to 31 Dec 2019).

Cross-sectional: one visit (the MEC exam) per person. Fetch with
``python scripts/fetch_nhanes.py``; see ``data_cards/nhanes.md``.

Outputs (``data/interim/nhanes/``, via ``common.save_cohort``)
-------------------------------------------------------------
* ``long.parquet``      person_id, cohort, visit_age, feature, value, unit (common contract)
* ``outcomes.parquet``  person_id, age_at_baseline, age_at_death_or_censor, event, cause
* ``survey.parquet``    extra per-person columns NOT in the contract: cycle, survey weights
                        (WTMEC2YR, WTSAF2YR), design (SDMVPSU, SDMVSTRA), race/ethnicity,
                        LMF raw fields (mortstat, ucod_leading, permth_exm, mcod diabetes/
                        hypertension flags), age_topcoded.

Harmonisation (raw NHANES unit -> canonical unit of ``common.CANONICAL_FEATURES``)
-------------------------------------------------------------------------------
* lipids mg/dL -> mmol/L (chol/HDL/LDL / 38.67, TG / 88.57); LDL = Friedewald (fasting subsample)
* HbA1c % (NGSP) -> mmol/mol (IFCC): (pct - 2.15) * 10.929
* fasting plasma glucose mg/dL -> mmol/L (/ 18.016) [fasting morning subsample only]
* CRP: 1999-2010 LBXCRP mg/dL * 10 -> mg/L (Dade Behring nephelometry); 2015-2018 LBXHSCRP mg/L
  (Roche hs-CRP). No CRP in 2011-2014. Assay change => keep cycle as covariate.
* fibrinogen (1999-2002 only, adults 40+): LBDFBSI g/L
* creatinine mg/dL -> umol/L (* 88.42) after the NHANES-recommended recalibration to the
  standardised (IDMS-traceable) assay: 1999-2000: 1.013*Scr + 0.147; 2005-2006:
  -0.016 + 0.978*Scr (Selvin et al. 2007, AJKD; NHANES BIOPRO_D doc). Other cycles unchanged.
* eGFR: CKD-EPI 2021 race-free equation from standardised creatinine.
* albumin g/dL -> g/L (* 10); haemoglobin g/dL; WBC 10^9/L; lymph %, MCV fL, RDW %, ALP U/L
* SBP/DBP: mean of available auscultatory readings 1-4 (DBP == 0 treated as missing)
* grip: max single-hand trial (kg), 2011-2014 only; grip_combined = MGDCGSZ (sum of best hands)
* smoking: 0 never (SMQ020=2), 1 former (SMQ040=3), 2 current (SMQ040 in 1,2)
* diabetes: self-report DIQ010 == 1 (borderline = 0)
* bp_med: BPQ050A == 1 (0 if told no HBP / not told to take meds / not taking)
* lipid_med: BPQ100D == 1 (0 if no high chol / not told to take / not taking)
* prevalent_cvd: any of MCQ160B-F (CHF, CHD, angina, MI, stroke) == 1
* sex: 1 male, 0 female

Visit age: RIDAGEEX (age in months at exam, 1999-2010) / 12, else RIDEXAGM/12, else
RIDAGEYR + 0.5. RIDAGEYR is top-coded at 85 (1999-2006) and 80 (2007-2018): such people
get the top-code value (+0.5) and ``age_topcoded = 1`` in survey.parquet.

Outcomes: eligible adults (LMF ELIGSTAT == 1); time = PERMTH_EXM months from MEC exam.
cause: UCOD_LEADING 1 (heart) or 5 (cerebrovascular) -> "cvd"; 2 -> "cancer"; other codes
-> "other"; deceased without code -> "unknown". The 2019 public-use LMF has perturbed
(synthetic) follow-up time/cause for a small fraction of decedents to limit re-identification
(see NCHS public-use LMF documentation): fine for modelling, not for exact counts.

Usage
-----
>>> from mwm.data import nhanes
>>> long, outcomes = nhanes.build()     # raw -> data/interim/nhanes (idempotent overwrite)
>>> long, outcomes = nhanes.load()
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CANONICAL_FEATURES, load_cohort, save_cohort

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "nhanes"
INTERIM_ROOT = ROOT / "data" / "interim"
COHORT = "nhanes"

CYCLES = {1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E",
          2009: "_F", 2011: "_G", 2013: "_H", 2015: "_I", 2017: "_J"}
_EARLY = {
    "TCHOL": ("LAB13", "L13_B", "L13_C"), "TRIGLY": ("LAB13AM", "L13AM_B", "L13AM_C"),
    "GHB": ("LAB10", "L10_B", "L10_C"), "GLU": ("LAB10AM", "L10AM_B", "L10AM_C"),
    "CRP": ("LAB11", "L11_B", "L11_C"), "BIOPRO": ("LAB18", "L40_B", "L40_C"),
    "CBC": ("LAB25", "L25_B", "L25_C"),
}
UNITS = dict(CANONICAL_FEATURES)
UNITS.update({"grip_combined": "kg", "glucose_serum": "mmol/L", "prevalent_cvd": "flag",
              "race_eth": "code", "height": "cm", "weight": "kg"})


def _stem(year: int, comp: str) -> str | None:
    s = CYCLES[year]
    if comp in _EARLY and year <= 2003:
        return _EARLY[comp][(year - 1999) // 2]
    if comp == "CRP":
        return ("CRP" + s) if year <= 2009 else ("HSCRP" + s) if year >= 2015 else None
    if comp == "HDL":
        return ("HDL" + s) if year >= 2005 else None
    if comp == "MGX":
        return ("MGX" + s) if year in (2011, 2013) else None
    return comp + s


def _read(year: int, comp: str) -> pd.DataFrame | None:
    stem = _stem(year, comp)
    if stem is None:
        return None
    p = RAW / f"{year}-{year + 1}" / f"{stem}.xpt"
    if not p.exists():
        return None
    with warnings.catch_warnings():  # pandas' XPORT reader emits fragmentation warnings
        warnings.simplefilter("ignore")
        df = pd.read_sas(p, format="xport", encoding="latin-1")
    df["SEQN"] = df["SEQN"].astype(np.int64)
    # SAS XPORT stores exact zeros as ~5.4e-79 when read by pandas: snap them back to 0.
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    return df.set_index("SEQN")


def _col(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            return df[n].astype(float)
    return None


def _read_lmf(year: int) -> pd.DataFrame:
    p = RAW / "mortality" / f"NHANES_{year}_{year + 1}_MORT_2019_PUBLIC.dat"
    cols = [("seqn", 0, 6), ("eligstat", 14, 15), ("mortstat", 15, 16), ("ucod_leading", 16, 19),
            ("mcod_diabetes", 19, 20), ("mcod_hyperten", 20, 21),
            ("permth_int", 42, 45), ("permth_exm", 45, 48)]
    if not p.exists():
        warnings.warn(f"{p.name} missing (run scripts/fetch_nhanes.py): cycle {year} gets no outcomes")
        return pd.DataFrame(columns=[c for c, _, _ in cols[1:]], index=pd.Index([], name="seqn"),
                            dtype=float)
    df = pd.read_fwf(p, colspecs=[(a, b) for _, a, b in cols], names=[c for c, _, _ in cols],
                     na_values=["."], dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df["seqn"] = df["seqn"].astype(np.int64)
    return df.set_index("seqn")


def _ckd_epi_2021(scr_mgdl: pd.Series, age: pd.Series, male: pd.Series) -> pd.Series:
    k = np.where(male == 1, 0.9, 0.7)
    a = np.where(male == 1, -0.302, -0.241)
    r = scr_mgdl / k
    egfr = 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age
    return egfr * np.where(male == 1, 1.0, 1.012)


def _cycle_frame(year: int) -> pd.DataFrame:
    """One row per examined adult (>=18) with canonical features in canonical units."""
    demo = _read(year, "DEMO")
    demo = demo[(demo["RIDSTATR"] == 2) & (demo["RIDAGEYR"] >= 18)]
    out = pd.DataFrame(index=demo.index)
    out["cycle"] = year
    age_m = None
    for c in ("RIDAGEEX", "RIDEXAGM", "RIDAGEMN"):
        if c in demo.columns:
            age_m = demo[c] if age_m is None else age_m.fillna(demo[c])
    topcode = 85 if year <= 2005 else 80
    age = age_m / 12 if age_m is not None else pd.Series(np.nan, index=demo.index)
    age = age.where(age.notna(), demo["RIDAGEYR"] + 0.5)
    # Month variables are blank for top-coded ages; guard against inconsistent month data.
    age = age.where((age - demo["RIDAGEYR"]).abs() < 1.5, demo["RIDAGEYR"] + 0.5)
    out["visit_age"] = age
    out["age_topcoded"] = (demo["RIDAGEYR"] >= topcode).astype(int)
    out["sex"] = (demo["RIAGENDR"] == 1).astype(float)
    out["race_eth"] = demo.get("RIDRETH1")
    for c in ("WTMEC2YR", "SDMVPSU", "SDMVSTRA"):
        out[c] = demo.get(c)
    j = lambda s: s.reindex(out.index) if s is not None else np.nan  # noqa: E731

    bmx = _read(year, "BMX")
    out["bmi"] = j(_col(bmx, "BMXBMI"))
    out["waist"] = j(_col(bmx, "BMXWAIST"))
    out["height"] = j(_col(bmx, "BMXHT"))
    out["weight"] = j(_col(bmx, "BMXWT"))

    bpx = _read(year, "BPX")
    if bpx is not None:
        sy = bpx[[c for c in bpx.columns if c.startswith("BPXSY")]]
        di = bpx[[c for c in bpx.columns if c.startswith("BPXDI")]].mask(lambda d: d < 1)
        out["sbp"] = j(sy.mean(axis=1))
        out["dbp"] = j(di.mean(axis=1))
        out["heart_rate"] = j(_col(bpx, "BPXPLS"))

    tc = _read(year, "TCHOL")
    out["total_chol"] = j(_col(tc, "LBXTC")) / 38.67
    hdl = _col(_read(year, "HDL"), "LBDHDD") if year >= 2005 else _col(tc, "LBDHDL", "LBXHDD")
    out["hdl"] = j(hdl) / 38.67
    tg = _read(year, "TRIGLY")
    out["triglycerides"] = j(_col(tg, "LBXTR")) / 88.57
    out["ldl"] = j(_col(tg, "LBDLDL")) / 38.67
    out["WTSAF2YR"] = j(_col(tg, "WTSAF2YR"))
    out["hba1c"] = (j(_col(_read(year, "GHB"), "LBXGH")) - 2.15) * 10.929
    out["glucose"] = j(_col(_read(year, "GLU"), "LBXGLU")) / 18.016

    crp = _read(year, "CRP")
    if crp is not None:
        if "LBXHSCRP" in crp.columns:
            out["crp"] = j(crp["LBXHSCRP"].astype(float))
        else:
            out["crp"] = j(_col(crp, "LBXCRP")) * 10.0
        out["fibrinogen"] = j(_col(crp, "LBDFBSI"))

    bio = _read(year, "BIOPRO")
    scr = j(_col(bio, "LBXSCR", "LBDSCR"))
    if year == 1999:
        scr = 1.013 * scr + 0.147
    elif year == 2005:
        scr = -0.016 + 0.978 * scr
    out["creatinine"] = scr * 88.42
    out["egfr"] = _ckd_epi_2021(scr, out["visit_age"], out["sex"])
    out["albumin"] = j(_col(bio, "LBXSAL")) * 10.0
    out["alp"] = j(_col(bio, "LBXSAPSI", "LBDSAPSI"))
    out["glucose_serum"] = j(_col(bio, "LBXSGL")) / 18.016

    cbc = _read(year, "CBC")
    out["haemoglobin"] = j(_col(cbc, "LBXHGB"))
    out["wbc"] = j(_col(cbc, "LBXWBCSI"))
    out["lymph_pct"] = j(_col(cbc, "LBXLYPCT"))
    out["mcv"] = j(_col(cbc, "LBXMCVSI"))
    out["rdw"] = j(_col(cbc, "LBXRDW"))

    mgx = _read(year, "MGX")
    if mgx is not None:
        trials = [c for c in mgx.columns if c.startswith("MGXH") and not c.endswith("E")]
        out["grip"] = j(mgx[trials].max(axis=1))
        out["grip_combined"] = j(_col(mgx, "MGDCGSZ"))

    smq = _read(year, "SMQ")
    s020, s040 = j(_col(smq, "SMQ020")), j(_col(smq, "SMQ040"))
    out["smoking"] = np.select([s020 == 2, (s020 == 1) & (s040 == 3), (s020 == 1) & s040.isin([1, 2])],
                               [0.0, 1.0, 2.0], default=np.nan)

    diq = j(_col(_read(year, "DIQ"), "DIQ010"))
    out["diabetes"] = np.select([diq == 1, diq.isin([2, 3])], [1.0, 0.0], default=np.nan)

    bpq = _read(year, "BPQ")
    b020, b040, b050 = (j(_col(bpq, c)) for c in ("BPQ020", "BPQ040A", "BPQ050A"))
    out["bp_med"] = np.select([b050 == 1, (b050 == 2) | (b040 == 2) | (b020 == 2)], [1.0, 0.0],
                              default=np.nan)
    b080, b090, b100 = (j(_col(bpq, c)) for c in ("BPQ080", "BPQ090D", "BPQ100D"))
    out["lipid_med"] = np.select([b100 == 1, (b100 == 2) | (b090 == 2) | (b080 == 2)], [1.0, 0.0],
                                 default=np.nan)

    mcq = _read(year, "MCQ")
    if mcq is not None:
        m = mcq[[c for c in ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F") if c in mcq]]
        m = m.reindex(out.index)
        anyyes = (m == 1).any(axis=1)
        allno = (m == 2).all(axis=1)
        out["prevalent_cvd"] = np.select([anyyes, allno], [1.0, 0.0], default=np.nan)

    lmf = _read_lmf(year).reindex(out.index)
    for c in lmf.columns:
        out[c] = lmf[c]
    return out


FEATURES = ["sex", "race_eth", "bmi", "waist", "height", "weight", "sbp", "dbp", "heart_rate",
            "total_chol", "hdl", "ldl", "triglycerides", "hba1c", "glucose", "glucose_serum", "crp",
            "fibrinogen", "creatinine", "egfr", "albumin", "alp", "haemoglobin", "wbc", "lymph_pct",
            "mcv", "rdw", "grip", "grip_combined", "smoking", "diabetes", "bp_med", "lipid_med",
            "prevalent_cvd"]
SURVEY_COLS = ["cycle", "WTMEC2YR", "WTSAF2YR", "SDMVPSU", "SDMVSTRA", "age_topcoded", "eligstat",
               "mortstat", "ucod_leading", "mcod_diabetes", "mcod_hyperten", "permth_int", "permth_exm"]


def build_wide() -> pd.DataFrame:
    """All cycles, one row per examined adult, canonical units (index = SEQN)."""
    return pd.concat([_cycle_frame(y) for y in CYCLES])


def _cause(row_mort: pd.Series, ucod: pd.Series) -> pd.Series:
    c = pd.Series("none", index=ucod.index, dtype=object)
    dead = row_mort == 1
    c[dead] = "unknown"
    c[dead & ucod.isin([1, 5])] = "cvd"
    c[dead & (ucod == 2)] = "cancer"
    c[dead & ucod.notna() & ~ucod.isin([1, 2, 5])] = "other"
    return c


def build(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = build_wide()
    wide = wide[wide["visit_age"].notna()]
    pid = wide.index.astype(str)
    long = wide[FEATURES].copy()
    long.index = pid
    long["visit_age"] = wide["visit_age"].to_numpy()
    long = long.reset_index(names="person_id").melt(
        id_vars=["person_id", "visit_age"], var_name="feature", value_name="value").dropna(subset=["value"])
    long["cohort"] = COHORT
    long["unit"] = long["feature"].map(UNITS)

    elig = wide[(wide["eligstat"] == 1) & wide["permth_exm"].notna()]
    outcomes = pd.DataFrame({
        "person_id": elig.index.astype(str),
        "age_at_baseline": elig["visit_age"].to_numpy(),
        "age_at_death_or_censor": (elig["visit_age"] + elig["permth_exm"] / 12.0).to_numpy(),
        "event": (elig["mortstat"] == 1).astype(int).to_numpy(),
        "cause": _cause(elig["mortstat"], elig["ucod_leading"]).to_numpy(),
    })
    # Contract: every long-table person has the same baseline as outcomes; persons without
    # mortality eligibility are kept in the long table (useful for the encoder) but get no outcome.
    if save:
        d = save_cohort(long, outcomes, COHORT, INTERIM_ROOT)
        surv = wide[SURVEY_COLS].copy()
        surv.index = pid
        surv.reset_index(names="person_id").to_parquet(d / "survey.parquet", index=False)
    return long, outcomes


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_cohort(COHORT, INTERIM_ROOT)


def load_survey() -> pd.DataFrame:
    return pd.read_parquet(INTERIM_ROOT / COHORT / "survey.parquet")


if __name__ == "__main__":
    lg, oc = build()
    print(lg.shape, oc.shape)
    print(oc["event"].value_counts(), oc["cause"].value_counts())
