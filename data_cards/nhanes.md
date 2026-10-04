# Data card: NHANES 1999–2018 + public-use Linked Mortality Files (2019)

Status: **downloaded, harmonised, loader tested** (2026-10-03).

| Field | Value |
|---|---|
| Source | CDC/NCHS continuous NHANES, https://wwwn.cdc.gov/nchs/nhanes/ ; file URLs `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/<FILE>.xpt` |
| Mortality linkage | NCHS public-use LMF (NDI linkage, follow-up to **31 Dec 2019**), https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/ ; docs https://www.cdc.gov/nchs/linked-data/mortality-files/index.html |
| Licence | US federal public domain. NCHS data-use restrictions apply: use for statistical research only, **no attempt to re-identify** respondents or link to other individual-level data to that end. Publishing trained weights/latents is allowed (public-use data). |
| Access | Direct download, no registration |
| Fetch | `python scripts/fetch_nhanes.py` (idempotent; 147 XPT files + 10 LMF `.dat`). ftp.cdc.gov throttles hard (≈3 KB/s, then stops answering); the script uses parallel byte ranges and falls back to the Internet Archive's verbatim copy (`web.archive.org/web/2024id_/<cdc url>`). The four 2011–2018 LMF files came from that fallback on 2026-10-03; their byte sizes equal the CDC directory listing exactly. MD5s: `data/raw/nhanes/mortality/MD5SUMS.txt`. |
| Loader | `src/mwm/data/nhanes.py`: `build()` (raw → interim), `load()`, `load_survey()` |
| Size on disk | raw 273 MB (`data/raw/nhanes/`), interim 9.5 MB (`data/interim/nhanes/{long,outcomes,survey}.parquet`) |
| Restricted-use alternative | LMF with follow-up to 31 Dec 2022, exact dates and full ICD-10 cause: NCHS Research Data Center only (application). Not needed for v1. |

## Cohort numbers (examined adults ≥ 18, cycles 1999-2000 … 2017-2018)

| | Value |
|---|---|
| People in long table | 56,367 (MEC-examined, age ≥ 18) |
| People with mortality outcome (LMF ELIGSTAT = 1) | 56,253 |
| Deaths | 8,366 (CVD 2,604 · cancer 1,850 · other 3,911 · missing cause 1) |
| Follow-up | median 9.4 y (max ≈ 20.8 y); 550k person-years. By cycle: 1999-00 median 19.5 y / 1,454 deaths … 2017-18 median 2.0 y / 130 deaths |
| Age ≥ 40 at exam | 34,367 people, 7,977 deaths |
| **N with ≥ 2 visits** | **0**: NHANES is cross-sectional (one MEC exam per person). Δt distribution: n/a |
| Long table | 1.59 M rows, 34 features |

`cause` mapping (UCOD_LEADING): 1 heart disease (I00–I09, I11, I13, I20–I51) and 5 cerebrovascular (I60–I69) → `cvd`;
2 malignant neoplasms → `cancer`; 3, 4, 6–10 → `other`. Multiple-cause flags for diabetes/hypertension
are kept in `survey.parquet` (`mcod_diabetes`, `mcod_hyperten`). The public-use LMF perturbs follow-up time
and cause for a small, unstated fraction of decedents to prevent re-identification. That's fine for
modelling, but exact counts will differ slightly from restricted-use analyses.

## Features (people with a non-missing value, of 56,367)

| Canonical feature | Unit | N | Coverage / notes |
|---|---|---|---|
| sex, race_eth | code | 56,367 | sex 1 = male; RIDRETH1 codes 1–5 |
| bmi, waist, height, weight | kg/m², cm | 53–55k | all cycles |
| sbp, dbp, heart_rate | mmHg, bpm | 53–54k | mean of up to 4 auscultatory readings; DBP 0 → missing |
| total_chol, hdl | mmol/L | 52.8k | all cycles |
| triglycerides, ldl | mmol/L | 25k | **fasting morning subsample only** (Friedewald LDL); weight WTSAF2YR |
| hba1c | mmol/mol | 53.4k | NGSP % → IFCC |
| glucose | mmol/L | 25.9k | fasting plasma glucose, fasting subsample |
| glucose_serum | mmol/L | 52.6k | non-fasting serum glucose (biochemistry profile) |
| crp | mg/L | 42.0k | 1999–2010 standard CRP (mg/dL ×10); **none 2011–2014**; 2015–2018 hs-CRP. Assay change, so adjust for cycle |
| fibrinogen | g/L | 5.7k | 1999–2002 only, age ≥ 40 |
| creatinine, egfr | µmol/L, mL/min/1.73m² | 52.6k | creatinine recalibrated to the IDMS standard for 1999-2000 and 2005-2006 (NCHS-recommended equations); eGFR is CKD-EPI 2021 (race-free) |
| albumin | g/L | 52.6k | |
| alp | U/L | 52.6k | analyser change in 2017-18 (median 66 → 76 U/L) |
| haemoglobin, wbc, lymph_pct, mcv, rdw | g/dL, 10⁹/L, %, fL, % | 53.4k | RDW shifts +0.9 from 2013-14 (instrument) |
| grip, grip_combined | kg | 10.6k | 2011–2014 only (MGX). grip = best single-hand trial |
| smoking | code 0/1/2 | 53.2k | never / former / current |
| diabetes | flag | 56.3k | self-reported doctor diagnosis (DIQ010) |
| bp_med, lipid_med | flag | 56.1k / 43.3k | self-reported current use (BPQ050A / BPQ100D) |
| prevalent_cvd | flag | 52.1k | self-reported CHF/CHD/angina/MI/stroke |

Not in v1: prescription-level meds (RXQ_RX), PhenoAge is computable (albumin, creatinine, glucose_serum,
crp, lymph_pct, mcv, rdw, alp, wbc, age), cystatin C (only some cycles), diet, accelerometry (2003–06, 2011–14).
NHANES III (1988–94, LMF also available) is a possible older extension (not downloaded).

Cross-cycle medians, checked after harmonisation: total cholesterol 5.1 → 4.7 mmol/L (a real secular decline),
HbA1c 34–38 mmol/mol, creatinine 72–80 µmol/L, CRP 1.8–2.4 mg/L. No unit discontinuities apart from the
assay notes above.

## Caveats
- Survey design: the sample is complex and oversamples some groups. `survey.parquet` holds `WTMEC2YR`, `WTSAF2YR`,
  `SDMVPSU` and `SDMVSTRA`. Pooled multi-cycle weights = WTMEC2YR / n_cycles (1999–2002 have 4-year
  weights in the raw files, not used here).
- Age: exact age in months at exam for 1999–2010 (RIDAGEEX). For 2011–2018 only integer years are public
  (we add +0.5). Top-coded at 85 (1999–2006) and 80 (2007–2018): `age_topcoded` flag. That's about 4% of people.
- Mortality: public LMF covers adults only. Follow-up ends 2019-12-31.

## Verified/corrected PLAN.md §1.1 row

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **NHANES 1999–2018 + public-use LMF** | Human (US civilian) | **No** (one exam per person); 56k adults over 10 cycles | Full core panel: lipids, HbA1c, glucose, CRP (not 2011–14), creatinine/eGFR, albumin, CBC, BP, BMI, waist, smoking, diabetes, BP/lipid meds; fibrinogen 1999–2002 only; grip 2011–14 only | All-cause mortality (8,366 deaths, median 9.4 y follow-up to end-2019) + leading cause (heart, cerebrovascular, cancer …) | Public domain, direct download (no registration). Confirmed |

## Usage
```python
from mwm.data import nhanes
long, outcomes = nhanes.build()      # rebuild from data/raw/nhanes
long, outcomes = nhanes.load()       # validated via mwm.data.common
survey = nhanes.load_survey()        # cycle, weights, design, raw LMF fields
```
