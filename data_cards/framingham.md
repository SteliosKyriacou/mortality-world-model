# Data card: Framingham Heart Study teaching dataset (`frmgham2`, NHLBI BioLINCC)

Status: **downloaded 2026-10-03** from a legitimate no-login source (CRAN). Counts below were computed from the file.

## TL;DR

- **Downloaded:** `data/raw/framingham/frmgham2.csv` (11,627 rows × 39 cols, 1.4 MB) plus the original `framingham_riskCommunicator_1.0.1.rda`. Provenance and sha256 are in `data/raw/framingham/SOURCE.txt`.
  - Source: CRAN package **riskCommunicator 1.0.1** (GPL-3). Its documentation states the data are "the teaching dataset from the Framingham Heart Study (No. N01-HC-25195), provided with permission from the National Heart, Lung, and Blood Institute (NHLBI)" [1][2].
  - Row and column counts and per-period counts match the BioLINCC documentation exactly (4,434 / 3,930 / 3,263) [3].
- **BioLINCC route needs login.** The teaching-dataset request page (https://biolincc.nhlbi.nih.gov/requests/teaching-dataset-request/) now sits behind **NIH RAS login (Login.gov / ID.me identity verification)**. NIH "does not approve accounts using free email providers", so an institutional e-mail is required [4].
  - The CRAN copy makes this unnecessary for development use.
- **Gate:** N = 4,434; **≥2 visits = 3,987; ≥3 visits = 3,206**; 1,550 deaths over 24 y. This is **below the 5,000 gate**, and the data are **not publishable** (below). Use it only as a **pipeline/smoke-test cohort**.
- **Licence verdict on publishing weights:** BioLINCC states: "specific methods were employed to ensure an anonymous dataset that protects patient confidentiality; therefore, this dataset is inappropriate for publication purposes" [3]. The data are permuted/anonymised (BioLINCC teaching page: "completely unsuitable for publication purposes" [5]).
  - Weights trained on it may be shipped as **test fixtures/demo checkpoints** (the data themselves are redistributed under GPL-3 on CRAN).
  - They must **not** be presented as a scientific model or used for scientific claims.

## Source / access

| Item | Value |
|---|---|
| Canonical source | NHLBI BioLINCC teaching datasets, https://biolincc.nhlbi.nih.gov/teaching/ [5] |
| Canonical access | Request form at `/requests/teaching-dataset-request/` behind NIH RAS login with an institutional e-mail [4]. Turnaround not stated (**UNVERIFIED**; historically same-day/automatic) |
| Used source | CRAN `riskCommunicator` 1.0.1, published 2022-05-31, GPL-3, "provided with permission from NHLBI" [1][2]. Another CRAN copy exists in `LocalControl` (`framingham`) [6] |
| Documentation | https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf [3] |
| Kaggle "framingham.csv" | A **different**, cross-sectional 10-year-CHD subset (~4,240 rows, ~16 columns, single exam, `TenYearCHD` label), with unclear provenance and licence (details **UNVERIFIED**, not inspected). **Not useful** (no repeat visits, no mortality time) |

## Cohort, visits, Δt (computed from the downloaded file)

| | Value |
|---|---|
| Participants | 4,434 (5,209 enrolled originally; this is a subset) |
| Rows by PERIOD | 1: 4,434 · 2: 3,930 · 3: 3,263 |
| Visits per person | 1: 447 · 2: 781 · 3: 3,206 → **≥2 = 3,987, ≥3 = 3,206** |
| Exam years | ≈1956–1968 [3] |
| Δt between consecutive exams | median 5.96 y, IQR 5.87–6.06, 5–95% 5.63–6.29, range 4.25–12.67 (from `TIME` in days) |
| Age at period 1 | 32–70, median 49 |
| Follow-up | 24 y for everyone (`TIMEDTH` max 8,766 d = 24.0 y) |
| Events (ever) | DEATH 1,550 · CVD 1,157 · ANYCHD 1,240 |

## Features (mapped to PLAN.md §3.2)

| §3.2 feature | Variable | Unit / coding | Missing (rows) |
|---|---|---|---|
| age, sex | `AGE`, `SEX` (1 M, 2 F) | years | 0 |
| BMI | `BMI` | kg/m² | 52 |
| waist | — | not available | |
| SBP/DBP | `SYSBP`, `DIABP` (mean of last 2 of 3) | mmHg | 0 |
| total chol | `TOTCHOL` | mg/dL | 409 |
| HDL / LDL | `HDLC`, `LDLC` (**period 3 only**) | mg/dL | 8,600 / 8,601 |
| triglycerides | — | not available | |
| glucose | `GLUCOSE` (casual) | mg/dL | 1,440 |
| HbA1c, CRP, fibrinogen, creatinine, Hb, albumin, grip | — | not available | |
| smoking | `CURSMOKE` (0/1), `CIGPDAY` | cig/day | 79 (CIGPDAY) |
| diabetes | `DIABETES` (0/1) | | 0 |
| BP meds | `BPMEDS` (0/1) | | 593 |
| other | `HEARTRTE` (bpm), `educ` (1–4), `PREVCHD`, `PREVAP`, `PREVMI`, `PREVSTRK`, `PREVHYP` | | |

## Outcomes

- Event indicators are person-level and identical on every row of a person (checked): `DEATH`, `ANGINA`, `HOSPMI`, `MI_FCHD`, `ANYCHD`, `STROKE`, `CVD`, `HYPERTEN`.
- Times are in days from baseline (PERIOD 1) to first event or censoring: `TIMEAP`, `TIMEMI`, `TIMEMIFC`, `TIMECHD`, `TIMESTRK`, `TIMECVD`, `TIMEDTH`, `TIMEHYP`.
- Cause of death: only CVD/CHD-specific events. Death from CVD = `DEATH==1` & `CVD==1` with `TIMECVD==TIMEDTH` (approximation).

## Expected size on disk

1.4 MB CSV and 0.3 MB rda (actual).

## Verified PLAN.md §1.1 row (new row)

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **Framingham teaching (frmgham2)** | Human | Yes: 3 exams ≈6 y apart (1956–68); 3,987 with ≥2, 3,206 with 3 | Age, sex, SBP/DBP, TC, casual glucose, BMI, smoking, diabetes, BP meds, HR; HDL/LDL period 3 only | 24-y all-cause death + adjudicated CHD/CVD/stroke with times | BioLINCC (NIH RAS login, institutional e-mail) or CRAN `riskCommunicator` (no login). **Anonymised/permuted: not for publication** |

## Loader notes

**Implemented and tested:** `src/mwm/data/framingham.py` (`build()` / `load()`), fetch script
`scripts/fetch_framingham.py` (CRAN tarball → .rda → CSV; CSV conversion needs `pyreadr` or R).
Output `data/interim/framingham/{long,outcomes,events}.parquet`: 154,330 long rows (15 features, canonical units),
4,434 outcomes (1,550 deaths; `cause` = `cvd` for 244 deaths where a CVD/CHD/stroke event is dated on the death
day, else `unknown`, a heuristic lower bound), `events.parquet` = incident CVD/CHD/MI/stroke/angina/hypertension
flags + ages. visit_age = AGE(period 1) + 0.5 + TIME/365.25. Smoking is stored as `current_smoker` flag
(the canonical 0/1/2 code is not derivable because never vs former is unknown); glucose as `glucose_nonfasting`.

Original notes from the data-card research:

- File: `data/raw/framingham/frmgham2.csv`. One row per (person, exam). Key `RANDID` + `PERIOD`. Visit time is `TIME` (days since period-1 exam). Visit age is `AGE` (integer years), or better `AGE_at_P1 + TIME/365.25`.
- The outcome table (§3.1) is built from the PERIOD 1 row:
  - `age_at_death_or_censor = AGE_p1 + TIMEDTH/365.25`
  - `event = DEATH`
  - `cause = 'CVD'` if `CVD==1 and TIMECVD==TIMEDTH`, else all-cause
- Units are US (mg/dL). Convert to mmol/L: TC/HDL/LDL ÷ 38.67, glucose ÷ 18.016.
- Empty cells are NA. All columns are numeric.

## Manual steps for the user

1. None needed for development. The file is already in `data/raw/framingham/`.
2. Optional: request the official copy via BioLINCC (log in with NIH RAS using an institutional e-mail, then submit the teaching-dataset request form) to have a first-party copy and accept NHLBI's terms directly.
3. Keep it out of any published results. Label any demo checkpoints trained on it as "teaching data, not for scientific use".

## Sources

1. CRAN riskCommunicator package page. https://cran.r-project.org/web/packages/riskCommunicator/index.html . Source tarball: https://cran.r-project.org/src/contrib/riskCommunicator_1.0.1.tar.gz
2. riskCommunicator `framingham` reference manual. https://search.r-project.org/CRAN/refmans/riskCommunicator/html/framingham.html
3. BioLINCC FHS Teaching Longitudinal Data Documentation (2021a). https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf
4. BioLINCC teaching dataset request page (login wall, NOT-OD-25-083 notice). https://biolincc.nhlbi.nih.gov/requests/teaching-dataset-request/
5. BioLINCC Teaching Datasets. https://biolincc.nhlbi.nih.gov/teaching/
6. LocalControl `framingham` (RDocumentation). https://www.rdocumentation.org/packages/LocalControl/versions/1.1.1/topics/framingham
