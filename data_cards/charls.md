# Data card: CHARLS (China Health and Retirement Longitudinal Study, Peking University)

Status: desk research on 2026-10-03, no data downloaded. Items not checked against a primary source are marked **UNVERIFIED**.

## TL;DR (gate-relevant)

- **Biomarker visits:** venous blood was drawn only twice, in **2011 (wave 1, n = 11,847)** and **2015 (wave 3, n = 13,420; 13,013 cross-sectional)**. Δt ≈ 4 years [1][2][3].
  - No blood in 2013, 2018 or 2020 (2020 was a COVID-adapted wave) [3][4].
  - Non-blood measures (BP, grip, anthropometry) exist for 2011, 2013 and 2015 [3].
- **N with blood in both 2011 and 2015:** not published directly. One CHARLS paper analysed **7,064 CVD-free participants with cystatin C change 2011→2015** [5]. Cystatin C in 2011 was assayed on only a subsample (n = 8,878: everyone aged 70+ plus a random subset of the rest) [2]. So **≥2 blood visits is likely ~8–9k (UNVERIFIED)**. **≥3 blood visits = 0** by design.
- **Mortality:**
  - Exit interviews, with WHO 2012 verbal autopsy for cause of death, were run in waves 2–4 (2013, 2015, 2018) [3] and wave 5 (2020) [4].
  - The wave 5 `Sample_Infor.dta` flags "whether died" [4].
  - Death year/month come from the exit interview (exact variable names **UNVERIFIED**).
  - Follow-up runs to 2020 (latest public release: Wave 5, released 16 Nov 2023 [6]).
- **Gate:** passes on numbers (≥5,000 with 2 blood visits + mortality to 2020). Weaknesses: only one Δt step; cystatin C has a gap; deaths come from verbal autopsy and proxy reports, not a registry.
- **Licence verdict on publishing weights:** the terms are shown only inside the JS sign-up flow and could not be read. **UNVERIFIED.** Typical CHARLS terms are non-commercial academic use, no redistribution, and citation. Ask charls_info@pku.edu.cn before publishing weights.

## Source / access

| Item | Value |
|---|---|
| Portal | https://charls.charlsdata.com (current), legacy https://charls.pku.edu.cn/en |
| Releases | 2008 pilot, 2011 W1 (baseline), 2012 pilot W2, 2013 W2, 2014 Life History, 2015 W3, 2018 W4, 2020 W5 (16 Nov 2023), Harmonized CHARLS [6] |
| Access | Register online (agree to terms, give real identity), confirm email, then apply for each dataset. Applications are "reviewed and approved in 3 working days" (FAQ, via search summary) [7]. Over 88,000 registered users by Oct 2023 [4] |
| Terms | Agreement text is loaded by JavaScript at https://charls.charlsdata.com/users/sign_up/agreement/en.html and was not retrievable. **UNVERIFIED** |
| Citation requirements | Cite Zhao et al. (IJE 2014) for the main data and Chen et al. (AJE 2019) for venous blood [3][4] |
| Harmonized CHARLS | Gateway to Global Aging. Version D (June 2021); coverage 2011–2018 **UNVERIFIED**. Free to registered CHARLS users [8] |

## Cohort size, visits, Δt

| Wave | Year | Interviewed | Blood |
|---|---|---|---|
| W1 | 2011–12 | 17,708 individuals / 10,257 households | **11,847** (67%) [2] |
| W2 | 2013 | 18,264 (19,055 in release note) | none [1] |
| W3 | 2015 (one PSU 2016) | 21,100 (20,284 cross-sectional) | **13,420** (64%) [1] |
| W4 | 2018 | 19,816 | none (non-blood biomarkers also dropped after W3) [3] |
| W5 | 2020 | 19,395 | none [4] |

Respondent counts are from the W5 user guide, Table 1 [4].

- Δt between blood draws: ~4 years (2011/12 → 2015, mostly July–Oct 2015) [1].
- Exact interview month is in the data (W5 `Sample_Infor` holds the interview date [4]).

## Features (mapped to PLAN.md §3.2)

| §3.2 feature | CHARLS | 2015 variable [1] | Unit |
|---|---|---|---|
| age, sex | Demographic_Background | | |
| BMI, waist | physical measures W1–W3 | biomarker module (**UNVERIFIED** names) | |
| SBP/DBP | W1–W3 (3 readings) | | mmHg |
| total/HDL/LDL chol, TG | 2011, 2015 | `bl_cho`, `bl_hdl`, `bl_ldl`, `bl_tg` (+`bl_top_coding_tg`, TG top-coded at 500) | **mg/dL** |
| HbA1c / glucose | 2011, 2015 | `bl_hbalc` (note the "l"), `bl_glu`; `bl_fasting` flag | %, mg/dL |
| CRP | 2011, 2015 | `bl_crp` | mg/L |
| fibrinogen | **not available** | | |
| creatinine / cystatin C | 2011 (cystatin C subsample 8,878), 2015 | `bl_crea`, `bl_cysc` | mg/dL, mg/L |
| BUN, uric acid | 2011, 2015 | `bl_bun`, `bl_ua` | mg/dL |
| haemoglobin | CBC 2011, 2015 (county labs) | `bl_hgb` (+`bl_wbc`, `bl_hct`, `bl_mcv`, `bl_plt`) | g/dL |
| albumin | **not available** | | |
| grip | W1–W3 (2 per hand, dynamometer) [3] | | kg |
| smoking, diabetes, meds | Health_Status_and_Functioning (self-report) | | |

2011 variable names, from the 2011 blood user guide [2]:
- CBC: `qc1_vb002` … `qc1_vb009` (e.g. `qc1_vb004` = haemoglobin plot label)
- `newcrp`, `newhba1c`, `newcho`, `newhdl`, `newldl`, `newtg`, `newbun`, `newcrea`, `newglu`, `newua`, `newcysc`? The cystatin C name is **UNVERIFIED**.
- Units are mg/dL as in 2015. Creatinine used the Jaffe method in 2011 [2]. The 2015 method is **UNVERIFIED**; check before pooling.
- HbA1c in 2011 was measured on frozen samples (biased low) [2].

## Outcomes

- Death: exit interviews in 2013, 2015, 2018 and 2020, plus a "whether died" flag in the sample-information files [3][4].
- Cause of death: verbal autopsy (VA module), coded fields `VAS42`/`VAS43` in 2018 [3].
- Count of deaths 2011–2020: roughly 2,000–2,500 of 17,708 (**UNVERIFIED**; compute on download).

## Expected size on disk

Each wave is a set of Stata files of ~50–200 MB. All waves plus Harmonized CHARLS come to roughly 1 GB (**UNVERIFIED**).

## Verified/corrected PLAN.md §1.1 row (new row)

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **CHARLS** | Human | Yes, but only 2 venous blood waves (2011 n=11,847; 2015 n=13,420; Δt≈4 y; ~8–9k with both, UNVERIFIED). Physical measures 2011/13/15 | Lipids (mg/dL), HbA1c, glucose, CRP, creatinine, cystatin C, BUN, uric acid, CBC (Hb); BP, grip, BMI, waist. No fibrinogen/albumin | All-cause death via exit interviews 2013–2020; verbal-autopsy cause | Free registration + per-dataset approval (~3 working days); terms UNVERIFIED |

## Loader notes

- **IDs:** `ID` (individual), `householdID`, `communityID`. All waves are Stata (`.dta`; W3 Stata 13, W5 Stata 14) [1][4].
- **ID fix:** the baseline (2011) IDs must be adjusted to match later waves, "as noted in the release note of wave 2" [1][4]. Commonly this means appending "0" to the 2011 `householdID` and rebuilding `ID` (**UNVERIFIED** exact rule; read the 2013 release note).
- **2015 blood file:** contains `ID`, `bl_*` vars, `Blood_weight` [1]. File name likely `Blood.dta` (**UNVERIFIED**).
- **2011 blood file:** likely `Blood_20140429.dta` with `new*` vars and `qc1_vb*` CBC (**UNVERIFIED** file name).
- **2020 files:** `Demographic_Background`, `Family_Information`, `Health_Status_and_Functioning`, `Work_Retirement`, `Household_Income`, `Individual_Income`, `COVID_Module`, `Exit_Module`, `Weights.dta`, `Sample_Infor.dta` [4].
- **Unit conversion:** cholesterol mg/dL ÷ 38.67 = mmol/L; TG mg/dL ÷ 88.57; glucose ÷ 18.016; creatinine mg/dL × 88.4 = µmol/L.

## Manual steps for the user (priority order)

1. Register at https://charls.charlsdata.com/users/sign_up/agreement/en.html. **Read and save a copy of the terms** (they could not be fetched automatically).
2. After the email is confirmed, apply for: 2011 W1 (incl. blood + biomarkers), 2013 W2, 2015 W3 (incl. blood), 2018 W4, 2020 W5 (incl. exit and Sample_Infor), and Harmonized CHARLS. Wait about 3 working days.
3. Download to `data/raw/charls/`. Paste the agreement's redistribution, derived-data and commercial-use clauses into this card.
4. If the terms are silent on model weights, email charls_info@pku.edu.cn.

## Sources

1. CHARLS 2015 Blood Data Release Note (VersionID 20190620). https://charls.charlsdata.com/Public/ashelf/public/uploads/document/2015-charls-wave4/application/CHARLS_2015_Blood_Data_Release_Note.pdf
2. CHARLS 2011–2012 National Baseline Blood Data Users' Guide (2014). https://charls.charlsdata.com/Public/ashelf/public/uploads/document/2011-charls-wave1/application/blood_user_guide_en_20140429.pdf
3. CHARLS Wave 4 (2018) User's Guide. https://charls.charlsdata.com/Public/ashelf/public/uploads/document/2018-charls-wave4/application/CHARLS_2018_Users_Guide.pdf
4. CHARLS Wave 5 (2020) User Guide (VersionID 20231106). https://charls.charlsdata.com/Public/ashelf/public/uploads/document/2020-charls-wave5/application/CHARLS_2020_User_Guide_English.pdf
5. "Associations of serum cystatin C and its change with new-onset cardiovascular disease…" NMCD 2022 (abstract via search). https://www.sciencedirect.com/science/article/abs/pii/S0939475322002265
6. CHARLS data page. https://charls.charlsdata.com/pages/data/111/en.html
7. CHARLS FAQ (search-result summary). https://charls.pku.edu.cn/en/Faq/Frequently_Asked_Questions.htm
8. Harmonized CHARLS documentation version D. https://charls.charlsdata.com/Public/ashelf/public/uploads/document/harmonized_charls/application/Harmonized_CHARLS_D.pdf
9. Chen X et al. "Venous Blood-Based Biomarkers in CHARLS: 2015 wave", AJE 2019. https://academic.oup.com/aje/article/188/11/1871/5542023
