# Data card: ELSA (English Longitudinal Study of Ageing)

Status: desk research on 2026-10-03, no data downloaded. Anything not checked against a primary source is marked **UNVERIFIED**.

## TL;DR (gate-relevant)

- **Biomarker visits are fine.** Venous blood was collected at the nurse visits in waves 2 (2004/05), 4 (2008/09), 6 (2012/13), and 8 and 9 (2016–19, each wave covering half the sample). About 6,200–6,400 people gave blood at each of waves 2, 4 and 6. Δt ≈ 4 yrs.
- **Mortality is the blocker.** Under the End User Licence, ELSA no longer ships mortality data.
  The UKDS user guide says the ELSA Index File "Up until 2021 … also contained mortality status. Due to changes in UK Data Protection legislation, on 08/07/2021 the file was updated to remove mortality data. We do not currently have an onward sharing agreement for mortality data with NHS England … we kindly request that any data users who downloaded the version of the index file containing mortality data (pre 08/07/2021) delete it" [1, p.28].
  It also says: "NHS-England data linked to the ELSA sample (including mortality data) is not archived, and we do not have permission from NHS-England for onward sharing. Therefore, we are currently unable to fulfil any requests for linked health data." [1, p.29].
  This is not an EUL vs Special Licence question. No archived licence tier currently carries linked date or cause of death.
- **What remains for mortality:**
  - Deaths found during fieldwork: interview and nurse outcome codes, plus Harmonised ELSA interview status `RwIWSTAT`. Whether death codes are still present in current EUL files is **UNVERIFIED** and must be checked after download.
  - End-of-Life (EOL) proxy interviews: small, and only about half of known deaths get one. Productive interviews by wave: W2 ≈ 204 issued, W3 375, W4 242, W6 240, HCAP2 177, W11 347 [2].
  - So the right-censored survival outcome is incomplete and probably biased. No date of death, no cause (except EOL proxy reports).
- **Gate verdict for ELSA:** fails the "≥2 biomarker visits **and mortality follow-up**" criterion as currently released. It is usable as a longitudinal *biomarker-dynamics* cohort, without hazard supervision. Use HRS (see `hrs.md`) as the mortality-supervised cohort.
- **Licence verdict on publishing weights:** **No, not without written permission from UKDS/ELSA.** Details below. The EUL also **forbids using "Online Data Tools" (explicitly including AI tools) with the data** without written permission [3, cl.5]. Do not paste or upload ELSA microdata into any cloud LLM, including coding assistants.

## Source / access

| Item | Value |
|---|---|
| Study | UK Data Service SN **5050**, "English Longitudinal Study of Ageing: Waves 0–11, 1998–2024" [4] |
| Catalogue | https://datacatalogue.ukdataservice.ac.uk/studies/study/5050 (JS page, contents not machine-readable) |
| Latest edition | Jan 2026: Wave 11 core updated with weights. Wave 11 IFS/financial derived, Life History 2 and **Wave 11 Health Visit** data deposited [5] |
| Licence | **End User Licence (EUL)**, "Safeguarded" data. Free, needs a UKDS account [3] |
| Higher tiers | From wave 8, main-interview data come in EUL, Primary **Special Licence (SL)** (extra sensitive variables, state-pension-age, geography) and Primary **Secure Access (SA)** (lower-level geography, exact dates) versions [1, p.27]. Geography SL/SA: GN 33542 [1, p.28] |
| Non-archived variables | Request through NatCen Data Release Panel (elsadata@natcen.ac.uk). **Minimum £1,000**, delete after 1 year, approval not guaranteed [1, p.28–29] |
| Related SNs | HCAP: SN 8502. Harmonized ELSA-HCAP: SN 9081. ELSA COVID-19: SN 8688. Harmonized ELSA COVID: SN 9228 [1, p.28] |
| Harmonised ELSA | Built by the Gateway to Global Aging (RAND/USC) and distributed inside SN 5050 under EUL [1, p.28]. Version G.3 (June 2023; waves 0–9) was used in recent papers [6]. Newer version **UNVERIFIED**. Includes Harmonized ELSA End of Life and Life History [7] |

### EUL clauses that matter (UKDS CD137 EUL v16.00, 25 Feb 2026 [3])

- cl.4: "Not to give access to the Data Collection(s), in whole or in part, or to any Dataset(s) derived from the Data Collection(s) (including synthetic Dataset(s)), except to Registered Users …"
- cl.5: "To abstain from using any Online Data Tools in connection with your use of the Data Collection(s), unless explicit written permission is granted by the Data Service Provider." The definition of Online Data Tools covers tools "that employ 'artificial intelligence' (AI) technologies, including generative AI technologies".
- cl.7: "Not to use the data to attempt to obtain or derive information relating specifically to an individual or household …"
- cl.8: "To adhere to the statistical disclosure control standards … in any outputs I produce and publish."
- cl.13: offer derived data collections for deposit. cl.18: at end of access period destroy all copies, "including … derived Datasets".

**Analysis: can trained weights be published?**
- Aggregate outputs that pass statistical disclosure control are allowed (cl.8): metrics, coefficients, plots with adequate cell sizes.
- Weights of a **generative** model (latent SDE or flow) that can sample realistic multivariate trajectories behave like a *synthetic dataset* generator. cl.4 explicitly bars sharing synthetic datasets derived from the collection with non-registered users.
- Over-parameterised models can also memorise individual records (membership-inference risk), which cuts against cl.7/cl.8.
- **Verdict:** publishing ELSA-trained weights is **not clearly permitted**. Treat it as prohibited unless UKDS/ELSA give written permission (help@ukdataservice.ac.uk; elsadata@natcen.ac.uk).
- Safe default: publish code, plus weights trained only on permissive data (e.g. NHANES). Report ELSA results as aggregate metrics only.

### Registration steps (manual, user)

1. Create a UK Data Service account at https://beta.ukdataservice.ac.uk/myaccount (UK federated/institutional login, or a non-UK-institution username registration). Accept the EUL.
2. Search SN 5050 and "Add to account" or download. You must state a project description ("Project Information"). Usage must match it [3, cl.2]. Pick format: Stata (.dta), SPSS (.sav) or tab-delimited (.tab).
3. Commercial use is a separate category ([3] defines Commercial Use). If the project is done through a company (saisho.ai), declare usage honestly. Commercial use of ELSA may need separate permission (**UNVERIFIED**).
4. **Optional:** Special Licence for SN 5050 SL files needs a UKDS SL application and depositor approval. Not needed for v1, and it would **not** add mortality.

## Cohort size, visits, Δt

| Nurse/health visit | Fieldwork | Nurse visit N | Blood sample N (core members) | Source |
|---|---|---|---|---|
| Wave 0 (HSE 1998/99/2001) | 1998–2001 | — | HSE bloods: total chol, HDL, ferritin, Hb, etc. (`cholval`, `hdlval`, `ferval`, `haemval`) for Cohort 1 | [8] |
| Wave 2 | 2004–05 | 7,666 | **6,231** (81% of nurse visits) | [9] |
| Wave 4 | 2008–09 | 8,643 | **6,438** | [10] |
| Wave 6 | 2012–13 | 8,054 (7,699 core in private households) | **6,180** | [11] |
| Wave 8 (half sample) | 2016–17 | 3,525 (3,471 core) | **2,479** usable | [12] |
| Wave 9 (other half) | 2018–19 | 3,047 | ~1,868 (4,347 W8+W9 combined minus 2,479) | [13] |
| Wave 10 | 2021–23 | **no health visit** (COVID) | — | [14] |
| Wave 11 (whole sample) | 2023–24 | 4,928 health visits by core members | released Jan 2026; blood analytes in EUL file **UNVERIFIED** | [5][14] |

- People with ≥1 nurse visit in waves 2/4/6: 12,291; 8,906 of them had complete biomarker data [15].
- **N with ≥2 blood visits: not published directly. Estimate ~6,500–7,500 (UNVERIFIED).**
- **N with ≥3 blood visits: estimate ~4,000–5,000 (UNVERIFIED).** For comparison, 2,437 people had CRP on 2–3 occasions across W0/W2/W4 [16], and 4,923 respondents were analysed for CRP across W2/W4/W6 [16b].
- Compute exact counts on download with `idauniq` + `bloodr==1` (see loader notes).
- Δt: nurse waves are about **4 years apart** (W2→W4→W6). With the W8/W9 split design, W6→W8 is ~4 yrs and W6→W9 is ~6 yrs [12][13]. Exact interview month is in EUL; full interview date is removed [2].
- Wave 8 selection was purposive: those who "had responded to all previous nurse visits" were prioritised for W8 [12]. That is informative selection, so model it or weight it.

## Features (mapped to PLAN.md §3.2 core panel)

| §3.2 feature | ELSA availability (nurse waves) | Variable (W4 naming [10]) | Unit |
|---|---|---|---|
| age, sex | all | main interview (`indager`, `indsex` **UNVERIFIED**; top-coded 90+) | yrs |
| BMI | W2, W4, W6. W8+ weight in main interview, **no height/waist at W8/9** [12] | `bmival` (`htval`, `wtval`) | kg/m² |
| waist | W2, W4, W6 (not W8/9) | `wstval` | cm |
| SBP/DBP | all nurse waves | `sysval`, `diaval` (mean of valid readings), raw `sys1-3`, `dias1-3` | mmHg |
| total/HDL/LDL chol, TG | all blood waves (LDL from W2? Listed at W4, W6, W8) | `chol`, `hdl`, `ldl`, `trig` | mmol/L |
| HbA1c / glucose | HbA1c W2+. Fasting glucose fasted subsample only | `hba1c` (%), `fglu` (mmol/L) | |
| CRP | all blood waves | `hscrp` | mg/L |
| fibrinogen | all blood waves | `cfib` | g/L |
| creatinine/eGFR | **not in standard ELSA panel** (absent from W2/4/6/8 analyte lists [9][10][11][12]) | — | |
| haemoglobin | all blood waves | `hgb` | g/dL |
| albumin | **not in standard panel** | — | |
| grip | W2+ (3 per hand) | `mmgsd1-3`, `mmgsn1-3` | kg |
| smoking, diabetes, BP/lipid meds | main interview (self-report). Nurse medication codes (**UNVERIFIED** names) | | |
| extras | ferritin `rtin` (ng/mL), WBC `wbc`, MCH `mch`, DHEAS `dheas`, IGF-1 `igf1` (W4, W6), vitamin D (W8+), ApoE genotype | | |

Missing codes: `-1` not applicable/invalid. Other negatives (-2 … -9) are standard ELSA missing codes [10]. `bloodr` is the derived flag for blood taken and received by lab.

## Outcomes

- All-cause mortality: **incomplete under EUL** (see TL;DR). As of the end of 2012 (last linked mortality release), 2,763 deaths = 24% of the original sample [17]. 4,709 respondents were known dead before W6 fieldwork [2].
- Cause of death: only EOL proxy interviews (n≈1,600 across all EOL waves) [2].
- Linked NHS data (HES, cancer, ONS mortality) exist at NatCen but are "not archived". Future access through a DEA-accredited secure environment is mentioned in participant materials [18]. **No current route.**

## Expected size on disk

The full SN 5050 Stata bundle (all waves, IFS derived, nurse, EOL, harmonised) is probably ~1–3 GB unzipped (**UNVERIFIED**). Nurse files alone are tens of MB.

## Verified/corrected PLAN.md §1.1 row

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **ELSA** | Human | Yes. Nurse-visit venous blood W2/W4/W6 (~6.2–6.4k each, Δt≈4 y), W8/W9 half-samples (~4.3k combined), W11 whole sample (4.9k visits). W0 = HSE bloods | Lipids, HbA1c, CRP, fibrinogen, Hb, ferritin, WBC, (IGF-1, DHEAS, vit D); BP, grip, BMI/waist (not W8/9). **No creatinine/albumin** | **Mortality NOT released since 07/2021** (only fieldwork-identified deaths + small EOL proxy interviews; no linked date/cause) | UKDS SN 5050, free EUL. EUL bans AI "Online Data Tools" and sharing derived/synthetic datasets |

## Loader notes

- Files: one `.tab`/`.dta` per wave, all keyed on **`idauniq`** (unique individual analytical serial number, valid across all files) [10][2].
- File names below are **UNVERIFIED** (from search-result snippets; check `ls` after download):
  - `wave_2_nurse_data_v2`, `wave_4_nurse_data`, `wave_6_elsa_nurse_data_v2`, `wave_8_elsa_nurse_data_eul_v1`, `wave_9_elsa_nurse_data_eul_*`
  - main `wave_N_elsa_data_*`, `index_file_wave_0-wave_5_v2`
  - harmonised `h_elsa_g3.dta` (or newer), `wave_0_common_variables`
- Wave 8+ main-interview files include `w8nurout`/`w9nurout`/`w11nurout` and `w8bsout`… blood-outcome codes [1].
- W4 nurse-file variables, confirmed in the W4 nurse user guide [10]:
  - blood: `cfib` (g/l), `chol`, `hdl`, `trig`, `ldl` (mmol/l), `rtin` ferritin (ng/ml), `hscrp` (mg/l), `dheas` (µmol/l), `igf1` (nmol/l), `fglu` (mmol/l, fasting only), `hgb` (g/dl), `hba1c` (%), `wbc` (10⁹/L), `mch` (pg), `bloodr`
  - anthropometry: `htval`, `wtval`, `bmival`, `wstval`
  - BP: `sysval`, `diaval`
  - grip: `mmgsd1-3`, `mmgsn1-3`
  - fasting: `fastask`/`fasteli`
- Names are lower-case in the files (guide prints them upper-case). Check whether W2/W6/W8 use the same names. Expect minor suffix differences.
- Fasting matters: TG/LDL/glucose are valid only for fasted samples, and over-80s and treated diabetics were not asked to fast [10][12].
- Lab and assay changes across waves are possible. Check the nurse user guide before pooling. CRP >10 mg/L is usually excluded as acute.
- Harmonised ELSA gives `RwIWSTAT`, `RADYEAR`/`RADMONTH` (death year/month from EOL or spouse report, per the Gateway docs pattern [19]), plus `RwAGEY`, `RwBMI` etc. **Check death-variable coverage in the current release.**
- Weights: nurse and blood weights exist per wave (e.g. W6 blood weight for 6,180) [11].

## Manual steps for the user (priority order)

1. Register at UK Data Service, accept the EUL, and download **SN 5050 in Stata format**. Include nurse files, Harmonised ELSA and EOL. Put it under `data/raw/elsa/`. Never load it into cloud AI tools (EUL cl.5).
2. After download, run a local count of `idauniq` with `bloodr==1` per wave. Check which death indicators survive (`RwIWSTAT` codes, outcome codes) so the gate number can be recomputed.
3. Email elsadata@natcen.ac.uk to ask: (a) is any mortality/date-of-death data available, or planned, to external researchers; (b) is publishing trained generative-model weights acceptable. Email help@ukdataservice.ac.uk to ask for written permission regarding cl.4/cl.5.
4. Re-scope PLAN.md §1.2: HRS becomes the primary mortality-supervised cohort; ELSA becomes a secondary biomarker-dynamics and replication cohort.

## Sources

1. ELSA User Guide to the Main Interview Datasets Waves 1–11 (NatCen/UKDS), pp.27–29, 33. https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_elsa_waves_1-11_interviewer_data_user_guide.pdf
2. ELSA EoL User Guide (waves 2,3,4,6,11, HCAP2). https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_elsa_eol_user_guide_waves_2_3_4_6_11_hcap2.pdf
3. UK Data Service End User Licence Agreement, CD137 v16.00, 25 Feb 2026. https://ukdataservice.ac.uk/app/uploads/cd137-enduserlicence.pdf
4. ELSA Data & documentation. https://www.elsa-project.ac.uk/data-and-documentation
5. "New edition of ELSA Wave 11 dataset now available" (30 Jan 2026). https://www.elsa-project.ac.uk/post/new-edition-of-elsa-wave-11-dataset-now-available
6. JOGH 2025 APC analysis citing Harmonized ELSA G.3. https://jogh.org/2025/jogh-15-04260
7. Gateway to Global Aging, harmonized data overview. https://g2aging.org/harmonized-data/overview
8. ELSA Wave 0 (HSE) User Guide. https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_Wave_0_User_Guide.pdf
9. ELSA Wave 2 Technical Report §6.2.3. https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_wave_2_technical_report.pdf
10. ELSA Wave 4 Nurse Dataset User Guide v1 (variables §7.4–7.7). https://ifs.org.uk/sites/default/files/output_url_files/wave_4_nurse_dataset.pdf . Wave 4 Technical Report §8.3.2: https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_wave_4_technical_report.pdf
11. ELSA Wave 6 Technical Report §8.5.2. https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_elsa_wave_6_technical_report_v1.pdf
12. ELSA Wave 8 Technical Report (§4, weights). https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_elsa_w8_technical_report_v7.pdf
13. ELSA Wave 9 Technical Report. https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_elsa_w9_technical_report_v3.pdf
14. ELSA Wave 10 technical report / Wave 11 summaries. https://www.ucl.ac.uk/population-health-sciences/sites/population_health_sciences/files/elsa_w10_technical_report.pdf
15. Sex and education differences in trajectories of physiological ageing (ELSA). https://pmc.ncbi.nlm.nih.gov/articles/PMC11741463/
16. 10-year CRP trajectories and healthy ageing (ELSA). https://pmc.ncbi.nlm.nih.gov/articles/PMC6333942/ . 16b: https://pubmed.ncbi.nlm.nih.gov/31400408 (search-result summary; abstract not directly read: **UNVERIFIED** wording)
17. IFS, ELSA data linked to administrative data sets (search-result summary). https://ifs.org.uk/elsa-data-linked-administrative-data-sets
18. ELSA "Linking your data". https://www.elsa-project.ac.uk/linking-your-data
19. Harmonized ELSA-HCAP documentation (RADYEAR/RADMONTH definitions). https://doc.ukdataservice.ac.uk/doc/9081/mrdoc/pdf/9081_harmonized_elsa-hcap_a2_2018.pdf
