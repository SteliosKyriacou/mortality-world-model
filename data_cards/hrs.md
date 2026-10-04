# Data card: HRS (Health and Retirement Study, University of Michigan)

Status: desk research on 2026-10-03, no data downloaded. The hrs.isr.umich.edu and hrsdata.isr.umich.edu sites block automated fetching (bot challenge / HTTP 403), so several items come from search-result snippets or peer-reviewed papers. They are marked **UNVERIFIED** where the primary page could not be read.

## TL;DR (gate-relevant)

- **Biomarkers:** dried blood spots (DBS) from ~6,000 people per wave on rotating halves. One half was measured in 2006/2010/2014, the other in 2008/2012/2016. So each person has DBS every **4 years**. Assays: total chol, HDL, HbA1c, CRP, cystatin C; IL-6 added 2014/16 [1].
  - **N with ≥2 DBS visits ≈ 7,286; N with ≥3 ≈ 5,801.** These come from a 2006/08 baseline of 10,408 with HbA1c, with 3,070 deaths over mean 8.9 y follow-up to 2018 [2].
  - The **gate (≥5,000 with ≥2 biomarker visits + mortality) is passed.**
  - 2016 was the last DBS year [1]. 2016 also had a **Venous Blood Study (VBS)**: CMP incl. creatinine and albumin, CBC, lipids, HbA1c, CRP, cystatin C, plus a ~4,000 random subsample with novel aging assays [3]. VBS 2016 Final v2.0 was released 1 Oct 2024 [4].
  - Venous blood collection was planned to continue (2020/2022 VBS). Release status of a second VBS wave is **UNVERIFIED** [4].
- **Mortality:** public Tracker file and RAND HRS give death year/month from exit interviews, spouse reports and NDI-assisted searches. **Cause of death (NDI) is restricted data** [5].
- **Access:**
  - Core/RAND/Tracker files are free after public registration.
  - **DBS and VBS biomarker files are "Sensitive Health Data".** They need a supplemental order form plus a signed **HRS Sensitive Data Access Use Agreement** countersigned by your institution ("Receiving Entity") [1][6][7].
- **Licence verdict on publishing weights:** not explicitly addressed in the documents read. Public Conditions of Use forbid attempting to identify participants and transferring HRS data to third parties [8]. Aggregate model weights are not "data", but a generative model that emits synthetic records is a grey zone, and the sensitive-data agreement is stricter (text not readable; **UNVERIFIED**). **Get written confirmation from HRS (hrsquestions@umich.edu) before publishing weights trained on DBS/VBS.**

## Source / access

| Item | Value |
|---|---|
| Portal | https://hrsdata.isr.umich.edu (data downloads), https://hrs.isr.umich.edu |
| Public registration | Free account, agree to Conditions of Use [8]. Public files download immediately |
| Conditions of Use (public) | "make no attempts to identify study participants"; "must not transfer HRS Public Release data to any third party other than staff or students for whom you are directly responsible"; "certify the destruction of any downloaded Public Release data file as well as any data files derived from the downloaded file when requested" [8] |
| Sensitive Health Data | Biomarker 2006–2016, DBS CRP and IL-6 crosswave files, VBS 2016 (+ flow cytometry, subsample, supplemental), epigenetic clocks, etc. Order via https://hrsdata.isr.umich.edu/data-products/sensitive-health/order-form . Requires a signed Sensitive Data Access Use Agreement per individual user, "with authorization of the Receiving Entity" (institution) [6][7]. The Kim et al. 2024 paper: "publicly available. Researchers who want to use biomarker data are required to submit a Sensitive Data Access Use Agreement." [1] |
| Restricted Data | NDI cause of death, geography, linked Medicare/SSA etc. Needs an institutionally countersigned confidentiality agreement, **proof of current federal funding**, IRB, research plan and data security plan, or use of the MiCDA enclave [7] |
| RAND HRS Longitudinal File 2022 (V1) | Released May 2025. Waves 1992–2022, **45,234 respondents**, one row per respondent, Stata/SAS/SPSS [9]. Public |
| Harmonized HRS | Gateway to Global Aging; public (version B docs) [10] |

**Concern for this project:** sensitive and restricted tiers assume an institution (university/research org) countersigns. Whether a small company (saisho.ai) qualifies as Receiving Entity is **UNVERIFIED**. Ask HRS.

## Cohort size, visits, Δt

- Total ever-respondents: 45,234 (RAND 2022) [9]. Biennial core interviews 1992–2022.
- Enhanced face-to-face (EFTF) interview with physical measures, saliva and DBS on a random half each wave from 2006. Each half repeats every 4 years [1].
- DBS samples: ~6,000 per wave, completion 81–90% [1].
- HbA1c longitudinal counts (Kim/Shi-type terminal trajectory analysis) [2]:

| | N |
|---|---|
| Baseline 2006/2008 with HbA1c | 10,408 |
| Also measured 2010/2012 | 7,286 (70.0%) |
| Also measured 2014/2016 | 5,801 (55.7%) |
| Deaths (to 2018) | 3,070; mean follow-up 8.9 y |

  These counts start from the 2006/08 baseline. People entering at 2010/12 (new cohorts) add more with ≥2 visits (2010/12 → 2014/16), so ≥2 visits is probably ~8–9k (**UNVERIFIED**).
- Δt between DBS: **4 years** by design (±interview timing, about ±1 y).
- VBS 2016: N ≈ 9,900 with venous blood (**UNVERIFIED** exact number). Single time point unless a later VBS is released. It can be linked to the person's 2012/2014 DBS, but assay matrices differ.

## Features (mapped to PLAN.md §3.2)

| §3.2 feature | HRS source |
|---|---|
| age, sex | RAND `RwAGEY_B`, `RAGENDER`, `RABYEAR` |
| BMI | self-report every wave (`RwBMI`). Measured height/weight in EFTF physical measures 2006+ (`RwPMBMI`, **UNVERIFIED** RAND name) |
| waist | EFTF physical measures |
| SBP/DBP | EFTF physical measures (3 readings) |
| total/HDL chol | DBS 2006–16; VBS 2016 (+LDL, TG in VBS) |
| HbA1c / glucose | DBS HbA1c 2006–16; VBS HbA1c + glucose |
| CRP | DBS 2006–16; VBS |
| cystatin C / creatinine / eGFR | DBS cystatin C 2006–16; VBS creatinine + cystatin C |
| haemoglobin, albumin | VBS 2016 only (CBC, CMP) [3] |
| fibrinogen | **not available** |
| grip | EFTF physical measures (2 per hand) |
| smoking, diabetes, BP/lipid meds | core self-report (RAND `RwSMOKEN`, `RwDIAB`, `RwHIBP`; med flags in core section C) |
| extras | IL-6 (2014/16), VBS flow cytometry, epigenetic clocks (sensitive), polygenic scores |

Physical-measure files sit in the public core (section I/PM) in most years, but HRS also lists "physical measures" among sensitive products. Tier is **UNVERIFIED**.

## Outcomes

- **All-cause death:** exit interviews with proxies (from 1994/95) plus tracker updates. Year and month of death are in the public Tracker and RAND (`RADYEAR`, `RADMONTH`; `RwIWSTAT` 5 = died this wave, 6 = died previous wave) [9][10].
  - Kim et al. note mortality "determined by exit interviews with proxy respondents as well as the National Death Index data linked to respondents" [2].
- **Cause of death:** NDI-linked cause of death is **Restricted** [7]. Exit interviews give proxy-reported cause. CVD as an endpoint can be approximated from self-reported heart disease/stroke.
- Follow-up through 2022 core/exit (RAND 2022 V1) [9].

## Expected size on disk

- RAND HRS Longitudinal 2022 Stata: ~1–2 GB (**UNVERIFIED**).
- Tracker: ~100 MB.
- Biomarker files: a few MB each.
- Core raw files (fixed-width `.da` + `.dct` dictionaries + SAS/Stata/SPSS setup programs): several GB for all years.

## Verified/corrected PLAN.md §1.1 row

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **HRS** | Human | Yes. Biennial core; EFTF physical measures + DBS on rotating halves 2006–2016 (each person every 4 y; ~7.3k with ≥2, ~5.8k with ≥3 DBS from 2006/08 baseline); VBS 2016 (~9.9k, single time) | BP, grip, waist, BMI; DBS TC, HDL, HbA1c, CRP, cystatin C (+IL-6 2014/16); VBS adds CMP (creatinine, albumin, glucose), CBC, lipids, novel aging assays | All-cause mortality with year/month (public tracker/RAND); NDI cause of death restricted | Free public registration; **DBS/VBS = Sensitive Health Data (signed agreement + institutional authorisation)**; restricted tier needs federal funding/IRB or enclave |

## Loader notes

- **IDs:** `HHID` (6-char household) + `PN` (3-char person number) → person key `HHIDPN` (numeric in RAND = HHID*1000+PN). All HRS files merge on HHID+PN [9].
- **Wave letter prefixes in raw files:** 2006 = K, 2008 = L, 2010 = M, 2012 = N, 2014 = O, 2016 = P (e.g. `OA1C_ADJ` = 2014 NHANES-equivalent HbA1c).
- **Biomarker files:** `BIOMK06BL` … `BIOMK16BL` (confirmed for 06 and 08 by HRS codebook URLs [11]). Each has raw (`xA1CUW`?) and **NHANES-equivalent `_ADJ` variables**, e.g. `KCRP_ADJ`, `KA1C_ADJ`, `KCHOL_ADJ`, `KHDL_ADJ`, `KCYSC_ADJ`. The example equation is `OA1C_ADJ = -2.444194 + OA1CUW*1.511696` [11]. Exact names per year are **UNVERIFIED**; use the `_ADJ` (venous/NHANES-equivalent) values for longitudinal analysis [1].
- **Units:** chol/HDL mg/dL; HbA1c %; CRP µg/mL (= mg/L); cystatin C mg/L [1].
- **Lab changes:** Biosafe (2006) → FlexSite/Univ. Washington (2008) → Heritage/UW (2010–12) → UW (2014–16). Use the `_ADJ` values [1].
- **Mortality:** Tracker `TRK20xx` has `KNOWNDECEASEDMO/YR`, `EXDEATHMO/YR` (**UNVERIFIED** names). RAND has `RADYEAR`, `RADMONTH`, `RAXYEAR`?, `RwIWSTAT`.
- **Formats:** raw HRS = ASCII `.da` + `.dct` (Stata dictionary) / `.sas` / `.sps`. RAND and Harmonized come as ready `.dta`/`.sas7bdat`/`.sav`.

## Manual steps for the user (priority order)

1. Register free at https://hrsdata.isr.umich.edu (public), accept the Conditions of Use, and download: **RAND HRS Longitudinal File 2022 (V1)** (Stata), **Tracker 2022** (or latest), and the core physical-measure sections 2006–2016. Place them in `data/raw/hrs/`.
2. Submit the **Sensitive Health Data order form** for Biomarker 2006, 2008, 2010, 2012, 2014, 2016 (+DBS CRP/IL-6 crosswave) and VBS 2016. Sign the Sensitive Data Access Use Agreement and get the institutional signature (Receiving Entity). If saisho.ai is the institution, confirm with HRS that a company qualifies.
3. Email hrsquestions@umich.edu asking whether publishing trained (generative) model weights and derived latents is allowed under the sensitive data agreement.
4. Optional, later: restricted NDI cause-of-death. Realistically only through an academic collaborator with federal funding, or the MiCDA enclave.

## Sources

1. Kim JK et al. "Dried blood spot based biomarkers in the Health and Retirement Study: 2006 to 2016." Am J Hum Biol 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC10873048/ (abstract via Europe PMC: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:37803815)
2. "Terminal trajectory of HbA1c for 10 years supports the HbA1c paradox" (HRS). https://pmc.ncbi.nlm.nih.gov/articles/PMC11070457/
3. HRS 2016 VBS documentation (search-result summaries; PDFs blocked to automated fetch). https://hrsdata.isr.umich.edu/sites/default/files/documentation/data-descriptions/1726681096/HRS2016VBSV2DD.pdf
4. HRS Data Announcements (search summary: VBS 2016 Final v2.0, 1 Oct 2024; VBS neuropathological/supplemental 22 Dec 2025). https://hrs.isr.umich.edu/news/data-announcements
5. HRS Available Restricted Data Products. https://hrs.isr.umich.edu/data-products/restricted-data/available-products
6. Sensitive Health Data Order Form. https://hrsdata.isr.umich.edu/data-products/sensitive-health/order-form . Sensitive Biomarker and Health Data page: https://hrsdata.isr.umich.edu/data-products/sensitive-health
7. HRS Sensitive Data Access Use Agreement (PDF; not readable by tool, summary from search). https://hrs.isr.umich.edu/sites/default/files/HRS-Sensitive-Data-Access-Use-Agreement.pdf
8. HRS Conditions of Use. https://hrsdata.isr.umich.edu/data-products/conditions-of-use
9. RAND HRS Longitudinal File 2022 (V1). https://hrsdata.isr.umich.edu/data-products/rand-hrs-longitudinal-file-2022 . Docs: https://www.rand.org/content/dam/rand/www/external/labor/aging/dataprod/randhrs1992_2022v1.pdf
10. Harmonized HRS documentation (version B). https://hrsdata.isr.umich.edu/sites/default/files/documentation/other/Harmonized_HRS_B.pdf
11. HRS 2006/2008 Biomarker codebooks. https://hrs.isr.umich.edu/sites/default/files/meta/bio2006/codebook/biomk06bl_r.htm , https://hrs.isr.umich.edu/sites/default/files/meta/bio2008/codebook/biomk08bl_r.htm
