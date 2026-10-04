# Phase 0 data summary and go/no-go (PLAN.md §1.2 gate)
Date: 2026-10-03. Detail and sources are in the per-dataset cards.

## Bottom line
**GO, with a re-scoped human arm.**
- The gate (≥ ~5,000 people with ≥2 biomarker visits AND mortality follow-up) is met by **HRS**: about 7,300 with ≥2 dried-blood-spot visits, about 5,800 with ≥3, and deaths in the public files.
- **CHARLS** meets it more weakly: about 7–9k with both 2011 and 2015 bloods (unverified), one Δt step, deaths from exit interviews.
- **ELSA fails the mortality half.** Death status was removed from End User Licence data in July 2021, and no licence tier provides linked deaths. ELSA is usable for biomarker dynamics without a hazard head.
- HRS and CHARLS need user registration. HRS biomarkers also need a Sensitive Data agreement countersigned by an institution.

Suggested roles:
- **HRS:** primary cohort (dynamics and mortality).
- **CHARLS:** replication.
- **ELSA:** dynamics only.
- **NHANES:** cross-sectional data for the encoder, snapshot flow and hazard head.
- **Framingham teaching data:** smoke tests only.
- **Molecular arm:** TACO, TMS, GTEx and gene sets.

## Status
| Dataset | Status | N | N ≥2 visits | Δt | Mortality | Publish weights |
|---|---|---|---|---|---|---|
| NHANES 1999–2018 + LMF | Downloaded, `nhanes.py` tested | 56,367 adults | 0 | — | 8,366 deaths, median 9.4 y, leading cause | Yes (public domain) |
| Framingham teaching | Downloaded (CRAN), `framingham.py` tested | 4,434 | 3,987 (3,206 ≥3) | ~6 y | 1,550 deaths / 24 y + CVD events | No (test fixtures only) |
| HRS | Needs registration + Sensitive Data agreement | 45,234 ever | ≈7,286 DBS (≈5,801 ≥3) | 4 y | Death date public; cause restricted | Ask HRS |
| CHARLS | Needs registration | bloods 11,847 / 13,420 | ≈7–9k (UNVERIFIED) | 4 y (1 step) | Exit interviews to 2020 | Ask CHARLS |
| ELSA | Needs UKDS registration | ~6.2–6.4k bloods per wave | ≈6.5–7.5k (est.) | ~4 y | Not available since 2021 | No without permission; no AI tools on data |
| TACO | Downloaded (Zenodo), `taco.py` tested | 4,539 rodent samples | n/a | — | Intervention lifespan effect | Non-commercial (MGB OAL 1.0) |
| Tabula Muris Senis | Downloaded, `tms.py` tested | 1,023 pseudobulk / 925 bulk | n/a | — | — | Yes |
| GTEx v10 | Downloaded, `gtex.py` tested | 946 donors / 19,616 samples | 0 | — | Hardy scale only | Yes (open tier) |
| Gene sets | Downloaded | — | — | — | — | Yes (CC BY 4.0) |

## TACO (verified)
- **Paper:** Tyshkovskiy …, Gladyshev, Nature 654:173–188, online 27 May 2026, doi:10.1038/s41586-026-10542-3.
- **True:** 4,539 rodent samples, 26 tissues, 79 interventions, 96 studies, the web app and `tAge`.
- **Corrections:** 973 samples are microarray, not RNA-seq. "TACO" is the web calculator; the dataset is Zenodo 10.5281/zenodo.18763485. The licence is non-commercial.
- **New GEO data:** GSE292885.

## Disk
- `data/raw` ≈ 4.2 GB; `data/interim` ≈ 1.2 GB.
- Skipped on purpose: TACO `.pkl` models (~62 GB), TMS droplet h5ad (4 GB), all FASTQ/BAM.

## Manual steps (priority order)
1. **HRS:** register free at hrsdata.isr.umich.edu. Download RAND HRS 2022, Tracker and 2006–16 physical measures. Submit the Sensitive Health Data order form for DBS 2006–16 and VBS 2016 (signed agreement, institution countersigns). Email hrsquestions@umich.edu about saisho.ai as Receiving Entity and about publishing weights.
2. **CHARLS:** register, save the terms, request waves 2011–2020 and Harmonized CHARLS. Ask about publishing weights.
3. **ELSA:** register with UKDS and download SN 5050 (Stata). Never put ELSA microdata into any AI tool (EUL cl.5); run loaders locally. Optionally ask NatCen about mortality data and publishing weights.
4. **Optional:** official BioLINCC Framingham copy; UK Biobank application.
5. **Plan:** update PLAN.md §1.1/§1.2 from the corrected rows in each card.
