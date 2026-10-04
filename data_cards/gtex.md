# Data card: GTEx v10 (open-access bulk RNA-seq + public phenotypes)

Status: **downloaded, processed, loader tested** (2026-10-03).

| Field | Value |
|---|---|
| Source | GTEx Portal, Adult GTEx **v10** (RNA-SeQC 2.4.2), https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression ; files at `https://storage.googleapis.com/adult-gtex/...` |
| Licence / access | Open-access tier: free download, no login. GTEx portal data are released for research use; cite the GTEx Consortium (Science 2020, 369:1318). Protected tier (dbGaP **phs000424**: exact age, cause of death, medical history, genotypes) needs a dbGaP application and was not used. Publishing models trained on open-access expression is fine. |
| Fetch | `python scripts/fetch_gtex.py` (idempotent, ~2.3 GB) |
| Loader | `src/mwm/data/gtex.py`: `build()`, `load(tissue=..., tissue_detail=...)`, `to_symbols(X)` |
| Raw on disk | 2.2 GB: `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz` (2.26 GB), SampleAttributesDS (38 MB), SubjectPhenotypesDS (20 kB), 2 data dictionaries |
| Interim on disk | 461 MB: `expression.parquet` (19,616 samples × 20,545 genes, log1p TPM rounded to 0.01, genes with mean TPM ≥ 1), `genes.parquet`, `metadata.parquet` |

## Numbers
- 19,616 RNA-seq samples, 946 donors with RNA-seq (981 in the subject file), 54 tissue sites (30 broad tissues).
  Largest: brain (3,234, 13 regions), skin, oesophagus, blood vessel, adipose, whole blood (Blood, 1,130). Smallest: kidney medulla (11).
- Donor age bins (public, 10 y): 20-29: 81 · 30-39: 75 · 40-49: 145 · 50-59: 305 · 60-69: 308 · 70-79: 32.
  `visit_age` = bin midpoint, so age resolution is coarse. Exact age is protected.
- Hardy scale (DTHHRDY): 0 ventilator case 496 · 1 violent/fast 35 · 2 fast natural 232 · 3 intermediate 54 ·
  4 slow death 113 · missing 16. It describes *death circumstances* for post-mortem donors. It's not a survival outcome.
- Sex: 1 = male (≈ 2/3 of samples).
- **N with ≥ 2 visits: 0** (post-mortem, one time point per donor; several tissues per donor).

## Features / outcomes
Bulk gene expression for about 20.5k expressed genes (Ensembl IDs, mapped to HGNC symbols via `genes.parquet`),
plus sample covariates: RIN, ischaemic time, tissue. There's no mortality outcome; age bin and Hardy scale only.
Use: the human molecular reference for the rodent arm (same human-gene coordinates as TACO/TMS after ortholog
mapping), age-associated pathway directions per tissue, and a cross-species consistency check (PLAN.md §7).
Covariates to regress out: Hardy scale (ventilator cases differ strongly), ischaemic time, RIN, sex.

## Verified/corrected PLAN.md §1.1 row

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **GTEx v10** | Human (post-mortem, 946 donors aged 20–79) | No | Bulk RNA-seq, 19.6k samples, 54 tissue sites; public age only in 10-y bins | Death circumstances (Hardy 0–4), not survival | Open: expression + sex/age-bin/Hardy, direct download. Exact age, cause of death and history need dbGaP (phs000424). Confirmed |

## Usage
```python
from mwm.data import gtex
X, meta = gtex.load(tissue_detail="Whole Blood")   # (n, 20545) log1p TPM, metadata
Xs = gtex.to_symbols(X)                            # columns -> HGNC symbols
```
