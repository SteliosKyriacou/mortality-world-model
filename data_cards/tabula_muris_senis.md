# Data card: Tabula Muris Senis (TMS), FACS pseudobulk + bulk RNA-seq atlas

*Last verified: 2026-10-03.* Molecular arm (PLAN.md §3.3), mouse only, cross-sectional (no individual is
measured twice). No outcomes: there are no lifespan or survival labels. Age is the only label.

## Sources

| Component | Reference | Where | Licence |
|---|---|---|---|
| Single-cell FACS / Smart-seq2, raw counts | Tabula Muris Consortium, *A single-cell transcriptomic atlas characterizes ageing tissues in the mouse*, Nature 583:590–595 (2020), doi:10.1038/s41586-020-2496-1 | figshare 10.6084/m9.figshare.8273102 (v3), file `tabula-muris-senis-facs-official-raw-obj.h5ad` (2.37 GB), https://ndownloader.figshare.com/files/23939711 | MIT (figshare record); also mirrored in figshare 12827615 (CC BY 4.0) |
| Bulk RNA-seq organ atlas, raw counts | Schaum et al., *Ageing hallmarks exhibit organ-specific temporal signatures*, Nature 583:596–602 (2020), doi:10.1038/s41586-020-2499-y | GEO **GSE132040**: `GSE132040_190214_A00111_0269_AHH3J3DSXX_190214_A00111_0270_BHHMFWDSXX.csv.gz` (38 MB) + `GSE132040_MACA_Bulk_metadata.csv.gz` | GEO public, no restrictions |

Access: open, no registration. **Not downloaded** (size rule: ≤ 3 GB per file): droplet raw object
(`tabula-muris-senis-droplet-official-raw-obj.h5ad`, 4.06 GB; adds 1 m and 30 m ages, ~245k cells), processed
FACS (4.8 GB), droplet (8.2 GB) and BBKNN (12.9 GB) objects. Per-tissue processed h5ads (figshare 12654728,
0.05–1.3 GB each) hold log-normalised, not raw, data.

## What is on disk

| Path | Content | Size |
|---|---|---|
| `data/raw/tms/GSE132040_counts.csv.gz` | 54,352 genes (GENCODE vM19 symbols + HTSeq `__` rows) × 947 bulk samples, raw counts | 40 MB |
| `data/raw/tms/GSE132040_MACA_Bulk_metadata.csv.gz` | sample → organ_mouseID, age (months), sex, SRR | 17 kB |
| `data/raw/tms/FACS_H5AD_DELETED.txt` | note: FACS h5ad was downloaded, reduced to pseudobulk, **then deleted** to save disk | – |
| `data/raw/orthologs/HOM_AllOrganism.rpt` (+ `mouse_human_1to1.tsv`, `rat_human_1to1.tsv`) | MGI homology report; 17,178 mouse↔human and 15,830 rat↔human strict one-to-one pairs | 23 MB |
| `data/interim/tms/facs_pseudobulk_counts.parquet` | 1,023 groups × 22,966 mouse genes, summed raw counts | 73 MB |
| `data/interim/tms/facs_pseudobulk_meta.parquet` | group_id, mouse_id, tissue, cell_type, cell_ontology_id, sex, age_months, visit_age, age_frac_lifespan, n_cells, total_counts | 36 kB |
| `data/interim/tms/bulk_counts.parquet` | 947 samples × 54,352 mouse genes, raw counts | 99 MB |
| `data/interim/tms/bulk_meta.parquet` | sample_id, mouse_id, tissue, sex, age_months, visit_age, age_frac_lifespan, SRR, total_counts | 33 kB |

## Contents

**FACS pseudobulk.** The source object has 110,824 cells, 23 tissues, 21 mice, ages 3, 18, 21 and 24 months (the
FACS arm has no 1 m or 30 m mice; the 21 m group is tiny), 120 cell_ontology_class labels.
Pseudobulk = sum of raw counts per **mouse × tissue × cell_ontology_class**, keeping groups with ≥ 20 cells:
**1,023 groups** (of 2,268) covering 102,011 cells, 23 tissues, 97 cell types, 21 mice.
By age: 3 m: 410 groups / 10 mice; 18 m: 307 / 4; 21 m: 6 / 3; 24 m: 300 / 4. Sex: 682 male, 341 female groups.
Cells per group: median 58 (20–1,222). Every group has ≥ 4.8 × 10⁵ counts.

**Bulk (GSE132040).** 947 samples, 17 organs (BAT, GAT, MAT, SCAT, bone, brain, heart, kidney, limb muscle,
liver, lung, marrow, pancreas, skin, small intestine, spleen, white blood cells), 10 ages (1, 3, 6, 9, 12, 15,
18, 21, 24, 27 months), both sexes (≈ 72% male). The loader's default QC (≥ 1 × 10⁵ counts) keeps **925 samples**.
**Overlap:** 752 of these samples are already part of the TACO meta-dataset (source `GSE132040`, see
`data_cards/taco.md`); don't double-count them when pooling the two.

**Features.** `load_*` returns log1p CPM (CPM computed on *all* mouse genes, then subset) mapped to human
genes by MGI one-to-one orthologs on the mouse **symbol**: 14,556 human genes for pseudobulk (2019-era symbols in
the h5ad; some renamed genes are lost) and 16,221 for bulk.

**Time axis.** `age_frac_lifespan = age / species median lifespan`, using `mwm.data.common.lifespan_fraction`
(mouse median = 26 months, the shared constant set in `common.py`; C57BL/6J median lifespan is reported at roughly
26–30 months depending on sex and colony, e.g. Yuan et al., Aging Cell 2009). `visit_age` is in years.

## Verified/corrected PLAN.md §1.1 row

| Dataset | Species | Longitudinal? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **Tabula Muris Senis** | Mouse (C57BL/6JN) | No: age snapshots. FACS: 3/18/21/24 m (droplet adds 1 & 30 m); bulk GSE132040: 1–27 m, 10 ages | scRNA-seq, 23 tissues (FACS, ~111k cells) (droplet arm covers a smaller tissue set, not fetched); plus bulk RNA-seq of 17 organs (947 samples) | None (age only) | Open: figshare (MIT / CC BY 4.0), GEO |

## Usage

```bash
python scripts/fetch_orthologs.py      # MGI homology -> data/raw/orthologs/
python scripts/fetch_tms.py            # bulk + FACS h5ad -> pseudobulk, deletes the 2.4 GB h5ad
python scripts/fetch_tms.py --keep-h5ad   # keep the h5ad (to rebuild, delete facs_pseudobulk_counts.parquet first)
```
```python
from mwm.data import tms
X, meta = tms.load_pseudobulk()      # (1023, 14556) log1p CPM, human genes; meta (1023, 13)
Xb, mb = tms.load_bulk()             # (925, 16221);  mb (925, 12)
Xm, _ = tms.load_bulk(human=False)   # mouse-gene space
```
The h5ad reader uses only `h5py` (the legacy anndata 0.6 layout of the official object), so `anndata` isn't needed.

## Caveats
- Pseudobulk groups are not independent: each mouse contributes many cell types. Split by **mouse**, not by group.
- The 18/21/24 m FACS cohorts have only 4/3/4 mice. Age effects are confounded with mouse and sort batch.
- Cell-type labels follow the 2020 official annotation (`cell_ontology_class`), and the labels differ between tissues.
