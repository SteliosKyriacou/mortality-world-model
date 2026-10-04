# Data card: pathway gene sets (MSigDB Hallmark / Reactome / GO:BP, Reactome native)

| Field | Value |
|---|---|
| Purpose | Pool human-ortholog gene expression (TACO, TMS, GTEx) into ~50-300 pathway scores (PLAN.md §3.3), GSEA of Jacobian eigenvectors (§7) |
| Fetch | `python scripts/fetch_genesets.py` (idempotent) |
| Location | `data/raw/genesets/` (6.2 MB total) |
| Access | Direct download, no login |
| Identifiers | Human HGNC gene symbols (`*.symbols.gmt`), GMT format: `name<TAB>url/desc<TAB>gene1<TAB>gene2...` |
| Reader | `scripts/fetch_genesets.py::read_gmt(path) -> dict[name, list[symbol]]` |

## Files (downloaded 2026-10-03)

| File | Source | Sets | Licence |
|---|---|---|---|
| `h.all.v2026.1.Hs.symbols.gmt` | MSigDB 2026.1.Hs, Hallmark collection | 50 | CC BY 4.0 |
| `c2.cp.reactome.v2026.1.Hs.symbols.gmt` | MSigDB 2026.1.Hs, C2:CP:REACTOME | 1,839 | CC BY 4.0 (MSigDB) / Reactome CC BY 4.0 |
| `c5.go.bp.v2026.1.Hs.symbols.gmt` | MSigDB 2026.1.Hs, C5:GO:BP | 7,538 | CC BY 4.0 (GO is CC BY 4.0) |
| `ReactomePathways.gmt` (+ `.zip`) | reactome.org/download/current | 2,868 | CC BY 4.0 |

URLs: `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/<file>`,
`https://reactome.org/download/current/ReactomePathways.gmt.zip`.

## Licence notes
- MSigDB v2022.1 and later are under **CC BY 4.0** (attribution: cite Liberzon et al. 2015 Cell Syst
  for Hallmarks, Subramanian et al. 2005 PNAS for MSigDB, plus the source database), per
  https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp. The only restricted subsets are
  KEGG-derived ones (KEGG_LEGACY: Kanehisa licence; KEGG_MEDICUS: CC BY-SA 4.0) and those were
  **not** downloaded. Redistribution of derived pathway scores and model weights is fine with attribution.
- Reactome: CC BY 4.0 (https://reactome.org/license).

## Suggested v1 usage
- Hallmark 50 sets as the primary, interpretable pathway space (covers mTORC1, OXPHOS, UPR,
  inflammatory response, IL6/JAK/STAT3, TNFa/NFkB, DNA repair, p53, G2M, E2F ... i.e. hallmark-of-aging
  relevant). Reactome (filtered to 15-500 genes, ~1,000 sets) as the ~300-dim extended space after
  redundancy pruning. GO:BP for post-hoc enrichment only.
- Rodent genes: map with `data/raw/orthologs/mouse_human_1to1.tsv` / `rat_human_1to1.tsv` first.
