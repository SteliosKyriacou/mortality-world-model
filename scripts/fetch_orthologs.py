#!/usr/bin/env python3
"""Download the MGI vertebrate homology report and build rodent->human 1:1 ortholog tables.

Source: MGI (Mouse Genome Informatics, The Jackson Laboratory) homology report
        https://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt
        (Alliance of Genome Resources / HGNC-curated homology classes; mouse, human, rat,
        zebrafish, ...). Free to use with citation (MGI terms of use).

Outputs (data/raw/orthologs/):
  HOM_AllOrganism.rpt                 raw report (~22 MB)
  mouse_human_1to1.tsv                mouse_entrez, mouse_symbol, human_entrez, human_symbol
  rat_human_1to1.tsv                  rat_entrez,   rat_symbol,   human_entrez, human_symbol

"1:1" = the homology class (DB Class Key) contains exactly one gene of the rodent species
and exactly one human gene. Idempotent: skips download if the report exists.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "orthologs"
URL = "https://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt"
TAXA = {"mouse": 10090, "rat": 10116, "human": 9606}


def fetch(dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] {dest.name} present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    head = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
    print(f"[get]  {URL} ({size/1e6:.1f} MB)")
    if size > 200_000_000:
        raise SystemExit("unexpectedly large ortholog report; aborting")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def one_to_one(hom: pd.DataFrame, species: str) -> pd.DataFrame:
    tax = TAXA[species]
    sub = hom[hom["NCBI Taxon ID"].isin([tax, TAXA["human"]])]
    counts = sub.groupby(["DB Class Key", "NCBI Taxon ID"]).size().unstack(fill_value=0)
    ok = counts[(counts.get(tax, 0) == 1) & (counts.get(TAXA["human"], 0) == 1)].index
    sub = sub[sub["DB Class Key"].isin(ok)]
    rod = sub[sub["NCBI Taxon ID"] == tax][["DB Class Key", "EntrezGene ID", "Symbol"]]
    hum = sub[sub["NCBI Taxon ID"] == TAXA["human"]][["DB Class Key", "EntrezGene ID", "Symbol"]]
    out = rod.merge(hum, on="DB Class Key", suffixes=("_r", "_h"))
    out = out.rename(columns={"EntrezGene ID_r": f"{species}_entrez", "Symbol_r": f"{species}_symbol",
                              "EntrezGene ID_h": "human_entrez", "Symbol_h": "human_symbol"})
    out = out.dropna(subset=[f"{species}_entrez", "human_entrez"])
    # MGI classes are built per rodent gene, so one human gene can sit in several classes
    # (e.g. ABCB1 <- Abcb1a, Abcb1b). Strict 1:1: each gene appears exactly once on both sides.
    for c in (f"{species}_entrez", "human_entrez", "human_symbol"):
        out = out[~out[c].duplicated(keep=False)]
    for c in (f"{species}_entrez", "human_entrez"):
        out[c] = out[c].astype("int64")
    return out.drop(columns="DB Class Key").sort_values(f"{species}_entrez").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rpt = OUT / "HOM_AllOrganism.rpt"
    fetch(rpt)
    hom = pd.read_csv(rpt, sep="\t", dtype={"EntrezGene ID": "Int64"}, low_memory=False)
    for sp in ("mouse", "rat"):
        tab = one_to_one(hom, sp)
        tab.to_csv(OUT / f"{sp}_human_1to1.tsv", sep="\t", index=False)
        print(f"[ok]   {sp}_human_1to1.tsv: {len(tab):,} one-to-one pairs")


if __name__ == "__main__":
    main()
