#!/usr/bin/env python3
"""Download pathway gene sets (human gene symbols) into data/raw/genesets/.

- MSigDB Hallmark (h.all), MSigDB C2:CP:Reactome, MSigDB C5:GO:BP  (release 2026.1.Hs)
- Reactome native GMT (current release, ReactomePathways.gmt.zip)

Licences (see data_cards/genesets.md): MSigDB Hallmark/Reactome/GO collections are distributed
under CC BY 4.0 (MSigDB licence terms; KEGG/BioCarta-derived sets are the restricted ones and are
NOT downloaded here). Reactome data is CC BY 4.0 (formerly CC0 for parts). GO is CC BY 4.0.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

MSIGDB = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{rel}/{name}"
REL = "2026.1.Hs"
FILES = [f"h.all.v{REL}.symbols.gmt", f"c2.cp.reactome.v{REL}.symbols.gmt", f"c5.go.bp.v{REL}.symbols.gmt"]
REACTOME = "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print("cached", dest.name)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mwm data fetch)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())
    print("ok", dest.name, dest.stat().st_size)


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    sets = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data/raw/genesets"
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        get(MSIGDB.format(rel=REL, name=f), out / f)
    z = out / "ReactomePathways.gmt.zip"
    get(REACTOME, z)
    if not (out / "ReactomePathways.gmt").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    for f in [*FILES, "ReactomePathways.gmt"]:
        print(f, len(read_gmt(out / f)), "sets")


if __name__ == "__main__":
    main()
