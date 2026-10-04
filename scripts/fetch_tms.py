#!/usr/bin/env python3
"""Download compact Tabula Muris Senis (TMS) data and build per-mouse pseudobulk.

Sources
-------
1. TMS bulk RNA-seq atlas (Schaum et al., Nature 2020, "Ageing hallmarks exhibit
   organ-specific temporal signatures"), GEO GSE132040: raw gene counts for ~950 samples,
   17 organs, ages 1-27 months, both sexes (38 MB csv.gz + 17 kB metadata).
2. TMS single-cell FACS / Smart-seq2 (Tabula Muris Consortium, Nature 2020, "A single-cell
   transcriptomic atlas characterizes ageing tissues in the mouse"), figshare article
   8273102 file `tabula-muris-senis-facs-official-raw-obj.h5ad` (2.37 GB, raw counts,
   ~110k cells, 23 tissues, 1/3/18/21/24 months). Licence: MIT (figshare record).
   The droplet raw object (4.06 GB) exceeds our 3 GB per-file rule and is NOT fetched.

After download the FACS h5ad is reduced to pseudobulk (sum of raw counts per
mouse x tissue x cell_ontology_class, groups with >= 20 cells) by
`mwm.data.tms.build_facs_pseudobulk`, written to data/interim/tms/, and the h5ad is then
DELETED to save disk unless --keep-h5ad is given. Re-running is idempotent: if the
pseudobulk parquet already exists, the h5ad is not re-downloaded.

Usage:
    python scripts/fetch_tms.py               # bulk + FACS -> pseudobulk, delete h5ad
    python scripts/fetch_tms.py --keep-h5ad   # keep the 2.4 GB h5ad in data/raw/tms
    python scripts/fetch_tms.py --bulk-only
    python scripts/fetch_tms.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw" / "tms"
INTERIM = ROOT / "data" / "interim" / "tms"
BUDGET_BYTES = 7_000_000_000  # shared cap (7 GB) for data/raw/{taco,tms,orthologs}
MAX_SINGLE_FILE = 3_000_000_000
UA = {"User-Agent": "mortality-world-model/0.1 (research data fetch)"}

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132040/suppl/"
BULK_FILES = {
    "GSE132040_counts.csv.gz":
        GEO + "GSE132040_190214_A00111_0269_AHH3J3DSXX_190214_A00111_0270_BHHMFWDSXX.csv.gz",
    "GSE132040_MACA_Bulk_metadata.csv.gz": GEO + "GSE132040_MACA_Bulk_metadata.csv.gz",
}
FACS_NAME = "tabula-muris-senis-facs-official-raw-obj.h5ad"
FACS_URL = "https://ndownloader.figshare.com/files/23939711"  # figshare article 8273102 v3


def dir_size(*dirs: Path) -> int:
    return sum(f.stat().st_size for d in dirs if d.exists() for f in d.rglob("*") if f.is_file())


def remote_size(url: str) -> int | None:
    """Content-Length after redirects (figshare redirects to S3; HEAD on S3 works)."""
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={**UA, "Range": "bytes=0-0"}
                                         if method == "GET" else UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    return int(cr.split("/")[-1])
                n = r.headers.get("Content-Length")
                if n and int(n) > 1:
                    return int(n)
        except Exception:
            continue
    return None


def download(url: str, dest: Path, dry: bool) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} present ({dest.stat().st_size/1e6:.1f} MB)")
        return
    size = remote_size(url)
    raw = ROOT / "data" / "raw"
    used = dir_size(raw / "taco", raw / "tms", raw / "orthologs")
    print(f"[plan] {dest.name}: {(size or 0)/1e6:.1f} MB (budget used {used/1e9:.2f}/7.00 GB)")
    if size is None:
        sys.exit(f"[abort] could not determine size of {url}")
    if size > MAX_SINGLE_FILE:
        sys.exit(f"[abort] {dest.name} is {size/1e9:.2f} GB > 3 GB rule")
    if used + size > BUDGET_BYTES:
        sys.exit(f"[abort] {dest.name} would exceed the 7 GB budget")
    if dry:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1200) as r, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    if tmp.stat().st_size != size:
        tmp.unlink()
        sys.exit(f"[abort] size mismatch for {dest.name}")
    tmp.rename(dest)
    print(f"[ok]   {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-h5ad", action="store_true", help="do not delete the FACS h5ad after pseudobulk")
    ap.add_argument("--bulk-only", action="store_true")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    for name, url in BULK_FILES.items():
        download(url, RAW / name, args.dry_run)
    if args.bulk_only:
        return

    from mwm.data import tms  # local import: needs h5py

    pb = INTERIM / tms.FACS_PSEUDOBULK_COUNTS
    h5 = RAW / FACS_NAME
    if pb.exists() and not h5.exists():
        print(f"[skip] {pb.relative_to(ROOT)} exists; FACS h5ad not re-downloaded")
        return
    download(FACS_URL, h5, args.dry_run)
    if args.dry_run:
        return
    if not pb.exists():
        tms.build_facs_pseudobulk(h5, INTERIM, min_cells=args.min_cells)
    if not args.keep_h5ad:
        h5.unlink()
        (RAW / "FACS_H5AD_DELETED.txt").write_text(
            f"{FACS_NAME} ({FACS_URL}) was downloaded, reduced to pseudobulk in "
            f"data/interim/tms/ and deleted to save disk. Re-run scripts/fetch_tms.py "
            f"--keep-h5ad after removing data/interim/tms/{tms.FACS_PSEUDOBULK_COUNTS} to rebuild.\n")
        print(f"[del]  {FACS_NAME} removed (pseudobulk kept)")


if __name__ == "__main__":
    main()
