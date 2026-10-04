#!/usr/bin/env python3
"""Download the TACO rodent gene-expression meta-dataset (Tyshkovskiy et al., Nature 2026).

Paper : "Universal transcriptomic hallmarks of mammalian ageing and mortality",
        Nature 654:173-188, doi:10.1038/s41586-026-10542-3 (published 27 May 2026).
Data  : Zenodo record 10.5281/zenodo.18763485 ("Transcriptomic clock models and rodent
        gene expression meta-dataset"). Licence: MGB Open Access License 1.0
        (non-commercial academic use only).

The Zenodo record is ~64 GB in total, but almost all of that is the Bayesian-ridge clock
models (*.pkl, ~2 GB each), which we do NOT need. We fetch only:

  * Data_annotation_absolute_rodents.xlsx   (0.4 MB)  per-sample metadata, 4,539 samples
  * Data_annotation_relative_rodents.xlsx   (0.5 MB)  + differences vs matched controls
  * Expression_data_absolute_rodents_Scaled.csv (1.12 GB) genes (mouse Entrez) x samples
  * Expression_data_relative_rodents_Scaled.csv (0.72 GB) same, relative to study controls
    (only with --relative; on by default)
  * Nature Supplementary Table 1 (MOESM3, 63 kB): dataset/source list per group.

Everything lands in data/raw/taco/. The script is idempotent: a file whose size (and md5,
where Zenodo provides one) already matches is skipped. A hard byte budget is enforced
before each download using the server-reported size.

Usage:
    python scripts/fetch_taco.py                 # annotation + absolute + relative
    python scripts/fetch_taco.py --no-relative   # skip the relative matrix (saves 0.72 GB)
    python scripts/fetch_taco.py --dry-run       # print plan + sizes only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "taco"
ZENODO_RECORD = "18763485"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
SUPP_TABLE_1 = ("https://media.springernature.com/original/springer-static/esm/"
                "art%3A10.1038%2Fs41586-026-10542-3/MediaObjects/41586_2026_10542_MOESM3_ESM.xlsx")
BUDGET_BYTES = 7_000_000_000  # shared cap (7 GB) for data/raw/{taco,tms,orthologs}

CORE_FILES = [
    "Data_annotation_absolute_rodents.xlsx",
    "Data_annotation_relative_rodents.xlsx",
    "Expression_data_absolute_rodents_Scaled.csv",
]
RELATIVE_FILES = ["Expression_data_relative_rodents_Scaled.csv"]
UA = {"User-Agent": "mortality-world-model/0.1 (research data fetch)"}


def dir_size(*dirs: Path) -> int:
    return sum(f.stat().st_size for d in dirs if d.exists() for f in d.rglob("*") if f.is_file())


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None, md5: str | None, dry: bool) -> None:
    raw_root = ROOT / "data" / "raw"
    if dest.exists() and (size is None or dest.stat().st_size == size):
        if md5 and md5sum(dest) != md5:
            print(f"[warn] md5 mismatch for existing {dest.name}; re-downloading")
        else:
            print(f"[skip] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
            return
    used = dir_size(raw_root / "taco", raw_root / "tms", raw_root / "orthologs")
    need = size or 0
    print(f"[plan] {dest.name}: {need/1e6:.1f} MB (budget used {used/1e9:.2f}/{BUDGET_BYTES/1e9:.2f} GB)")
    if used + need > BUDGET_BYTES:
        sys.exit(f"[abort] downloading {dest.name} would exceed the 7 GB budget")
    if dry:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    if size is not None and tmp.stat().st_size != size:
        tmp.unlink()
        sys.exit(f"[abort] size mismatch for {dest.name}")
    if md5 and md5sum(tmp) != md5:
        tmp.unlink()
        sys.exit(f"[abort] md5 mismatch for {dest.name}")
    tmp.rename(dest)
    print(f"[ok]   {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-relative", action="store_true", help="skip the relative-to-control matrix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(urllib.request.Request(ZENODO_API, headers=UA), timeout=60) as r:
        rec = json.load(r)
    (OUT / "zenodo_record.json").write_text(json.dumps(rec, indent=1))
    files = {f["key"]: f for f in rec["files"]}
    wanted = CORE_FILES + ([] if args.no_relative else RELATIVE_FILES)
    for name in wanted:
        f = files[name]
        md5 = f.get("checksum", "").removeprefix("md5:") or None
        download(f["links"]["self"], OUT / name, int(f["size"]), md5, args.dry_run)

    download(SUPP_TABLE_1, OUT / "Tyshkovskiy2026_SuppTable1_datasets.xlsx",
             remote_size(SUPP_TABLE_1), None, args.dry_run)
    print(f"[done] data/raw/taco = {dir_size(OUT)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
