#!/usr/bin/env python3
"""Download continuous NHANES (1999-2018) component files + public-use Linked Mortality Files.

Public domain (US federal government work), no registration needed.

Usage:
    python scripts/fetch_nhanes.py [--out data/raw/nhanes]

Idempotent: files already present (and valid SAS XPORT) are skipped.
Files that do not exist for a given cycle (HTTP 404) are recorded in
`data/raw/nhanes/manifest.csv` with status "missing".
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
LMF_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
WAYBACK = "https://web.archive.org/web/2024id_/"
LMF_FILE = "NHANES_{a}_{b}_MORT_2019_PUBLIC.dat"

CYCLES = {  # start year -> file suffix
    1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E",
    2009: "_F", 2011: "_G", 2013: "_H", 2015: "_I", 2017: "_J",
}
EARLY = {1999: 0, 2001: 1, 2003: 2}  # index into early-name tuples below


def component_files(year: int) -> dict[str, str]:
    """Return {component: file stem} for one cycle (only components that exist in that cycle)."""
    s = CYCLES[year]
    f: dict[str, str] = {}
    for comp in ("DEMO", "BMX", "BPX", "SMQ", "DIQ", "BPQ", "MCQ"):
        f[comp] = comp + s
    early = {
        "TCHOL": ("LAB13", "L13_B", "L13_C"),      # total chol (+HDL 1999-2004)
        "TRIGLY": ("LAB13AM", "L13AM_B", "L13AM_C"),  # TG + LDL (fasting subsample)
        "GHB": ("LAB10", "L10_B", "L10_C"),          # HbA1c
        "GLU": ("LAB10AM", "L10AM_B", "L10AM_C"),    # fasting glucose
        "CRP": ("LAB11", "L11_B", "L11_C"),          # CRP (+fibrinogen 1999-2002)
        "BIOPRO": ("LAB18", "L40_B", "L40_C"),       # biochemistry (creatinine, albumin, ...)
        "CBC": ("LAB25", "L25_B", "L25_C"),          # complete blood count
    }
    for comp, names in early.items():
        if year in EARLY:
            f[comp] = names[EARLY[year]]
        else:
            if comp == "CRP":
                if year <= 2009:
                    f[comp] = "CRP" + s
                elif year >= 2015:
                    f[comp] = "HSCRP" + s
                # 2011-2014: no CRP measured
            else:
                f[comp] = comp + s
    if year >= 2005:
        f["HDL"] = "HDL" + s
    if year in (2011, 2013):
        f["MGX"] = "MGX" + s  # handgrip strength
    return f


def is_xpt(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            return fh.read(80).startswith(b"HEADER RECORD")
    except OSError:
        return False


def download(url: str, dest: Path, retries: int = 3) -> str:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            tmp.rename(dest)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 * (attempt + 1))
    print(f"FAILED {url}: {err}", file=sys.stderr)
    return "failed"


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length"))
    except Exception:  # noqa: BLE001
        return None


def download_ranged(url: str, dest: Path, chunk: int = 32 * 1024, workers: int = 16) -> str:
    """ftp.cdc.gov serves ~3 KB/s per connection; fetch byte ranges in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    size = remote_size(url)
    if size is None:
        # ftp.cdc.gov sometimes stops answering (rate limiting). Fall back to the Internet
        # Archive's verbatim copy (sizes verified equal to the CDC directory listing, 2026-10-03).
        return download(WAYBACK + url, dest)
    if dest.exists() and dest.stat().st_size == size:
        return "cached"

    def part(start: int) -> bytes:
        end = min(start + chunk, size) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - start + 1:
                    return data
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"range {start}-{end} failed for {url}")

    with ThreadPoolExecutor(workers) as ex:
        blobs = list(ex.map(part, range(0, size, chunk)))
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(b"".join(blobs))
    tmp.rename(dest)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data/raw/nhanes"))
    ap.add_argument("--no-mortality", action="store_true",
                    help="skip the Linked Mortality Files (ftp.cdc.gov is slow; they are fetched "
                         "with parallel byte-range requests)")
    args = ap.parse_args()
    out = Path(args.out)
    rows = []
    for year in CYCLES:
        cdir = out / f"{year}-{year + 1}"
        cdir.mkdir(parents=True, exist_ok=True)
        for comp, stem in component_files(year).items():
            dest = cdir / f"{stem}.xpt"
            url = BASE.format(year=year, name=stem)
            if dest.exists() and is_xpt(dest):
                status = "cached"
            else:
                status = download(url, dest)
                if status == "ok" and not is_xpt(dest):
                    dest.unlink()
                    status = "not_xpt"
            print(f"{year} {comp:7s} {stem:10s} {status}")
            rows.append(dict(cycle=year, component=comp, file=stem, url=url, status=status))
        # mortality linkage
        if args.no_mortality:
            continue
        mdir = out / "mortality"
        mdir.mkdir(parents=True, exist_ok=True)
        name = LMF_FILE.format(a=year, b=year + 1)
        dest = mdir / name
        status = download_ranged(LMF_BASE + name, dest)
        print(f"{year} LMF     {name} {status}")
        rows.append(dict(cycle=year, component="LMF", file=name, url=LMF_BASE + name, status=status))
    rprog = out / "mortality" / "R_ReadInProgramAllSurveys.R"
    if not rprog.exists():
        download(LMF_BASE + rprog.name, rprog)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
