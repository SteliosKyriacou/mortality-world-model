#!/usr/bin/env python3
"""Fetch the Framingham Heart Study *teaching* dataset (frmgham2; 4,434 people x 3 exams).

Source: CRAN package riskCommunicator 1.0.1 (GPL-3), which ships the NHLBI BioLINCC teaching
dataset "with permission from the National Heart, Lung, and Blood Institute". No login needed.
The official route (BioLINCC teaching-dataset request) needs an NIH-verified institutional login.

RESTRICTION: the teaching dataset is anonymised/perturbed and "inappropriate for publication
purposes" (BioLINCC). Pipeline development and smoke tests only. See data_cards/framingham.md.

Converting the .rda to CSV needs `pyreadr` (pip install pyreadr) or R (Rscript).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://cran.r-project.org/src/contrib/riskCommunicator_1.0.1.tar.gz"
SHA256_TGZ = "bf1a7c6d89c297845e78d569275499880b73e8ef6d16f3d7e340de569cc22e1f"
OUT = Path(__file__).resolve().parents[1] / "data/raw/framingham"
RDA = OUT / "framingham_riskCommunicator_1.0.1.rda"
CSV = OUT / "frmgham2.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not RDA.exists():
        with tempfile.TemporaryDirectory() as td:
            tgz = Path(td) / "pkg.tar.gz"
            urllib.request.urlretrieve(URL, tgz)
            h = hashlib.sha256(tgz.read_bytes()).hexdigest()
            if h != SHA256_TGZ:
                print(f"WARNING: sha256 mismatch ({h}); CRAN tarball may have changed")
            with tarfile.open(tgz) as tf:
                member = tf.getmember("riskCommunicator/data/framingham.rda")
                with tf.extractfile(member) as src, open(RDA, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        print("saved", RDA)
    if CSV.exists():
        print("cached", CSV)
        return
    try:
        import pyreadr  # type: ignore

        df = pyreadr.read_r(str(RDA))["framingham"]
        for c in df.columns:  # write integer-valued float columns as integers
            s = df[c].dropna()
            if len(s) and (s == s.round()).all():
                df[c] = df[c].astype("Int64")
        df.to_csv(CSV, index=False)
    except ImportError:
        if shutil.which("Rscript") is None:
            raise SystemExit("Need `pip install pyreadr` or R to convert the .rda to CSV")
        subprocess.run(["Rscript", "-e", f"load('{RDA}'); write.csv(framingham, '{CSV}', "
                        "row.names=FALSE, na='')"], check=True)
    print("saved", CSV)


if __name__ == "__main__":
    main()
