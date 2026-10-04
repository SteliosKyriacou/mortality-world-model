#!/usr/bin/env python3
"""Download GTEx v10 open-access gene TPM matrix + public sample/subject attributes (~2.3 GB).

Open access (GTEx Portal, no login): https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression
Protected phenotypes (exact age, cause of death, medical history) need dbGaP phs000424 - not used.
Then run: python -m mwm.data.gtex  (builds data/interim/gtex/).
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/adult-gtex"
FILES = {
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz": f"{BASE}/bulk-gex/v10/rna-seq/",
    "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt": f"{BASE}/annotations/v10/metadata-files/",
    "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt": f"{BASE}/annotations/v10/metadata-files/",
    "GTEx_Analysis_v10_Annotations_SampleAttributesDD.xlsx": f"{BASE}/annotations/v10/metadata-files/",
    "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDD.xlsx": f"{BASE}/annotations/v10/metadata-files/",
}


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data/raw/gtex"
    out.mkdir(parents=True, exist_ok=True)
    for name, base in FILES.items():
        dest = out / name
        url = base + name
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                size = int(r.headers["Content-Length"])
        except Exception as e:  # noqa: BLE001
            print("skip", name, e)
            continue
        if dest.exists() and dest.stat().st_size == size:
            print("cached", name)
            continue
        print(f"downloading {name} ({size / 1e6:.0f} MB)")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 22)
        tmp.rename(dest)


if __name__ == "__main__":
    main()
