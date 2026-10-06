"""Momentum worlds with ANNUAL visits (like linked GP lab records): 50k people seen every ~year,
5-10 follow-ups each, same equation, hallmarks and momentum / matched-null settings as
data/synthetic_momentum.

    conda run -n mwm python scripts/make_synthetic_annual.py --out data/synthetic_momentum_annual
"""
import argparse
from pathlib import Path

import numpy as np

from mwm.data.synthetic import SynthConfig, simulate_cohort

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic_momentum_annual")
ap.add_argument("--n-persons", type=int, default=50_000)
args = ap.parse_args()
fu = np.zeros(11)
fu[5:] = 1 / 6                                   # 5..10 follow-ups, equally likely
base = dict(n_persons=args.n_persons, seed=0, followup_probs=tuple(float(x) for x in fu), gap_range=(0.9, 1.1),
            miss_growth=0.03, omics_fraction=0.10, omics_repeat=0.3)
for name, kw in (("momentum", {"momentum": True}), ("null", {"momentum_null": True})):
    simulate_cohort(SynthConfig(**base, **kw), Path(args.out) / name)
