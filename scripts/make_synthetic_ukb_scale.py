"""Synthetic cohorts at UK Biobank scale: same ground-truth dynamics and planted hallmarks as
data/synthetic/{on,off}, but 500k people with a UKB-like visit structure.

UKB (approximate, from public descriptions; verify against the UKB showcase): ~500k recruited at
40-69; most have only the baseline assessment; ~20k attended the 2012-13 repeat, ~60-100k an imaging
visit from 2014, a few thousand a second imaging visit; ~54k have Olink proteomics, mostly at
baseline; ~15 years of linked death follow-up.

    conda run -n mwm python scripts/make_synthetic_ukb_scale.py --out data/synthetic_ukb
"""
import argparse
import time
from pathlib import Path

from mwm.data.synthetic import SynthConfig, hallmarks_off, simulate_cohort

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic_ukb")
ap.add_argument("--n-persons", type=int, default=500_000)
ap.add_argument("--cohorts", nargs="*", default=["on", "off"])
args = ap.parse_args()

base = SynthConfig(
    n_persons=args.n_persons,
    seed=0,
    baseline_age=(40.0, 70.0),
    followup_probs=(0.76, 0.19, 0.045, 0.005),   # ~24% scheduled for >= 1 repeat visit
    gap_range=(3.0, 12.0),                        # repeat (~4 y) and imaging (~5-14 y) visits
    admin_followup=(13.0, 17.0),                  # linked follow-up to ~2022-23
    omics_fraction=0.11,                          # ~54k of 500k with proteomics
    omics_repeat=0.10,                            # few repeat proteomics
)
for c in args.cohorts:
    cfg = base if c == "on" else hallmarks_off(base)
    t0 = time.time()
    simulate_cohort(cfg, Path(args.out) / c)
    print(f"{c}: {time.time() - t0:.0f}s", flush=True)
