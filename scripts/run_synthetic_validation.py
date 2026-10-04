"""Synthetic validation (validation ladder step 1): generate hallmarks-on / hallmarks-off
cohorts, run the full pipeline for each seed, save runs/synthetic/<cohort>/seed<k>/results.json.

    conda run -n mwm python scripts/run_synthetic_validation.py --config configs/synthetic.yaml
"""
import argparse
from pathlib import Path

import yaml

from mwm.data.synthetic import SynthConfig, hallmarks_off, load_truth, simulate_cohort
from mwm.pipeline import Logger, run_cohort, run_truth_hallmarks

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/synthetic.yaml")
ap.add_argument("--cohorts", default="on,off")
ap.add_argument("--seeds", default=None, help="comma list overriding config")
ap.add_argument("--out", default="runs/synthetic")
ap.add_argument("--quick", action="store_true", help="tiny epochs for smoke testing")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
if args.quick:
    cfg["encoder"]["epochs"] = 3
    cfg["model_a"]["epochs"] = 2
    cfg["model_b"]["epochs"] = 3
    cfg["model_bprime"].update(dsm_iters=50, diff_epochs=1)
    cfg["snapshot"]["iters"] = 20
seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
root = Path(cfg["data"]["root"])
log = Logger(Path(args.out) / "log.txt")
for cohort in args.cohorts.split(","):
    cdir = root / cohort
    if not (cdir / "long.parquet").exists():
        c = SynthConfig(n_persons=cfg["data"]["n_persons"], seed=cfg["data"]["seed"])
        simulate_cohort(hallmarks_off(c) if cohort == "off" else c, cdir)
    tcfg, meta, tl, _ = load_truth(cdir)
    # sanity check of the tests themselves: run them on the TRUE dynamics in the TRUE latent
    run_truth_hallmarks(cdir, Path(args.out) / cohort / "truth_hallmarks.json", cfg["device"], log)
    for seed in seeds:
        run_cohort(cdir, Path(args.out) / cohort / f"seed{seed}", cfg, seed, log,
                   truth=(tcfg, meta, tl), controls=(seed == cfg["seeds"][0]),
                   oracle=cfg.get("oracle", False) and seed == cfg["seeds"][0])
