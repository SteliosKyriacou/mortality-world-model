"""Train one named Model A variant (all cohorts x seeds) from its config into models/<name>/.

    conda run -n mwm python scripts/train_model_a.py --config configs/model_a_transformer.yaml
    conda run -n mwm python scripts/train_model_a.py --config configs/model_a_hetgnn.yaml --cohorts on --seeds 0
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--cohorts", nargs="*", default=None)
ap.add_argument("--seeds", nargs="*", type=int, default=None)
ap.add_argument("--no-hallmarks", action="store_true")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
e, a = cfg["encoder"], cfg["model_a"]
for cohort in args.cohorts or cfg["cohorts"]:
    for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
        cmd = [sys.executable, "-u", "scripts/train_model_a_long.py", "--encoder", e["kind"], "--cohort", cohort,
               "--seed", str(seed), "--d", str(e["d_latent"]), "--enc-epochs", str(e["epochs"]),
               "--a-epochs", str(a["epochs"]), "--eval-every", str(a["eval_every"]), "--lr", str(a["lr"]),
               "--jac-penalty", str(a["jac_penalty"]), "--select", a["select"], "--out", cfg["out"],
               "--name", f"seed{seed}"]
        if args.no_hallmarks:
            cmd.append("--no-hallmarks")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        # train_model_a_long writes <out>/<cohort>/seed<k>/
        print(f"done {cfg['name']} {cohort} seed {seed} -> {Path(cfg['out']) / cohort / f'seed{seed}'}", flush=True)
