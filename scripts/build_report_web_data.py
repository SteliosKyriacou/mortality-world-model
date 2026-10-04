"""Assemble reports/model_a/figure_data_web.json for the explainer page.

Merges figure_data.json with the intervention-vs-reality analysis and, when present, the
long-run convergence history; NaN/inf are written as null (browsers reject them).

    python scripts/build_report_web_data.py --reality e80
    python scripts/build_report_web_data.py --reality long --convergence runs/long/on/seed0/convergence.json
"""
import argparse
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="reports/model_a")
ap.add_argument("--reality", default="e80", help="tag of intervention_reality_<tag>.json")
ap.add_argument("--reality-compare", default=None, help="optional second tag shown for comparison")
ap.add_argument("--convergence", default=None)
ap.add_argument("--baseline-history", default="runs/synthetic/log.txt",
                help="log of the original 80-epoch run (Model A val curve is parsed from it)")
args = ap.parse_args()
d = Path(args.dir)


def clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 5)
    if isinstance(o, list):
        return [clean(x) for x in o]
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    return o


data = json.loads((d / "figure_data.json").read_text())
data.pop("landscape_true_infl", None)
data["reality"] = json.loads((d / f"intervention_reality_{args.reality}.json").read_text())
data["reality"]["tag"] = args.reality
if args.reality_compare:
    data["reality_compare"] = json.loads((d / f"intervention_reality_{args.reality_compare}.json").read_text())
    data["reality_compare"]["tag"] = args.reality_compare
if args.convergence and Path(args.convergence).exists():
    c = json.loads(Path(args.convergence).read_text())
    h = c["a_history"]
    step = max(1, len(h) // 2000)          # thin the 5-epoch history to <= ~2000 points
    c["a_history"] = h[::step] + ([h[-1]] if h and h[-1] is not h[::step][-1] else [])
    data["convergence"] = c
s = json.dumps(clean(data), separators=(",", ":"), allow_nan=False)
(d / "figure_data_web.json").write_text(s)
print("wrote", d / "figure_data_web.json", len(s) // 1000, "kB")
