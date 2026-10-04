"""Collect training-convergence histories of all runs into one file for the convergence page.

    python scripts/build_convergence_data.py
"""
import json
import math
from pathlib import Path

OUT = Path("reports/convergence/convergence_data.json")
RUNS = {
    "a80_hetgnn": ("Model A, 80 epochs, HetGNN encoder", "runs/encoder_cmp/on/hetgnn_seed0/convergence.json"),
    "a80_transformer": ("Model A, 80 epochs, transformer encoder", "runs/encoder_cmp/on/transformer_seed0/convergence.json"),
    "a_long": ("Model A long run (lr 5e-4, stiffness penalty)", "runs/long/on/seed0/convergence.json"),
    "a_long_hetgnn": ("Model A long run, HetGNN encoder (lr 5e-4, stiffness penalty)", "runs/long/on/hetgnn_seed0/convergence.json"),
    "a_long_diverged": ("Model A first long run (lr 2e-3, diverged)", "runs/long/on/seed0_diverged_2026-10-04/convergence.json"),
    "b_long": ("Model B long run", "runs/long_b/on/seed0/convergence.json"),
}


def clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 5)
    if isinstance(o, list):
        return [clean(x) for x in o]
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    return o


data = {}
for key, (label, path) in RUNS.items():
    p = Path(path)
    if not p.exists():
        continue
    c = json.loads(p.read_text())
    h = c.get("a_history", [])
    step = max(1, len(h) // 1500)
    entry = {"label": label, "history": h[::step], "checkpoints": c.get("checkpoints", []),
             "encoder_history": c.get("encoder_history", [])}
    # final results (best epoch) when the run finished
    res = p.parent / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        entry["best_epoch"] = r.get("best_epoch")
        entry["final"] = {k: r["forecast"][m]["mae"] for k, m in (("model_mae", next(iter(r["forecast"]))),
                                                                  ("gbm_mae", "base_direct_gbm"))}
    data[key] = entry
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(clean(data), separators=(",", ":"), allow_nan=False))
print("wrote", OUT, {k: (len(v["history"]), len(v["checkpoints"])) for k, v in data.items()})
