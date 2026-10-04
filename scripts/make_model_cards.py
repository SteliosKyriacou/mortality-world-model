"""Write models/<variant>/MODEL_CARD.md for each Model A variant from its own results files.

    python scripts/make_model_cards.py
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ARCH = {
    "model_a_transformer": """**Encoder: masked tabular transformer** (`src/mwm/encoders/tabular.py`, 400,976 parameters)
- One token per clinical marker: `value × val_w[j] + feat_e[j]`; a missing marker becomes a learned mask
  token and is excluded from attention (33 × 64 each for `val_w`, `feat_e`, `mask_e`).
- Gene panel: `[x·m, m]` (1,000 values) → Linear 1000→256 → GELU → Linear 256→256 → 4 tokens of width 64
  (+ learned positions); padded out when no genes were measured. 322,048 parameters.
- CLS token + 2 pre-norm transformer layers (width 64, 4 heads, feed-forward 128, ReLU, no dropout): 66,944.
- Readout from CLS: LayerNorm → Linear 64→64 → GELU → Linear 64→16 (5,328). Latent whitened per dimension.""",
    "model_a_hetgnn": """**Encoder: heterogeneous graph neural network** (`src/mwm/encoders/hetgnn.py`, 157,456 parameters)
- A graph per visit from measured values only. Nodes: `visit` (learned constant start), one `clin` node per
  measured clinical value (`clin_e[j] + value × clin_w[j]`), one `prot` node per measured gene value
  (`prot_e[g] + value × prot_w[g]`), one `path` node per (visit, pathway) with ≥ 1 measured member (`path_e[p]`).
- Edges: clin→visit, prot→path (membership), prot→visit, path→visit, visit→clin.
- 2 rounds of HeteroConv: one GraphSAGE layer per edge type (mean aggregation + root weight, 8,256 parameters
  each), summed over edge types, residual `h ← h + ReLU(new)`. 82,560 parameters.
- Readout from the visit node: LayerNorm → Linear 64→64 → GELU → Linear 64→16 (5,328). Latent whitened.
- Pathway membership on synthetic data = the cohort's planted pathway lists (this partly hands the Jacobian
  pathway test its answer; judge that test against the hallmarks-OFF cohort). Real data: MSigDB/Reactome.""",
}
SHARED = """**Shared heads (trained with the encoder, then frozen)**
- Decoder: Linear 16→256 → GELU → Linear 256→256 → GELU → Linear 256→533, per-marker learned σ (207,658 parameters).
- Hazard head: Linear 16→64 → GELU → Linear 64→1, log death rate from z only (1,154 parameters).

**Model A dynamics** (`src/mwm/dynamics/sde.py`, 46,193 parameters): `dz = f(z,a,u)da + Σ(z,a)dW`,
`f = −∇V(z) + J(z,a,u)`.
- V: MLP 16→128→128→1 (SiLU), 18,817 parameters. J: MLP [z, (a−60)/10, u] 18→128→128→16, 21,008 parameters.
- Σ: diagonal, MLP [z, (a−60)/10] 17→64→64→16, softplus + 1e-4, initial σ = 0.2; 6,352 parameters.
- Observation noise r² fixed from encoder input-perturbation (errors-in-variables start jitter).
- Training: consecutive visit pairs, 32 Euler–Maruyama steps per pair in normalised time, K = 32 futures,
  Gaussian NLL of the next visit + 1e-3‖J‖² + 1e-3‖Σ‖²; AdamW lr 2e-3, wd 1e-5, cosine, batch 512, clip 5."""


def ms(v, fmt="{:.3f}"):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "—"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


for variant in ("model_a_transformer", "model_a_hetgnn"):
    root = Path("models") / variant
    cfg = yaml.safe_load(open(f"configs/{variant}.yaml"))
    R = {c: [json.loads((root / c / f"seed{s}" / "results.json").read_text()) for s in cfg["seeds"]
             if (root / c / f"seed{s}" / "results.json").exists()] for c in cfg["cohorts"]}
    on, off = R["on"], R["off"]
    L = [f"# {variant}", "", f"{cfg['description']}. Card generated {date.today().isoformat()} from "
         f"`models/{variant}/*/seed*/results.json` by `scripts/make_model_cards.py`.", "",
         "**Status:** trained and evaluated on the synthetic cohort only (validation ladder step 1). "
         "Nothing here is evidence about human biology.", "",
         "## Load / retrain", "", "```python", "from mwm.models import load_model_a",
         f'm = load_model_a("{variant}", cohort="on", seed=0)   # m.bundle, m.sde, m.latents, m.results', "```",
         "```bash", f"conda run -n mwm python scripts/train_model_a.py --config configs/{variant}.yaml", "```", "",
         "## Architecture", "", ARCH[variant], "", SHARED, "",
         "## Training settings", "",
         f"- Encoder: {cfg['encoder']['epochs']} epochs (masked autoencoder 30% hidden + hazard 0.5 + omics-consistency 0.5).",
         f"- Model A: {cfg['model_a']['epochs']} epochs, lr {cfg['model_a']['lr']}, checkpoint kept by "
         f"{cfg['model_a']['select']} (checked every {cfg['model_a']['eval_every']} epochs).",
         f"- Seeds {cfg['seeds']}, cohorts {cfg['cohorts']}; person-level split 75/10/15 (seed 20261003).",
         "- Convergence: both encoder and Model A converge within these budgets; longer Model A training overfits "
         "(see `reports/convergence`).", "",
         "## Results on held-out people (mean ± sd over seeds)", "",
         "| Metric | hallmarks-ON | hallmarks-OFF |", "|---|---|---|"]
    rows = [
        ("Forecast MAE, first → last visit (SD units)", lambda r: r["forecast"]["A"]["mae"]),
        ("Gradient-boosted trees MAE (same task)", lambda r: r["forecast"]["base_direct_gbm"]["mae"]),
        ("CRPS", lambda r: r["forecast"]["A"]["crps"]),
        ("50% interval coverage", lambda r: r["forecast"]["A"]["cov50"]),
        ("90% interval coverage", lambda r: r["forecast"]["A"]["cov90"]),
        ("Survival C-index (Harrell)", lambda r: r["survival"]["A"]["c_harrell"]),
        ("Cox on baseline markers C-index", lambda r: r["survival"]["cox_baseline_features"]["c_harrell"]),
        ("True hidden state recovered (mean affine R²)", lambda r: float(np.mean(r["ground_truth"]["affine_r2_per_true_dim"]))),
        ("Drift relative MSE vs true equation", lambda r: r["ground_truth"]["A"]["drift_rel_mse"]),
        ("Noise ratio age ≥70 / ≤50 (true ON 3.5, OFF 1.0)", lambda r: r["ground_truth"]["A"].get("diff_trace_ratio_old_young_model")),
        ("Best epoch", lambda r: float(r["best_epoch"])),
    ]
    for name, f in rows:
        def col(rs):
            vals = []
            for r in rs:
                try:
                    vals.append(f(r))
                except (KeyError, TypeError):
                    pass
            return ms(vals, "{:.0f}" if name == "Best epoch" else "{:.3f}")
        L.append(f"| {name} | {col(on)} | {col(off)} |")
    L += ["", "## Hallmark tests (decision rule fired?)", "",
          "| Test | " + " | ".join(f"ON s{s}" for s in cfg["seeds"]) + " | " + " | ".join(f"OFF s{s}" for s in cfg["seeds"]) + " |",
          "|---|" + "---|" * (2 * len(cfg["seeds"]))]
    for t in ("inflammaging", "dispersion", "irreversibility", "intervention", "jacobian_enrichment", "attractors"):
        cells = []
        for rs in (on, off):
            for r in rs:
                h = r.get("hallmarks", {}).get("A", {}).get(t, {})
                if t == "jacobian_enrichment":
                    cells.append(f"{'yes' if h.get('detected_flag') else 'no'} ({h.get('true_pos')}/7, {h.get('false_pos')} FP)")
                else:
                    cells.append("yes" if h.get("detected") else "no")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "Desired pattern: yes on every ON seed, no on every OFF seed.", ""]
    ir = root / "on" / "seed0" / "intervention_reality.json"
    if ir.exists():
        d = json.loads(ir.read_text())
        h, s, pp = d["trial"]["hba1c_rate"], d["trial"]["survival"], d["per_person"]["summary"]
        i = s["grid"].index(15.0)
        L += ["## Intervention predictions vs reality (ON, seed 0)", "",
              "| | Observed | Model | True equation |", "|---|---|---|---|",
              f"| Treated − untreated HbA1c change (mmol/mol/yr) | {h['observed_diff']:.3f} | {h['model_diff']:.3f} | {h['truth_diff']:.3f} |",
              f"| Alive at 15 y, untreated arm | {s['observed_km']['0'][i]:.3f} | {s['model']['0'][i]:.3f} | {s['truth']['0'][i]:.3f} |",
              f"| Per-person HbA1c effect at +10 y (mean) | — | {pp['hba1c']['pred_mean']:.2f} | {pp['hba1c']['true_mean']:.2f} |",
              f"| Per-person years gained over 15 y (mean; corr with truth) | — | {pp['years_alive']['pred_mean']:.3f} "
              f"(r = {pp['years_alive']['corr']:.2f}) | {pp['years_alive']['true_mean']:.3f} |", ""]
    L += ["## Known limitations", "",
          "- Survival is ranked well but miscalibrated (too pessimistic at 15 y); per-person treatment benefit on "
          "survival does not track the truth. The static hazard head is the likely cause.",
          "- Misses the frailty double well (one attractor instead of two).",
          "- Noise growth with age is overstated; the V/J split of the drift is not identifiable.",
          "- Jacobian pathway test is seed-sensitive.", ""]
    (root / "MODEL_CARD.md").write_text("\n".join(L))
    print("wrote", root / "MODEL_CARD.md")
