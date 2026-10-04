"""Aggregate runs/synthetic/<cohort>/seed*/results.json into tables + figures.

    conda run -n mwm python scripts/make_synthetic_report.py --runs runs/synthetic
Writes reports/synthetic_tables.md (auto-generated tables) and reports/figures/*.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "lines.linewidth": 2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/synthetic")
ap.add_argument("--out", default="reports")
args = ap.parse_args()
runs = Path(args.runs)
fig_dir = Path(args.out) / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

R = {c: [json.load(open(p)) for p in sorted((runs / c).glob("seed*/results.json"))] for c in ("on", "off")}
lines = []


def ms(vals, fmt="{:.3f}"):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return fmt.format(v[0])
    return (fmt + " ± " + fmt).format(v.mean(), v.std())


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")


MODELS = ["A", "B", "Bp", "snapshot"]
NAMES = {"A": "A (Neural SDE)", "B": "B (CFM)", "Bp": "B′ (stochastic CFM)", "snapshot": "Snapshot OT-CFM",
         "A_constSigma": "A, constant Σ (control)", "A_noPotential": "A, no potential (control)",
         "A_shuffledAge": "A, shuffled age (control)", "Bp_noScore": "B′ without score correction"}

for coh in ("on", "off"):
    rs = R[coh]
    if not rs:
        continue
    lines.append(f"## Cohort: hallmarks-{coh.upper()} ({len(rs)} seeds)\n")
    # ---- oracle
    orc = [r["oracle"] for r in rs if "oracle" in r]
    if orc:
        lines.append("### Ground-truth recovery, oracle latent (models trained on the true z*, seed 0)\n")
        rows = []
        for m in ("A", "B", "Bp", "Bp_noScore"):
            if m not in orc[0]:
                continue
            o = orc[0][m]
            rows.append([NAMES.get(m, m), f"{o['drift_rel_mse']:.3f}", f"{o['drift_cos']:.3f}",
                         " ".join(f"{x:.2f}" for x in o["drift_nrmse_per_dim"]),
                         " ".join(f"{x:.3f}" for x in o["u_effect_on_true_dims"][2:3]),
                         " ".join(f"{x:.2f}" for x in o.get("diff_var_rel_err_per_dim", [])) or "n/a",
                         f"{o.get('diff_trace_ratio_old_young_model', float('nan')):.2f} / {o.get('diff_trace_ratio_old_young_true', float('nan')):.2f}"])
        table(["model", "drift rel. MSE", "drift cos", "drift NRMSE per true dim (0..7)", "u effect on z2 (true: "
               + ("-0.020" if coh == "on" else "0") + ")", "diffusion var rel. err per dim", "Σ² ratio old/young (model / true)"], rows)
    # ---- encoder latent recovery
    lines.append("### Ground-truth recovery through the learned encoder (affine map learned→true)\n")
    lines.append("Affine R² of true latent from learned latent (test visits, per true dim): "
                 + ms([np.mean(r["ground_truth"]["affine_r2_per_true_dim"]) for r in rs]) + " (mean over dims); seed 0: "
                 + " ".join(f"{x:.2f}" for x in rs[0]["ground_truth"]["affine_r2_per_true_dim"]) + "\n")
    rows = []
    for m in MODELS + ["A_constSigma", "A_noPotential", "A_shuffledAge"]:
        g = [r["ground_truth"][m] for r in rs if m in r["ground_truth"]]
        if not g:
            continue
        rows.append([NAMES.get(m, m), ms([x["drift_rel_mse"] for x in g]), ms([x["drift_cos"] for x in g]),
                     " ".join(f"{np.mean([x['drift_nrmse_per_dim'][k] for x in g]):.2f}" for k in range(8)),
                     ms([x["u_effect_on_true_dims"][2] for x in g], "{:.4f}"),
                     ms([x.get("diff_trace_ratio_old_young_model") for x in g], "{:.2f}"),
                     ms([x.get("diff_trace_rel_err") for x in g], "{:.2f}")])
    table(["model", "drift rel. MSE", "drift cos", "NRMSE per true dim", "u effect on z2",
           "Σ² ratio old/young (true " + f"{rs[0]['ground_truth']['A'].get('diff_trace_ratio_old_young_true', float('nan')):.2f})",
           "Σ² trace rel. err"], rows)
    # ---- forecasting
    lines.append("### Forecasting held-out people (first → last visit; decoded clinical features in z-units)\n")
    keys = [k for k in rs[0]["forecast"]]
    rows = []
    for k in keys:
        f = [r["forecast"][k] for r in rs]
        rows.append([NAMES.get(k, k), ms([x.get("latent_mse") for x in f]), ms([x.get("mae") for x in f]),
                     ms([x.get("rmse") for x in f]), ms([x.get("crps") for x in f]),
                     ms([x.get("cov50") for x in f], "{:.2f}"), ms([x.get("cov90") for x in f], "{:.2f}"),
                     ms([x.get("latent_energy") for x in f])])
    table(["model", "latent MSE", "MAE", "RMSE", "CRPS", "cov50", "cov90", "latent energy score"], rows)
    lines.append("MAE by Δt bin (years), seed-mean:\n")
    bins = list(rs[0]["forecast"]["A"]["mae_vs_dt"].keys())
    rows = [[NAMES.get(k, k)] + [ms([r["forecast"][k].get("mae_vs_dt", {}).get(b) for r in rs]) for b in bins]
            for k in keys if "mae" in rs[0]["forecast"][k]]
    table(["model"] + bins, rows)
    # ---- survival
    lines.append("### Survival (baseline visit → death; held-out people)\n")
    rows = []
    for k in rs[0]["survival"]:
        s = [r["survival"][k] for r in rs]
        rows.append([NAMES.get(k, k), ms([x.get("c_harrell") for x in s]), ms([x.get("c_uno") for x in s]),
                     ms([x.get("auc_5y") for x in s]), ms([x.get("auc_10y") for x in s]), ms([x.get("ibs") for x in s])])
    table(["model", "Harrell C", "Uno C", "AUC 5y", "AUC 10y", "IBS"], rows)
    # ---- cost
    lines.append("### Cost\n")
    rows = []
    for k in rs[0]["costs"]:
        c = [r["costs"][k] for r in rs if k in r["costs"]]
        rows.append([NAMES.get(k, k), ms([x.get("train_seconds") for x in c], "{:.0f}"),
                     ms([x.get("peak_vram_mb") for x in c], "{:.0f}"),
                     ms([x.get("nfe_10y_rollout") for x in c], "{:.0f}"),
                     ms([x.get("nfe_10y_dopri5") for x in c], "{:.0f}")])
    table(["model", "train s", "peak VRAM MB", "NFE / 10y rollout (fixed step)", "NFE dopri5"], rows)
    if "constSigma_vs_ageSigma_val_nll" in rs[0]:
        lines.append(f"Validation NLL, age-dependent Σ vs constant Σ (seed 0): {rs[0]['constSigma_vs_ageSigma_val_nll']}\n")

# ---- hallmark matrix
lines.append("## Hallmark tests: detection (fraction of seeds) on hallmarks-ON vs hallmarks-OFF\n")
lines.append("Desired pattern: ON = 1.00, OFF = 0.00. For jacobian_enrichment the ON statistic is the number "
             "of planted aging pathways detected (of 7); on OFF any detected pathway counts as a false positive. "
             "Controls were trained for seed 0 only.\n")
TESTS = [("inflammaging", "detected", "statistic"), ("dispersion", "detected", "statistic"),
         ("irreversibility", "detected", "statistic"), ("intervention", "detected", "statistic"),
         ("jacobian_enrichment", "detected_flag", "true_pos"), ("attractors", "detected", "statistic")]
TRUTH = {c: json.load(open(runs / c / "truth_hallmarks.json")) for c in ("on", "off")
         if (runs / c / "truth_hallmarks.json").exists()}
rows = []
for t, dk, sk in TESTS:
    if len(TRUTH) == 2:
        rows.append([t, "**ground truth** (true dynamics, true latent)"] +
                    sum([[str(bool(TRUTH[c][t][dk])), f"{TRUTH[c][t].get(sk) if not isinstance(TRUTH[c][t].get(sk), list) else ''}"[:6]]
                         for c in ("on", "off")], []))
    for m in ["A", "B", "Bp", "A_constSigma", "A_noPotential", "A_shuffledAge"]:
        cells = []
        for coh in ("on", "off"):
            hs = [r["hallmarks"][m][t] for r in R[coh] if m in r["hallmarks"] and t in r["hallmarks"][m]]
            if not hs:
                cells += ["–", "–"]
                continue
            # in the OFF cohort pathway names carry no structure: ANY detected pathway is a false positive
            det = [(h["n_detected"] > 0) if (t == "jacobian_enrichment" and coh == "off") else h[dk] for h in hs]
            cells.append(f"{np.mean(det):.2f} ({len(hs)})")
            cells.append(ms([h.get(sk) if not isinstance(h.get(sk), list) else None for h in hs], "{:.3g}"))
        rows.append([t, NAMES.get(m, m)] + cells)
for coh_idx, coh in enumerate(("on", "off")):
    pass
att_rows = []
for coh in ("on", "off"):
    hs = [r["hallmarks"]["attribution_encoder_hazard"] for r in R[coh]]
    if hs:
        att_rows.append(f"{np.mean([h['detected'] for h in hs]):.2f} ({len(hs)})")
        att_rows.append(ms([h["statistic"] for h in hs], "{:.3g}"))
    else:
        att_rows += ["–", "–"]
rows.append(["attribution (encoder hazard)", "encoder"] + att_rows)
table(["test", "model", "ON detected", "ON statistic", "OFF detected", "OFF statistic"], rows)

Path(args.out, "synthetic_tables.md").write_text("<!-- auto-generated by scripts/make_synthetic_report.py -->\n"
                                                 "# Synthetic validation: auto-generated tables\n\n" + "\n".join(lines))

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(fig_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# 1) inflammaging curves
if R["on"] and R["off"]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=False)
    for ax, coh in zip(axes, ("on", "off")):
        h = R[coh][0]["hallmarks"]
        for i, m in enumerate(["A", "B", "Bp"]):
            x = h[m]["inflammaging"]
            ax.plot(x["age_grid"], np.array(x["curve"]) - x["curve"][0], color=SERIES[i], label=NAMES[m])
        x = h["A"]["inflammaging"]
        ax.plot(x["raw_age"], np.array(x["raw_curve"]) - x["raw_curve"][0], "o", ms=4, color=MUTED,
                label="observed held-out cross-section")
        ax.set_title(f"Hallmarks-{coh.upper()}", color=INK, loc="left")
        ax.set_xlabel("age (years)")
        ax.set_ylabel("Δ inflammatory composite (z)")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "inflammaging_rollouts.png")

# 2) diffusion vs age, oracle
if R["on"] and "oracle" in R["on"][0]:
    o = R["on"][0]["oracle"]
    ages = [int(a) + 5 for a in o["true_diffusion_by_age"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, k, nm in zip(axes, [0, 1, 6], ["z0 age core", "z1 inflammation", "z6 OU"]):
        ax.plot(ages, [v[k] for v in o["true_diffusion_by_age"].values()], color=INK, label="true σ")
        for i, m in enumerate(["A", "Bp"]):
            ax.plot(ages, [v[k] for v in o[m]["diffusion_by_age"].values()], "o-", color=SERIES[i], label=NAMES[m], ms=4)
        ax.set_title(nm, loc="left", color=INK)
        ax.set_xlabel("age")
    axes[0].set_ylabel("σ (oracle latent)")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "diffusion_vs_age_oracle.png")

# 3) MAE vs dt
if R["on"]:
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    keys = ["A", "B", "Bp", "base_locf", "base_linear_drift", "base_direct_gbm"]
    bins = list(R["on"][0]["forecast"]["A"]["mae_vs_dt"].keys())
    for i, k in enumerate(keys):
        y = [np.mean([r["forecast"][k]["mae_vs_dt"].get(b, np.nan) for r in R["on"]]) for b in bins]
        ax.plot(range(len(bins)), y, "o-", color=SERIES[i], label=NAMES.get(k, k), ms=5)
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlabel("Δt bin (years)")
    ax.set_ylabel("decoded MAE (z-units)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "mae_vs_dt.png")

# 4) hallmark detection heatmap (A, B, B')
if R["on"] and R["off"]:
    tests = [t for t, _, _ in TESTS]
    ms_ = ["A", "B", "Bp"]
    M = np.full((len(tests), 2 * len(ms_)), np.nan)
    for i, (t, dk, _) in enumerate(TESTS):
        for j, m in enumerate(ms_):
            for c, coh in enumerate(("on", "off")):
                hs = [(r["hallmarks"][m][t]["n_detected"] > 0) if (t == "jacobian_enrichment" and coh == "off")
                      else r["hallmarks"][m][t][dk] for r in R[coh] if t in r["hallmarks"][m]]
                if hs:
                    M[i, 2 * j + c] = np.mean(hs)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f3f3f0", "#2a78d6"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), [f"{NAMES[m].split(' ')[0]}\n{c}" for m in ms_ for c in ("ON", "OFF")])
    ax.set_yticks(range(len(tests)), tests)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Detection rate across seeds (want ON=1, OFF=0)", loc="left", color=INK)
    save(fig, "hallmark_detection.png")
print("wrote", Path(args.out, "synthetic_tables.md"), "and figures in", fig_dir)
