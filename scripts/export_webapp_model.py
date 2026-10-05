"""Export one trained Model A for the in-browser simulator (reports/model_a/webapp/models/<key>.json).

Contents: network weights (drift J, landscape V, diffusion, decoder rows for the displayed
markers, hazard head, latent whitening), example held-out people (baseline latent, true hidden
state, observed visits), two landscape planes, the synthetic ground-truth parameters, run
metadata/results, and Python reference outputs ("tests") the JavaScript engine is checked against.

    conda run -n mwm python scripts/export_webapp_model.py --run runs/fix_sweep/on/baseline \
        --data data/synthetic_ukb/on --key transformer_500k --label "Transformer encoder · 500k people"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mwm.data import synthetic as syn
from mwm.data.common import Standardizer
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.encoders.train import EncoderBundle
from mwm.interrogate.common import T, decode_np, drift_at, log_hazard_np
from mwm.interrogate.irreversibility import arrow_direction
from mwm.viz.landscape import learned_surface, plane_direction

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", default="reports/model_a/webapp/models")
ap.add_argument("--n-people", type=int, default=12)
ap.add_argument("--enc-epochs", type=int, default=None,
                help="true encoder epochs when the run reused a frozen encoder (its args show the default)")
args = ap.parse_args()
dev = "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)
run, data = Path(args.run), Path(args.data)
SHOW = ["hba1c", "glucose", "crp", "il6", "sbp", "egfr", "ldl", "grip", "walk_speed", "frailty_index"]

# ---------------- load ----------------
emeta = json.loads((run / "encoder_meta.json").read_text())
bundle = EncoderBundle.load(run / "encoder.pt", map_location=dev).to(dev)
lat = pd.read_parquet(run / "latents.parquet")
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)
m = NeuralSDE.from_run_dir(run, d, dev)
m.set_obs_noise(emeta["latent_obs_noise_var"])
m.load_state_dict(torch.load(run / "model_A.pt", map_location=dev, weights_only=True))
m.eval()
res = json.loads((run / "results.json").read_text())
st = Standardizer.from_dict(emeta["standardizer"])
feats = emeta["features"]
fidx = [feats.index(f) for f in SHOW]
tcfg, meta, tl, L = load_truth(data)
key = pd.MultiIndex.from_arrays([lat.person_id, lat.visit_age.round(3)])
Zt = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()


def rnd(a, k=6):
    return np.round(np.asarray(a, dtype=np.float64), k).tolist()


def mlp_layers(seq):
    """nn.Sequential(Linear, act, Linear, act, Linear) -> list of {W (out x in), b}."""
    out = []
    for mod in seq:
        if isinstance(mod, torch.nn.Linear):
            out.append({"W": rnd(mod.weight.detach().numpy()), "b": rnd(mod.bias.detach().numpy())})
    return out


# ---------------- weights ----------------
dec = mlp_layers(bundle.decoder.net)
dec[-1] = {"W": [dec[-1]["W"][j] for j in fidx], "b": [dec[-1]["b"][j] for j in fidx]}
weights = {
    "d": d,
    "J": mlp_layers(m.field.J), "V": mlp_layers(m.field.V), "diff": mlp_layers(m.diff.net),
    "decoder": dec, "hazard": mlp_layers(bundle.hazard.net),
    "hazard_bias": float(bundle.hazard.bias.item()),
    "z_mean": rnd(bundle.z_mean.numpy()), "z_std": rnd(bundle.z_std.numpy()),
    "show": SHOW, "use_age": bool(m.field.use_age), "diffusion_mode": m.diff.mode,
    "std_mean": rnd([st.mean[st.features.index(f)] for f in SHOW]),
    "std_std": rnd([st.std[st.features.index(f)] for f in SHOW]),
    "std_log": [f in st.log_features for f in SHOW],
}

# ---------------- example people (held-out) ----------------
te = lat[lat.split == "test"]
nvis = te.groupby("person_id").size()
base = te.groupby("person_id").head(1)
base = base[(base.visit_age >= 40) & (base.visit_age <= 64) & base.person_id.map(nvis).ge(2).to_numpy()]
bi = base.index.to_numpy()
zt_base = Zt[bi]
ok = np.isfinite(zt_base).all(1)
base, zt_base = base[ok], zt_base[ok]
risk = syn.true_log_hazard(zt_base, tcfg)
order = np.argsort(risk)
pick = order[np.linspace(0, len(order) - 1, args.n_people).round().astype(int)]
people_rows = base.iloc[pick]
ids = people_rows.person_id.tolist()
long = pd.read_parquet(data / "long.parquet", filters=[("person_id", "in", ids)])
outc = pd.read_parquet(data / "outcomes.parquet", filters=[("person_id", "in", ids)]).set_index("person_id")
people = []
for (i, row), zt in zip(people_rows.iterrows(), zt_base[pick]):
    pid = row.person_id
    lp = long[long.person_id == pid]
    visits = []
    for a, g in lp.groupby("visit_age"):
        vals = {f: float(g.loc[g.feature == f, "value"].iloc[0]) for f in SHOW if (g.feature == f).any()}
        visits.append({"age": round(float(a), 2), "values": {k: round(v, 3) for k, v in vals.items()}})
    o = outc.loc[pid]
    people.append({"id": pid, "age0": round(float(row.visit_age), 2), "u_actual": float(row.u),
                   "z0": rnd(row[zc].to_numpy(np.float64)), "ztrue0": rnd(zt),
                   "visits": visits, "event": int(o.event), "end_age": round(float(o.age_at_death_or_censor), 2),
                   "cause": str(o.cause)})

# ---------------- landscape planes ----------------
Zte, Ate = te[zc].to_numpy(), te.visit_age.to_numpy()
f_te = drift_at(m, Zte, Ate, np.zeros(len(Zte)), dev)
w = arrow_direction(f_te)
origin = Zte.mean(0)
proj = (Zte - origin) @ w
slope, icpt = np.polyfit(proj, Ate, 1)
age_of_x = lambda x: icpt + slope * np.asarray(x)
trm = (lat.split == "train").to_numpy()
okt = trm & np.isfinite(Zt).all(1)
planes = {}
for name, k in (("nutrient", 2), ("inflammation", 1)):
    e2 = plane_direction(lat[zc].to_numpy()[okt], Zt[okt, k], w)
    sy = (Zte - origin) @ e2
    x_hi = (95.0 - icpt) / slope
    xr = (float(np.quantile(proj, 0.005)) - 0.3, float(max(x_hi, np.quantile(proj, 0.995))) + 0.3)
    yr = (float(np.quantile(sy, 0.005)) - 1.0, float(np.quantile(sy, 0.995)) + 1.0)
    S = learned_surface(m, bundle, w, e2, xr, yr, origin, age_of_x, dev, n=61)
    planes[name] = {"w": rnd(w), "e2": rnd(e2), "origin": rnd(origin), "x": rnd(S["x"], 4), "y": rnd(S["y"], 4),
                    "U": rnd(S["U"], 4), "log_hazard": rnd(S["log_hazard"], 4), "curl_frac": round(S["curl_frac"], 3),
                    "true_axis": syn.LATENT_NAMES[k]}
age_map = {"slope": float(slope), "intercept": float(icpt)}

# ---------------- ground truth (synthetic world) ----------------
feats_true = meta["clinical_features"]
tj = [feats_true.index(f) for f in SHOW]
truth = {"cfg": tcfg.to_json(), "Wc": rnd(L["Wc"][tj]), "quad": rnd(L["quad"][tj]),
         "prim": [int(np.argmax(np.abs(L["Wc"][j]))) for j in tj],
         "ref_mean": rnd(L["ref_mean"]), "ref_scale": rnd(L["ref_scale"]),
         "clin": {f: {"m": syn.CLINICAL[f][1], "s": syn.CLINICAL[f][2], "log": syn.CLINICAL[f][4],
                      "unit": syn.CLINICAL[f][3]} for f in SHOW},
         "latent_names": syn.LATENT_NAMES}

# ---------------- reference outputs for the JS engine ----------------
ti = rng.choice(len(Zte), 16, replace=False)
Zs, As = Zte[ti], Ate[ti] + rng.uniform(-5, 15, 16)
Us = (rng.uniform(size=16) < 0.5).astype(float)
with torch.no_grad():
    sig = m.diffusion(T(Zs, dev), T(As, dev)).numpy()
Xs = decode_np(bundle, Zs, device=dev)[:, fidx]
raw = np.where(weights["std_log"], np.exp(Xs * weights["std_std"] + weights["std_mean"]), Xs * weights["std_std"] + weights["std_mean"])
zt_s = Zt[(lat.split == "test").to_numpy()][ti]
xs_true = syn.decode_clinical_std(zt_s, L["Wc"], L["quad"])[:, tj]
raw_true = np.stack([np.exp(truth["clin"][f]["m"] + truth["clin"][f]["s"] * xs_true[:, i]) if truth["clin"][f]["log"]
                     else truth["clin"][f]["m"] + truth["clin"][f]["s"] * xs_true[:, i] for i, f in enumerate(SHOW)], 1)
tests = {"z": rnd(Zs), "age": rnd(As), "u": rnd(Us),
         "drift": rnd(drift_at(m, Zs, As, Us, dev)), "sigma": rnd(sig), "decoded_raw": rnd(raw),
         "log_hazard": rnd(log_hazard_np(bundle, Zs, device=dev)),
         "ztrue": rnd(zt_s), "true_drift": rnd(syn.true_drift(zt_s, As, Us, tcfg)),
         "true_sigma": rnd(syn.true_diffusion(zt_s, As, tcfg)), "true_decoded_raw": rnd(raw_true),
         "true_log_hazard": rnd(syn.true_log_hazard(zt_s, tcfg))}

# ---------------- metadata ----------------
fc, sv, gt = res["forecast"], res["survival"], res.get("ground_truth", {})
hm = (res.get("hallmarks") or {}).get("A", {})
summary = {
    "forecast_mae": fc["A"]["mae"], "gbm_mae": fc.get("base_direct_gbm", {}).get("mae"),
    "cov90": fc["A"]["cov90"], "c_index": sv["A"]["c_harrell"],
    "cox_c_index": sv.get("cox_baseline_features", {}).get("c_harrell"),
    "drift_rel_mse": gt.get("A", {}).get("drift_rel_mse"),
    "affine_r2": float(np.mean(gt["affine_r2_per_true_dim"])) if gt.get("affine_r2_per_true_dim") else None,
    "best_epoch": res.get("best_epoch"),
    "hallmarks": {k: bool(v.get("detected", v.get("detected_flag"))) for k, v in hm.items()},
}
out = {"key": args.key, "label": args.label, "run": str(run), "data": str(data),
       "encoder": bundle.cfg["kind"],
       "n_people_cohort": int(lat.person_id.nunique()),
       "train_args": {**res.get("args", {}), **({"enc_epochs": args.enc_epochs} if args.enc_epochs else {})},
       "summary": summary, "weights": weights, "people": people, "planes": planes, "age_map": age_map,
       "truth": truth, "tests": tests}
p = Path(args.out) / f"{args.key}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
print("wrote", p, round(p.stat().st_size / 1e6, 2), "MB; people", len(people), "curl", {k: v["curl_frac"] for k, v in planes.items()})

# ---------------- manifest of all exported models ----------------
man = []
for f in sorted(Path(args.out).glob("*.json")):
    if f.name == "index.json":
        continue
    j = json.loads(f.read_text())
    man.append({"key": j["key"], "label": j["label"], "encoder": j["encoder"], "n_people_cohort": j["n_people_cohort"],
                "train_args": {k: j["train_args"].get(k) for k in ("a_epochs", "enc_epochs", "lr", "jac_penalty", "select")},
                "summary": j["summary"]})
order = {"transformer_500k": 0, "hetgnn_500k": 1, "transformer_20k": 2, "hetgnn_20k": 3}
man.sort(key=lambda r: order.get(r["key"], 9))
(Path(args.out) / "index.json").write_text(json.dumps(man, indent=1))
print("manifest:", [r["key"] for r in man])
