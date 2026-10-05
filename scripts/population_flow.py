"""Population view of Model A: a cohort of held-out people evolved as weighted particles.

Each particle follows Model A's SDE; its weight decays as exp(-∫λ(z)da) with the death-risk head.
The weighted particle cloud is a Monte Carlo (Feynman–Kac) solution of the Fokker–Planck equation
with mortality as a killing term:

    ∂p/∂a = −∇·(f p) + ½ ∇∇:(ΣΣᵀ p) − λ(z) p

Outputs per scenario: alive fraction, density and death flux on the landscape planes, cohort mean path,
and the distribution of decoded markers among survivors at each age. The same cohort is run through the
true equation (from each person's true state); observed survival (Kaplan–Meier with delayed entry) and
observed marker cross-sections come from the held-out data.

    conda run -n mwm python scripts/population_flow.py --run runs/fix_sweep/on/baseline \
        --data data/synthetic_ukb/on --key transformer_500k
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from mwm.data import synthetic as syn
from mwm.data.common import Standardizer
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.encoders.train import EncoderBundle

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--entry", nargs=2, type=float, default=[40.0, 45.0], help="cohort: baseline age range")
ap.add_argument("--max-n", type=int, default=15000)
ap.add_argument("--end-age", type=float, default=95.0)
ap.add_argument("--grid", type=int, default=31)
ap.add_argument("--out", default="reports/model_a/webapp/population")
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = "cpu"
rng = np.random.default_rng(0)
torch.manual_seed(0)
run, data = Path(args.run), Path(args.data)
web = json.loads(Path(f"reports/model_a/webapp/models/{args.key}.json").read_text())
SHOW = web["weights"]["show"]
emeta = json.loads((run / "encoder_meta.json").read_text())
bundle = EncoderBundle.load(run / "encoder.pt", map_location=dev).to(dev)
lat = pd.read_parquet(run / "latents.parquet")
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)
m = NeuralSDE.from_run_dir(run, d, dev)
m.set_obs_noise(emeta["latent_obs_noise_var"])
m.load_state_dict(torch.load(run / "model_A.pt", map_location=dev, weights_only=True))
m.eval()
st = Standardizer.from_dict(emeta["standardizer"])
fidx = [emeta["features"].index(f) for f in SHOW]
sig_x = bundle.feature_sigma.detach().numpy()[fidx]                 # decoder per-marker noise (std units)
smean = np.array([st.mean[st.features.index(f)] for f in SHOW])
sstd = np.array([st.std[st.features.index(f)] for f in SHOW])
slog = np.array([f in st.log_features for f in SHOW])
tcfg, meta, tl, L = load_truth(data)
tj = [meta["clinical_features"].index(f) for f in SHOW]

# ---------------- cohort ----------------
te = lat[lat.split == "test"]
base = te.groupby("person_id").head(1)
base = base[(base.visit_age >= args.entry[0]) & (base.visit_age < args.entry[1])]
key = pd.MultiIndex.from_arrays([base.person_id, base.visit_age.round(3)])
zt0 = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
ok = np.isfinite(zt0).all(1)
base, zt0 = base[ok], zt0[ok]
if len(base) > args.max_n:
    sel = rng.choice(len(base), args.max_n, replace=False)
    base, zt0 = base.iloc[sel], zt0[sel]
N = len(base)
Z0, A0, U0 = base[zc].to_numpy(np.float32), base.visit_age.to_numpy(), base["u"].to_numpy()
print(f"cohort: {N} held-out people entering at {args.entry}, actually treated {U0.mean():.1%}")

SCEN = {"actual": lambda a, U: U, "none": lambda a, U: np.zeros_like(U), "all": lambda a, U: np.ones_like(U),
        "from55": lambda a, U: (a >= 55).astype(float), "from65": lambda a, U: (a >= 65).astype(float)}
AGES = np.arange(np.ceil(args.entry[1]), args.end_age + 0.01, 1.0)        # yearly read-out ages
AGES2 = AGES[::2]                                                          # density snapshots every 2 years
QS = [0.1, 0.25, 0.5, 0.75, 0.9]
planes = web["planes"]


def wquant(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return [float("nan")] * len(qs)
    c = c / c[-1]
    return [float(np.interp(q, c, x)) for q in qs]


def decode_model(Z):
    Xs = bundle.decode(torch.as_tensor(Z, dtype=torch.float32)).numpy()[:, fidx]
    Xs = Xs + sig_x * rng.normal(size=Xs.shape)                            # measurement noise, like a real visit
    v = Xs * sstd + smean
    return np.where(slog, np.exp(v), v)


def decode_true(Z):
    xs = syn.decode_clinical_std(Z, L["Wc"], L["quad"])[:, tj] + tcfg.feat_noise * rng.normal(size=(len(Z), len(tj)))
    return np.stack([np.exp(syn.CLINICAL[f][1] + syn.CLINICAL[f][2] * xs[:, i]) if syn.CLINICAL[f][4]
                     else syn.CLINICAL[f][1] + syn.CLINICAL[f][2] * xs[:, i] for i, f in enumerate(SHOW)], 1)


def grid_edges(pl, n):
    x, y = np.array(pl["x"]), np.array(pl["y"])
    return np.linspace(x[0], x[-1], n + 1), np.linspace(y[0], y[-1], n + 1)


EDGES = {k: grid_edges(v, args.grid) for k, v in planes.items()}


def density(P, w, pl, ex, ey):
    q = P - np.array(pl["origin"])
    x, y = q @ np.array(pl["w"]), q @ np.array(pl["e2"])
    H, _, _ = np.histogram2d(np.clip(x, ex[0], ex[-1] - 1e-9), np.clip(y, ey[0], ey[-1] - 1e-9), bins=[ex, ey], weights=w)
    return gaussian_filter(H, 0.8) / N, float((x * w).sum() / max(w.sum(), 1e-12)), float((y * w).sum() / max(w.sum(), 1e-12))


def simulate_states(model_kind, name):
    """Evolve the cohort and read out each particle when ITS OWN age reaches each read-out age
    (people enter at different ages). Returns {age: (Z, survival weight since entry, hazard)}."""
    model = model_kind == "model"
    h = 0.25 if model else 0.05
    n = int(np.ceil((args.end_age - A0.min()) / h)) + 2
    a = A0.astype(np.float64).copy()
    if model:
        g = torch.Generator().manual_seed(1)
        z = torch.as_tensor(Z0)
        lam = torch.exp(bundle.log_hazard(z)).numpy()
        D = Z0.shape[1]
    else:
        r = np.random.default_rng(2)
        z = zt0.astype(np.float64).copy()
        lam = np.exp(syn.true_log_hazard(z, tcfg))
        D = 8
    cum = np.zeros(N)
    res = {A: [np.zeros((N, D)), np.zeros(N), np.zeros(N), np.zeros(N, bool)] for A in AGES}
    for _ in range(n):
        zn = z.numpy() if model else z
        for A in AGES:
            Rz, Rw, Rl, done = res[A]
            hit = (a >= A - 1e-9) & ~done
            if hit.any():
                Rz[hit], Rw[hit], Rl[hit], done[hit] = zn[hit], np.exp(-cum[hit]), lam[hit], True
        if all(r_[3].all() for r_ in res.values()):
            break
        if model:
            u = torch.as_tensor(SCEN[name](a, U0), dtype=torch.float32).reshape(-1, 1)
            at = torch.as_tensor(a, dtype=torch.float32)
            with torch.enable_grad():
                f = m.drift(z, at, u).detach()
            s_ = m.diffusion(z, at)
            z = z + f * h + s_ * np.sqrt(h) * torch.randn(z.shape, generator=g)
            lam_new = torch.exp(bundle.log_hazard(z)).numpy()
        else:
            u = SCEN[name](a, U0)
            z = z + syn.true_drift(z, a, u, tcfg) * h + syn.true_diffusion(z, a, tcfg) * np.sqrt(h) * r.normal(size=z.shape)
            lam_new = np.exp(syn.true_log_hazard(z, tcfg))
        cum += 0.5 * (lam + lam_new) * h
        lam = lam_new
        a = a + h
    return {A: (v[0], v[1], v[2]) for A, v in res.items()}


t0 = time.time()
out = {"key": args.key, "run": str(run), "entry": args.entry, "n": N, "treated_actual": float(U0.mean()),
       "ages": AGES.tolist(), "density_ages": AGES2.tolist(), "grid": args.grid,
       "edges": {k: [e[0].round(4).tolist(), e[1].round(4).tolist()] for k, e in EDGES.items()},
       "markers": SHOW, "quantiles": QS, "scenarios": {}}
for name in SCEN:
    S = simulate_states("model", name)
    T = simulate_states("truth", name)
    sc = {"model": {"alive": [], "deaths_per_year": [], "markers": {f: [] for f in SHOW}},
          "truth": {"alive": [], "deaths_per_year": [], "markers": {f: [] for f in SHOW}},
          "density": {k: [] for k in planes}, "flux": {k: [] for k in planes}, "mean": {k: [] for k in planes}}
    for A in AGES:
        for kind, R, dec in (("model", S, decode_model), ("truth", T, decode_true)):
            Z, w, lam = R[A]
            sc[kind]["alive"].append(float(w.mean()))
            sc[kind]["deaths_per_year"].append(float((w * lam).mean()))      # sink flux, fraction of the cohort per year
            X = dec(Z)
            for i, f in enumerate(SHOW):
                sc[kind]["markers"][f].append([round(v, 4) for v in wquant(X[:, i], w, QS)])
        if A in AGES2:
            Z, w, lam = S[A]
            for k, pl in planes.items():
                D, mx, my = density(Z, w, pl, *EDGES[k])
                F, _, _ = density(Z, w * lam, pl, *EDGES[k])
                sc["density"][k].append(np.round(D * 1e6).astype(int).tolist())   # mass per cell x 1e6
                sc["flux"][k].append(np.round(F * 1e6).astype(int).tolist())      # deaths/yr per cell x 1e6
                sc["mean"][k].append([round(mx, 4), round(my, 4)])
    out["scenarios"][name] = sc
    print(f"{name}: model alive@{AGES[-1]:.0f} {sc['model']['alive'][-1]:.3f}  truth {sc['truth']['alive'][-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)

# ---------------- observed: Kaplan–Meier with delayed entry, marker cross-sections ----------------
oc = pd.read_parquet(data / "outcomes.parquet", filters=[("person_id", "in", base.person_id.tolist())]).set_index("person_id")
entry = A0
exit_ = oc.reindex(base.person_id)["age_at_death_or_censor"].to_numpy()
ev = oc.reindex(base.person_id)["event"].to_numpy()
km, Ssurv = [], 1.0
grid = np.arange(np.ceil(args.entry[1]), exit_.max(), 0.25)
death_ages = np.sort(exit_[ev == 1])
prev = args.entry[1]
for A in AGES:
    if A > np.nanmax(exit_):
        break
    dd = death_ages[(death_ages > prev) & (death_ages <= A)]
    for da in dd:
        at_risk = ((entry <= da) & (exit_ >= da)).sum()
        Ssurv *= (1 - 1 / max(at_risk, 1))
    km.append([float(A), float(Ssurv)])
    prev = A
# condition on alive at the first read-out age (the simulations start each person at entry; normalise both)
out["observed_km"] = km
test_ids = set(te.person_id.unique())
long = pd.read_parquet(data / "long.parquet", filters=[("feature", "in", SHOW)], columns=["person_id", "visit_age", "feature", "value"])
long = long[long.person_id.isin(test_ids)]
obs = {}
for f in SHOW:
    g = long[long.feature == f]
    rows = []
    for A in AGES:
        v = g.value[(g.visit_age >= A - 1) & (g.visit_age < A + 1)].to_numpy()
        rows.append([round(x, 4) for x in np.quantile(v, QS)] if len(v) >= 50 else None)
    obs[f] = rows
out["observed_markers"] = obs
out["observed_marker_note"] = "all held-out visits within ±1 year of each age (any entry age), with measurement noise"
p = Path(args.out) / f"{args.key}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
print("wrote", p, round(p.stat().st_size / 1e6, 2), "MB in", round(time.time() - t0), "s")
