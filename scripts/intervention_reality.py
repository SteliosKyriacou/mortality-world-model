"""How do Model A's intervention predictions compare with reality?

1. Per person (synthetic only): for held-out people, the predicted effect of treating from
   baseline (u = 1 vs u = 0, paired futures) on HbA1c at +10 y and on years alive over 15 y,
   vs the effect of the TRUE equation from the same person's true state.
2. Trial-style (also possible on real data): u was randomised, so held-out people form a
   treated and an untreated arm. Compare the observed arm difference in annualised HbA1c
   change (first -> last visit) and the observed Kaplan-Meier survival by arm with what the
   model predicts for the same people.

    conda run -n mwm python scripts/intervention_reality.py --run runs/long/on/seed0 --tag long
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mwm.data import synthetic as syn
from mwm.data.dataset import load_cohort_dir, make_pairs
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.encoders.train import EncoderBundle
from mwm.interrogate.common import T, decode_np, log_hazard_np
from mwm.pipeline import jsonable

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/synthetic/on/seed0")
ap.add_argument("--data", default="data/synthetic/on")
ap.add_argument("--n-people", type=int, default=600)
ap.add_argument("--tag", default="v1")
ap.add_argument("--out", default="reports/model_a")
args = ap.parse_args()
dev = "cuda"
rng = np.random.default_rng(0)
run = Path(args.run)

tcfg, meta, tl, L = load_truth(args.data)
ca = load_cohort_dir(args.data)
bundle = EncoderBundle.load(run / "encoder.pt").to(dev)
lat = pd.read_parquet(run / "latents.parquet")
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)
m = NeuralSDE(d, hidden=128, solver="native", n_steps=32).to(dev)
m.set_obs_noise(json.loads((run / "encoder_meta.json").read_text())["latent_obs_noise_var"])
m.load_state_dict(torch.load(run / "model_A.pt", map_location=dev, weights_only=True))
m.eval()
feats_true = meta["clinical_features"]
J_HBA = feats_true.index("hba1c")
std_feats = ca.standardizer.features
J_HBA_M = std_feats.index("hba1c")


def model_hba1c(Z):
    """Decoded HbA1c (mmol/mol) for latents (..., d)."""
    Xs = decode_np(bundle, Z)
    full = np.zeros(Xs.shape[:-1] + (len(std_feats),))
    full[..., :Xs.shape[-1]] = Xs
    return ca.standardizer.inverse(full.reshape(-1, len(std_feats))).reshape(full.shape)[..., J_HBA_M]


def true_hba1c(Z):
    xs = syn.decode_clinical_std(Z, L["Wc"], L["quad"])
    return syn.std_to_raw(xs, feats_true)[..., J_HBA]


def sim_model(Z0, A0, U, years, K, seed, spy=4):
    """Paired-noise EM for many people at once; U per person (0/1). Returns path (n+1, K, N, d)."""
    N = len(Z0)
    n = int(years * spy)
    h = years / n
    g = torch.Generator(device=dev).manual_seed(seed)
    z = T(np.repeat(Z0[None], K, 0).reshape(K * N, d), dev)
    a0 = T(np.tile(A0, K), dev)
    u = T(np.tile(U, K).reshape(-1, 1), dev)
    path = [z.cpu().numpy()]
    for i in range(n):
        a = a0 + i * h
        f = m.drift(z, a, u).detach()
        with torch.no_grad():
            s = m.diffusion(z, a)
        eps = torch.randn(z.shape, generator=g, device=dev)
        z = (z + f * h + s * np.sqrt(h) * eps).detach()
        path.append(z.cpu().numpy())
    return np.stack(path).reshape(n + 1, K, N, d), h


def sim_true(Z0, A0, U, years, K, seed, dt=0.05):
    r = np.random.default_rng(seed)
    N = len(Z0)
    n = int(round(years / dt))
    z = np.repeat(Z0[None], K, 0).reshape(K * N, 8).astype(np.float64)
    a0, u = np.tile(A0, K), np.tile(U, K)
    out = [z.copy()]
    for i in range(n):
        a = a0 + i * dt
        z = z + syn.true_drift(z, a, u, tcfg) * dt + syn.true_diffusion(z, a, tcfg) * np.sqrt(dt) * r.normal(size=z.shape)
        out.append(z.copy())
    return np.stack(out).reshape(n + 1, K, N, 8), dt


def rmst(lam, h):
    cum = np.concatenate([np.zeros((1,) + lam.shape[1:]), np.cumsum(0.5 * (lam[1:] + lam[:-1]) * h, 0)])
    S = np.exp(-cum)
    return (0.5 * (S[1:] + S[:-1]) * h).sum(0), S   # years alive over the horizon


# ---------------- 1. per-person predicted vs true effect ----------------
te = lat[lat.split == "test"]
base = te.groupby("person_id").head(1)
base = base.sample(min(args.n_people, len(base)), random_state=0)
Z0, A0 = base[zc].to_numpy(), base.visit_age.to_numpy()
key = pd.MultiIndex.from_arrays([base.person_id, base.visit_age.round(3)])
Zt0 = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
ok = np.isfinite(Zt0).all(1)
Z0, A0, Zt0 = Z0[ok], A0[ok], Zt0[ok]
N = len(Z0)
H_MARK, H_SURV = 10.0, 15.0
res_pp = {}
Pm, hm_ = {}, None
for u in (0, 1):
    P, hm_ = sim_model(Z0, A0, np.full(N, float(u)), H_SURV, 64, seed=11)
    Pm[u] = P
i10 = int(round(H_MARK / hm_))
hb_m = {u: model_hba1c(Pm[u][i10]).mean(0) for u in (0, 1)}               # (N,)
lam_m = {u: np.exp(log_hazard_np(bundle, Pm[u])) for u in (0, 1)}        # (n+1, K, N)
rm_m = {u: rmst(lam_m[u], hm_)[0].mean(0) for u in (0, 1)}
del Pm
Pt_, ht_ = {}, None
hb_t, rm_t = {}, {}
for u in (0, 1):
    P, ht_ = sim_true(Zt0, A0, np.full(N, float(u)), H_SURV, 128, seed=12)
    j10 = int(round(H_MARK / ht_))
    hb_t[u] = true_hba1c(P[j10]).mean(0)
    lam = np.exp(syn.true_log_hazard(P[::5], tcfg))                       # 0.25 y grid
    rm_t[u] = rmst(lam, ht_ * 5)[0].mean(0)
    del P
pp = {"age0": A0.round(2).tolist(),
      "true_nutr_z2": Zt0[:, 2].round(3).tolist(),
      "pred_d_hba1c_10y": (hb_m[1] - hb_m[0]).round(3).tolist(),
      "true_d_hba1c_10y": (hb_t[1] - hb_t[0]).round(3).tolist(),
      "pred_d_years_alive_15y": (rm_m[1] - rm_m[0]).round(4).tolist(),
      "true_d_years_alive_15y": (rm_t[1] - rm_t[0]).round(4).tolist(),
      "pred_years_alive_15y_untreated": rm_m[0].round(3).tolist(),
      "true_years_alive_15y_untreated": rm_t[0].round(3).tolist()}


def summ(p, t):
    p, t = np.asarray(p), np.asarray(t)
    slope = np.polyfit(t, p, 1)[0] if t.std() > 0 else float("nan")
    return {"pred_mean": float(p.mean()), "true_mean": float(t.mean()), "corr": float(np.corrcoef(p, t)[0, 1]),
            "slope_pred_on_true": float(slope), "mean_abs_err": float(np.abs(p - t).mean())}


pp["summary"] = {"hba1c": summ(pp["pred_d_hba1c_10y"], pp["true_d_hba1c_10y"]),
                 "years_alive": summ(pp["pred_d_years_alive_15y"], pp["true_d_years_alive_15y"]),
                 "untreated_years_alive": summ(pp["pred_years_alive_15y_untreated"], pp["true_years_alive_15y_untreated"])}
print("per-person:", json.dumps(pp["summary"], indent=1))

# ---------------- 2. trial-style: observed arms vs model ----------------
pairs = make_pairs(te.reset_index(), zc, "first_last")
idx0 = te.index.to_numpy()[pairs["i0"]]
idx1 = te.index.to_numpy()[pairs["i1"]]
hb_obs0 = ca.visits.X[idx0, ca.visits.features.index("hba1c")]
hb_obs1 = ca.visits.X[idx1, ca.visits.features.index("hba1c")]
dt = pairs["a1"] - pairs["a0"]
u_arm = pairs["u"][:, 0]
okp = np.isfinite(hb_obs0) & np.isfinite(hb_obs1) & (dt > 0.5)
obs_rate = (hb_obs1 - hb_obs0) / dt


def boot_diff(x, g, reps=2000, seed=0):
    r = np.random.default_rng(seed)
    a, b = x[g == 1], x[g == 0]
    v = [r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(reps)]
    return float(a.mean() - b.mean()), [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


obs_diff, obs_ci = boot_diff(obs_rate[okp], u_arm[okp])
# model prediction for the same people with their actual arm, from their first visit
z0p, a0p, a1p = pairs["z0"][okp], pairs["a0"][okp], pairs["a1"][okp]
with torch.no_grad():
    S = m.sample(T(z0p, dev), T(a0p, dev), T(a1p, dev), T(u_arm[okp].reshape(-1, 1), dev), n_samples=64, n_steps=40).cpu().numpy()
pred_hb1 = model_hba1c(S).mean(0)
pred_rate = (pred_hb1 - hb_obs0[okp]) / dt[okp]
pred_diff, pred_ci = boot_diff(pred_rate, u_arm[okp])
# truth expectation for the same people (true state at first visit, true equation)
k0 = pd.MultiIndex.from_arrays([lat.person_id.to_numpy()[idx0], lat.visit_age.to_numpy()[idx0].round(3)])
zt_first = tl.set_index(["person_id", "visit_age"]).reindex(k0)[[f"ztrue_{k}" for k in range(8)]].to_numpy()[okp]
r = np.random.default_rng(5)
z = np.repeat(zt_first[None], 64, 0).reshape(-1, 8)
a = np.tile(a0p, 64)
uu = np.tile(u_arm[okp], 64)
step = 0.05
dtt = np.tile(dt[okp], 64)
nst = int(np.ceil(dtt.max() / step))
for i in range(nst):
    hstep = np.clip(dtt - i * step, 0, step)
    live = hstep > 0
    z[live] = z[live] + syn.true_drift(z[live], a[live], uu[live], tcfg) * hstep[live, None] + \
        syn.true_diffusion(z[live], a[live], tcfg) * np.sqrt(hstep[live, None]) * r.normal(size=(live.sum(), 8))
    a = a + hstep
true_hb1 = true_hba1c(z.reshape(64, -1, 8)).mean(0)
true_rate = (true_hb1 - hb_obs0[okp]) / dt[okp]
true_diff, true_ci = boot_diff(true_rate, u_arm[okp])
print(f"arm difference in HbA1c change (mmol/mol per year): observed {obs_diff:.3f} {obs_ci}, "
      f"model {pred_diff:.3f}, truth {true_diff:.3f}")

# survival by arm from baseline: observed KM vs model vs truth expectation
outc = ca.outcomes.reindex(base.person_id)
allbase = te.groupby("person_id").head(1)
ob = ca.outcomes.reindex(allbase.person_id)
t_obs = (ob["age_at_death_or_censor"].to_numpy() - allbase.visit_age.to_numpy()).clip(1e-3)
e_obs = ob["event"].to_numpy()
u_base = allbase["u"].to_numpy()
grid = np.linspace(0, 15, 61)


def km(t, e):
    o = np.argsort(t)
    t, e = t[o], e[o]
    S, out, j = 1.0, [], 0
    atrisk = len(t)
    for g in grid:
        while j < len(t) and t[j] <= g:
            if e[j]:
                S *= (1 - 1 / atrisk)
            atrisk -= 1
            j += 1
        out.append(S)
    return out


km_obs = {int(u): km(t_obs[u_base == u], e_obs[u_base == u]) for u in (0, 1)}
Zb, Ab = allbase[zc].to_numpy(), allbase.visit_age.to_numpy()
km_model = {}
for u in (0, 1):
    sel = u_base == u
    P, h = sim_model(Zb[sel], Ab[sel], np.full(sel.sum(), float(u)), 15.0, 32, seed=21)
    lam = np.exp(log_hazard_np(bundle, P))
    S = rmst(lam, h)[1].mean((1, 2))                                     # (n+1,)
    km_model[u] = np.interp(grid, np.linspace(0, 15, len(S)), S).tolist()
    del P
km_true = {}
kb = pd.MultiIndex.from_arrays([allbase.person_id, allbase.visit_age.round(3)])
Ztb = tl.set_index(["person_id", "visit_age"]).reindex(kb)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
for u in (0, 1):
    sel = (u_base == u) & np.isfinite(Ztb).all(1)
    P, h = sim_true(Ztb[sel], Ab[sel], np.full(sel.sum(), float(u)), 15.0, 16, seed=31)
    lam = np.exp(syn.true_log_hazard(P[::5], tcfg))
    S = rmst(lam, h * 5)[1].mean((1, 2))
    km_true[u] = np.interp(grid, np.linspace(0, 15, len(S)), S).tolist()
    del P
trial = {"hba1c_rate": {"observed_diff": obs_diff, "observed_ci95": obs_ci, "model_diff": pred_diff, "model_ci95": pred_ci,
                        "truth_diff": true_diff, "truth_ci95": true_ci, "n_treated": int((u_arm[okp] == 1).sum()),
                        "n_untreated": int((u_arm[okp] == 0).sum()),
                        "observed_rate_by_arm": {str(u): float(obs_rate[okp][u_arm[okp] == u].mean()) for u in (0, 1)},
                        "model_rate_by_arm": {str(u): float(pred_rate[u_arm[okp] == u].mean()) for u in (0, 1)},
                        "truth_rate_by_arm": {str(u): float(true_rate[u_arm[okp] == u].mean()) for u in (0, 1)}},
         "survival": {"grid": grid.tolist(), "observed_km": km_obs, "model": km_model, "truth": km_true,
                      "n": {str(u): int((u_base == u).sum()) for u in (0, 1)},
                      "deaths": {str(u): int(e_obs[u_base == u].sum()) for u in (0, 1)}}}
outp = Path(args.out) / f"intervention_reality_{args.tag}.json"
outp.write_text(json.dumps(jsonable({"run": str(run), "per_person": pp, "trial": trial}), allow_nan=False))
print("wrote", outp)
