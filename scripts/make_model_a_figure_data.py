"""Figure data for the Model A (Neural SDE) explainer: 3D landscapes with simulated paths,
a single-patient forecast fan, noise-vs-age, arrow-of-aging distributions, and the
hallmark/forecast numbers from runs/synthetic/*/seed*/results.json.

    conda run -n mwm python scripts/make_model_a_figure_data.py --out reports/model_a/figure_data.json

Landscapes: the learned drift is evaluated on a 2D plane through the latent space (axis 1 =
the max-margin "arrow of aging" direction w, axis 2 = the latent direction that best predicts a
chosen ground-truth axis, orthogonalised against w). The plotted height is the least-squares
(Helmholtz) potential U of the in-plane drift: -grad U is the curl-free part of the flow; the
remaining rotational fraction is reported. Model A's own V network is NOT used: it is not
identified (see reports/synthetic_validation.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import LinearRegression

from mwm.data import synthetic as syn
from mwm.data.dataset import load_cohort_dir, make_pairs
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.encoders.train import EncoderBundle
from mwm.interrogate.common import T, decode_np, drift_at, log_hazard_np
from mwm.interrogate.irreversibility import arrow_direction
from mwm.pipeline import jsonable

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/synthetic")
ap.add_argument("--data-root", default="data/synthetic")
ap.add_argument("--config", default="configs/synthetic.yaml")
ap.add_argument("--seed-dir", default="seed0")
ap.add_argument("--out", default="reports/model_a/figure_data.json")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
dev = cfg["device"]
torch.manual_seed(0)
rng = np.random.default_rng(0)
NG = 61  # grid resolution


def load(cohort):
    sd = Path(args.runs) / cohort / args.seed_dir
    cdir = Path(args.data_root) / cohort
    tcfg, meta, tl, L = load_truth(cdir)
    ca = load_cohort_dir(cdir)
    bundle = EncoderBundle.load(sd / "encoder.pt").to(dev)
    lat = pd.read_parquet(sd / "latents.parquet")
    zc = [c for c in lat.columns if c.startswith("z_")]
    a = cfg["model_a"]
    m = NeuralSDE(len(zc), hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"]).to(dev)
    m.set_obs_noise(json.loads((sd / "encoder_meta.json").read_text())["latent_obs_noise_var"])
    m.load_state_dict(torch.load(sd / "model_A.pt", map_location=dev, weights_only=True))
    m.eval()
    key = pd.MultiIndex.from_arrays([lat.person_id, lat.visit_age.round(3)])
    Zt = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
    return dict(tcfg=tcfg, meta=meta, tl=tl, ca=ca, bundle=bundle, lat=lat, zc=zc, m=m, Zt=Zt,
                res=json.loads((sd / "results.json").read_text()))


def ls_potential(fx, fy, hx, hy):
    """Least-squares U with dU/dx ~ -fx, dU/dy ~ -fy on a (nx, ny) grid (edge midpoints)."""
    nx, ny = fx.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals, b = [], [], [], []
    r = 0
    for i in range(nx - 1):
        for j in range(ny):
            rows += [r, r]; cols += [idx[i + 1, j], idx[i, j]]; vals += [1 / hx, -1 / hx]
            b.append(-0.5 * (fx[i, j] + fx[i + 1, j])); r += 1
    for i in range(nx):
        for j in range(ny - 1):
            rows += [r, r]; cols += [idx[i, j + 1], idx[i, j]]; vals += [1 / hy, -1 / hy]
            b.append(-0.5 * (fy[i, j] + fy[i, j + 1])); r += 1
    rows.append(r); cols.append(idx[0, 0]); vals.append(1.0); b.append(0.0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r + 1, nx * ny))
    U = lsqr(A, np.array(b), atol=1e-10, btol=1e-10, iter_lim=20000)[0].reshape(nx, ny)
    gx = np.gradient(U, hx, axis=0)
    gy = np.gradient(U, hy, axis=1)
    curl_frac = float(np.sqrt(((fx + gx) ** 2 + (fy + gy) ** 2).sum() / (fx ** 2 + fy ** 2).sum()))
    return U - U.min(), curl_frac


def plane_dir(D, true_k, w):
    """Latent direction best predicting true axis k (train visits), orthogonalised vs w."""
    tr = (D["lat"].split == "train").to_numpy()
    Z = D["lat"][D["zc"]].to_numpy()
    c = LinearRegression().fit(Z[tr], D["Zt"][tr, true_k]).coef_
    c = c - (c @ w) * w
    return c / np.linalg.norm(c)


def learned_surface(D, w, e2, xr, yr, origin, age_of_x):
    xs, ys = np.linspace(*xr, NG), np.linspace(*yr, NG)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = origin[None] + X.reshape(-1, 1) * w[None] + Y.reshape(-1, 1) * e2[None]
    Ag = age_of_x(X.reshape(-1))
    f = drift_at(D["m"], P, Ag, np.zeros(len(P)), dev)
    fx, fy = (f @ w).reshape(NG, NG), (f @ e2).reshape(NG, NG)
    U, curl = ls_potential(fx, fy, xs[1] - xs[0], ys[1] - ys[0])
    lh = log_hazard_np(D["bundle"], P).reshape(NG, NG)
    return dict(x=xs, y=ys, U=U, curl_frac=curl, log_hazard=lh, fx=fx, fy=fy,
                age_x=age_of_x(xs))


def true_surface(D, k2, xr, yr, ref, age_of_x):
    tc = D["tcfg"]
    xs, ys = np.linspace(*xr, NG), np.linspace(*yr, NG)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.tile(ref, (NG * NG, 1))
    P[:, 0], P[:, k2] = X.reshape(-1), Y.reshape(-1)
    Ag = age_of_x(X.reshape(-1))
    f = syn.true_drift(P, Ag, np.zeros(len(P)), tc)
    fx, fy = f[:, 0].reshape(NG, NG), f[:, k2].reshape(NG, NG)
    U, curl = ls_potential(fx, fy, xs[1] - xs[0], ys[1] - ys[0])
    lh = syn.true_log_hazard(P, tc).reshape(NG, NG)
    return dict(x=xs, y=ys, U=U, curl_frac=curl, log_hazard=lh, age_x=age_of_x(xs))


@torch.no_grad()
def simulate(D, Z0, A0, end_age, u, K, steps_per_year=4):
    h = end_age - A0.min()
    n = int(h * steps_per_year)
    a1 = np.full(len(A0), end_age)
    _, path = D["m"].sample(T(Z0, dev), T(A0, dev), T(a1, dev), T(np.full((len(A0), 1), u), dev),
                            n_samples=K, n_steps=n, return_path=True)
    ages = A0[None, :] + np.linspace(0, 1, n + 1)[:, None] * (a1 - A0)[None, :]
    return path.cpu().numpy(), ages  # (n+1, K, B, d), (n+1, B)


def simulate_true(D, z0, a0, end_age, u, K, dt=0.05, seed=0):
    r = np.random.default_rng(seed)
    tc = D["tcfg"]
    n = int((end_age - a0) / dt)
    z = np.tile(z0, (K, 1)).astype(np.float64)
    out = [z.copy()]
    for s in range(n):
        a = a0 + s * dt
        z = z + syn.true_drift(z, np.full(K, a), np.full(K, u), tc) * dt + \
            syn.true_diffusion(z, np.full(K, a), tc) * np.sqrt(dt) * r.normal(size=z.shape)
        out.append(z.copy())
    return np.stack(out), a0 + np.arange(n + 1) * dt


def raw_markers(D, Zw, feats):
    """Decode whitened latents to raw clinical units for selected features."""
    ca = D["ca"]
    Xs = decode_np(D["bundle"], Zw)
    nc = len(ca.clinical)
    full = np.full(Xs.shape[:-1] + (len(ca.standardizer.features),), 0.0)
    full[..., :Xs.shape[-1]] = Xs
    raw = ca.standardizer.inverse(full.reshape(-1, full.shape[-1])).reshape(full.shape)
    return {f: raw[..., ca.standardizer.features.index(f)] for f in feats}


out = {}
ON, OFF = load("on"), load("off")
MARKERS = ["crp", "il6", "sbp", "egfr", "hba1c", "glucose", "grip", "walk_speed", "frailty_index"]

for tag, D in (("on", ON), ("off", OFF)):
    lat = D["lat"]
    te = lat[lat.split == "test"]
    Zte, Ate = te[D["zc"]].to_numpy(), te.visit_age.to_numpy()
    f = drift_at(D["m"], Zte, Ate, np.zeros(len(Zte)), dev)
    w = arrow_direction(f)
    proj = Zte @ w
    reg = LinearRegression().fit(proj[:, None], Ate)
    age_of_x = lambda x, o=None: reg.predict(np.asarray(x).reshape(-1, 1) + (0 if o is None else o))
    origin = Zte.mean(0)
    po = origin @ w
    age_of_xrel = lambda x: reg.predict((np.asarray(x) + po).reshape(-1, 1))
    # arrow-of-aging distribution (held-out states): projection of drift on w
    half = len(f) // 2
    w_half = arrow_direction(f[:half])
    out[f"arrow_{tag}"] = {"wf": (f[half:] @ w_half).tolist(),
                           "frac_nonpos": float((f[half:] @ w_half <= 0).mean())}
    # diffusion trace vs age at fixed held-out states
    ages = np.arange(40, 91, 2.5)
    sub = Zte[rng.choice(len(Zte), 1500, replace=False)]
    with torch.no_grad():
        tr = [float((D["m"].diffusion(T(sub, dev), T(np.full(len(sub), a), dev)) ** 2).sum(-1).mean())
              for a in ages]
    true_tr = [float((syn.true_diffusion(np.zeros((1, 8)), np.array([a]), D["tcfg"]) ** 2).sum()) for a in ages]
    out[f"diffusion_{tag}"] = {"ages": ages.tolist(), "model_trace": tr, "true_trace": true_tr}

    if tag == "off":
        continue
    # ---------------- landscapes (ON) ----------------
    e_infl = plane_dir(D, 1, w)
    e_frail = plane_dir(D, 5, w)
    # paths: held-out people aged 40-46, simulate to 90 (u = 0 and u = 1)
    base = te.groupby("person_id").head(1)
    cand = base[(base.visit_age >= 40) & (base.visit_age < 46)]
    pick = cand.sample(10, random_state=3)
    Z0, A0 = pick[D["zc"]].to_numpy(), pick.visit_age.to_numpy()
    path0, pages = simulate(D, Z0, A0, 90.0, 0.0, 96)
    torch.manual_seed(0)
    path1, _ = simulate(D, Z0, A0, 90.0, 1.0, 96)
    allp = path0.reshape(-1, path0.shape[-1]) - origin
    for name, e2, k in (("infl", e_infl, 1), ("frail", e_frail, 5)):
        px, py = allp @ w, allp @ e2
        sy = (Zte - origin) @ e2
        xr = (np.quantile(np.r_[px, (Zte - origin) @ w], 0.005) - 0.3, np.quantile(px, 0.995) + 0.3)
        yr = (min(np.quantile(py, 0.01), np.quantile(sy, 0.01)) - 0.4,
              max(np.quantile(py, 0.99), np.quantile(sy, 0.99)) + 0.4)
        S = learned_surface(D, w, e2, xr, yr, origin, age_of_xrel)
        interp = RegularGridInterpolator((S["x"], S["y"]), S["U"], bounds_error=False, fill_value=None)

        def lift(p):
            q = p - origin
            x, y = q @ w, q @ e2
            return np.stack([x, y, interp(np.stack([np.clip(x, *xr), np.clip(y, *yr)], -1))], -1)

        paths = []
        for b in range(len(A0)):
            mean0 = path0[:, :, b].mean(1)
            mean1 = path1[:, :, b].mean(1)
            paths.append({"age": pages[:, b].round(2).tolist(),
                          "mean_u0": lift(mean0).round(4).tolist(),
                          "mean_u1": lift(mean1).round(4).tolist(),
                          "samples_u0": [lift(path0[:, k, b]).round(4).tolist() for k in range(6)]})
        states = lift(Zte[rng.choice(len(Zte), 600, replace=False)])
        out[f"landscape_learned_{name}"] = {**{k: (v.round(5).tolist() if isinstance(v, np.ndarray) else v)
                                               for k, v in S.items() if k not in ("fx", "fy")},
                                            "paths": paths, "states": states.round(4).tolist(),
                                            "state_proj_y": sy.round(4).tolist()}
        # ground truth, true latent: plane (z0, z_k), other coords at test mean
        tt = D["Zt"][(lat.split == "test").to_numpy()]
        ok = np.isfinite(tt).all(1)
        tt, ta = tt[ok], Ate[ok]
        treg = LinearRegression().fit(tt[:, :1], ta)
        t_age = lambda x: treg.predict(np.asarray(x).reshape(-1, 1))
        ref = tt.mean(0)
        b0 = pick.index[0]
        # true paths from the same people's true baseline states
        tz0 = D["Zt"][pick.index.to_numpy()]
        tpaths = []
        for b in range(len(A0)):
            Ptrue, tages = simulate_true(D, tz0[b], A0[b], 90.0, 0.0, 24, seed=b)
            tpaths.append({"age": tages[::5].round(2).tolist(), "mean": Ptrue.mean(1)[::5][:, [0, k]],
                           "samples": [Ptrue[::5, s][:, [0, k]] for s in range(6)]})
        allt = np.concatenate([p["mean"] for p in tpaths] + [tt[:, [0, k]]])
        txr = (np.quantile(allt[:, 0], 0.005) - 0.2, np.quantile(allt[:, 0], 0.995) + 0.2)
        tyr = (np.quantile(allt[:, 1], 0.005) - 0.3, np.quantile(allt[:, 1], 0.995) + 0.3)
        TS = true_surface(D, k, txr, tyr, ref, t_age)
        tint = RegularGridInterpolator((TS["x"], TS["y"]), TS["U"], bounds_error=False, fill_value=None)
        tl3 = lambda q: np.c_[q, tint(np.c_[np.clip(q[:, 0], *txr), np.clip(q[:, 1], *tyr)])]
        out[f"landscape_true_{name}"] = {**{kk: (v.round(5).tolist() if isinstance(v, np.ndarray) else v)
                                            for kk, v in TS.items()},
                                         "paths": [{"age": p["age"], "mean": tl3(p["mean"]).round(4).tolist(),
                                                    "samples": [tl3(s).round(4).tolist() for s in p["samples"]]}
                                                   for p in tpaths],
                                         "state_proj_y": tt[:, k].round(4).tolist()}
    # milestones along person 0's mean path (decoded raw markers, hazard)
    b = 0
    mk_ages = [A0[b], 50, 60, 65, 70, 75, 80, 85, 90]
    ms = []
    for u, P in ((0, path0), (1, path1)):
        mp = P[:, :, b].mean(1)
        idx = [int(np.argmin(np.abs(pages[:, b] - a))) for a in mk_ages]
        Zm = mp[idx]
        rawm = raw_markers(D, Zm, MARKERS)
        lh = log_hazard_np(D["bundle"], Zm)
        # sample spread of hazard
        lhs = log_hazard_np(D["bundle"], P[idx, :, b])
        ms.append({"u": u, "ages": [round(float(pages[i, b]), 1) for i in idx],
                   "markers": {k: v.round(3).tolist() for k, v in rawm.items()},
                   "annual_death_prob": (1 - np.exp(-np.exp(lh))).round(4).tolist(),
                   "annual_death_prob_q10_q90": [np.quantile(1 - np.exp(-np.exp(lhs)), q, axis=1).round(4).tolist()
                                                 for q in (0.1, 0.9)]})
    out["milestones"] = ms
    # ---------------- single-patient forecast fan ----------------
    lt = te.copy()
    nvis = lt.groupby("person_id").size()
    pid = nvis[nvis >= 4].index
    first = lt[lt.person_id.isin(pid)].groupby("person_id").head(1)
    first = first[(first.visit_age > 45) & (first.visit_age < 55)]
    pp = first.person_id.iloc[0] if len(first) else pid[0]
    rows = lt[lt.person_id == pp]
    z0, a0 = rows[D["zc"]].to_numpy()[:1], rows.visit_age.to_numpy()[:1]
    P, ages_f = simulate(D, z0, a0, float(rows.visit_age.max()) + 2.0, 0.0, 200)
    fan = {}
    feats = ["crp", "sbp", "hba1c", "egfr", "grip"]
    rawS = raw_markers(D, P[:, :, 0], feats)
    ca = D["ca"]
    vis_idx = rows.index.to_numpy()
    obs_raw = ca.visits.X[vis_idx]
    for fn in feats:
        q = np.quantile(rawS[fn], [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
        j = ca.visits.features.index(fn)
        fan[fn] = {"q": q.round(3).tolist(), "obs": [None if not np.isfinite(v) else round(float(v), 3)
                                                     for v in obs_raw[:, j]]}
    tz = D["Zt"][vis_idx]
    out["fan"] = {"person": str(pp), "ages": ages_f[:, 0].round(2).tolist(),
                  "visit_ages": rows.visit_age.round(2).tolist(), "features": fan,
                  "true_infl_latent": tz[:, 1].round(3).tolist()}

# ---------------- numbers from results.json (A only) ----------------
summ = {}
for cohort in ("on", "off"):
    for sd in sorted((Path(args.runs) / cohort).glob("seed*")):
        r = json.loads((sd / "results.json").read_text())
        h = r["hallmarks"]["A"]
        summ[f"{cohort}/{sd.name}"] = {
            "hallmarks": {k: {kk: vv for kk, vv in v.items()
                              if kk not in ("basis", "age_grid", "curve", "raw_age", "raw_curve")}
                          for k, v in h.items()},
            "inflammaging_curve": {k: h["inflammaging"].get(k) for k in ("age_grid", "curve", "raw_age", "raw_curve")},
            "attribution": r["hallmarks"]["attribution_encoder_hazard"],
            "forecast": {k: {kk: vv for kk, vv in v.items() if kk != "mae_per_feature"}
                         for k, v in r["forecast"].items() if k in ("A", "base_locf", "base_linear_drift",
                                                                    "base_direct_gbm", "base_direct_mlp")},
            "mae_per_feature_A": r["forecast"]["A"]["mae_per_feature"],
            "survival": {k: v for k, v in r["survival"].items() if k in ("A", "encoder_static",
                                                                          "cox_baseline_features", "cox_age_only")},
            "ground_truth_A": r.get("ground_truth", {}).get("A"),
            "affine_r2": r.get("ground_truth", {}).get("affine_r2_per_true_dim"),
            "oracle_A": (r.get("oracle") or {}).get("A"),
            "oracle_true_diffusion": (r.get("oracle") or {}).get("true_diffusion_by_age"),
            "costs_A": r["costs"]["A"], "n_pairs": r["n_pairs"],
            "obs_noise_var": r["latent_obs_noise_var"],
            "controls": {k: r["hallmarks"].get(k) for k in ("A_constSigma", "A_noPotential", "A_shuffledAge")
                         if k in r["hallmarks"]},
        }
out["results"] = summ
truth_h = {}
for cohort in ("on", "off"):
    f = Path(args.runs) / cohort / "truth_hallmarks.json"
    if f.exists():
        truth_h[cohort] = json.loads(f.read_text())
out["truth_hallmarks"] = {c: {k: {kk: vv for kk, vv in v.items() if kk not in ("basis", "age_grid", "curve")}
                              for k, v in h.items()} for c, h in truth_h.items()}
out["clinical_features"] = ON["ca"].clinical
out["cohort"] = {"n_persons": int(ON["lat"].person_id.nunique()), "n_visits": int(len(ON["lat"])),
                 "events_on": int(ON["ca"].outcomes.event.sum()), "events_off": int(OFF["ca"].outcomes.event.sum())}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
def finite(o):
    """Browsers reject NaN/Infinity in JSON: write them as null."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, list):
        return [finite(x) for x in o]
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    return o


Path(args.out).write_text(json.dumps(finite(jsonable(out)), allow_nan=False))
print("wrote", args.out, round(Path(args.out).stat().st_size / 1e6, 2), "MB")
for k in out:
    if k.startswith("landscape"):
        print(k, "curl_frac", round(out[k]["curl_frac"], 3))
