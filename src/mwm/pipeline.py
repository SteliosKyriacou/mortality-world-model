"""End-to-end experiment pipeline: encoder -> latents -> A / B / B' / snapshot / baselines ->
forecast + survival metrics -> ground-truth recovery -> hallmark interrogation.

Used by scripts/run_synthetic_validation.py; written cohort-agnostic so real cohorts can
reuse it (ground-truth parts are skipped when no truth is available)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.dataset import CohortArrays, load_cohort_dir, make_pairs
from .dynamics.flow import FlowModel, SnapshotOTFlow, StochasticFlow
from .dynamics.sde import NeuralSDE
from .dynamics.train import fit_bprime, pairs_to_tensors, train_pairs, train_snapshot
from .encoders.train import export_latents, train_encoder
from .eval import baselines as bl
from .eval.metrics import (crps_samples, energy_score_np, gaussian_samples, interval_coverage,
                           latent_mse, masked_mae_rmse, survival_metrics)
from .interrogate.attractors import test_attractors
from .interrogate.attribution import test_attribution
from .interrogate.common import LatentAgeProbe, T, decode_np, log_hazard_np
from .interrogate.dispersion import observed_dispersion, test_dispersion
from .interrogate.inflammaging import test_inflammaging
from .interrogate.intervention import test_intervention
from .interrogate.irreversibility import test_irreversibility
from .interrogate.jacobian import test_jacobian_enrichment

DT_BINS = [(0, 3), (3, 6), (6, 9), (9, 30)]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------------- models
def build_models(d, cfg):
    a, b = cfg["model_a"], cfg["model_b"]
    return {
        "A": NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"]),
        "B": FlowModel(d, hidden=b["hidden"], jac_penalty=b.get("jac_penalty", 0.1)),
        "Bp": StochasticFlow(d, hidden=b["hidden"], jac_penalty=b.get("jac_penalty", 0.1),
                             dsm_sigma=cfg["model_bprime"].get("dsm_sigma", 0.5)),
    }


def fit_model(name, m, P, states, cfg, log, device, seed):
    a, b = cfg["model_a"], cfg["model_b"]
    if isinstance(m, NeuralSDE):
        return train_pairs(m, P["train"], P["val"], epochs=a["epochs"], batch=a["batch"], lr=a["lr"],
                           device=device, log=log, loss_kw={"K": a["K"]}, patience=25, seed=seed)
    r = train_pairs(m, P["train"], P["val"], epochs=b["epochs"], batch=b["batch"], lr=b["lr"],
                    device=device, log=log, val_every=10, patience=60, seed=seed)
    if isinstance(m, StochasticFlow):
        bp = cfg["model_bprime"]
        r2 = fit_bprime(m, P["train"], P["val"], *states, dsm_iters=bp["dsm_iters"],
                        diff_epochs=bp["diff_epochs"], device=device, log=log, seed=seed,
                        K=a["K"], n_steps=a["n_steps"])
        r = {**r, "stage2": r2, "train_seconds": r["train_seconds"] + r2["train_seconds"],
             "peak_vram_mb": max(r["peak_vram_mb"], r2["peak_vram_mb"]),
             "train_nfe": r.get("train_nfe", 0) + r2.get("train_nfe", 0)}
    return r


# ---------------------------------------------------------------------------------- eval
@torch.no_grad()
def predict_latent(m, p, K, device, n_steps=40):
    z0, z1, a0, a1, u = pairs_to_tensors(p, device)
    S = m.sample(z0, a0, a1, u, n_samples=K, n_steps=n_steps)
    return S.cpu().numpy()


def nfe_10y(m, d, device):
    m.nfe = 0
    z = torch.zeros(64, d, device=device)
    a = torch.full((64,), 60.0, device=device)
    m.sample(z, a, a + 10, torch.zeros(64, 1, device=device), n_samples=1, n_steps=40)
    return m.nfe


def forecast_eval(name, S_lat, p, bundle, Xs, nc, sigma_x, rng, extra=None):
    """S_lat: (K, N, d) latent samples at a1 for pairs p."""
    z1 = p["z1"]
    out = {"latent_mse": latent_mse(S_lat.mean(0), z1),
           "latent_energy": float(energy_score_np(S_lat[:16], z1).mean()) if S_lat.shape[0] > 1
           else float(np.linalg.norm(S_lat[0] - z1, axis=1).mean())}
    X1 = Xs[p["i1"], :nc]
    Xd = decode_np(bundle, S_lat)[..., :nc]           # (K, N, Fc)
    pt = masked_mae_rmse(Xd.mean(0), X1)
    Kx = max(S_lat.shape[0], 32)
    Xsamp = np.concatenate([Xd] * int(np.ceil(Kx / Xd.shape[0])))[:Kx]
    Xsamp = Xsamp + sigma_x[None, None, :nc] * rng.normal(size=Xsamp.shape)
    crps = crps_samples(Xsamp, X1)
    out.update({"mae": pt["mae"], "rmse": pt["rmse"], "crps": float(np.nanmean(crps)),
                **interval_coverage(Xsamp, X1), "mae_per_feature": pt["mae_per_feature"]})
    dt = p["a1"] - p["a0"]
    out["mae_vs_dt"] = {f"{lo}-{hi}": float(np.nanmean(np.abs(Xd.mean(0) - X1)[(dt >= lo) & (dt < hi)]))
                        for lo, hi in DT_BINS if ((dt >= lo) & (dt < hi)).sum() > 5}
    return out


def baseline_eval(name, mu, sd, p, Xs, nc, rng, latent=False):
    if latent:
        S = gaussian_samples(mu, sd, 32, rng)
        return {"latent_mse": latent_mse(mu, p["z1"]),
                "latent_energy": float(energy_score_np(S[:16], p["z1"]).mean())}
    X1 = Xs[p["i1"], :nc]
    S = gaussian_samples(mu, sd, 32, rng)
    pt = masked_mae_rmse(mu, X1)
    dt = p["a1"] - p["a0"]
    return {"mae": pt["mae"], "rmse": pt["rmse"], "crps": float(np.nanmean(crps_samples(S, X1))),
            **interval_coverage(S, X1), "mae_per_feature": pt["mae_per_feature"],
            "mae_vs_dt": {f"{lo}-{hi}": float(np.nanmean(np.abs(mu - X1)[(dt >= lo) & (dt < hi)]))
                          for lo, hi in DT_BINS if ((dt >= lo) & (dt < hi)).sum() > 5}}


@torch.no_grad()
def model_survival(m, bundle, z0, a0, horizon, device, K=32, steps_per_year=4):
    grid_n = int(horizon * steps_per_year)
    zz, aa = T(z0, device), T(a0, device)
    u = torch.zeros(len(z0), 1, device=device)
    surv = []
    for s in range(0, len(z0), 1024):
        _, path = m.sample(zz[s:s + 1024], aa[s:s + 1024], aa[s:s + 1024] + horizon, u[s:s + 1024],
                           n_samples=K, n_steps=grid_n, return_path=True)   # (T+1, K, B, d)
        lh = log_hazard_np(bundle, path.cpu().numpy())                       # (T+1, K, B)
        lam = np.exp(lh)
        cum = np.concatenate([np.zeros((1,) + lam.shape[1:]),
                              np.cumsum(0.5 * (lam[1:] + lam[:-1]) / steps_per_year, 0)])
        surv.append(np.exp(-cum).mean(1).T)                                   # (B, T+1)
    return np.linspace(0, horizon, grid_n + 1), np.concatenate(surv)


def survival_eval(models, bundle, ca, lat, zc, cfg, device, train_cox_X, rng):
    """Baseline visit of each person -> time to death/censor from baseline."""
    base = lat.groupby("person_id").head(1)
    o = ca.outcomes.reindex(base["person_id"])
    time_ = (o["age_at_death_or_censor"].to_numpy() - base["visit_age"].to_numpy()).clip(1e-3)
    ev = o["event"].to_numpy()
    sp = base["split"].to_numpy()
    tr, te = sp == "train", sp == "test"
    nc = len(ca.clinical)
    Xb = ca.Xs[base.index.to_numpy(), :nc]
    Z = base[zc].to_numpy()
    A = base["visit_age"].to_numpy()
    H = cfg["eval"]["survival_horizon"]
    res = {}
    grid = np.linspace(0, H, int(H * 4) + 1)
    # static encoder hazard (z frozen at baseline)
    lh = log_hazard_np(bundle, Z[te])
    S_static = np.exp(-np.exp(lh)[:, None] * grid[None])
    res["encoder_static"] = survival_metrics(time_[te], ev[te], lh, time_[tr], ev[tr], S_static, grid)
    for name, m in models.items():
        g, S = model_survival(m, bundle, Z[te], A[te], H, device,
                              K=cfg["eval"]["rollout_samples"] if m.stochastic else 1)
        risk = 1 - S[:, np.searchsorted(g, 10.0)]
        res[name] = survival_metrics(time_[te], ev[te], risk, time_[tr], ev[tr], S, g)
    cox = bl.CoxBaseline().fit(Xb[tr], A[tr], time_[tr], ev[tr])
    res["cox_baseline_features"] = survival_metrics(time_[te], ev[te], cox.risk(Xb[te], A[te]),
                                                    time_[tr], ev[tr], cox.survival(Xb[te], A[te], grid), grid)
    cox_age = bl.CoxBaseline().fit(np.zeros((tr.sum(), 0)), A[tr], time_[tr], ev[tr])
    res["cox_age_only"] = survival_metrics(time_[te], ev[te], cox_age.risk(np.zeros((te.sum(), 0)), A[te]),
                                           time_[tr], ev[tr])
    return res, cox


# ---------------------------------------------------------------------------------- run
def run_cohort(cohort_dir, out_dir, cfg, seed, log, truth=None, controls=False, oracle=False):
    """truth: (cfg_true, meta, truth_latents) for synthetic cohorts, else None."""
    device = cfg["device"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    results = {"seed": seed, "cohort_dir": str(cohort_dir)}
    ca = load_cohort_dir(cohort_dir)
    nc = len(ca.clinical)
    e = cfg["encoder"]
    log(f"== {cohort_dir} seed {seed}: visits {len(ca.Xs)} features {len(ca.features)}")
    bundle = train_encoder(ca, d_latent=e["d_latent"], kind=e["kind"], epochs=e["epochs"], seed=seed,
                           device=device, log=log, d_model=e["d_model"], n_layers=e["n_layers"])
    lat = export_latents(bundle, ca, out, device)
    results["encoder"] = {"train_seconds": bundle.train_seconds, "final": bundle.history[-1]}
    zc = [c for c in lat.columns if c.startswith("z_")]
    d = len(zc)
    lat_sorted = lat.sort_values(["person_id", "visit_age"])
    assert (lat_sorted.index.to_numpy() == np.arange(len(lat))).all(), "visit order mismatch"
    P = {s: make_pairs(lat[lat.split == s].reset_index(), zc, "consecutive") for s in ("train", "val")}
    for s in P:  # map indices back to global rows
        sub = lat[lat.split == s].reset_index()
        P[s]["i0"], P[s]["i1"] = sub["index"].to_numpy()[P[s]["i0"]], sub["index"].to_numpy()[P[s]["i1"]]
    sub = lat[lat.split == "test"].reset_index()
    Pt = make_pairs(sub, zc, "first_last")
    Pt["i0"], Pt["i1"] = sub["index"].to_numpy()[Pt["i0"]], sub["index"].to_numpy()[Pt["i1"]]
    results["n_pairs"] = {"train": len(P["train"]["a0"]), "val": len(P["val"]["a0"]), "test_first_last": len(Pt["a0"])}
    trs = lat[lat.split == "train"]
    states = (trs[zc].to_numpy(), trs.visit_age.to_numpy(), trs[["u"]].to_numpy())

    obs_var = json.loads((out / "encoder_meta.json").read_text())["latent_obs_noise_var"]
    results["latent_obs_noise_var"] = obs_var
    use_fixed = cfg.get("obs_noise", "fixed") == "fixed"

    # ---- models
    models = build_models(d, cfg)
    if use_fixed:
        for m in models.values():
            if m.stochastic:
                m.to(device).set_obs_noise(obs_var)
    costs = {}
    for name, m in models.items():
        log(f"-- training {name}")
        r = fit_model(name, m, P, states, cfg, log, device, seed)
        costs[name] = {k: v for k, v in r.items() if k not in ("history",)}
        costs[name]["nfe_10y_rollout"] = nfe_10y(m, d, device)
        torch.save(m.state_dict(), out / f"model_{name}.pt")
    # adaptive-solver NFE for B (information only)
    mB = models["B"]
    mB.nfe = 0
    z = torch.zeros(64, d, device=device)
    a = torch.full((64,), 60.0, device=device)
    mB.odeint(z, a, a + 10, torch.zeros(64, 1, device=device), method="dopri5")
    costs["B"]["nfe_10y_dopri5"] = mB.nfe
    # snapshot OT-CFM on baseline (cross-sectional) visits
    snap = SnapshotOTFlow(d, hidden=cfg["model_b"]["hidden"])
    base_tr = trs.groupby("person_id").head(1)
    costs["snapshot"] = train_snapshot(snap, base_tr[zc].to_numpy(), base_tr.visit_age.to_numpy(),
                                       base_tr[["u"]].to_numpy(), iters=cfg["snapshot"]["iters"],
                                       device=device, seed=seed, log=log)
    results["costs"] = costs

    # ---- forecasting on held-out people (first -> last visit)
    sigma_x = bundle.feature_sigma.detach().cpu().numpy()
    fc = {}
    for name, m in {**models, "snapshot": snap}.items():
        S = predict_latent(m, Pt, cfg["eval"]["rollout_samples"], device)
        fc[name] = forecast_eval(name, S, Pt, bundle, ca.Xs, nc, sigma_x, rng)
    # feature-space baselines
    X0 = ca.Xs[:, :nc].copy()
    Xdec0 = decode_np(bundle, lat[zc].to_numpy())[:, :nc]
    X0f = np.where(np.isnan(X0), Xdec0, X0)
    U = ca.u[:, 0]

    trP = P["train"]
    dt_tr = trP["a1"] - trP["a0"]
    dt_te = Pt["a1"] - Pt["a0"]
    for B in (bl.LOCF(), bl.LinearDrift(), bl.DirectMLP(device=device, seed=seed), bl.DirectGBM(seed=seed)):
        x0tr = X0f[trP["i0"]] if not isinstance(B, bl.DirectGBM) else X0[trP["i0"]]
        x0te = X0f[Pt["i0"]] if not isinstance(B, bl.DirectGBM) else X0[Pt["i0"]]
        B.fit(x0tr, ca.Xs[trP["i1"], :nc], trP["a0"], dt_tr, U[trP["i0"]])
        mu, sd = B.predict(x0te, Pt["a0"], dt_te, U[Pt["i0"]])
        fc[f"base_{B.name}"] = baseline_eval(B.name, mu, sd, Pt, ca.Xs, nc, rng)
    # latent-space baselines
    for B in (bl.LOCF(), bl.LinearDrift(), bl.DirectMLP(device=device, seed=seed)):
        B.fit(trP["z0"], trP["z1"], trP["a0"], dt_tr, trP["u"])
        mu, sd = B.predict(Pt["z0"], Pt["a0"], dt_te, Pt["u"])
        fc[f"latent_{B.name}"] = baseline_eval(B.name, mu, sd, Pt, ca.Xs, nc, rng, latent=True)
    results["forecast"] = fc
    log("forecast: " + ", ".join(f"{k}: mae={v.get('mae', float('nan')):.3f} lmse={v.get('latent_mse', float('nan')):.3f}"
                                 for k, v in fc.items()))

    # ---- survival
    surv, cox = survival_eval({"A": models["A"], "B": models["B"], "Bp": models["Bp"]}, bundle, ca, lat, zc,
                              cfg, device, None, rng)
    results["survival"] = surv
    log("survival: " + ", ".join(f"{k}: C={v['c_harrell']:.3f}" for k, v in surv.items()))

    # ---- control / ablation models (null models for hallmark tests)
    ctrl = {}
    if controls:
        a = cfg["model_a"]
        ctrl_specs = {}
        if cfg["controls"].get("constant_sigma"):
            ctrl_specs["A_constSigma"] = (NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"],
                                                    diffusion_mode="constant"), P)
        if cfg["controls"].get("no_potential"):
            ctrl_specs["A_noPotential"] = (NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"],
                                                     use_potential=False), P)
        if cfg["controls"].get("shuffled_age"):
            Ps = {k: dict(v) for k, v in P.items()}
            for k in Ps:
                perm = rng.permutation(len(Ps[k]["a0"]))
                shift = Ps[k]["a0"][perm] - Ps[k]["a0"]
                Ps[k]["a0"], Ps[k]["a1"] = Ps[k]["a0"] + shift, Ps[k]["a1"] + shift
            ctrl_specs["A_shuffledAge"] = (NeuralSDE(d, hidden=a["hidden"], solver=a["solver"],
                                                     n_steps=a["n_steps"]), Ps)
        for name, (m, PP) in ctrl_specs.items():
            if use_fixed:
                m.to(device).set_obs_noise(obs_var)
            log(f"-- training control {name}")
            r = fit_model(name, m, PP, states, cfg, log, device, seed)
            ctrl[name] = m
            torch.save(m.state_dict(), out / f"model_{name}.pt")
            costs[name] = {k: v for k, v in r.items() if k != "history"}
        if "A_constSigma" in ctrl:
            results["constSigma_vs_ageSigma_val_nll"] = {
                "A": models["A"].val_loss(pairs_to_tensors(P["val"], device), K=64),
                "A_constSigma": ctrl["A_constSigma"].val_loss(pairs_to_tensors(P["val"], device), K=64)}

    # ---- ground truth recovery
    if truth is not None:
        from .eval.groundtruth import drift_recovery, fit_affine, affine_r2
        tcfg, meta, tl = truth
        key = pd.MultiIndex.from_arrays([lat.person_id, lat.visit_age.round(3)])
        Zt = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
        trm = (lat.split == "train").to_numpy()
        tem = (lat.split == "test").to_numpy()
        W, b0 = fit_affine(lat[zc].to_numpy()[trm], Zt[trm])
        r2 = affine_r2(lat[zc].to_numpy()[tem], Zt[tem], W, b0)
        gt = {"affine_r2_per_true_dim": r2.round(3).tolist()}
        tz = lat[tem]
        for name, m in {**models, "snapshot": snap, **ctrl}.items():
            gt[name] = drift_recovery(m, tz[zc].to_numpy(), tz.visit_age.to_numpy(), tz[["u"]].to_numpy(),
                                      Zt[tem], tcfg, W=W, device=device)
            # intervention effect on the nutrient latent (true: -kappa*beta2 if on)
            Zx, Ax = tz[zc].to_numpy(), tz.visit_age.to_numpy()
            from .interrogate.common import drift_at
            dv = (drift_at(m, Zx, Ax, np.ones(len(Zx)), device) - drift_at(m, Zx, Ax, np.zeros(len(Zx)), device)) @ W
            gt[name]["u_effect_on_true_dims"] = dv.mean(0).round(4).tolist()
        results["ground_truth"] = gt
        if oracle:
            results["oracle"] = run_oracle(tl, lat, cfg, tcfg, device, log, seed)

    # ---- hallmark interrogation on held-out people
    meta = truth[1] if truth is not None else None
    hm, probe_r2 = run_hallmarks({**models, **ctrl}, bundle, ca, lat, zc, meta, cox, device, log)
    results["latent_age_probe_r2_test"] = probe_r2
    results["hallmarks"] = hm
    (out / "results.json").write_text(json.dumps(jsonable(results), indent=1))
    log(f"saved {out/'results.json'}")
    return results


def run_hallmarks(all_models, bundle, ca, lat, zc, meta, cox, device, log=print):
    """All PLAN §7 tests on held-out people for every model (+ encoder attribution)."""
    nc = len(ca.clinical)
    trs = lat[lat.split == "train"]
    te_lat = lat[lat.split == "test"]
    Zte, Ate = te_lat[zc].to_numpy(), te_lat.visit_age.to_numpy()
    Xte = ca.Xs[te_lat.index.to_numpy()]
    probe = LatentAgeProbe().fit(trs[zc].to_numpy(), trs.visit_age.to_numpy())
    probe_r2 = float(1 - ((probe(Zte) - Ate) ** 2).mean() / Ate.var())
    infl = [ca.clinical.index(f) for f in (meta["inflammatory_features"] if meta else []) if f in ca.clinical]
    metab = [ca.clinical.index(f) for f in (meta["metabolic_features"] if meta else []) if f in ca.clinical]
    base_te = te_lat.groupby("person_id").head(1)
    ev_te = ca.outcomes.reindex(base_te.person_id)["event"].to_numpy()
    Pte_cons = make_pairs(te_lat.reset_index(drop=True), zc, "consecutive")
    hm = {}
    for name, m in all_models.items():
        log(f"-- interrogating {name}")
        h = {}
        h["inflammaging"] = test_inflammaging(m, bundle, Zte, Ate, infl, Xte, Ate, device)
        h["dispersion"] = test_dispersion(m, Zte, Ate, None, device)
        h["irreversibility"] = test_irreversibility(m, probe, Zte, Ate, Pte_cons, device)
        h["intervention"] = test_intervention(m, bundle, Zte, Ate, metab, slice(0, nc), device)
        if meta is not None and ca.omics:
            h["jacobian_enrichment"] = test_jacobian_enrichment(
                m, bundle, Zte, Ate, ca.omics, slice(nc, nc + len(ca.omics)), meta["pathways"],
                meta["aging_pathways"], device=device)
        h["attractors"] = test_attractors(m, bundle, base_te[zc].to_numpy(), base_te.visit_age.to_numpy(),
                                          ev_te, device=device)
        hm[name] = h
    hm["observed_dispersion_test"] = observed_dispersion(Zte, Ate)
    cox_attr = cox.attributions(Xte[:, :nc], Ate)
    hm["attribution_encoder_hazard"] = test_attribution(bundle, Xte, Ate, infl, nc, cox_attr, device)
    return hm, probe_r2


def run_oracle(tl, lat, cfg, tcfg, device, log, seed):
    """Train A and B' directly on the true latent (no encoder) -> clean recovery check."""
    from .eval.groundtruth import drift_recovery
    tl = tl.copy()
    sp = lat.drop_duplicates("person_id").set_index("person_id")["split"]
    tl["split"] = sp.reindex(tl.person_id).to_numpy()
    zc = [f"ztrue_{k}" for k in range(8)]
    P = {s: make_pairs(tl[tl.split == s], zc) for s in ("train", "val")}
    trs = tl[tl.split == "train"]
    states = (trs[zc].to_numpy(), trs.visit_age.to_numpy(), trs[["u"]].to_numpy())
    te = tl[tl.split == "test"]
    a, b = cfg["model_a"], cfg["model_b"]
    ms = {"A": NeuralSDE(8, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"], learn_obs_noise=False),
          "B": FlowModel(8, hidden=b["hidden"], jac_penalty=b.get("jac_penalty", 0.1)),
          "Bp": StochasticFlow(8, hidden=b["hidden"], jac_penalty=b.get("jac_penalty", 0.1),
                               dsm_sigma=cfg["model_bprime"].get("dsm_sigma_oracle", 0.2)),
          "Bp_noScore": StochasticFlow(8, hidden=b["hidden"], score_correction=False,
                                       jac_penalty=b.get("jac_penalty", 0.1))}
    out = {}
    for name, m in ms.items():
        log(f"-- oracle {name}")
        fit_model(name, m, P, states, cfg, log, device, seed)
        out[name] = drift_recovery(m, te[zc].to_numpy(), te.visit_age.to_numpy(), te[["u"]].to_numpy(),
                                   te[zc].to_numpy(), tcfg, device=device)
        from .interrogate.common import drift_at
        Zx, Ax = te[zc].to_numpy(), te.visit_age.to_numpy()
        out[name]["u_effect_on_true_dims"] = (drift_at(m, Zx, Ax, np.ones(len(Zx)), device) -
                                              drift_at(m, Zx, Ax, np.zeros(len(Zx)), device)).mean(0).round(4).tolist()
        if m.stochastic:
            S = m.diffusion(T(Zx, device), T(Ax, device)).detach().cpu().numpy()
            out[name]["diffusion_by_age"] = {
                str(lo): S[(Ax >= lo) & (Ax < lo + 10)].mean(0).round(3).tolist() for lo in (40, 50, 60, 70, 80)}
    import mwm.data.synthetic as syn
    out["true_diffusion_by_age"] = {str(lo): syn.true_diffusion(np.zeros((1, 8)), np.array([lo + 5.0]), tcfg)[0].round(3).tolist()
                                    for lo in (40, 50, 60, 70, 80)}
    return out


def run_truth_hallmarks(cohort_dir, out_path, device="cuda", log=print):
    """Run every hallmark test on the TRUE dynamics in the TRUE latent (test persons)."""
    from .data.common import load_splits
    from .data.synthetic import load_truth
    from .eval.truth_model import TrueBundle, TrueDynamics
    tcfg, meta, tl, L = load_truth(cohort_dir)
    sp = load_splits("synth", cohort_dir).set_index("person_id")["split"]
    tl = tl.copy()
    tl["split"] = sp.reindex(tl.person_id).to_numpy()
    zc = [f"ztrue_{k}" for k in range(8)]
    model = TrueDynamics(tcfg).to(device)
    bundle = TrueBundle(tcfg, L).to(device)
    feats = meta["clinical_features"]
    nc = len(feats)
    infl = [feats.index(f) for f in meta["inflammatory_features"]]
    metab = [feats.index(f) for f in meta["metabolic_features"]]
    te, trs = tl[tl.split == "test"], tl[tl.split == "train"]
    Zte, Ate = te[zc].to_numpy(), te.visit_age.to_numpy()
    probe = LatentAgeProbe().fit(trs[zc].to_numpy(), trs.visit_age.to_numpy())
    import pandas as pd
    outc = pd.read_parquet(Path(cohort_dir) / "outcomes.parquet").set_index("person_id")
    base = te.groupby("person_id").head(1)
    ev = outc.reindex(base.person_id)["event"].to_numpy()
    h = {"inflammaging": test_inflammaging(model, bundle, Zte, Ate, infl, None, None, device),
         "dispersion": test_dispersion(model, Zte, Ate, None, device),
         "irreversibility": test_irreversibility(model, probe, Zte, Ate, None, device),
         "intervention": test_intervention(model, bundle, Zte, Ate, metab, slice(0, nc), device),
         "jacobian_enrichment": test_jacobian_enrichment(model, bundle, Zte, Ate, meta["genes"],
                                                         slice(nc, nc + len(meta["genes"])), meta["pathways"],
                                                         meta["aging_pathways"], device=device),
         "attractors": test_attractors(model, bundle, base[zc].to_numpy(), base.visit_age.to_numpy(), ev,
                                       device=device)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(jsonable(h), indent=1))
    log(f"truth hallmarks: " + ", ".join(f"{k}={v.get('detected', v.get('detected_flag'))}" for k, v in h.items()))
    return h
