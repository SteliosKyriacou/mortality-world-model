"""Train Model A2 (second-order, age-free) and the first-order age-free Model A on the same frozen
encoder, then test momentum: forecast each held-out person's 3rd visit from their first two.

    conda run -n mwm python scripts/train_model_a2.py --cohort momentum
    conda run -n mwm python scripts/train_model_a2.py --cohort null --encoder-from runs/momentum/null/a2_seed0
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mwm.data.dataset import load_cohort_dir, make_pairs
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.dynamics.sde2 import SecondOrderSDE
from mwm.dynamics.train import pairs_to_tensors, train_pairs
from mwm.encoders.train import EncoderBundle, export_latents, train_encoder
from mwm.interrogate.common import LatentAgeProbe
from mwm.pipeline import Logger, forecast_eval, jsonable, run_hallmarks, survival_eval

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data/synthetic_momentum")
ap.add_argument("--cohort", default="momentum")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--enc-epochs", type=int, default=40)
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--encoder-from", default=None)
ap.add_argument("--out", default="runs/momentum")
ap.add_argument("--name", default=None)
ap.add_argument("--no-hallmarks", action="store_true")
ap.add_argument("--oracle", action="store_true",
                help="train on the TRUE hidden states (no encoder, tiny observation noise): can the model learn momentum at all?")
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)
cdir = Path(args.data_root) / args.cohort
out = Path(args.out) / args.cohort / (args.name or (f"a2_oracle_seed{args.seed}" if args.oracle else f"a2_seed{args.seed}"))
out.mkdir(parents=True, exist_ok=True)
log = Logger(out / "log.txt")
tcfg, meta, tl, _ = load_truth(cdir)
ca = load_cohort_dir(cdir)
nc = len(ca.clinical)
log(f"== {cdir} seed {args.seed} momentum={tcfg.momentum} null={tcfg.momentum_null}")

# ---------------- encoder (trained here or reused) ----------------
if args.oracle:
    vis = pd.DataFrame({"person_id": ca.visits.person_id, "visit_age": ca.visits.visit_age, "split": ca.split, "u": ca.u[:, 0]})
    vis["visit_age"] = vis.visit_age.round(3)
    tt = tl.copy()
    tt["visit_age"] = tt.visit_age.round(3)
    lat = vis.merge(tt[["person_id", "visit_age"] + [f"ztrue_{k}" for k in range(8)]], on=["person_id", "visit_age"], how="inner")
    lat = lat.rename(columns={f"ztrue_{k}": f"z_{k}" for k in range(8)}).sort_values(["person_id", "visit_age"]).reset_index(drop=True)
    (out / "encoder_meta.json").write_text(json.dumps({"latent_obs_noise_var": [1e-4] * 8, "oracle": True}))
    lat.to_parquet(out / "latents.parquet", index=False)
    bundle = None
    log(f"ORACLE: training on true hidden states ({len(lat)} visits)")
elif args.encoder_from:
    import shutil
    src = Path(args.encoder_from)
    for f in ("encoder.pt", "encoder_meta.json", "latents.parquet"):
        shutil.copy(src / f, out / f)
    bundle = EncoderBundle.load(out / "encoder.pt").to(dev)
    lat = pd.read_parquet(out / "latents.parquet")
else:
    bundle = train_encoder(ca, d_latent=16, kind="transformer", epochs=args.enc_epochs, seed=args.seed, device=dev,
                           log=log, d_model=64, n_layers=2)
    lat = export_latents(bundle, ca, out, dev)
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)
obs_var = json.loads((out / "encoder_meta.json").read_text())["latent_obs_noise_var"]
sigma_x = bundle.feature_sigma.detach().cpu().numpy() if bundle is not None else None
lat = lat.reset_index(drop=True)
assert (lat.sort_values(["person_id", "visit_age"]).index.to_numpy() == np.arange(len(lat))).all()
Z, A, U = lat[zc].to_numpy(np.float32), lat.visit_age.to_numpy(np.float32), lat["u"].to_numpy(np.float32)
pid = lat.person_id.to_numpy()
same_next = np.r_[pid[1:] == pid[:-1], False]
same_prev = np.r_[False, pid[1:] == pid[:-1]]


def pairs_with_history(split):
    i0 = np.nonzero(same_next & (lat.split.to_numpy() == split))[0]
    i1 = i0 + 1
    hp = same_prev[i0]
    ip = np.where(hp, i0 - 1, i0)
    return {"z0": Z[i0], "z1": Z[i1], "a0": A[i0], "a1": A[i1], "u": U[i0, None], "zp": Z[ip],
            "dtp": np.where(hp, A[i0] - A[ip], 1.0).astype(np.float32), "hp": hp, "i0": i0, "i1": i1}


def tens(p):
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    return (t(p["z0"]), t(p["z1"]), t(p["a0"]), t(p["a1"]), t(p["u"]).reshape(len(p["a0"]), -1),
            t(p["zp"]), t(p["dtp"]), t(p["hp"], torch.bool))


P = {s: pairs_with_history(s) for s in ("train", "val")}
log(f"pairs: train {len(P['train']['a0'])} (with history {P['train']['hp'].mean():.0%}), val {len(P['val']['a0'])}")

# ---------------- Model A2 ----------------
m2 = SecondOrderSDE(d).to(dev)
m2.set_obs_noise(obs_var)
m2.save_config(out)
tr, va = tens(P["train"]), tens(P["val"])
params = [p for p in m2.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
n = tr[0].shape[0]
best, best_state, best_ep, hist = np.inf, None, -1, []
t0 = time.time()
for ep in range(args.epochs):
    m2.train()
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for s in range(0, n, 512):
        b = perm[s:s + 512]
        loss, _ = m2.loss(tuple(x[b] for x in tr), K=32)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    if (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.epochs - 1:
        vl = float(np.mean([m2.val_loss(va, K=64) for _ in range(2)]))
        hist.append({"epoch": ep + 1, "train": tot / n, "val": vl, "seconds": time.time() - t0})
        log(f"[A2] ep {ep + 1} train {tot / n:.4f} val {vl:.4f}")
        if vl < best:
            best, best_state, best_ep = vl, copy.deepcopy(m2.state_dict()), ep + 1
m2.load_state_dict(best_state)
m2.eval()
torch.save(m2.state_dict(), out / "model_A2.pt")
log(f"A2 best epoch {best_ep} val {best:.4f} ({time.time() - t0:.0f}s)")

# ---------------- first-order age-free Model A (same latents, same budget) ----------------
m1 = NeuralSDE(d, hidden=128, solver="native", n_steps=32, use_age=False).to(dev)
m1.set_obs_noise(obs_var)
m1.save_config(out)
strip = lambda p: {k: p[k] for k in ("z0", "z1", "a0", "a1", "u")}
r1 = train_pairs(m1, strip(P["train"]), strip(P["val"]), epochs=args.epochs, lr=args.lr, device=dev, log=log,
                 loss_kw={"K": 32}, val_every=args.eval_every, patience=10 ** 9, seed=args.seed)
torch.save(m1.state_dict(), out / "model_A.pt")
log(f"A best val {r1['best_val']:.4f} ({r1['train_seconds']:.0f}s)")

# ---------------- momentum test: 3rd visit from the first two (held-out people) ----------------
te = np.nonzero(lat.split.to_numpy() == "test")[0]
first = te[(~same_prev[te]) & np.r_[same_next[te]]]
first = first[(first + 2 < len(lat)) & (pid[np.minimum(first + 2, len(lat) - 1)] == pid[first])]
i1, i2, i3 = first, first + 1, first + 2
log(f"3-visit test people: {len(first)}")
probe = LatentAgeProbe().fit(Z[lat.split.to_numpy() == "train"], A[lat.split.to_numpy() == "train"])
T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


@torch.no_grad()
def forecast(model, history):
    kw = {"z_prev": T(Z[i1]), "dt_prev": T(A[i2] - A[i1])} if history else {}
    S = model.sample(T(Z[i2]), T(A[i2]), T(A[i3]), T(U[i2, None]), n_samples=64, n_steps=40, **kw)
    return S.cpu().numpy()


p3 = {"z0": Z[i2], "z1": Z[i3], "a0": A[i2], "a1": A[i3], "i1": i3}
res = {"args": vars(args), "n_test_people_3visits": int(len(first)), "a2_best_epoch": best_ep, "a2_history": hist,
       "a_best_val": r1["best_val"], "a2_best_val": best, "forecast_visit3": {}, "momentum": {}}
rate1 = (probe(Z[i2]) - probe(Z[i1])) / (A[i2] - A[i1])
rate2_obs = (probe(Z[i3]) - probe(Z[i2])) / (A[i3] - A[i2])
res["momentum"]["data_corr_rate1_rate2"] = float(np.corrcoef(rate1, rate2_obs)[0, 1])
for name, model, hist_ in (("A2_with_history", m2, True), ("A2_no_history", m2, False), ("A_first_order", m1, False)):
    S = forecast(model, hist_)
    if bundle is not None:
        fc = forecast_eval(name, S, p3, bundle, ca.Xs, nc, sigma_x, np.random.default_rng(0))
    else:
        fc = {"mae": float("nan"), "crps": float("nan"), "cov90": float("nan"),
              "latent_mse": float(((S.mean(0) - Z[i3]) ** 2).sum(-1).mean())}
    var = S.var(0) + np.asarray(obs_var)
    nll = float((0.5 * (Z[i3] - S.mean(0)) ** 2 / var + 0.5 * np.log(var)).sum(-1).mean())
    rate2_pred = (probe(S.mean(0)) - probe(Z[i2])) / (A[i3] - A[i2])
    res["forecast_visit3"][name] = {"mae": fc["mae"], "latent_mse": fc["latent_mse"], "crps": fc["crps"],
                                    "cov90": fc["cov90"], "nll": nll}
    res["momentum"][name] = {"corr_rate1_pred_rate2": float(np.corrcoef(rate1, rate2_pred)[0, 1]),
                             "slope_pred_rate2_on_rate1": float(np.polyfit(rate1, rate2_pred, 1)[0])}
    log(f"{name}: visit-3 MAE {fc['mae']:.4f} latent MSE {fc['latent_mse']:.3f} NLL {nll:.3f} | "
        f"corr(prev rate, predicted next rate) {res['momentum'][name]['corr_rate1_pred_rate2']:.3f} "
        f"(data {res['momentum']['data_corr_rate1_rate2']:.3f})")
res["momentum"]["data_slope_rate2_on_rate1"] = float(np.polyfit(rate1, rate2_obs, 1)[0])

# recovery against the hidden truth (momentum world only): inferred pace vs true pace, persistence time
key = pd.MultiIndex.from_arrays([lat.person_id.to_numpy()[i2], lat.visit_age.to_numpy()[i2].round(3)])
tli = tl.set_index(["person_id", "visit_age"]).reindex(key)
with torch.no_grad():
    mu, _ = m2.v_start(T(Z[i2]), T(Z[i1]), T(A[i2] - A[i1]), torch.ones(len(i2), dtype=torch.bool, device=dev))
    pace_inferred = mu.cpu().numpy() @ probe.w
    G = m2.friction(T(Z[te])).cpu().numpy()
w = probe.w / np.linalg.norm(probe.w)
res["momentum"]["persistence_years_along_age_direction"] = float(1.0 / ((G * w ** 2).sum(1)).mean())
if "vtrue_0" in tli.columns:
    vt = tli["vtrue_0"].to_numpy()
    ok = np.isfinite(vt)
    res["momentum"]["corr_inferred_pace_true_pace"] = float(np.corrcoef(pace_inferred[ok], vt[ok])[0, 1])
    res["momentum"]["corr_fd_rate_true_pace"] = float(np.corrcoef(rate1[ok], vt[ok])[0, 1])
    res["momentum"]["true_persistence_years"] = 1.0 / tcfg.gamma_v
if args.oracle:   # true coordinates: read pace and persistence directly on the true axes
    res["momentum"]["persistence_years_true_axis0"] = float(1.0 / G[:, 0].mean())
    res["momentum"]["persistence_years_true_axis2"] = float(1.0 / G[:, 2].mean())
    if "vtrue_0" in tli.columns:
        vt0 = tli["vtrue_0"].to_numpy()
        ok0 = np.isfinite(vt0)
        res["momentum"]["corr_inferred_v_axis0_true_pace"] = float(np.corrcoef(mu.cpu().numpy()[ok0, 0], vt0[ok0])[0, 1])
        fd0 = (Z[i2, 0] - Z[i1, 0]) / (A[i2] - A[i1])
        res["momentum"]["corr_fd_axis0_true_pace"] = float(np.corrcoef(fd0[ok0], vt0[ok0])[0, 1])
log("momentum: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["momentum"].items()
                               if not isinstance(v, dict)}))

# ---------------- survival + hallmarks (overdamped view of A2) ----------------
cox = None
if bundle is not None:
    try:
        surv, cox = survival_eval({"A2": m2, "A": m1}, bundle, ca, lat, zc,
                                  {"eval": {"survival_horizon": 15, "rollout_samples": 32}}, dev, None, rng)
        res["survival"] = surv
    except ValueError as e:   # e.g. NaN risk scores if long A2 rollouts diverge
        log(f"survival evaluation failed: {e}")
        res["survival_error"] = str(e)
if not args.no_hallmarks and bundle is not None and cox is not None:
    hm, _ = run_hallmarks({"A2": m2, "A": m1}, bundle, ca, lat, zc, meta, cox, dev, log)
    res["hallmarks"] = hm
(out / "results.json").write_text(json.dumps(jsonable(res), indent=1))
log(f"saved {out / 'results.json'}")
