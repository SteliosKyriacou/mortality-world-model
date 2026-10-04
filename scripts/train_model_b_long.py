"""Long training of Model B (paired conditional flow matching) on a frozen encoder, with
convergence tracking on a log-spaced schedule (to see early fitting and any late grokking).

Reuses the encoder/latents of a previous run (default: the long Model A run) so A and B are
compared on identical latents. Keeps the checkpoint with the best validation forecast error.

    conda run -n mwm python scripts/train_model_b_long.py --encoder-run runs/long/on/seed0
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
from mwm.dynamics.flow import FlowModel
from mwm.dynamics.train import pairs_to_tensors
from mwm.encoders.train import EncoderBundle
from mwm.eval import baselines as bl
from mwm.eval.groundtruth import affine_r2, drift_recovery, fit_affine
from mwm.interrogate.common import drift_at
from mwm.pipeline import (Logger, baseline_eval, forecast_eval, jsonable, predict_latent, run_hallmarks,
                          survival_eval)

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="on")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--encoder-run", default="runs/long/on/seed0")
ap.add_argument("--epochs", type=int, default=300000)
ap.add_argument("--n-evals", type=int, default=150, help="log-spaced checkpoint evaluations")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-5)
ap.add_argument("--jac-penalty", type=float, default=0.1)
ap.add_argument("--max-hours", type=float, default=16.0)
ap.add_argument("--data-root", default="data/synthetic")
ap.add_argument("--out", default="runs/long_b")
ap.add_argument("--no-hallmarks", action="store_true")
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

cdir = Path(args.data_root) / args.cohort
enc = Path(args.encoder_run)
out = Path(args.out) / args.cohort / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
log = Logger(out / "log.txt")
tcfg, meta, tl, _ = load_truth(cdir)
ca = load_cohort_dir(cdir)
nc = len(ca.clinical)
bundle = EncoderBundle.load(enc / "encoder.pt").to(dev)
lat = pd.read_parquet(enc / "latents.parquet")
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)
log(f"== Model B long: {cdir} seed {args.seed} encoder {enc} epochs {args.epochs} lr {args.lr} "
    f"wd {args.wd} jac {args.jac_penalty}")


def pairs(split, mode):
    sub = lat[lat.split == split].reset_index()
    p = make_pairs(sub, zc, mode)
    p["i0"], p["i1"] = sub["index"].to_numpy()[p["i0"]], sub["index"].to_numpy()[p["i1"]]
    return p


P = {"train": pairs("train", "consecutive"), "val": pairs("val", "consecutive")}
Pv, Pt = pairs("val", "first_last"), pairs("test", "first_last")
key = pd.MultiIndex.from_arrays([lat.person_id, lat.visit_age.round(3)])
Zt = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
trm, tem = (lat.split == "train").to_numpy(), (lat.split == "test").to_numpy()
W, b0 = fit_affine(lat[zc].to_numpy()[trm], Zt[trm])
r2 = affine_r2(lat[zc].to_numpy()[tem], Zt[tem], W, b0)
tz = lat[tem]
sigma_x = bundle.feature_sigma.detach().cpu().numpy()


def checkpoint_metrics(m):
    m.eval()
    vl = m.val_loss(pairs_to_tensors(P["val"], dev))
    fv = forecast_eval("B", predict_latent(m, Pv, 1, dev), Pv, bundle, ca.Xs, nc, sigma_x, np.random.default_rng(0))
    fc = forecast_eval("B", predict_latent(m, Pt, 1, dev), Pt, bundle, ca.Xs, nc, sigma_x, np.random.default_rng(0))
    gt = drift_recovery(m, tz[zc].to_numpy(), tz.visit_age.to_numpy(), tz[["u"]].to_numpy(), Zt[tem], tcfg, W=W, device=dev)
    Zx, Ax = tz[zc].to_numpy(), tz.visit_age.to_numpy()
    dv = (drift_at(m, Zx, Ax, np.ones(len(Zx)), dev) - drift_at(m, Zx, Ax, np.zeros(len(Zx)), dev)) @ W
    return {"val_cfm": vl, "val_mae": fv["mae"], "test_mae": fc["mae"], "test_latent_mse": fc["latent_mse"],
            "test_crps": fc["crps"], "cov90": fc["cov90"], "drift_rel_mse": gt["drift_rel_mse"],
            "drift_nrmse_per_dim": gt["drift_nrmse_per_dim"], "u_effect_true_nutr": float(dv.mean(0)[2])}


m = FlowModel(d, hidden=128, jac_penalty=args.jac_penalty).to(dev)
tr, va = pairs_to_tensors(P["train"], dev), pairs_to_tensors(P["val"], dev)
params = list(m.parameters())
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
n = tr[0].shape[0]
eval_at = set(np.unique(np.geomspace(1, args.epochs, args.n_evals).astype(int)).tolist())
hist, ckpts = [], []
best, best_state, best_ep = np.inf, None, -1
hist_every = max(1, args.epochs // 4000)
t0 = time.time()
for ep in range(args.epochs):
    m.train()
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for s in range(0, n, 512):
        b = perm[s:s + 512]
        loss, _ = m.loss(tuple(x[b] for x in tr))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    e1 = ep + 1
    if e1 % hist_every == 0 or ep == 0:
        hist.append({"epoch": e1, "train": tot / n, "val": m.val_loss(va), "lr": sched.get_last_lr()[0],
                     "seconds": time.time() - t0})
    timeout = args.max_hours and time.time() - t0 > args.max_hours * 3600
    if e1 in eval_at or e1 == args.epochs or timeout:
        cm = checkpoint_metrics(m)
        cm["epoch"] = e1
        ckpts.append(cm)
        log(f"[B] ep {e1} val_cfm {cm['val_cfm']:.4f} val_mae {cm['val_mae']:.4f} test_mae {cm['test_mae']:.4f} "
            f"lmse {cm['test_latent_mse']:.3f} drift_relmse {cm['drift_rel_mse']:.3f} u {cm['u_effect_true_nutr']:.4f}")
        if np.isfinite(cm["val_mae"]) and cm["val_mae"] < best:
            best, best_state, best_ep = cm["val_mae"], copy.deepcopy(m.state_dict()), e1
            torch.save(best_state, out / "model_B_best_sofar.pt")
        (out / "convergence.json").write_text(json.dumps(jsonable({"a_history": hist, "checkpoints": ckpts}),
                                                         allow_nan=True))
    if timeout:
        log(f"time limit reached at epoch {e1}")
        break
train_seconds = time.time() - t0
torch.save(m.state_dict(), out / "model_B_final.pt")
m.load_state_dict(best_state)
m.eval()
torch.save(m.state_dict(), out / "model_B.pt")
log(f"best checkpoint epoch {best_ep} (val_mae {best:.4f}); train {train_seconds:.0f}s")

res = {"args": vars(args), "best_epoch": best_ep, "train_seconds": train_seconds, "checkpoints": ckpts,
       "history": hist}
fc = {"B": forecast_eval("B", predict_latent(m, Pt, 1, dev), Pt, bundle, ca.Xs, nc, sigma_x, rng)}
X0, U, trP = ca.Xs[:, :nc], ca.u[:, 0], P["train"]
for Bm in (bl.LOCF(), bl.DirectGBM(seed=args.seed)):
    Bm.fit(X0[trP["i0"]], ca.Xs[trP["i1"], :nc], trP["a0"], trP["a1"] - trP["a0"], U[trP["i0"]])
    mu, sd = Bm.predict(X0[Pt["i0"]], Pt["a0"], Pt["a1"] - Pt["a0"], U[Pt["i0"]])
    fc[f"base_{Bm.name}"] = baseline_eval(Bm.name, mu, sd, Pt, ca.Xs, nc, rng)
res["forecast"] = fc
surv, cox = survival_eval({"B": m}, bundle, ca, lat, zc, {"eval": {"survival_horizon": 15, "rollout_samples": 32}},
                          dev, None, rng)
res["survival"] = surv
res["ground_truth"] = {"affine_r2_per_true_dim": r2.round(3).tolist(),
                       "B": drift_recovery(m, tz[zc].to_numpy(), tz.visit_age.to_numpy(), tz[["u"]].to_numpy(),
                                           Zt[tem], tcfg, W=W, device=dev)}
log(f"final: mae {fc['B']['mae']:.4f} (GBM {fc['base_direct_gbm']['mae']:.4f}) C {surv['B']['c_harrell']:.3f}")
if not args.no_hallmarks:
    hm, _ = run_hallmarks({"B": m}, bundle, ca, lat, zc, meta, cox, dev, log)
    res["hallmarks"] = hm
(out / "results.json").write_text(json.dumps(jsonable(res), indent=1))
log(f"saved {out / 'results.json'}")
