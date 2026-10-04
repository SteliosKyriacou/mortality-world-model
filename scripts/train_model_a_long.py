"""Long training of the encoder + Model A only, with convergence tracking.

Every --eval-every epochs it records validation NLL (K=128), held-out forecast error and
drift error against the synthetic ground truth, so the convergence plot shows what extra
training buys. No early stopping: the checkpoint with the best K=128 validation NLL is kept,
and the final-epoch weights are saved too.

    conda run -n mwm python scripts/train_model_a_long.py --cohort on --seed 0
    conda run -n mwm python scripts/train_model_a_long.py --cohort on --seed 0 --d 8 --a-epochs 300 \
        --no-hallmarks --out runs/dimsweep
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from mwm.data.dataset import load_cohort_dir, make_pairs
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.dynamics.train import pairs_to_tensors
from mwm.encoders.train import export_latents, train_encoder
from mwm.eval import baselines as bl
from mwm.eval.groundtruth import affine_r2, drift_recovery, fit_affine
from mwm.interrogate.common import drift_at
from mwm.pipeline import (Logger, baseline_eval, forecast_eval, jsonable, predict_latent, run_hallmarks,
                          survival_eval)

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="on")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--d", type=int, default=16)
ap.add_argument("--enc-epochs", type=int, default=200)
ap.add_argument("--a-epochs", type=int, default=600)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--jac-penalty", type=float, default=0.0,
                help="Hutchinson ||d drift/dz||_F^2 penalty; stops stiff fast-reverting solutions")
ap.add_argument("--select", default="val_mae", choices=["val_mae", "val_nll"],
                help="which validation metric picks the kept checkpoint")
ap.add_argument("--data-root", default="data/synthetic")
ap.add_argument("--out", default="runs/long")
ap.add_argument("--no-hallmarks", action="store_true")
ap.add_argument("--max-hours", type=float, default=None, help="stop training after this wall time")
ap.add_argument("--snapshot-every", type=int, default=0, help="save weights every N epochs (0 = off)")
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)

cdir = Path(args.data_root) / args.cohort
out = Path(args.out) / args.cohort / (f"seed{args.seed}" if args.d == 16 else f"d{args.d}_seed{args.seed}")
out.mkdir(parents=True, exist_ok=True)
log = Logger(out / "log.txt")
tcfg, meta, tl, _ = load_truth(cdir)
ca = load_cohort_dir(cdir)
nc = len(ca.clinical)
log(f"== {cdir} seed {args.seed} d {args.d} enc_epochs {args.enc_epochs} a_epochs {args.a_epochs}")

# ---------------- encoder ----------------
bundle = train_encoder(ca, d_latent=args.d, kind="transformer", epochs=args.enc_epochs, seed=args.seed,
                       device=dev, log=log, d_model=64, n_layers=2)
lat = export_latents(bundle, ca, out, dev)
zc = [c for c in lat.columns if c.startswith("z_")]
obs_var = json.loads((out / "encoder_meta.json").read_text())["latent_obs_noise_var"]

P = {s: make_pairs(lat[lat.split == s].reset_index(), zc, "consecutive") for s in ("train", "val")}
for s in P:
    sub = lat[lat.split == s].reset_index()
    P[s]["i0"], P[s]["i1"] = sub["index"].to_numpy()[P[s]["i0"]], sub["index"].to_numpy()[P[s]["i1"]]
subv = lat[lat.split == "val"].reset_index()
Pv = make_pairs(subv, zc, "first_last")
Pv["i0"], Pv["i1"] = subv["index"].to_numpy()[Pv["i0"]], subv["index"].to_numpy()[Pv["i1"]]
sub = lat[lat.split == "test"].reset_index()
Pt = make_pairs(sub, zc, "first_last")
Pt["i0"], Pt["i1"] = sub["index"].to_numpy()[Pt["i0"]], sub["index"].to_numpy()[Pt["i1"]]

# ground-truth affine map learned -> true
import pandas as pd
key = pd.MultiIndex.from_arrays([lat.person_id, lat.visit_age.round(3)])
Zt = tl.set_index(["person_id", "visit_age"]).reindex(key)[[f"ztrue_{k}" for k in range(8)]].to_numpy()
trm, tem = (lat.split == "train").to_numpy(), (lat.split == "test").to_numpy()
W, b0 = fit_affine(lat[zc].to_numpy()[trm], Zt[trm])
r2 = affine_r2(lat[zc].to_numpy()[tem], Zt[tem], W, b0)
log(f"affine R2 to truth: {np.round(r2, 3).tolist()} mean {r2.mean():.3f}")
tz = lat[tem]
sigma_x = bundle.feature_sigma.detach().cpu().numpy()


def checkpoint_metrics(m):
    m.eval()
    va = pairs_to_tensors(P["val"], dev)
    nll = float(np.mean([m.val_loss(va, K=128) for _ in range(2)]))
    Sv = predict_latent(m, Pv, 32, dev)
    fv = forecast_eval("A", Sv, Pv, bundle, ca.Xs, nc, sigma_x, np.random.default_rng(0))
    S = predict_latent(m, Pt, 32, dev)
    fc = forecast_eval("A", S, Pt, bundle, ca.Xs, nc, sigma_x, np.random.default_rng(0))
    gt = drift_recovery(m, tz[zc].to_numpy(), tz.visit_age.to_numpy(), tz[["u"]].to_numpy(), Zt[tem], tcfg, W=W, device=dev)
    Zx, Ax = tz[zc].to_numpy(), tz.visit_age.to_numpy()
    dv = (drift_at(m, Zx, Ax, np.ones(len(Zx)), dev) - drift_at(m, Zx, Ax, np.zeros(len(Zx)), dev)) @ W
    return {"val_nll_K128": nll, "val_mae": fv["mae"], "test_latent_mse": fc["latent_mse"], "test_mae": fc["mae"], "test_crps": fc["crps"],
            "cov90": fc["cov90"], "drift_rel_mse": gt["drift_rel_mse"], "drift_nrmse_per_dim": gt["drift_nrmse_per_dim"],
            "diff_trace_ratio_old_young": gt.get("diff_trace_ratio_old_young_model"),
            "u_effect_true_nutr": float(dv.mean(0)[2])}


# ---------------- Model A ----------------
m = NeuralSDE(args.d, hidden=128, solver="native", n_steps=32).to(dev)
m.set_obs_noise(obs_var)
tr, va = pairs_to_tensors(P["train"], dev), pairs_to_tensors(P["val"], dev)
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.a_epochs)
n = tr[0].shape[0]
hist, ckpts = [], []
best, best_state, best_ep = np.inf, None, -1
best_val5, last_good, n_skipped, n_rollbacks = np.inf, copy.deepcopy(m.state_dict()), 0, 0
t0 = time.time()
for ep in range(args.a_epochs):
    m.train()
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for s in range(0, n, 512):
        b = perm[s:s + 512]
        batch = tuple(x[b] for x in tr)
        loss, _ = m.loss(batch, K=32)
        if args.jac_penalty > 0:
            zz = batch[0].clone().requires_grad_(True)
            f = m.drift(zz, batch[2], batch[4])
            v = torch.randn_like(f)
            vJ = torch.autograd.grad(f, zz, grad_outputs=v, create_graph=True)[0]
            loss = loss + args.jac_penalty * (vJ ** 2).sum(-1).mean()
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    if ep % 5 == 0 or ep == args.a_epochs - 1:
        m.eval()
        vl = m.val_loss(va, K=32)
        hist.append({"epoch": ep, "train": tot / n, "val": vl, "lr": sched.get_last_lr()[0],
                     "seconds": time.time() - t0})
        # divergence guard: roll back to the last good weights and halve the learning rate
        if not np.isfinite(vl) or vl > best_val5 + 3.0 or n_skipped > 0:
            m.load_state_dict(last_good)
            for gp in opt.param_groups:
                gp["lr"] *= 0.5
            sched.base_lrs = [lr * 0.5 for lr in sched.base_lrs]
            n_rollbacks += 1
            log(f"[guard] epoch {ep}: val {vl:.3f} (best {best_val5:.3f}), skipped {n_skipped} -> rollback #{n_rollbacks}, lr x0.5")
            n_skipped = 0
            continue
        best_val5 = min(best_val5, vl)
        last_good = copy.deepcopy(m.state_dict())
    if (ep + 1) % args.eval_every == 0 or ep == 0 or ep == args.a_epochs - 1:
        cm = checkpoint_metrics(m)
        cm["epoch"] = ep + 1
        ckpts.append(cm)
        log(f"[A] ep {ep + 1} val128 {cm['val_nll_K128']:.4f} val_mae {cm['val_mae']:.4f} test_mae {cm['test_mae']:.4f} "
            f"lmse {cm['test_latent_mse']:.3f} drift_relmse {cm['drift_rel_mse']:.3f} u {cm['u_effect_true_nutr']:.4f}")
        score = cm["val_mae"] if args.select == "val_mae" else cm["val_nll_K128"]
        if np.isfinite(score) and score < best:
            best, best_state, best_ep = score, copy.deepcopy(m.state_dict()), ep + 1
            torch.save(best_state, out / "model_A_best_sofar.pt")
        (out / "convergence.json").write_text(json.dumps(jsonable({"a_history": hist, "checkpoints": ckpts,
                                                                    "encoder_history": bundle.history})))
    if args.snapshot_every and (ep + 1) % args.snapshot_every == 0:
        torch.save(m.state_dict(), out / f"model_A_ep{ep + 1}.pt")
    if args.max_hours and time.time() - t0 > args.max_hours * 3600:
        log(f"time limit reached at epoch {ep + 1}")
        if not ckpts or ckpts[-1]["epoch"] != ep + 1:
            cm = checkpoint_metrics(m)
            cm["epoch"] = ep + 1
            ckpts.append(cm)
            score = cm["val_mae"] if args.select == "val_mae" else cm["val_nll_K128"]
            if np.isfinite(score) and score < best:
                best, best_state, best_ep = score, copy.deepcopy(m.state_dict()), ep + 1
        break
train_seconds = time.time() - t0
torch.save(m.state_dict(), out / "model_A_final.pt")
m.load_state_dict(best_state)
m.eval()
torch.save(m.state_dict(), out / "model_A.pt")
log(f"best checkpoint epoch {best_ep} ({args.select} {best:.4f}); rollbacks {n_rollbacks}; train {train_seconds:.0f}s")

# ---------------- final evaluation (best checkpoint) ----------------
res = {"args": vars(args), "best_epoch": best_ep, "n_rollbacks": n_rollbacks, "train_seconds": train_seconds,
       "n_pairs": {"train": len(P["train"]["a0"]), "val": len(P["val"]["a0"]), "test_first_last": len(Pt["a0"])},
       "latent_obs_noise_var": obs_var, "encoder_history": bundle.history, "a_history": hist,
       "checkpoints": ckpts}
S = predict_latent(m, Pt, 32, dev)
fc = {"A": forecast_eval("A", S, Pt, bundle, ca.Xs, nc, sigma_x, rng)}
X0 = ca.Xs[:, :nc]
U = ca.u[:, 0]
trP = P["train"]
for B in (bl.LOCF(), bl.DirectGBM(seed=args.seed)):
    B.fit(X0[trP["i0"]], ca.Xs[trP["i1"], :nc], trP["a0"], trP["a1"] - trP["a0"], U[trP["i0"]])
    mu, sd = B.predict(X0[Pt["i0"]], Pt["a0"], Pt["a1"] - Pt["a0"], U[Pt["i0"]])
    fc[f"base_{B.name}"] = baseline_eval(B.name, mu, sd, Pt, ca.Xs, nc, rng)
res["forecast"] = fc
cfg = {"eval": {"survival_horizon": 15, "rollout_samples": 32}}
surv, cox = survival_eval({"A": m}, bundle, ca, lat, zc, cfg, dev, None, rng)
res["survival"] = surv
gt = drift_recovery(m, tz[zc].to_numpy(), tz.visit_age.to_numpy(), tz[["u"]].to_numpy(), Zt[tem], tcfg, W=W, device=dev)
res["ground_truth"] = {"affine_r2_per_true_dim": r2.round(3).tolist(), "A": gt}
log(f"final: mae {fc['A']['mae']:.4f} (GBM {fc['base_direct_gbm']['mae']:.4f}) C {surv['A']['c_harrell']:.3f} "
    f"drift_relmse {gt['drift_rel_mse']:.3f}")
if not args.no_hallmarks:
    hm, probe_r2 = run_hallmarks({"A": m}, bundle, ca, lat, zc, meta, cox, dev, log)
    res["hallmarks"] = hm
    log("hallmarks: " + ", ".join(f"{k}={v.get('detected', v.get('detected_flag'))}" for k, v in hm["A"].items()))
(out / "results.json").write_text(json.dumps(jsonable(res), indent=1))
log(f"saved {out / 'results.json'}")
