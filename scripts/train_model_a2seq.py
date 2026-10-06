"""Momentum with annual visits: Model A2 with a sequence filter vs first-order Model A.

    conda run -n mwm python scripts/train_model_a2seq.py --cohort momentum --oracle     # true states
    conda run -n mwm python scripts/train_model_a2seq.py --cohort momentum              # through the encoder

Evaluates on held-out people with >= 7 annual visits:
  * one-step-ahead forecast after seeing 1..6 visits (A2-seq uses the whole history; Model A the last visit)
  * 3-year-ahead forecast from the 4th visit
  * recovery of each person's true pace as a function of how many visits the filter has seen
  * persistence time of the learned velocity (true 5 y) and the lag autocorrelation of simulated rates
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mwm.data.dataset import load_cohort_dir
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde import NeuralSDE
from mwm.dynamics.sde2 import SeqSecondOrderSDE
from mwm.dynamics.train import train_pairs
from mwm.encoders.train import EncoderBundle, export_latents, train_encoder
from mwm.pipeline import Logger, jsonable
from sklearn.linear_model import LinearRegression

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data/synthetic_momentum_annual")
ap.add_argument("--cohort", default="momentum")
ap.add_argument("--oracle", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--enc-epochs", type=int, default=40)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--encoder-from", default=None)
ap.add_argument("--out", default="runs/momentum_annual")
ap.add_argument("--wearables", action="store_true", help="feed wearable summaries between visits into the filter")
ap.add_argument("--horizons", type=int, default=1, help="train on predictions 1..H visits ahead")
ap.add_argument("--name", default=None)
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cdir = Path(args.data_root) / args.cohort
out = Path(args.out) / args.cohort / (args.name or (f"oracle_seed{args.seed}" if args.oracle else f"encoder_seed{args.seed}"))
out.mkdir(parents=True, exist_ok=True)
log = Logger(out / "log.txt")
tcfg, meta, tl, _ = load_truth(cdir)
ca = load_cohort_dir(cdir)
log(f"== {cdir} oracle={args.oracle} momentum={tcfg.momentum} null={tcfg.momentum_null}")

tt = tl.copy()
tt["visit_age"] = tt.visit_age.round(3)
if args.oracle:
    lat = pd.DataFrame({"person_id": ca.visits.person_id, "visit_age": np.round(ca.visits.visit_age, 3),
                        "split": ca.split, "u": ca.u[:, 0]})
    lat = lat.merge(tt[["person_id", "visit_age"] + [f"ztrue_{k}" for k in range(8)]], on=["person_id", "visit_age"])
    lat = lat.rename(columns={f"ztrue_{k}": f"z_{k}" for k in range(8)})
    obs_var = [1e-4] * 8
elif args.encoder_from:
    import shutil
    for f in ("encoder.pt", "encoder_meta.json", "latents.parquet"):
        shutil.copy(Path(args.encoder_from) / f, out / f)
    lat = pd.read_parquet(out / "latents.parquet")
    obs_var = json.loads((out / "encoder_meta.json").read_text())["latent_obs_noise_var"]
else:
    bundle = train_encoder(ca, d_latent=16, kind="transformer", epochs=args.enc_epochs, seed=args.seed, device=dev,
                           log=log, d_model=64, n_layers=2)
    lat = export_latents(bundle, ca, out, dev)
    obs_var = json.loads((out / "encoder_meta.json").read_text())["latent_obs_noise_var"]
lat["visit_age"] = lat.visit_age.round(3)
lat = lat.merge(tt[["person_id", "visit_age", "ztrue_0", "ztrue_2"] + [c for c in tt.columns if c.startswith("vtrue")]],
                on=["person_id", "visit_age"], how="left").sort_values(["person_id", "visit_age"]).reset_index(drop=True)
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)

# ---------------- wearable summaries for the interval before each visit ----------------
W_DIM = 13 if args.wearables else 0
Wrow = np.zeros((len(lat), max(W_DIM, 1)), np.float32)
if args.wearables:
    wd = pd.read_parquet(cdir / "wearable.parquet")
    feats = ["resting_hr", "steps", "sleep_eff", "fitness"]
    wd["steps"] = np.log(wd["steps"])
    trp = set(lat.person_id[lat.split == "train"])
    mu_w = wd[wd.person_id.isin(trp)][feats].mean().to_numpy()
    sd_w = wd[wd.person_id.isin(trp)][feats].std().to_numpy()
    rows_by_p = lat.groupby("person_id").indices
    va_ = lat.visit_age.to_numpy()

    def slope(t, x):
        if len(t) < 3:
            return np.zeros(x.shape[1])
        tc = t - t.mean()
        return (tc[:, None] * (x - x.mean(0))).sum(0) / max((tc ** 2).sum(), 1e-6)

    for p_, g in wd.groupby("person_id"):
        if p_ not in rows_by_p:
            continue
        t = g.age.to_numpy()
        X = (g[feats].to_numpy() - mu_w) / sd_w
        okw = np.isfinite(X).all(1)
        t, X = t[okw], X[okw]
        prev = -np.inf
        for r in rows_by_p[p_]:
            a = va_[r]
            m_int = (t > prev) & (t <= a)
            m1, m2_ = (t > a - 1) & (t <= a), (t > a - 2) & (t <= a)
            if m_int.any():
                Wrow[r, 0:4] = X[m_int].mean(0)
            Wrow[r, 4:8] = slope(t[m1], X[m1]) if m1.sum() >= 3 else 0
            Wrow[r, 8:12] = slope(t[m2_], X[m2_]) if m2_.sum() >= 3 else 0
            Wrow[r, 12] = m1.sum() / 20.0
            prev = a
    has_w = set(wd.person_id.unique())
    log(f"wearables: {len(has_w)} people; visits with a wearable summary {(Wrow[:, 12] > 0).mean():.1%}")
else:
    has_w = set()

# ---------------- sequences ----------------
Tmax = int(lat.groupby("person_id").size().max())
people = lat.person_id.unique()
idx_by_p = lat.groupby("person_id").indices
Z = lat[zc].to_numpy(np.float32)
Aa = lat.visit_age.to_numpy(np.float32)
Uu = lat["u"].to_numpy(np.float32)
split_p = lat.groupby("person_id")["split"].first()


def pack(pids):
    B = len(pids)
    Zs = np.zeros((B, Tmax, d), np.float32)
    As = np.zeros((B, Tmax), np.float32)
    Us = np.zeros((B, Tmax, 1), np.float32)
    Ws = np.zeros((B, Tmax, max(W_DIM, 1)), np.float32)
    L = np.zeros(B, np.int64)
    rows = np.full((B, Tmax), -1)
    for b, p in enumerate(pids):
        ii = idx_by_p[p]
        n = len(ii)
        Zs[b, :n], As[b, :n], Us[b, :n, 0], L[b], rows[b, :n] = Z[ii], Aa[ii], Uu[ii], n, ii
        Ws[b, :n] = Wrow[ii]
        if n < Tmax:   # pad by repeating the last visit (masked out by L)
            Zs[b, n:], As[b, n:] = Z[ii[-1]], Aa[ii[-1]] + 1.0
    T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    return T(Zs), T(As), T(Us), T(L, torch.long), rows, (T(Ws) if W_DIM else None)


tr_p = [p for p in people if split_p[p] == "train" and len(idx_by_p[p]) >= 2]
va_p = [p for p in people if split_p[p] == "val" and len(idx_by_p[p]) >= 2]
te_p = [p for p in people if split_p[p] == "test" and len(idx_by_p[p]) >= 7]
VA = pack(va_p)
log(f"sequences: train {len(tr_p)}, val {len(va_p)}, test (>=7 visits) {len(te_p)}, max visits {Tmax}")

# ---------------- A2-seq ----------------
m2 = SeqSecondOrderSDE(d, w_dim=W_DIM, horizons=args.horizons).to(dev)
m2.set_obs_noise(obs_var)
(out / "model_A2seq_config.json").write_text(json.dumps(m2.cfg))
params = [p for p in m2.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
best, best_state, best_ep, hist = np.inf, None, -1, []
t0 = time.time()
rng = np.random.default_rng(args.seed)
for ep in range(args.epochs):
    m2.train()
    order = rng.permutation(len(tr_p))
    tot, nb = 0.0, 0
    for s in range(0, len(order), args.batch):
        Zs, As, Us, L, _, Ws = pack([tr_p[i] for i in order[s:s + args.batch]])
        loss, nll = m2.seq_loss(Zs, As, Us, L, K=16, Ws=Ws)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        tot += nll
        nb += 1
    sched.step()
    m2.train()   # keep start-noise jitter for a comparable validation NLL
    with torch.no_grad():
        vl = float(np.mean([m2.seq_loss(*VA[:4], K=32, j_penalty=0, sigma_penalty=0, Ws=VA[5])[1] for _ in range(2)]))
    hist.append({"epoch": ep + 1, "train": tot / max(nb, 1), "val": vl, "seconds": time.time() - t0})
    log(f"[A2seq] ep {ep + 1} train {tot / max(nb, 1):.4f} val {vl:.4f}")
    if vl < best:
        best, best_state, best_ep = vl, copy.deepcopy(m2.state_dict()), ep + 1
m2.load_state_dict(best_state)
m2.eval()
torch.save(m2.state_dict(), out / "model_A2seq.pt")
log(f"A2seq best epoch {best_ep} val {best:.4f} ({time.time() - t0:.0f}s)")

# ---------------- first-order Model A on consecutive pairs ----------------
def pairs(pids):
    i0 = np.concatenate([idx_by_p[p][:-1] for p in pids])
    i1 = i0 + 1
    return {"z0": Z[i0], "z1": Z[i1], "a0": Aa[i0], "a1": Aa[i1], "u": Uu[i0, None]}


m1 = NeuralSDE(d, hidden=128, solver="native", n_steps=8, use_age=False).to(dev)
m1.set_obs_noise(obs_var)
r1 = train_pairs(m1, pairs(tr_p), pairs(va_p), epochs=args.epochs, lr=args.lr, device=dev, log=log,
                 loss_kw={"K": 32}, val_every=5, patience=10 ** 9, seed=args.seed)
torch.save(m1.state_dict(), out / "model_A.pt")

# ---------------- evaluation ----------------
TE = pack(te_p)
Zs, As, Us, L, rows, Ws = TE
wear_mask = np.array([p in has_w for p in te_p])
ov = np.asarray(obs_var)


def nll_mse(S, target):
    S = S.cpu().numpy()
    tgt = target.cpu().numpy()
    var = S.var(0) + ov
    return (float((0.5 * (tgt - S.mean(0)) ** 2 / var + 0.5 * np.log(var)).sum(-1).mean()),
            float(((S.mean(0) - tgt) ** 2).sum(-1).mean()))


@torch.no_grad()
def a_forecast(k, a_target):
    return m1.sample(Zs[:, k], As[:, k], a_target, Us[:, k], n_samples=64, n_steps=8)


res = {"args": vars(args), "a2seq_best_epoch": best_ep, "a2seq_history": hist, "a_best_val": r1["best_val"],
       "n_test_people": len(te_p), "one_step": [], "three_year": {}}
for k in range(6):
    tgt = Zs[:, k + 1]
    n2, e2 = nll_mse(m2.forecast_from_history(Zs, As, Us, k, As[:, k + 1], Ws=Ws), tgt)
    n1, e1 = nll_mse(a_forecast(k, As[:, k + 1]), tgt)
    res["one_step"].append({"visits_seen": k + 1, "A2seq_nll": n2, "A_nll": n1, "A2seq_mse": e2, "A_mse": e1})
    log(f"one-step after {k + 1} visits: NLL A2seq {n2:.4f} A {n1:.4f} (gain {n1 - n2:+.4f}) | MSE {e2:.4f} vs {e1:.4f}")
n2, e2 = nll_mse(m2.forecast_from_history(Zs, As, Us, 3, As[:, 6], Ws=Ws), Zs[:, 6])
n1, e1 = nll_mse(a_forecast(3, As[:, 6]), Zs[:, 6])
res["three_year"] = {"A2seq_nll": n2, "A_nll": n1, "A2seq_mse": e2, "A_mse": e1}
log(f"3-year forecast from visit 4: NLL A2seq {n2:.4f} A {n1:.4f} (gain {n1 - n2:+.4f}) | MSE {e2:.4f} vs {e1:.4f}")

# direction of the true age-core axis in model coordinates (identity in the oracle)
trm = (lat.split == "train").to_numpy() & lat.ztrue_0.notna().to_numpy()
if args.oracle:
    e0 = np.eye(d)[0]
else:
    e0 = LinearRegression().fit(Z[trm], lat.ztrue_0.to_numpy()[trm]).coef_
    e0 = e0 / np.linalg.norm(e0)
mom = {}
if "vtrue_0" in lat.columns and lat.vtrue_0.notna().any():
    rec = []
    for k in range(6):
        with torch.no_grad():
            h = m2.filter_history(Zs, As, k, Ws)
            mu, _ = m2.v_from_filter(h, Zs[:, k])
        vin = mu.cpu().numpy() @ e0
        vt = lat.vtrue_0.to_numpy()[rows[:, k]]
        fd = ((Zs[:, k] - Zs[:, k - 1]).cpu().numpy() @ e0) / (As[:, k] - As[:, k - 1]).cpu().numpy() if k > 0 else np.full(len(vt), np.nan)
        ok = np.isfinite(vt)
        okw = ok & wear_mask
        rec.append({"visits_seen": k + 1, "corr_inferred_pace_true": float(np.corrcoef(vin[ok], vt[ok])[0, 1]),
                    "corr_inferred_pace_true_wearable_people": float(np.corrcoef(vin[okw], vt[okw])[0, 1]) if okw.sum() > 30 else None,
                    "corr_last_difference_true": float(np.corrcoef(fd[ok], vt[ok])[0, 1]) if k > 0 else None})
        log(f"pace recovery after {k + 1} visits: filter {rec[-1]['corr_inferred_pace_true']:.3f}"
            + (f" (people with wearables {rec[-1]['corr_inferred_pace_true_wearable_people']:.3f})" if rec[-1]['corr_inferred_pace_true_wearable_people'] is not None else "")
            + (f" | last two-visit difference {rec[-1]['corr_last_difference_true']:.3f}" if k > 0 else ""))
    mom["pace_recovery"] = rec
with torch.no_grad():
    G = m2.friction(Zs[:, 0]).cpu().numpy()
mom["persistence_years_age_core"] = float(1.0 / (G * e0 ** 2).sum(1).mean())
mom["true_persistence_years"] = 1.0 / tcfg.gamma_v if tcfg.momentum else None


@torch.no_grad()
def sim_lag_autocorr(model, seq):
    """Simulate 7 annual steps from each test person's first visit; autocorrelation of yearly rates along e0."""
    Zc = Zs[:, 0]
    if seq:
        h = model.filter_history(Zs, As, 0, Ws)
        mu, sd = model.v_from_filter(h, Zc)
        v = mu + sd * torch.randn_like(mu)
        z = Zc
        traj = [z]
        for _ in range(7):
            z, v = model.simulate2(z, v, torch.ones(len(z), device=dev), Us[:, 0], n_steps=8)
            traj.append(z)
    else:
        traj = [Zc]
        z = Zc
        for _ in range(7):
            z = model.sample(z, As[:, 0], As[:, 0] + 1.0, Us[:, 0], n_samples=1, n_steps=8)[0]
            traj.append(z)
    X = torch.stack(traj).cpu().numpy() @ e0
    R = np.diff(X, axis=0)
    return {L: float(np.corrcoef(R[:-L].ravel(), R[L:].ravel())[0, 1]) for L in range(1, 6)}


mom["simulated_rate_autocorr_A2seq"] = sim_lag_autocorr(m2, True)
mom["simulated_rate_autocorr_A"] = sim_lag_autocorr(m1, False)
tz = lat.groupby("person_id")["ztrue_0"].apply(lambda s: s.to_numpy())
R = [np.diff(v) for p, v in tz.items() if p in set(te_p)]
mom["true_rate_autocorr"] = {L: float(np.corrcoef(np.concatenate([r[:-L] for r in R if len(r) > L]),
                                                  np.concatenate([r[L:] for r in R if len(r) > L]))[0, 1]) for L in range(1, 6)}
res["momentum"] = mom
log("persistence (y): A2seq %.2f (true %s)" % (mom["persistence_years_age_core"], mom["true_persistence_years"]))
log("rate autocorr by lag: true " + str({k: round(v, 3) for k, v in mom["true_rate_autocorr"].items()}))
log("                      A2seq " + str({k: round(v, 3) for k, v in mom["simulated_rate_autocorr_A2seq"].items()}))
log("                      A     " + str({k: round(v, 3) for k, v in mom["simulated_rate_autocorr_A"].items()}))
(out / "results.json").write_text(json.dumps(jsonable(res), indent=1))
log(f"saved {out / 'results.json'}")
