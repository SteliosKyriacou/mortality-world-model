"""Fluid view of Model A2 (second-order): a held-out cohort as weighted phase-space particles (z, v).

Moments of the weighted particle cloud on a 2D landscape plane give the hydrodynamic fields of the
cohort at each age:
    density      ρ(x)           = Σ w δ(x − x_k)                       (mass; drains through the sink)
    mean velocity u(x)          = Σ w v_k / Σ w                         (local pace and direction of aging)
    velocity spread P(x)/ρ      = Σ w |v_k − u|² / Σ w                  ("pressure": heterogeneity of pace)
    mortality flux              = Σ w λ(z_k)                            (sink)
They obey the continuity and momentum (Euler/Navier–Stokes-type) equations of the phase-space
Fokker–Planck (Kramers) equation with a killing term; the pressure is the learned closure.

    conda run -n mwm python scripts/population_fluid.py --run runs/momentum/momentum/a2_seed0 \
        --data data/synthetic_momentum/momentum --key a2_momentum
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from mwm.data import synthetic as syn
from mwm.data.synthetic import load_truth
from mwm.dynamics.sde2 import SecondOrderSDE
from mwm.encoders.train import EncoderBundle
from mwm.interrogate.common import drift_at
from mwm.interrogate.irreversibility import arrow_direction
from mwm.viz.landscape import learned_surface, plane_direction

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--entry", nargs=2, type=float, default=[40.0, 50.0])
ap.add_argument("--end-age", type=float, default=95.0)
ap.add_argument("--grid", type=int, default=25)
ap.add_argument("--out", default="reports/model_a/webapp/fluid")
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)
run, data = Path(args.run), Path(args.data)
emeta = json.loads((run / "encoder_meta.json").read_text())
bundle = EncoderBundle.load(run / "encoder.pt", map_location=dev).to(dev)
lat = pd.read_parquet(run / "latents.parquet")
zc = [c for c in lat.columns if c.startswith("z_")]
d = len(zc)
m = SecondOrderSDE.from_run_dir(run, d, dev)
m.set_obs_noise(emeta["latent_obs_noise_var"])
m.load_state_dict(torch.load(run / "model_A2.pt", map_location=dev, weights_only=True))
m.eval()
tcfg, meta, tl, L = load_truth(data)
key = pd.MultiIndex.from_arrays([lat.person_id, lat.visit_age.round(3)])
Zt = tl.set_index(["person_id", "visit_age"]).reindex(key)
ZT = Zt[[f"ztrue_{k}" for k in range(8)]].to_numpy()

# ---------------- plane: arrow of aging x nutrient direction ----------------
te = lat[lat.split == "test"]
Zte, Ate = te[zc].to_numpy(), te.visit_age.to_numpy()
with torch.enable_grad():
    f_te = drift_at(m, Zte, Ate, np.zeros(len(Zte)), dev)
w = arrow_direction(f_te)
origin = Zte.mean(0)
proj = (Zte - origin) @ w
slope, icpt = np.polyfit(proj, Ate, 1)
trm = (lat.split == "train").to_numpy() & np.isfinite(ZT).all(1)
e2 = plane_direction(lat[zc].to_numpy()[trm], ZT[trm, 2], w)
sy = (Zte - origin) @ e2
xr = (float(np.quantile(proj, 0.005)) - 0.3, float(max((95 - icpt) / slope, np.quantile(proj, 0.995))) + 0.3)
yr = (float(np.quantile(sy, 0.005)) - 1.0, float(np.quantile(sy, 0.995)) + 1.0)
with torch.enable_grad():
    S = learned_surface(m, bundle, w, e2, xr, yr, origin, lambda x: icpt + slope * np.asarray(x), dev, n=61)

# ---------------- cohort ----------------
base = te.groupby("person_id").head(1)
base = base[(base.visit_age >= args.entry[0]) & (base.visit_age < args.entry[1])]
bi = base.index.to_numpy()
ok = np.isfinite(ZT[bi]).all(1)
base, bi = base[ok], bi[ok]
N = len(base)
Z0, A0, U0 = base[zc].to_numpy(np.float32), base.visit_age.to_numpy(), base["u"].to_numpy()
zt0 = ZT[bi]
vt0 = Zt[["vtrue_0", "vtrue_1"]].to_numpy()[bi] if "vtrue_0" in Zt.columns else None
print(f"cohort {N}, treated {U0.mean():.1%}")
SCEN = {"actual": lambda a: U0, "none": lambda a: np.zeros(N), "all": lambda a: np.ones(N),
        "from55": lambda a: (a >= 55).astype(float), "from65": lambda a: (a >= 65).astype(float)}
AGES = np.arange(np.ceil(args.entry[1]), args.end_age + 0.01, 2.0)
ex, ey = np.linspace(*xr, args.grid + 1), np.linspace(*yr, args.grid + 1)


def moments(Zp, Vp, wgt, lam):
    q = Zp - origin
    x, y = q @ w, q @ e2
    vx, vy = Vp @ w, Vp @ e2
    xi = np.clip(np.digitize(x, ex) - 1, 0, args.grid - 1)
    yi = np.clip(np.digitize(y, ey) - 1, 0, args.grid - 1)
    G = args.grid
    acc = lambda val: np.bincount(xi * G + yi, weights=val, minlength=G * G).reshape(G, G)
    M = acc(wgt)
    Mx, My = acc(wgt * vx), acc(wgt * vy)
    Mxx, Myy = acc(wgt * vx * vx), acc(wgt * vy * vy)
    F = acc(wgt * lam)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = Mx / M, My / M
        var = (Mxx / M - ux ** 2) + (Myy / M - uy ** 2)
    keep = M * N >= 3                     # at least ~3 particles' worth of mass for a velocity estimate
    ux, uy, var = [np.where(keep, a, np.nan) for a in (ux, uy, var)]
    sm = lambda a: gaussian_filter(a, 0.7)
    return {"density": np.round(sm(M) / N * 1e6).astype(int).tolist(), "flux": np.round(sm(F) / N * 1e6).astype(int).tolist(),
            "ux": np.round(ux, 4).tolist(), "uy": np.round(uy, 4).tolist(), "spread": np.round(np.sqrt(np.clip(var, 0, None)), 4).tolist(),
            "mean": [float((x * wgt).sum() / wgt.sum()), float((y * wgt).sum() / wgt.sum())],
            "mean_v": [float((vx * wgt).sum() / wgt.sum()), float((vy * wgt).sum() / wgt.sum())]}


def run_model(name):
    """Phase-space particles; each particle is read out when ITS OWN age crosses each read-out age."""
    h = 0.25
    z = torch.as_tensor(Z0)
    mu, sd = m.v_start(z)
    v = mu + sd * torch.randn(mu.shape)
    a = A0.copy()
    lam = torch.exp(bundle.log_hazard(z)).numpy()
    cum = np.zeros(N)
    R = {A: [np.zeros((N, d)), np.zeros((N, d)), np.zeros(N), np.zeros(N), np.zeros(N, bool)] for A in AGES}
    while True:
        zn, vn = z.numpy(), v.numpy()
        for A in AGES:
            Rz, Rv, Rw, Rl, done = R[A]
            hit = (a >= A - 1e-9) & ~done
            if hit.any():
                Rz[hit], Rv[hit], Rw[hit], Rl[hit], done[hit] = zn[hit], vn[hit], np.exp(-cum[hit]), lam[hit], True
        if all(r[4].all() for r in R.values()):
            break
        u = torch.as_tensor(SCEN[name](a), dtype=torch.float32).reshape(-1, 1)
        with torch.enable_grad():
            z, v = m._step(z, v, u, torch.full((N, 1), h), torch.randn(N, d))
        z, v = z.detach(), v.detach()
        lam_new = torch.exp(bundle.log_hazard(z)).numpy()
        cum += 0.5 * (lam + lam_new) * h
        lam = lam_new
        a = a + h
    out = {A: moments(r[0], r[1], r[2], r[3]) for A, r in R.items()}
    alive = {A: float(r[2].mean()) for A, r in R.items()}
    return out, alive


def run_truth(name):
    dt = 0.05
    z = zt0.astype(np.float64).copy()
    vel = vt0.copy() if (tcfg.momentum and vt0 is not None) else None
    vbase, sig_v = syn.momentum_base(tcfg), tcfg.pace_sd * np.sqrt(2 * tcfg.gamma_v)
    a = A0.copy()
    lam = np.exp(syn.true_log_hazard(z, tcfg))
    cum = np.zeros(N)
    W = {A: [np.zeros(N), np.zeros(N, bool)] for A in AGES}
    while True:
        for A in AGES:
            Rw, done = W[A]
            hit = (a >= A - 1e-9) & ~done
            Rw[hit], done[hit] = np.exp(-cum[hit]), True
        if all(r[1].all() for r in W.values()):
            break
        u = SCEN[name](a)
        z = z + syn.true_drift(z, a, u, tcfg, vel) * dt + syn.true_diffusion(z, a, tcfg) * np.sqrt(dt) * rng.normal(size=z.shape)
        if vel is not None:
            vel = vel - tcfg.gamma_v * (vel - vbase) * dt + sig_v * np.sqrt(dt) * rng.normal(size=vel.shape)
        lam_new = np.exp(syn.true_log_hazard(z, tcfg))
        cum += 0.5 * (lam + lam_new) * dt
        lam = lam_new
        a = a + dt
    return {A: float(r[0].mean()) for A, r in W.items()}


res = {"key": args.key, "run": str(run), "n": N, "ages": AGES.tolist(), "grid": args.grid,
       "edges": [ex.round(4).tolist(), ey.round(4).tolist()],
       "plane": {"x": np.round(S["x"], 4).tolist(), "y": np.round(S["y"], 4).tolist(), "U": np.round(S["U"], 4).tolist(),
                 "curl_frac": round(S["curl_frac"], 3)}, "scenarios": {}}
for name in SCEN:
    fl, al = run_model(name)
    tal = run_truth(name)
    res["scenarios"][name] = {"fields": [fl[A] for A in AGES], "alive_model": [al[A] for A in AGES],
                              "alive_truth": [tal[A] for A in AGES]}
    print(name, "alive@95 model", round(al[AGES[-1]] / al[AGES[0]], 3), "truth", round(tal[AGES[-1]] / tal[AGES[0]], 3), flush=True)
p = Path(args.out) / f"{args.key}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res, separators=(",", ":"), allow_nan=True).replace("NaN", "null"))
print("wrote", p, round(p.stat().st_size / 1e6, 2), "MB")
