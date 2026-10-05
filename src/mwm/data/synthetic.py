"""Synthetic cohort with a known ground-truth latent SDE and planted hallmarks of aging.

Ground-truth latent z* in R^8 (indices fixed, see LATENT_NAMES):

    0 age_core   "damage" axis. ON: constant tilt beta0 (irreversible, no confinement).
                 OFF (irreversibility off): OU mean-reversion to c_rev -> old states rejuvenate
                 (and the nutrient axis 2 also gets mean reversion k_rev, so that no latent
                 direction remains monotone).
    1 infl       inflammation. relaxes (rate k1) to a target driven by z0:
                 ON: gamma * w * softplus((z0-theta)/w)  (late-life inflection, ~age 65)
                 OFF: c_lin * (z0 - z0_ref) (linear rise, no inflection). Hazard weight b1 ON only.
    2 nutr       nutrient sensing. tilt beta2 * (1 - kappa*u). OFF: kappa = 0 (u does nothing).
    3,4 rot      damped rotation (non-gradient homeostatic oscillation).
    5 frail      ON: double well (robust -1 / frail +1, frail has higher hazard). OFF: single well.
    6,7 ou       stable OU nuisance dims.

    dz = f(z, a, u) dt + diag(sigma(a)) dW,  f = -grad V(z) + R(z) + C(z) + F(u)

Diffusion: ON sigma_i(a) = s_i * (0.4 + 1.2 * clip((a-30)/50, 0, 1.4)); OFF constant s_i.

Observation: ~33 clinical features and a 500-gene "omics" block (20 pathways x 25 genes),
decoded from z* with noise; irregular UKB-like visits, missingness, hazard-driven deaths.
Pathway hallmark (e) ON: genes load on the latent of their pathway; OFF: random loadings.

Everything needed to evaluate recovery (parameters, true latents per visit, loading
matrices, pathway membership) is saved under `<out>/truth/`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .common import save_splits, make_splits, validate_long, validate_outcomes

LATENT_NAMES = ["age_core", "infl", "nutr", "rot_a", "rot_b", "frail", "ou_a", "ou_b"]
D_TRUE = 8
HALLMARKS = ("inflammaging", "dispersion", "irreversibility", "nutrient", "pathways", "frailty_basin")

# feature -> (primary latent, loading, mean, sd, unit, log?, base missing rate)
CLINICAL = {
    # age core (0)
    "sbp":           ([(0, 0.8), (5, 0.1)], 132.0, 17.0, "mmHg", False, 0.02),
    "dbp":           ([(0, 0.4), (3, 0.3)], 80.0, 10.0, "mmHg", False, 0.02),
    "creatinine":    ([(0, 0.7)], 4.3, 0.22, "umol/L", True, 0.05),
    "egfr":          ([(0, -0.85)], 85.0, 15.0, "mL/min/1.73m2", False, 0.05),
    "cystatin_c":    ([(0, 0.8)], 0.9, 0.18, "mg/L", False, 0.45),
    "nt_probnp":     ([(0, 0.7), (1, 0.2)], 4.3, 0.8, "pg/mL", True, 0.5),
    "rdw":           ([(0, 0.5), (1, 0.2)], 13.4, 0.9, "%", False, 0.06),
    "haemoglobin":   ([(0, -0.4), (5, -0.2)], 14.0, 1.2, "g/dL", False, 0.05),
    "fev1":          ([(0, -0.7), (5, -0.2)], 2.9, 0.7, "L", False, 0.35),
    # inflammation (1)
    "crp":           ([(1, 0.9)], 0.4, 1.0, "mg/L", True, 0.06),
    "il6":           ([(1, 0.8)], 0.9, 0.6, "pg/mL", True, 0.5),
    "fibrinogen":    ([(1, 0.7), (0, 0.1)], 1.05, 0.2, "g/L", True, 0.3),
    "wbc":           ([(1, 0.6)], 1.9, 0.25, "10^9/L", True, 0.05),
    # nutrient sensing (2)
    "glucose":       ([(2, 0.8)], 1.65, 0.15, "mmol/L", True, 0.08),
    "hba1c":         ([(2, 0.85)], 36.0, 5.0, "mmol/mol", False, 0.06),
    "triglycerides": ([(2, 0.7)], 0.4, 0.45, "mmol/L", True, 0.06),
    "insulin":       ([(2, 0.75)], 2.2, 0.6, "pmol/L", True, 0.5),
    "hdl":           ([(2, -0.6)], 1.45, 0.38, "mmol/L", False, 0.06),
    "bmi":           ([(2, 0.6), (6, 0.3)], 27.4, 4.5, "kg/m2", False, 0.01),
    "waist":         ([(2, 0.6), (6, 0.3)], 90.0, 13.0, "cm", False, 0.02),
    "alt":           ([(2, 0.5), (7, 0.2)], 3.0, 0.4, "U/L", True, 0.1),
    # rotation (3, 4)
    "total_chol":    ([(3, 0.8)], 5.7, 1.1, "mmol/L", False, 0.05),
    "ldl":           ([(3, 0.8), (2, 0.15)], 3.55, 0.85, "mmol/L", False, 0.06),
    "heart_rate":    ([(4, 0.8)], 69.0, 11.0, "bpm", False, 0.03),
    "cortisol":      ([(4, 0.7), (3, 0.2)], 5.8, 0.4, "nmol/L", True, 0.55),
    # frailty (5)
    "grip":          ([(5, -0.6), (0, -0.4)], 31.0, 10.0, "kg", False, 0.04),
    "walk_speed":    ([(5, -0.7), (0, -0.2)], 1.2, 0.25, "m/s", False, 0.4),
    "albumin":       ([(5, -0.5), (1, -0.25)], 45.0, 2.6, "g/L", False, 0.06),
    "frailty_index": ([(5, 0.8)], 0.12, 0.08, "fraction", False, 0.2),
    # nuisance OU (6, 7)
    "platelets":     ([(6, 0.8)], 250.0, 58.0, "10^9/L", False, 0.05),
    "mcv":           ([(6, -0.6), (7, 0.3)], 91.0, 4.5, "fL", False, 0.05),
    "vitamin_d":     ([(7, 0.8)], 3.8, 0.45, "nmol/L", True, 0.12),
    "sodium":        ([(7, 0.6)], 140.0, 2.3, "mmol/L", False, 0.08),
}
INFLAMMATORY_FEATURES = ["crp", "il6", "fibrinogen", "wbc"]
METABOLIC_FEATURES = ["glucose", "hba1c", "triglycerides", "insulin", "hdl", "bmi", "waist", "alt"]
INTERVENTION_FEATURE = "u_nutrient"

# pathway -> (latent index or None, sign). 20 pathways x 25 genes.
PATHWAYS = {
    "OXIDATIVE_PHOSPHORYLATION": (0, -1), "UNFOLDED_PROTEIN_RESPONSE": (0, +1),
    "DNA_REPAIR": (0, +1), "SASP": (1, +1), "INFLAMMATORY_RESPONSE": (1, +1),
    "MTORC1_SIGNALING": (2, +1), "IGF1_INSULIN_SIGNALING": (2, +1),
    "CHOLESTEROL_HOMEOSTASIS": (3, +1), "E2F_TARGETS": (3, +1), "WNT_SIGNALING": (4, +1),
    "HYPOXIA": (4, -1), "MYOGENESIS": (5, -1), "HEME_METABOLISM": (6, +1),
    "COAGULATION": (6, +1), "XENOBIOTIC_METABOLISM": (7, +1), "BILE_ACID_METABOLISM": (7, -1),
    "SPERMATOGENESIS": (None, 0), "PANCREAS_BETA_CELLS": (None, 0),
    "PEROXISOME": (None, 0), "ANGIOGENESIS": (None, 0),
}
AGING_PATHWAYS = [p for p, (k, _) in PATHWAYS.items() if k in (0, 1, 2)]
GENES_PER_PATHWAY = 25


@dataclass
class SynthConfig:
    n_persons: int = 20000
    seed: int = 0
    hallmarks: dict = field(default_factory=lambda: {h: True for h in HALLMARKS})
    # dynamics
    start_age: float = 30.0
    dt: float = 0.05
    beta0: float = 0.06
    beta2: float = 0.04
    kappa: float = 0.5
    k_rev: float = 0.08
    c_rev: float = 2.0
    k1: float = 0.4
    gamma: float = 1.5
    theta: float = 2.1
    w: float = 0.25
    c_lin: float = 0.6
    z0_ref: float = 1.0
    k_rot: float = 0.25
    omega: float = 0.6
    h_frail: float = 0.4
    tilt_frail: float = 0.04
    k5_single: float = 0.6
    k_ou: tuple = (0.25, 0.35)
    sigma_base: tuple = (0.08, 0.25, 0.10, 0.30, 0.30, 0.45, 0.35, 0.35)
    # hazard log-lambda = alpha + b0 z0 + b1 z1 + b2 z2 + b5 * sigmoid(3 z5)
    alpha: float = -7.8
    b0: float = 1.2
    b1: float = 0.45
    b2: float = 0.3
    b5: float = 0.9
    cvd_share: float = 0.35
    # observation
    feat_noise: float = 0.35
    n_genes: int = 500
    gene_noise: float = 0.5
    omics_fraction: float = 0.25
    omics_repeat: float = 0.5
    # visits
    baseline_age: tuple = (40.0, 70.0)
    followup_probs: tuple = (0.55, 0.25, 0.12, 0.08)
    gap_range: tuple = (1.0, 8.0)
    admin_followup: tuple = (12.0, 16.0)
    p_intervention: float = 0.3
    # momentum ("pace of aging" as a persistent hidden velocity) on the age-core and nutrient axes.
    # momentum=True: each person's pace v follows dv = -gamma_v (v - base) dt + sigma_v dW, with
    #   stationary sd pace_sd, and dz_k = v_k dt replaces the constant tilt (persistence 1/gamma_v years).
    # momentum_null=True: matched memoryless null - no velocity state; extra white noise on the same
    #   axes with the same displacement variance over a 4-year gap.
    momentum: bool = False
    momentum_null: bool = False
    gamma_v: float = 0.2
    pace_sd: float = 0.04

    def on(self, h: str) -> bool:
        return bool(self.hallmarks.get(h, True))

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SynthConfig":
        d = dict(d)
        for k in ("k_ou", "sigma_base", "baseline_age", "followup_probs", "gap_range", "admin_followup"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def hallmarks_off(cfg: SynthConfig) -> SynthConfig:
    d = cfg.to_json()
    d["hallmarks"] = {h: False for h in HALLMARKS}
    return SynthConfig.from_json(d)


# --------------------------------------------------------------------------------------
# ground-truth dynamics (numpy; vectorised over leading dims; also used by evaluation)
# --------------------------------------------------------------------------------------
def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def true_potential(z: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """Gradient part V(z) (the non-gradient rotation/coupling/forcing are separate)."""
    z = np.asarray(z, dtype=np.float64)
    V = np.zeros(z.shape[:-1])
    if cfg.on("irreversibility"):
        V += -cfg.beta0 * z[..., 0]
    else:
        V += 0.5 * cfg.k_rev * (z[..., 0] - cfg.c_rev) ** 2
    V += 0.5 * cfg.k1 * z[..., 1] ** 2
    V += -cfg.beta2 * z[..., 2]
    if not cfg.on("irreversibility"):
        V += 0.5 * cfg.k_rev * z[..., 2] ** 2
    V += 0.5 * cfg.k_rot * (z[..., 3] ** 2 + z[..., 4] ** 2)
    if cfg.on("frailty_basin"):
        V += cfg.h_frail * (z[..., 5] ** 2 - 1.0) ** 2 - cfg.tilt_frail * z[..., 5]
    else:
        V += 0.5 * cfg.k5_single * (z[..., 5] + 1.0) ** 2
    V += 0.5 * cfg.k_ou[0] * z[..., 6] ** 2 + 0.5 * cfg.k_ou[1] * z[..., 7] ** 2
    return V


def infl_target(z0: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    if cfg.on("inflammaging"):
        return cfg.gamma * cfg.w * _softplus((z0 - cfg.theta) / cfg.w)
    return cfg.c_lin * (z0 - cfg.z0_ref)


MOMENTUM_AXES = (0, 2)          # age core, nutrient


def momentum_base(cfg: SynthConfig) -> np.ndarray:
    return np.array([cfg.beta0, cfg.beta2])


def momentum_null_sigma(cfg: SynthConfig, gap: float = 4.0) -> float:
    """White-noise sd giving the same displacement variance over `gap` years as the OU pace."""
    g, s = cfg.gamma_v, cfg.pace_sd
    var = 2 * s ** 2 / g ** 2 * (g * gap - 1 + np.exp(-g * gap))
    return float(np.sqrt(var / gap))


def true_drift(z: np.ndarray, age: np.ndarray, u: np.ndarray, cfg: SynthConfig, v: np.ndarray | None = None) -> np.ndarray:
    """f(z, a, u) = -grad V + rotation + inflammation coupling + intervention forcing.
    With cfg.momentum, pass the hidden pace v (..., 2); without it the mean pace is used."""
    z = np.asarray(z, dtype=np.float64)
    u = np.broadcast_to(np.asarray(u, dtype=np.float64), z.shape[:-1])
    f = np.zeros_like(z)
    if cfg.on("irreversibility"):
        f[..., 0] = cfg.beta0
    else:
        f[..., 0] = -cfg.k_rev * (z[..., 0] - cfg.c_rev)
    # inflammation: -k1 z1 is gradient; +k1*target(z0) is non-gradient coupling
    f[..., 1] = -cfg.k1 * (z[..., 1] - infl_target(z[..., 0], cfg))
    kap = cfg.kappa if cfg.on("nutrient") else 0.0
    f[..., 2] = cfg.beta2 * (1.0 - kap * u)
    if not cfg.on("irreversibility"):  # nutrient axis also mean-reverts (no monotone direction left)
        f[..., 2] -= cfg.k_rev * z[..., 2]
    if cfg.momentum and v is not None:
        v = np.asarray(v, dtype=np.float64)
        f[..., 0] = v[..., 0] if cfg.on("irreversibility") else f[..., 0] + (v[..., 0] - cfg.beta0)
        f[..., 2] = f[..., 2] + (v[..., 1] - cfg.beta2) * (1.0 - kap * u)
    f[..., 3] = -cfg.k_rot * z[..., 3] - cfg.omega * z[..., 4]
    f[..., 4] = -cfg.k_rot * z[..., 4] + cfg.omega * z[..., 3]
    if cfg.on("frailty_basin"):
        f[..., 5] = -4.0 * cfg.h_frail * z[..., 5] * (z[..., 5] ** 2 - 1.0) + cfg.tilt_frail
    else:
        f[..., 5] = -cfg.k5_single * (z[..., 5] + 1.0)
    f[..., 6] = -cfg.k_ou[0] * z[..., 6]
    f[..., 7] = -cfg.k_ou[1] * z[..., 7]
    return f


def diffusion_scale(age: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    age = np.asarray(age, dtype=np.float64)
    if cfg.on("dispersion"):
        return 0.4 + 1.2 * np.clip((age - 30.0) / 50.0, 0.0, 1.4)
    return np.ones_like(age)


def true_diffusion(z: np.ndarray, age: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """Diagonal sigma(z, a) (state independent here), shape like z."""
    z = np.asarray(z, dtype=np.float64)
    s = diffusion_scale(np.broadcast_to(age, z.shape[:-1]), cfg)
    out = s[..., None] * np.asarray(cfg.sigma_base)
    if cfg.momentum_null:
        extra = momentum_null_sigma(cfg)
        for k in MOMENTUM_AXES:
            out[..., k] = np.sqrt(out[..., k] ** 2 + extra ** 2)
    return out


def true_log_hazard(z: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    lh = cfg.alpha + cfg.b0 * z[..., 0] + cfg.b2 * z[..., 2]
    if cfg.on("inflammaging"):
        lh = lh + cfg.b1 * z[..., 1]
    if cfg.on("frailty_basin"):
        lh = lh + cfg.b5 * _sigmoid(3.0 * z[..., 5])
    return lh


# --------------------------------------------------------------------------------------
# observation model
# --------------------------------------------------------------------------------------
LATENT_REF_SCALE = np.array([1.0, 0.8, 0.6, 0.6, 0.6, 0.8, 0.55, 0.5])
LATENT_REF_MEAN = np.array([1.8, 0.3, 0.6, 0.0, 0.0, -0.6, 0.0, 0.0])


def build_loadings(cfg: SynthConfig, rng: np.random.Generator):
    feats = list(CLINICAL)
    Wc = np.zeros((len(feats), D_TRUE))
    for j, f in enumerate(feats):
        for k, l in CLINICAL[f][0]:
            Wc[j, k] = l
    Wc += rng.normal(0, 0.05, Wc.shape) * (Wc == 0)
    # mild nonlinearity: quadratic term on primary latent for a few features
    quad = np.zeros(len(feats))
    for f in ("crp", "glucose", "nt_probnp", "frailty_index", "triglycerides"):
        quad[feats.index(f)] = 0.12
    # omics
    genes, membership = [], {}
    gpp = cfg.n_genes // len(PATHWAYS)
    Wg = np.zeros((gpp * len(PATHWAYS), D_TRUE))
    gi = 0
    for p, (k, sgn) in PATHWAYS.items():
        members = []
        for i in range(gpp):
            g = f"G{gi:03d}_{p[:6]}" if cfg.on("pathways") else f"G{gi:03d}"
            genes.append(g)
            members.append(g)
            if k is not None:
                Wg[gi, k] = sgn * rng.uniform(0.4, 0.9)
            gi += 1
        membership[p] = members
    Wg += rng.normal(0, 0.08, Wg.shape)
    if not cfg.on("pathways"):
        # same marginal loading magnitudes, but no pathway structure
        Wg = rng.permutation(Wg.ravel()).reshape(Wg.shape)
    return feats, Wc, quad, genes, Wg, membership


def decode_clinical_std(z: np.ndarray, Wc: np.ndarray, quad: np.ndarray) -> np.ndarray:
    zt = (z - LATENT_REF_MEAN) / LATENT_REF_SCALE
    lin = zt @ Wc.T
    prim = np.argmax(np.abs(Wc), axis=1)
    return lin + quad * zt[..., prim] ** 2


def std_to_raw(xs: np.ndarray, feats: list[str]) -> np.ndarray:
    out = np.empty_like(xs)
    for j, f in enumerate(feats):
        _, m, s, _, is_log, _ = CLINICAL[f]
        out[..., j] = np.exp(m + s * xs[..., j]) if is_log else m + s * xs[..., j]
    return out


# --------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------
def simulate_cohort(cfg: SynthConfig, out_dir: str | Path | None = None, verbose: bool = True):
    rng = np.random.default_rng(cfg.seed)
    feats, Wc, quad, genes, Wg, membership = build_loadings(cfg, rng)
    n_over = int(cfg.n_persons * 1.6)

    base_age = rng.uniform(*cfg.baseline_age, n_over)
    n_fu = rng.choice(len(cfg.followup_probs), size=n_over, p=cfg.followup_probs)
    gaps = rng.uniform(*cfg.gap_range, (n_over, len(cfg.followup_probs) - 1))
    visit_ages = np.full((n_over, len(cfg.followup_probs)), np.nan)
    visit_ages[:, 0] = base_age
    for v in range(1, visit_ages.shape[1]):
        visit_ages[:, v] = np.where(n_fu >= v, visit_ages[:, v - 1] + gaps[:, v - 1], np.nan)
    censor_age = base_age + rng.uniform(*cfg.admin_followup, n_over)
    # follow-up visits after admin censoring are dropped
    visit_ages = np.where(visit_ages <= censor_age[:, None], visit_ages, np.nan)
    u = (rng.uniform(size=n_over) < cfg.p_intervention).astype(np.float64)

    # initial state at start_age
    z = np.zeros((n_over, D_TRUE))
    z[:, 0] = rng.normal(0.0, 0.3, n_over)
    z[:, 1] = rng.normal(0.0, 0.3, n_over)
    z[:, 2] = rng.normal(0.0, 0.3, n_over)
    z[:, 3:5] = rng.normal(0.0, 0.5, (n_over, 2))
    z[:, 5] = -1.0 + rng.normal(0.0, 0.15, n_over)
    z[:, 6:8] = rng.normal(0.0, 0.5, (n_over, 2))

    alive = np.ones(n_over, bool)
    death_age = np.full(n_over, np.nan)
    z_at_visit = np.full(visit_ages.shape + (D_TRUE,), np.nan)
    vbase = momentum_base(cfg)
    vel = vbase + cfg.pace_sd * rng.normal(size=(n_over, 2)) if cfg.momentum else None
    v_at_visit = np.full(visit_ages.shape + (2,), np.nan)
    sig_v = cfg.pace_sd * np.sqrt(2 * cfg.gamma_v)
    end_age = censor_age.max() + cfg.dt
    n_steps = int(np.ceil((end_age - cfg.start_age) / cfg.dt))
    sq = np.sqrt(cfg.dt)
    for step in range(n_steps):
        a = cfg.start_age + step * cfg.dt
        a_next = a + cfg.dt
        # record visits in (a, a_next]: linear interpolation not needed at dt=0.05
        hit = (visit_ages > a - 1e-9) & (visit_ages <= a_next + 1e-9) & alive[:, None]
        if hit.any():
            ii, vv = np.nonzero(hit)
            z_at_visit[ii, vv] = z[ii]
            if cfg.momentum:
                v_at_visit[ii, vv] = vel[ii]
        u_eff = np.where(a >= base_age, u, 0.0)
        f = true_drift(z, a, u_eff, cfg, vel)
        g = true_diffusion(z, a, cfg)
        lam = np.exp(true_log_hazard(z, cfg))
        die = alive & (rng.uniform(size=n_over) < 1.0 - np.exp(-lam * cfg.dt)) & (a < censor_age)
        death_age[die] = a + rng.uniform(0, cfg.dt, die.sum())
        alive &= ~die
        z = z + f * cfg.dt + g * sq * rng.normal(size=z.shape)
        if cfg.momentum:
            vel = vel - cfg.gamma_v * (vel - vbase) * cfg.dt + sig_v * sq * rng.normal(size=vel.shape)

    # keep people alive at baseline; drop visits after death
    keep = ~(death_age <= base_age)
    idx = np.nonzero(keep)[0][: cfg.n_persons]
    if len(idx) < cfg.n_persons:
        raise RuntimeError("not enough survivors to baseline; increase oversampling")
    visit_ages, z_at_visit, u = visit_ages[idx], z_at_visit[idx], u[idx]
    v_at_visit = v_at_visit[idx]
    death_age, censor_age, base_age = death_age[idx], censor_age[idx], base_age[idx]
    after_death = visit_ages >= np.where(np.isnan(death_age), np.inf, death_age)[:, None]
    visit_ages[after_death] = np.nan
    z_at_visit[after_death] = np.nan
    v_at_visit[after_death] = np.nan
    event = (~np.isnan(death_age)).astype(np.int8)
    end = np.where(event == 1, death_age, censor_age)

    # cause of death: CVD more likely when the age-core/BP axis is high relative to others
    cause = np.full(len(idx), "none", dtype=object)
    pc = np.clip(cfg.cvd_share + 0.1 * (z_at_visit[:, 0, 0] - 1.8), 0.05, 0.9)
    r = rng.uniform(size=len(idx))
    cause[(event == 1) & (r < pc)] = "cvd"
    cause[(event == 1) & (r >= pc) & (r < pc + 0.3)] = "cancer"
    cause[(event == 1) & (r >= pc + 0.3)] = "other"

    # ---- observations ------------------------------------------------------------------
    pid = np.array([f"S{ i:06d}" for i in range(len(idx))])
    rows = []
    truth_rows = []
    base_miss = np.array([CLINICAL[f][5] for f in feats])
    has_omics = rng.uniform(size=len(idx)) < cfg.omics_fraction
    for v in range(visit_ages.shape[1]):
        ok = ~np.isnan(visit_ages[:, v])
        if not ok.any():
            continue
        zz = z_at_visit[ok, v]
        xs = decode_clinical_std(zz, Wc, quad) + rng.normal(0, cfg.feat_noise, (ok.sum(), len(feats)))
        raw = std_to_raw(xs, feats)
        # per-visit dropout of whole optional panels + per-feature missingness
        miss_p = base_miss[None, :] * (1.0 + 0.3 * v)
        m = rng.uniform(size=raw.shape) >= np.clip(miss_p, 0, 0.95)
        zt = (zz - LATENT_REF_MEAN) / LATENT_REF_SCALE
        xg = zt @ Wg.T + rng.normal(0, cfg.gene_noise, (ok.sum(), Wg.shape[0]))
        om_visit = has_omics[ok] & ((v == 0) | (rng.uniform(size=ok.sum()) < cfg.omics_repeat))
        mg = (rng.uniform(size=xg.shape) > 0.03) & om_visit[:, None]
        p_ok, a_ok = pid[ok], visit_ages[ok, v]
        for j, f in enumerate(feats):
            sel = m[:, j]
            rows.append(pd.DataFrame({"person_id": p_ok[sel], "visit_age": a_ok[sel],
                                      "feature": f, "value": raw[sel, j],
                                      "unit": CLINICAL[f][3]}))
        ii, jj = np.nonzero(mg)
        rows.append(pd.DataFrame({"person_id": p_ok[ii], "visit_age": a_ok[ii],
                                  "feature": np.asarray(genes)[jj], "value": 5.0 + xg[ii, jj],
                                  "unit": "NPX"}))
        rows.append(pd.DataFrame({"person_id": p_ok, "visit_age": a_ok, "feature": INTERVENTION_FEATURE,
                                  "value": u[ok], "unit": "flag"}))
        tr = pd.DataFrame(zz, columns=[f"ztrue_{k}" for k in range(D_TRUE)])
        tr.insert(0, "visit_age", a_ok)
        tr.insert(0, "person_id", p_ok)
        tr["u"] = u[ok]
        if cfg.momentum:
            tr["vtrue_0"], tr["vtrue_1"] = v_at_visit[ok, v, 0], v_at_visit[ok, v, 1]
        tr["has_omics"] = om_visit
        truth_rows.append(tr)
    long = pd.concat(rows, ignore_index=True)
    long.insert(1, "cohort", "synth")
    long = validate_long(long.round({"visit_age": 3}))
    outcomes = validate_outcomes(pd.DataFrame({
        "person_id": pid, "age_at_baseline": visit_ages[:, 0].round(3),
        "age_at_death_or_censor": np.maximum(end, visit_ages[:, 0]).round(3), "event": event,
        "cause": cause}))
    truth = pd.concat(truth_rows, ignore_index=True)
    truth["visit_age"] = truth["visit_age"].round(3)
    truth = truth.sort_values(["person_id", "visit_age"]).reset_index(drop=True)

    if verbose:
        n_vis = truth.groupby("person_id").size()
        print(f"[synth] persons={len(pid)} visits={len(truth)} with>=2 visits={(n_vis >= 2).sum()} "
              f"deaths={event.sum()} ({event.mean():.1%}) omics persons={has_omics.sum()} "
              f"hallmarks={cfg.hallmarks}")

    if out_dir is not None:
        out = Path(out_dir)
        (out / "truth").mkdir(parents=True, exist_ok=True)
        long.to_parquet(out / "long.parquet", index=False)
        outcomes.to_parquet(out / "outcomes.parquet", index=False)
        truth.to_parquet(out / "truth" / "latents.parquet", index=False)
        np.savez(out / "truth" / "loadings.npz", Wc=Wc, quad=quad, Wg=Wg,
                 ref_mean=LATENT_REF_MEAN, ref_scale=LATENT_REF_SCALE)
        meta = {"config": cfg.to_json(), "latent_names": LATENT_NAMES, "clinical_features": feats,
                "genes": genes, "pathways": membership if cfg.on("pathways") else
                {p: genes[i * (len(genes) // len(PATHWAYS)):(i + 1) * (len(genes) // len(PATHWAYS))]
                 for i, p in enumerate(PATHWAYS)},
                "aging_pathways": AGING_PATHWAYS, "pathway_latent": {p: v[0] for p, v in PATHWAYS.items()},
                "inflammatory_features": INFLAMMATORY_FEATURES, "metabolic_features": METABOLIC_FEATURES,
                "intervention_feature": INTERVENTION_FEATURE}
        (out / "truth" / "meta.json").write_text(json.dumps(meta, indent=1))
        splits = make_splits(outcomes, seed=20261003)
        save_splits(splits, "synth", root=out)
    return long, outcomes, truth


def load_truth(out_dir: str | Path):
    out = Path(out_dir)
    meta = json.loads((out / "truth" / "meta.json").read_text())
    cfg = SynthConfig.from_json(meta["config"])
    lat = pd.read_parquet(out / "truth" / "latents.parquet")
    L = dict(np.load(out / "truth" / "loadings.npz"))
    return cfg, meta, lat, L


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/on")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--off", action="store_true", help="all hallmarks off (null cohort)")
    a = ap.parse_args()
    c = SynthConfig(n_persons=a.n, seed=a.seed)
    if a.off:
        c = hallmarks_off(c)
    simulate_cohort(c, a.out)
