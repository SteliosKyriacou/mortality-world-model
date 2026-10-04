import numpy as np
import pandas as pd
import pytest
import torch

from mwm.data import common
from mwm.data import synthetic as syn
from mwm.eval.metrics import crps_samples, interval_coverage, masked_mae_rmse
from mwm.interrogate.enrichment import bh, enrichment_test


# ------------------------------------------------------------------ data contract
def _long():
    return pd.DataFrame({"person_id": ["a", "a", "b"], "cohort": "x", "visit_age": [50.0, 54.0, 61.2],
                         "feature": ["crp", "crp", "sbp"], "value": [1.0, 2.0, 130.0],
                         "unit": ["mg/L", "mg/L", "mmHg"]})


def test_validate_long_and_outcomes(tmp_path):
    L = common.validate_long(_long())
    assert L["visit_age"].dtype == np.float64
    bad = _long()
    bad.loc[0, "value"] = np.nan
    with pytest.raises(ValueError):
        common.validate_long(bad)
    O = pd.DataFrame({"person_id": ["a", "b"], "age_at_baseline": [50, 61.2],
                      "age_at_death_or_censor": [60, 70], "event": [1, 0], "cause": ["cvd", "none"]})
    common.validate_outcomes(O)
    O2 = O.copy()
    O2.loc[1, "cause"] = "cvd"
    with pytest.raises(ValueError):
        common.validate_outcomes(O2)
    common.save_cohort(L, O, "toy", root=tmp_path)
    L2, O3 = common.load_cohort("toy", root=tmp_path)
    assert len(L2) == 3 and len(O3) == 2


def test_visits_standardizer_splits(tmp_path):
    vt = common.long_to_visits(common.validate_long(_long()))
    assert vt.X.shape == (3, 2) and np.isnan(vt.X).sum() == 3
    assert list(vt.visit_index) == [0, 1, 0]
    st = common.Standardizer(vt.features, {"crp"}).fit(vt.X)
    Z = st.transform(vt.X)
    assert np.allclose(st.inverse(np.nan_to_num(Z))[0, 0], 1.0)
    O = pd.DataFrame({"person_id": [f"p{i}" for i in range(400)], "age_at_baseline": np.linspace(40, 70, 400),
                      "age_at_death_or_censor": np.linspace(50, 80, 400), "event": np.arange(400) % 5 == 0,
                      "cause": "none"})
    O["event"] = O["event"].astype(int)
    O.loc[O.event == 1, "cause"] = "other"
    s = common.make_splits(O)
    assert set(s.split) == {"train", "val", "test"}
    assert abs((s.split == "train").mean() - 0.75) < 0.03
    common.save_splits(s, "toy", tmp_path)
    common.save_splits(s, "toy", tmp_path)  # identical -> ok
    s2 = s.copy()
    s2.loc[0, "split"] = "test" if s2.loc[0, "split"] != "test" else "train"
    with pytest.raises(FileExistsError):
        common.save_splits(s2, "toy", tmp_path)


# ------------------------------------------------------------------ synthetic ground truth
def test_synthetic_truth_functions():
    on = syn.SynthConfig()
    off = syn.hallmarks_off(on)
    z = np.zeros((5, 8))
    z[:, 0] = np.linspace(0, 4, 5)
    # irreversibility: ON drift along age core is positive everywhere, OFF reverts
    assert (syn.true_drift(z, 60.0, 0.0, on)[:, 0] > 0).all()
    assert (syn.true_drift(z, 60.0, 0.0, off)[-1, 0] < 0)
    # nutrient: u slows z2 only when ON
    assert syn.true_drift(z, 60, 1.0, on)[0, 2] < syn.true_drift(z, 60, 0.0, on)[0, 2]
    assert syn.true_drift(z, 60, 1.0, off)[0, 2] == syn.true_drift(z, 60, 0.0, off)[0, 2]
    # dispersion: grows with age when ON, constant when OFF
    assert syn.true_diffusion(z, 80.0, on)[0, 0] > syn.true_diffusion(z, 45.0, on)[0, 0]
    assert syn.true_diffusion(z, 80.0, off)[0, 0] == syn.true_diffusion(z, 45.0, off)[0, 0]
    # inflammaging target: convex (inflection) when ON, linear when OFF
    z0 = np.array([1.0, 2.0, 3.0])
    t_on, t_off = syn.infl_target(z0, on), syn.infl_target(z0, off)
    assert (t_on[2] - t_on[1]) > 3 * (t_on[1] - t_on[0])
    assert np.isclose(t_off[2] - t_off[1], t_off[1] - t_off[0])
    # drift = -grad V + non-gradient parts: numerical check for the gradient part on dims 3..7 w/o rotation
    zz = np.random.default_rng(0).normal(size=(4, 8))
    eps = 1e-5
    g = np.stack([(syn.true_potential(zz + eps * np.eye(8)[k], on) - syn.true_potential(zz - eps * np.eye(8)[k], on)) / (2 * eps)
                  for k in range(8)], 1)
    f = syn.true_drift(zz, 60, 0.0, on)
    assert np.allclose(f[:, [5, 6, 7]], -g[:, [5, 6, 7]], atol=1e-6)


def test_synthetic_cohort_small(tmp_path):
    long, out, truth = syn.simulate_cohort(syn.SynthConfig(n_persons=800, seed=1), tmp_path / "c", verbose=False)
    common.validate_long(long)
    common.validate_outcomes(out)
    assert out.event.mean() > 0.02
    n_vis = truth.groupby("person_id").size()
    assert (n_vis >= 2).mean() > 0.25
    assert (tmp_path / "c" / "truth" / "meta.json").exists()
    cfg, meta, lat, L = syn.load_truth(tmp_path / "c")
    assert len(meta["pathways"]) == 20 and lat.shape[0] == len(truth)


# ------------------------------------------------------------------ metrics
def test_crps_gaussian_closed_form():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(4000, 1, 1))
    y = np.array([[0.5]])
    # closed form CRPS of N(0,1) at y
    from scipy.stats import norm
    cf = 0.5 * (2 * norm.cdf(0.5) - 1) + 2 * norm.pdf(0.5) - 1 / np.sqrt(np.pi)
    assert abs(crps_samples(S, y)[0, 0] - cf) < 0.02
    cov = interval_coverage(rng.normal(size=(500, 2000, 1)), rng.normal(size=(2000, 1)))
    assert abs(cov["cov90"] - 0.9) < 0.03
    m = masked_mae_rmse(np.zeros((3, 2)), np.array([[1, np.nan], [1, 2], [np.nan, np.nan]]))
    assert np.isclose(m["mae"], 4 / 3)


# ------------------------------------------------------------------ enrichment
def test_enrichment_detects_planted_pathway():
    rng = np.random.default_rng(0)
    d, G = 6, 200
    J = rng.normal(0, 0.1, (G, d))
    J[:20, 0] += 1.0          # pathway P0 loads on latent 0
    J[20:40, 1] += 1.0        # P1 loads on latent 1
    genes = [f"g{i}" for i in range(G)]
    pw = {f"P{k}": genes[20 * k:20 * (k + 1)] for k in range(10)}
    V = np.eye(d)[:, :1]      # direction = latent 0
    res = enrichment_test(J, V, genes, pw, n_dir=300, seed=0)
    assert res["detected"] == ["P0"]
    assert np.all(np.diff(bh(np.array([0.01, 0.02, 0.5]))) >= 0)


# ------------------------------------------------------------------ dynamics
def test_models_learn_ou_drift():
    """1D-ish OU with constant tilt: A (SDE) and B' (flow + score) recover drift sign/scale."""
    from mwm.dynamics.flow import StochasticFlow
    from mwm.dynamics.sde import NeuralSDE
    from mwm.dynamics.train import fit_bprime, train_pairs
    rng = np.random.default_rng(0)
    n, k, s = 3000, 0.5, 0.3
    z0 = rng.normal(0, s / np.sqrt(2 * k), (n, 2))
    dt = rng.uniform(0.5, 3, n)
    m = z0 * np.exp(-k * dt)[:, None]
    v = s ** 2 / (2 * k) * (1 - np.exp(-2 * k * dt))
    z1 = m + np.sqrt(v)[:, None] * rng.normal(size=(n, 2))
    a0 = rng.uniform(40, 70, n)
    P = {"z0": z0, "z1": z1, "a0": a0, "a1": a0 + dt, "u": np.zeros((n, 1))}
    tr = {k_: v_[:2500] for k_, v_ in P.items()}
    va = {k_: v_[2500:] for k_, v_ in P.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zt = torch.tensor([[1.0, 0.0], [-1.0, 0.5]], device=dev)
    at = torch.tensor([55.0, 55.0], device=dev)
    ut = torch.zeros(2, 1, device=dev)
    A = NeuralSDE(2, hidden=32, solver="native", n_steps=10, learn_obs_noise=False)
    train_pairs(A, tr, va, epochs=40, batch=256, device=dev, loss_kw={"K": 16}, patience=100)
    fA = A.drift(zt, at, ut).detach().cpu().numpy()
    assert fA[0, 0] < -0.2 and fA[1, 0] > 0.2
    sA = A.diffusion(zt, at).detach().cpu().numpy()
    assert np.all((sA > 0.15) & (sA < 0.5))
    B = StochasticFlow(2, hidden=32)
    train_pairs(B, tr, va, epochs=60, batch=256, device=dev, val_every=10, patience=100)
    fit_bprime(B, tr, va, P["z0"], P["a0"], P["u"], dsm_iters=800, diff_epochs=8, device=dev, n_steps=10, K=16)
    fB = B.drift(zt, at, ut).detach().cpu().numpy()
    assert fB[0, 0] < -0.2 and fB[1, 0] > 0.2


def test_torchsde_solvers_run():
    from mwm.dynamics.sde import NeuralSDE
    for solver in ("euler", "reversible_heun"):
        m = NeuralSDE(3, hidden=16, solver=solver, n_steps=5)
        z0 = torch.randn(4, 3)
        a0 = torch.full((4,), 50.0)
        S = m.simulate(z0, a0, a0 + 2, torch.zeros(4, 1), K=3)
        assert S.shape == (3, 4, 3) and torch.isfinite(S).all()
        S.sum().backward()
