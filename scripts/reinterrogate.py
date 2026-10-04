"""Re-run only the hallmark interrogation on saved runs (no retraining), e.g. after a test is
re-specified. Rebuilds encoder, latents and models from runs/<...>/seed*/ and overwrites the
"hallmarks" block of results.json (the old block is kept as "hallmarks_previous").

    conda run -n mwm python scripts/reinterrogate.py --runs runs/synthetic --config configs/synthetic.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from mwm.data.dataset import load_cohort_dir
from mwm.data.synthetic import load_truth
from mwm.dynamics.flow import FlowModel, StochasticFlow
from mwm.dynamics.sde import NeuralSDE
from mwm.encoders.train import EncoderBundle
from mwm.eval import baselines as bl
from mwm.pipeline import Logger, jsonable, run_hallmarks

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs/synthetic")
ap.add_argument("--config", default="configs/synthetic.yaml")
ap.add_argument("--data-root", default="data/synthetic")
ap.add_argument("--seeds", default=None, help="comma list of seed dirs to redo (default all)")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
dev = cfg["device"]
log = Logger(Path(args.runs) / "reinterrogate_log.txt")
a, b = cfg["model_a"], cfg["model_b"]
for cohort in ("on", "off"):
    cdir = Path(args.data_root) / cohort
    tcfg, meta, tl, _ = load_truth(cdir)
    ca = load_cohort_dir(cdir)
    nc = len(ca.clinical)
    for sd in sorted((Path(args.runs) / cohort).glob("seed*")):
        if args.seeds and sd.name.replace("seed", "") not in args.seeds.split(","):
            continue
        res = json.loads((sd / "results.json").read_text())
        bundle = EncoderBundle.load(sd / "encoder.pt").to(dev)
        lat = pd.read_parquet(sd / "latents.parquet")
        zc = [c for c in lat.columns if c.startswith("z_")]
        d = len(zc)
        ov = json.loads((sd / "encoder_meta.json").read_text())["latent_obs_noise_var"]
        specs = {"A": lambda: NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"]),
                 "B": lambda: FlowModel(d, hidden=b["hidden"], jac_penalty=b.get("jac_penalty", 0.1)),
                 "Bp": lambda: StochasticFlow(d, hidden=b["hidden"], jac_penalty=b.get("jac_penalty", 0.1),
                                              dsm_sigma=cfg["model_bprime"].get("dsm_sigma", 0.5)),
                 "A_constSigma": lambda: NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"],
                                                   diffusion_mode="constant"),
                 "A_noPotential": lambda: NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"],
                                                    use_potential=False),
                 "A_shuffledAge": lambda: NeuralSDE(d, hidden=a["hidden"], solver=a["solver"], n_steps=a["n_steps"])}
        models = {}
        for name, mk in specs.items():
            f = sd / f"model_{name}.pt"
            if not f.exists():
                continue
            m = mk().to(dev)
            if m.stochastic and cfg.get("obs_noise", "fixed") == "fixed":
                m.set_obs_noise(ov)
            m.load_state_dict(torch.load(f, map_location=dev, weights_only=True))
            models[name] = m.eval()
        # Cox on baseline features for the attribution control
        base = lat.groupby("person_id").head(1)
        o = ca.outcomes.reindex(base["person_id"])
        t = (o["age_at_death_or_censor"].to_numpy() - base["visit_age"].to_numpy()).clip(1e-3)
        tr = (base["split"] == "train").to_numpy()
        cox = bl.CoxBaseline().fit(ca.Xs[base.index.to_numpy(), :nc][tr], base.visit_age.to_numpy()[tr],
                                   t[tr], o["event"].to_numpy()[tr])
        hm, r2 = run_hallmarks(models, bundle, ca, lat, zc, meta, cox, dev, log)
        res["hallmarks_previous"] = res.get("hallmarks")
        # models without saved weights (older runs' controls) keep their previous results
        res["hallmarks"] = {**(res.get("hallmarks") or {}), **hm}
        res["latent_age_probe_r2_test"] = r2
        (sd / "results.json").write_text(json.dumps(jsonable(res), indent=1))
        log(f"re-interrogated {sd}")
