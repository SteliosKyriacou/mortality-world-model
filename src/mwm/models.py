"""Named, trained Model A variants. Each lives in its own directory and is loaded by name:

    from mwm.models import load_model_a, VARIANTS
    m = load_model_a("model_a_hetgnn", cohort="on", seed=0)
    m.bundle   # frozen encoder + decoder + hazard head (EncoderBundle)
    m.sde      # Model A dynamics (NeuralSDE)
    m.latents  # per-visit latents (DataFrame)
    m.results  # evaluation results (dict)

Variants:
    model_a_transformer  - masked tabular transformer encoder  (configs/model_a_transformer.yaml)
    model_a_hetgnn       - heterogeneous graph network encoder (configs/model_a_hetgnn.yaml)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .dynamics.sde import NeuralSDE
from .encoders.train import EncoderBundle

ROOT = Path(__file__).resolve().parents[2] / "models"
VARIANTS = ("model_a_transformer", "model_a_hetgnn")


@dataclass
class TrainedModelA:
    variant: str
    cohort: str
    seed: int
    path: Path
    bundle: EncoderBundle
    sde: NeuralSDE
    latents: pd.DataFrame
    results: dict


def model_dir(variant: str, cohort: str = "on", seed: int = 0, root: Path | str = ROOT) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return Path(root) / variant / cohort / f"seed{seed}"


def load_model_a(variant: str, cohort: str = "on", seed: int = 0, device: str = "cuda",
                 root: Path | str = ROOT) -> TrainedModelA:
    d = model_dir(variant, cohort, seed, root)
    if not (d / "model_A.pt").exists():
        raise FileNotFoundError(f"{d} has no model_A.pt; train it with scripts/train_model_a.py "
                                f"--config configs/{variant}.yaml")
    bundle = EncoderBundle.load(d / "encoder.pt").to(device)
    lat = pd.read_parquet(d / "latents.parquet")
    dz = sum(c.startswith("z_") for c in lat.columns)
    sde = NeuralSDE.from_run_dir(d, dz, device)
    sde.set_obs_noise(json.loads((d / "encoder_meta.json").read_text())["latent_obs_noise_var"])
    sde.load_state_dict(torch.load(d / "model_A.pt", map_location=device, weights_only=True))
    sde.eval()
    res = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else {}
    return TrainedModelA(variant, cohort, seed, d, bundle, sde, lat, res)
