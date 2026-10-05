import json

import torch

from mwm.dynamics.sde import NeuralSDE


def test_age_free_model_ignores_age(tmp_path):
    torch.manual_seed(0)
    m = NeuralSDE(8, hidden=32, solver="native", n_steps=8, use_age=False)
    assert m.diff.mode == "state" and not m.uses_age
    z, u = torch.randn(5, 8), torch.zeros(5, 1)
    a1, a2 = torch.full((5,), 45.0), torch.full((5,), 85.0)
    assert torch.allclose(m.drift(z, a1, u), m.drift(z, a2, u))
    assert torch.allclose(m.diffusion(z, a1), m.diffusion(z, a2))
    # elapsed time still matters: a longer interval moves the state further on average
    m.eval()
    S1 = m.sample(z, a1, a1 + 1.0, u, n_samples=1, n_steps=8)
    S2 = m.sample(z, a1, a1 + 5.0, u, n_samples=1, n_steps=8)
    assert not torch.allclose(S1, S2)
    # the saved config rebuilds the same architecture
    m.save_config(tmp_path)
    m2 = NeuralSDE.from_run_dir(tmp_path, 8)
    assert json.loads((tmp_path / "model_A_config.json").read_text())["use_age"] is False
    assert m2.diff.mode == "state" and not m2.field.use_age


def test_age_aware_default_unchanged(tmp_path):
    m = NeuralSDE(8, hidden=32, solver="native", n_steps=8)
    assert m.uses_age and m.diff.mode == "state_age"
    m2 = NeuralSDE.from_run_dir(tmp_path, 8, hidden=32)  # no config file -> original defaults
    assert m2.uses_age
