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


def test_surface_model_is_pure_gradient_flow(tmp_path):
    torch.manual_seed(0)
    m = NeuralSDE(2, hidden=32, solver="native", n_steps=8, field_kind="surface")
    assert not m.uses_age and m.diff.mode == "state"
    z = torch.randn(64, 2, requires_grad=True)
    a, u0, u1 = torch.full((64,), 50.0), torch.zeros(64, 1), torch.ones(64, 1)
    f = m.drift(z, a, u0)
    # drift is minus the gradient of the surface, and has zero curl (symmetric Jacobian) in 2D
    g = torch.autograd.grad(m.field.potential(z, u0).sum(), z, create_graph=True)[0]
    assert torch.allclose(f, -g, atol=1e-5)
    J01 = torch.autograd.grad(f[:, 0].sum(), z, retain_graph=True)[0][:, 1]
    J10 = torch.autograd.grad(f[:, 1].sum(), z, retain_graph=True)[0][:, 0]
    assert torch.allclose(J01, J10, atol=1e-4)
    # the intervention changes the terrain, and age does not matter
    assert not torch.allclose(m.drift(z, a, u1), f)
    assert torch.allclose(m.drift(z, a + 30, u0), f)
    m.save_config(tmp_path)
    assert NeuralSDE.from_run_dir(tmp_path, 2).cfg["field_kind"] == "surface"
