# model_a_hetgnn

Latent neural SDE (Model A) on a heterogeneous graph neural network encoder. Card generated 2026-10-04 from `models/model_a_hetgnn/*/seed*/results.json` by `scripts/make_model_cards.py`.

**Status:** trained and evaluated on the synthetic cohort only (validation ladder step 1). Nothing here is evidence about human biology.

## Load / retrain

```python
from mwm.models import load_model_a
m = load_model_a("model_a_hetgnn", cohort="on", seed=0)   # m.bundle, m.sde, m.latents, m.results
```
```bash
conda run -n mwm python scripts/train_model_a.py --config configs/model_a_hetgnn.yaml
```

## Architecture

**Encoder: heterogeneous graph neural network** (`src/mwm/encoders/hetgnn.py`, 157,456 parameters)
- A graph per visit from measured values only. Nodes: `visit` (learned constant start), one `clin` node per
  measured clinical value (`clin_e[j] + value × clin_w[j]`), one `prot` node per measured gene value
  (`prot_e[g] + value × prot_w[g]`), one `path` node per (visit, pathway) with ≥ 1 measured member (`path_e[p]`).
- Edges: clin→visit, prot→path (membership), prot→visit, path→visit, visit→clin.
- 2 rounds of HeteroConv: one GraphSAGE layer per edge type (mean aggregation + root weight, 8,256 parameters
  each), summed over edge types, residual `h ← h + ReLU(new)`. 82,560 parameters.
- Readout from the visit node: LayerNorm → Linear 64→64 → GELU → Linear 64→16 (5,328). Latent whitened.
- Pathway membership on synthetic data = the cohort's planted pathway lists (this partly hands the Jacobian
  pathway test its answer; judge that test against the hallmarks-OFF cohort). Real data: MSigDB/Reactome.

**Shared heads (trained with the encoder, then frozen)**
- Decoder: Linear 16→256 → GELU → Linear 256→256 → GELU → Linear 256→533, per-marker learned σ (207,658 parameters).
- Hazard head: Linear 16→64 → GELU → Linear 64→1, log death rate from z only (1,154 parameters).

**Model A dynamics** (`src/mwm/dynamics/sde.py`, 46,193 parameters): `dz = f(z,a,u)da + Σ(z,a)dW`,
`f = −∇V(z) + J(z,a,u)`.
- V: MLP 16→128→128→1 (SiLU), 18,817 parameters. J: MLP [z, (a−60)/10, u] 18→128→128→16, 21,008 parameters.
- Σ: diagonal, MLP [z, (a−60)/10] 17→64→64→16, softplus + 1e-4, initial σ = 0.2; 6,352 parameters.
- Observation noise r² fixed from encoder input-perturbation (errors-in-variables start jitter).
- Training: consecutive visit pairs, 32 Euler–Maruyama steps per pair in normalised time, K = 32 futures,
  Gaussian NLL of the next visit + 1e-3‖J‖² + 1e-3‖Σ‖²; AdamW lr 2e-3, wd 1e-5, cosine, batch 512, clip 5.

## Training settings

- Encoder: 40 epochs (masked autoencoder 30% hidden + hazard 0.5 + omics-consistency 0.5).
- Model A: 80 epochs, lr 0.002, checkpoint kept by val_nll (checked every 10 epochs).
- Seeds [0, 1, 2], cohorts ['on', 'off']; person-level split 75/10/15 (seed 20261003).
- Convergence: both encoder and Model A converge within these budgets; longer Model A training overfits (see `reports/convergence`).

## Results on held-out people (mean ± sd over seeds)

| Metric | hallmarks-ON | hallmarks-OFF |
|---|---|---|
| Forecast MAE, first → last visit (SD units) | 0.658 ± 0.002 | 0.761 ± 0.001 |
| Gradient-boosted trees MAE (same task) | 0.665 ± 0.000 | 0.765 ± 0.000 |
| CRPS | 0.485 ± 0.002 | 0.558 ± 0.003 |
| 50% interval coverage | 0.492 ± 0.016 | 0.478 ± 0.021 |
| 90% interval coverage | 0.848 ± 0.014 | 0.841 ± 0.014 |
| Survival C-index (Harrell) | 0.781 ± 0.000 | 0.530 ± 0.004 |
| Cox on baseline markers C-index | 0.782 ± 0.000 | 0.519 ± 0.000 |
| True hidden state recovered (mean affine R²) | 0.857 ± 0.004 | 0.791 ± 0.001 |
| Drift relative MSE vs true equation | 0.635 ± 0.025 | 0.294 ± 0.015 |
| Noise ratio age ≥70 / ≤50 (true ON 3.5, OFF 1.0) | 4.623 ± 1.256 | 1.025 ± 0.018 |
| Best epoch | 77 ± 5 | 30 ± 14 |

## Hallmark tests (decision rule fired?)

| Test | ON s0 | ON s1 | ON s2 | OFF s0 | OFF s1 | OFF s2 |
|---|---|---|---|---|---|---|
| inflammaging | yes | yes | yes | no | no | no |
| dispersion | yes | yes | yes | no | no | no |
| irreversibility | yes | yes | yes | no | no | no |
| intervention | yes | yes | yes | no | no | no |
| jacobian_enrichment | yes (5/7, 0 FP) | no (0/7, 0 FP) | yes (2/7, 0 FP) | no (0/7, 0 FP) | no (0/7, 0 FP) | no (0/7, 0 FP) |
| attractors | no | no | no | no | no | no |

Desired pattern: yes on every ON seed, no on every OFF seed.

## Intervention predictions vs reality (ON, seed 0)

| | Observed | Model | True equation |
|---|---|---|---|
| Treated − untreated HbA1c change (mmol/mol/yr) | -0.113 | -0.114 | -0.128 |
| Alive at 15 y, untreated arm | 0.788 | 0.667 | 0.786 |
| Per-person HbA1c effect at +10 y (mean) | — | -1.10 | -1.42 |
| Per-person years gained over 15 y (mean; corr with truth) | — | 0.005 (r = -0.55) | 0.040 |

## Known limitations

- Survival is ranked well but miscalibrated (too pessimistic at 15 y); per-person treatment benefit on survival does not track the truth. The static hazard head is the likely cause.
- Misses the frailty double well (one attractor instead of two).
- Noise growth with age is overstated; the V/J split of the drift is not identifiable.
- Jacobian pathway test is seed-sensitive.
