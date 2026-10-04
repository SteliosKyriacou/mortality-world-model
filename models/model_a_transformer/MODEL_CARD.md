# model_a_transformer

Latent neural SDE (Model A) on a masked tabular transformer encoder. Card generated 2026-10-04 from `models/model_a_transformer/*/seed*/results.json` by `scripts/make_model_cards.py`.

**Status:** trained and evaluated on the synthetic cohort only (validation ladder step 1). Nothing here is evidence about human biology.

## Load / retrain

```python
from mwm.models import load_model_a
m = load_model_a("model_a_transformer", cohort="on", seed=0)   # m.bundle, m.sde, m.latents, m.results
```
```bash
conda run -n mwm python scripts/train_model_a.py --config configs/model_a_transformer.yaml
```

## Architecture

**Encoder: masked tabular transformer** (`src/mwm/encoders/tabular.py`, 400,976 parameters)
- One token per clinical marker: `value × val_w[j] + feat_e[j]`; a missing marker becomes a learned mask
  token and is excluded from attention (33 × 64 each for `val_w`, `feat_e`, `mask_e`).
- Gene panel: `[x·m, m]` (1,000 values) → Linear 1000→256 → GELU → Linear 256→256 → 4 tokens of width 64
  (+ learned positions); padded out when no genes were measured. 322,048 parameters.
- CLS token + 2 pre-norm transformer layers (width 64, 4 heads, feed-forward 128, ReLU, no dropout): 66,944.
- Readout from CLS: LayerNorm → Linear 64→64 → GELU → Linear 64→16 (5,328). Latent whitened per dimension.

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
| Forecast MAE, first → last visit (SD units) | 0.655 ± 0.002 | 0.758 ± 0.001 |
| Gradient-boosted trees MAE (same task) | 0.665 ± 0.000 | 0.765 ± 0.000 |
| CRPS | 0.481 ± 0.003 | 0.554 ± 0.002 |
| 50% interval coverage | 0.470 ± 0.004 | 0.447 ± 0.007 |
| 90% interval coverage | 0.836 ± 0.004 | 0.819 ± 0.006 |
| Survival C-index (Harrell) | 0.782 ± 0.002 | 0.522 ± 0.003 |
| Cox on baseline markers C-index | 0.782 ± 0.000 | 0.519 ± 0.000 |
| True hidden state recovered (mean affine R²) | 0.824 ± 0.037 | 0.773 ± 0.000 |
| Drift relative MSE vs true equation | 0.695 ± 0.023 | 0.294 ± 0.015 |
| Noise ratio age ≥70 / ≤50 (true ON 3.5, OFF 1.0) | 11.065 ± 0.590 | 1.035 ± 0.038 |
| Best epoch | 77 ± 5 | 33 ± 19 |

## Hallmark tests (decision rule fired?)

| Test | ON s0 | ON s1 | ON s2 | OFF s0 | OFF s1 | OFF s2 |
|---|---|---|---|---|---|---|
| inflammaging | yes | yes | yes | no | no | no |
| dispersion | yes | yes | yes | no | no | no |
| irreversibility | yes | yes | yes | no | no | no |
| intervention | yes | yes | yes | no | no | no |
| jacobian_enrichment | yes (2/7, 0 FP) | yes (3/7, 0 FP) | yes (2/7, 0 FP) | no (0/7, 0 FP) | no (0/7, 0 FP) | no (0/7, 0 FP) |
| attractors | no | no | no | no | no | no |

Desired pattern: yes on every ON seed, no on every OFF seed.

## Intervention predictions vs reality (ON, seed 0)

| | Observed | Model | True equation |
|---|---|---|---|
| Treated − untreated HbA1c change (mmol/mol/yr) | -0.113 | -0.109 | -0.128 |
| Alive at 15 y, untreated arm | 0.788 | 0.675 | 0.786 |
| Per-person HbA1c effect at +10 y (mean) | — | -1.14 | -1.42 |
| Per-person years gained over 15 y (mean; corr with truth) | — | 0.018 (r = -0.26) | 0.040 |

## Known limitations

- Survival is ranked well but miscalibrated (too pessimistic at 15 y); per-person treatment benefit on survival does not track the truth. The static hazard head is the likely cause.
- Misses the frailty double well (one attractor instead of two).
- Noise growth with age is overstated; the V/J split of the drift is not identifiable.
- Jacobian pathway test is seed-sensitive.
