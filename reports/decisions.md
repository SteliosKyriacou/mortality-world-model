# Design decisions log

Dated 2026-10-03. Each entry gives the decision, why it was made, and any alternative left open.

## Environment and repository

- **Conda env `mwm`:** Python 3.11, torch 2.5.1 with the cu124 wheel (runs on the driver's CUDA
  12.6). Pinned versions are in `environment.yml`. The `mwm` package is installed editable through
  `pyproject.toml`.
- **Configs and logging:** plain YAML configs (`configs/`) instead of Hydra, which keeps
  dependencies light; one command runs one experiment. Logs go to `runs/**/log.txt` and
  `results.json`. There is no external tracker.
- **Disk:** model checkpoints are under 2 MB each and synthetic cohorts about 42 MB each, all
  gitignored.

## Data contract (`src/mwm/data/common.py`)

- The long table and outcomes table follow PLAN §3.1 exactly. `cause` takes one of
  `none|cvd|cancer|other|unknown`, and censored rows must be `none`.
- Static covariates are repeated at each visit. Interventions are stored as long rows with unit
  `flag` (e.g. `u_nutrient`) and are excluded from encoder inputs.
- Splits are person-level, stratified on age band × event (plus optional strata), seed 20261003.
  `save_splits` refuses to overwrite a different split, so splits are never re-drawn.

## Synthetic cohort (`src/mwm/data/synthetic.py`)

- **Latent:** 8 true dimensions, each with a named role: age core, inflammation, nutrient,
  rotation pair, frailty, and two OU nuisance dimensions.
- **Drift:** −∇V plus rotation, inflammation coupling and the u forcing.
- **Diffusion:** diagonal, scaling ×0.4→×1.84 between ages 30 and 90 when dispersion is on.
- **Simulation:** Euler–Maruyama with dt = 0.05 y from age 30. Deaths come from the hazard; only
  people alive at baseline are kept.
- **Visits:** baseline age 40–70; 0–3 follow-ups with probabilities 0.55/0.25/0.12/0.08; gaps of
  1–8 y.
- **Missingness:** per-feature (2–55%), increasing at later visits.
- **Omics:** 500 genes in 20 pathways × 25, measured for 25% of people.
- **Added hallmark (f), frailty basin:** a double well on z5 whose frail well carries higher
  hazard. This was needed to give the attractor/basin test a positive control.
- **Irreversibility OFF:** z0 *and* the nutrient axis z2 both become mean-reverting. In the first
  version z2 kept its tilt, which left a monotone direction in the "null" cohort.
- **Inflammaging OFF:** the inflammation target becomes linear in z0 (a rise with no inflection)
  and inflammation drops out of the hazard. This makes the negative control stricter than "no
  rise at all".
- **Pathways OFF:** the gene loading matrix is permuted, so marginal loadings are unchanged but
  pathway structure is gone.
- **Sign convention:** drift is −∇V, so aging runs *downhill* in V. PLAN §7 says both "young
  samples sit uphill" and "V should increase along trajectories". These conflict, and we follow
  the former.

## Encoder

- **v1 architecture:** masked tabular transformer (2 layers, d_model 64, d = 16). The 500-gene
  omics block is projected into 4 tokens, so attention over 533 tokens is never needed.
- **Training loss:** masked autoencoder (30% of observed entries hidden; hidden entries weighted
  1, visible 0.5) plus a piecewise-exponential hazard on z.
  - The latent is held constant within each visit interval.
  - An omics-dropout consistency loss keeps z stable when omics are missing.
- **Exported latent:** whitened per dimension on the train split. Downstream heads accept the
  whitened z.
- **Latent observation-noise estimate:** we perturb inputs by the decoder's per-feature σ and take
  the per-dimension variance of z.
  - Dynamics models use this estimate as **fixed** observation noise, with errors-in-variables
    perturbation of the start state.
  - Reason: when the noise was learned, A and B′ absorbed encoder noise into a large, fast
    mean-reverting diffusion. With visit gaps ≥ 1 y, white measurement noise and a fast OU process
    cannot be told apart.
- **v2 HetGNN:** observed-only nodes (clinical, protein, pathway, visit). Missing modalities
  appear as absent nodes. It is unit-tested but not yet used in the synthetic benchmark.

## Dynamics

- **Model A training:** all pairs are integrated jointly in normalised time s∈[0,1], with
  per-pair Δt.
  - Default solver: native fixed-step Euler–Maruyama, 32 steps.
  - torchsde `euler` and `reversible_heun` are supported and tested. The torchsde path builds
    reference cycles that hold GPU graph memory (out-of-memory at batch 512), hence the `gc` and
    the native default.
- **Model A loss:** Gaussian NLL of z1 under the K = 32 sample mean/variance plus observation
  noise. Small penalties on ‖J‖² (so the potential carries as much of the drift as it can) and on
  Σ.
- **Not implemented:** the hazard loss along the path (PLAN §4.2). Survival comes from
  integrating the frozen hazard head along rollouts.
- **Model B:** paired CFM on the *full* velocity −∇V + g (V is trained). A Hutchinson
  ‖∂v/∂z‖_F² penalty (0.1) is required: without it the learned field expands (leading
  eigenvalues around +0.28/yr) and 40-year rollouts diverge (‖z‖ ~ 10³).
- **Theory finding, Model B:** paired CFM where the velocity depends only on (z_s, a_s) learns the
  **probability-flow** velocity of the population marginals, not the SDE drift. For a stationary
  OU dimension this is ≈ 0, so B cannot represent mean reversion. The straight-line target
  (z1−z0)/Δt also aliases dynamics faster than the gaps (rotation at ω = 0.6 rad/y).
- **B′ design:** v (CFM), plus a score network s(z, a, u) trained by DSM on visit latents, plus a
  diagonal g(z, a) fitted by the same simulated-pair NLL as A with v and s frozen. The drift is
  f = v + ½g²s, which is the identity linking probability flow and SDE drift.
  - The correction is soft-clamped to 1 latent unit per year.
  - DSM σ is 0.2 on the true latent and 0.5 on whitened encoder latents. σ = 0.2 on encoder
    latents produced a noisy score and poor recovery.
  - This is our extension. The naive "B + additive noise" variant is kept as the
    `Bp_noScore` ablation.
- **Snapshot OT-CFM:** exact EMD couplings (POT) between adjacent 5-year age bins of baseline
  visits. It is evaluated like the others (no score correction).

## Evaluation

- **Forecasting:** held-out people, first → last visit. Decoded clinical features in z-units are
  compared with *observed* values only.
- **Predictive samples:** decoded latent samples plus the decoder's per-feature σ.
- **Feature-space baselines:** fallback is the decoded z0 where x0 is missing; GBM uses the
  native NaN handling instead.
- **Survival:** baseline visit. Model risk = 1 − S(10 y) from rollouts (K = 32 for stochastic
  models), with hazard integrated by the trapezoid rule.
- **Drift recovery through the encoder:** an affine map learned→true is fit on train visits. The
  drift is pushed forward exactly (affine), and the diffusion as Wᵀdiag(σ²)W.
- **Oracle recovery:** A, B, B′ and B′-noScore are trained directly on the true latent.

## Hallmark tests (`src/mwm/interrogate/`)

Several tests were **re-specified after the first synthetic run**, because the first versions were
invalid on the ground truth itself. This is the purpose of the synthetic positive/negative-control
step. The thresholds were then frozen in `prereg/analysis_plan.md`.

- **Irreversibility, v1 (abandoned):** "latent-age probe velocity < 0". A linear age probe loads
  on mean-reverting dimensions, so even the true ON dynamics show many negative velocities.
- **Irreversibility, v2:** max-margin "arrow" direction (linear SVM through the origin on {f, −f})
  fit on half of the states and evaluated on the other half, plus 10-year rollouts along that
  direction.
- **Attractors:** the slice is orthogonal to the arrow direction rather than to the mean drift.
  The mean drift picks up the frail well's restoring force, so the slice tilted.
- **Intervention:** the decoded **state-averaged** Δv is compared with random directions of equal
  norm. Averaging |Δv| per state first diluted the share with estimation noise.
- **Intervention rollouts:** common random numbers for u = 1 vs u = 0.
- **Jacobian enrichment:** the direction null was raised to 2000 random subspaces. With 500 the
  p-value floor plus BH over 20 pathways made detection impossible.
- **Ground-truth wrappers:** `TrueDynamics` and `TrueBundle` (`src/mwm/eval/truth_model.py`) run
  every test on the true field in the true latent, as a check that the tests themselves work.
