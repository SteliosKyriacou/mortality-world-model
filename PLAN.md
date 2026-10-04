# Mortality World Model — Project Plan

Open-source side project: learn a continuous-time model of human aging in a latent space, train it two ways
(**Model A: CFD-inspired Neural SDE**, **Model B: Latent Flow Matching**), compare them on held-out people,
and then interrogate both to see whether the **Hallmarks of Aging** fall out of what they learned.

Background and the original design discussion: `previous discussion.md` (Gemini conversation, Turns 5–32).

---

## Execution order until UK Biobank access (validation ladder)

Hardware: single **RTX 4070 Ti, 12 GB** (not a 3090), enough for this data scale.

1. **Synthetic cohort with planted hallmarks** (known SDE ground truth, UKB-like sparsity/irregular visits,
   partial omics block). Validates both models *and* the hallmark tests (positive + negative controls).
2. **Framingham teaching dataset** (BioLINCC, if obtainable) / NHANES: end-to-end pipeline on real-shaped data.
3. **ELSA (+ HRS external test)**: the real A-vs-B benchmark. *Needs the user to register* (UKDS, HRS).
4. **Rodent molecular arm** (TACO GEO studies, Tabula Muris Senis): molecular hallmarks.
5. **UKB readiness**: loader against the real UKB field schema tested on a synthetic file in that format,
   HetGNN encoder, **pre-registered analysis plan** (`prereg/`) committed before any UKB data is seen,
   RAP budget + GPU download-exemption draft.

---

## 0. Goals and success criteria

| # | Question we want to answer | How we'll know |
|---|---|---|
| G1 | Can a latent continuous-time model forecast a person's future biomarker state from one snapshot? | Beats the baselines (§5.2) on held-out follow-up visits. |
| G2 | Does that latent trajectory predict mortality / ASCVD better than standard risk scores? | C-index and time-dependent AUC beat PCE/PREVENT and a Cox model on raw baseline features. |
| G3 | Neural SDE vs Flow Matching: which is better per unit of compute? | Same encoder, same split, same metrics, plus wall-clock and network evaluations (NFE). |
| G4 | Do the trained models reproduce the Hallmarks of Aging, beyond what's visible in the raw data? | Pre-registered tests in §7 pass against null/shuffled-age controls. |

Non-goals for v1: novel drug/SMILES prediction, commercial asset rating, high-frequency wearables, sequence encoders.

---

## 1. Data reality check (read before anything else)

The original design assumed **UK Biobank + UKB-PPP proteomics + TACO**. Three facts change that:

1. **UK Biobank is not open.** It needs an approved application, an MTA, access fees, and (per current UKB
   policy, to verify) analysis on their cloud platform (RAP/DNAnexus) rather than local download. Approval
   takes months. Fine to apply for in parallel; not something to block on.
2. **Flow Matching and the SDE both need *trajectories*.** The cleanest supervision is the *same person
   measured twice* (visit pairs). Purely cross-sectional data (e.g. NHANES) can still be used, but only via
   population-level snapshot matching (§4.3), which is weaker.
3. **TACO is mostly rodent bulk RNA-seq with no shared individuals with any human cohort.** You cannot
   "contrastively align" the same person across the two. Any mouse↔human link has to go through orthologous
   genes and a shared notion of biological age / hazard. Also: the TACO details in the Gemini chat
   (Nature, May 2026; 4,539 profiles; 79 interventions) are **unverified**: confirm against the paper.

### 1.1 Candidate datasets

| Dataset | Species | Longitudinal (same person)? | Content | Outcome | Access |
|---|---|---|---|---|---|
| **ELSA** (English Longitudinal Study of Ageing) | Human | Yes, nurse-visit bloods every ~4 yrs | Lipids, HbA1c, CRP, fibrinogen, BP, grip, BMI, meds | Mortality | UK Data Service, free End User Licence |
| **HRS** (Health & Retirement Study) | Human | Yes, biennial; physical measures/biomarkers every 4 yrs on rotating half | BP, grip, dried blood spot markers, 2016 venous blood (incl. some immune/omic) | Mortality, self-reported CVD | Free registration; some files restricted |
| **NHANES + public-use Linked Mortality Files** | Human | **No** (cross-sectional) | Deep labs, CRP, BP, meds, diet | Mortality + cause (NDI) | Public domain, direct download |
| **GTEx** | Human (post-mortem) | No | Bulk RNA-seq, ~50 tissues, age bins | Death circumstances (Hardy scale) | Open (summary/expression); individual phenotypes partly controlled |
| **TACO underlying studies (GEO)** | Mouse/rat (+ some primate/human) | Mostly no; some time-series | Bulk RNA-seq, interventions | Lifespan effect of interventions | Public GEO; verify TACO aggregated matrix availability |
| **Tabula Muris Senis** | Mouse | No (age snapshots 1–30 mo) | scRNA-seq, ~20 organs | — | Open |
| **UK Biobank (+ UKB-PPP)** | Human | Partly (~20–50k repeat visits) | Labs, ~3k–5k Olink proteins on ~54k | Mortality, HES/ICD-10 | Application + fees, cloud-only (verify) |
| MIMIC-IV | Human (ICU) | Yes but acute care | Labs, vitals | In-hospital death | Credentialed, not an aging cohort; skip for v1 |

Numbers/policies above are from memory: **Phase 0 verifies each one**.

### 1.2 Recommended data strategy

- **Human arm (primary, open):** ELSA as the main longitudinal training cohort, HRS as an external
  replication cohort, NHANES as a large cross-sectional cohort for the encoder and snapshot flow.
- **Molecular arm (secondary):** TACO/GEO rodent RNA-seq + Tabula Muris Senis, mapped to human orthologs and
  pooled to pathway scores. Trained as its own latent-flow problem first; joined to the human arm only through
  shared pathway/hazard heads (§3.3).
- **UK Biobank:** submit an application in parallel. If it clears, it becomes a third, much larger human
  cohort, and the same code runs on RAP.

**Decision gate (end of Phase 0):** if neither ELSA nor HRS gives ≥ ~5,000 people with ≥ 2 biomarker visits
and mortality follow-up, stop or re-scope to the molecular arm only. That's the "don't waste the time" check.

---

## 2. Phase 0 — Feasibility (≈1 week)

1. Register for UK Data Service (ELSA) and HRS; download NHANES + mortality files.
2. Check the TACO paper and Gladyshev lab GitHub (`tAge`): is the aggregated count matrix + metadata
   downloadable, under which licence?
3. For each human cohort, produce a one-page **data card**: N people, N with 2+/3+ biomarker visits,
   distribution of Δt between visits, which biomarkers overlap across waves, deaths and cause-of-death availability,
   licence terms (can derived latents / trained weights be published?).
4. Start the UK Biobank application (optional).
5. **Go/no-go** using the gate in §1.2.

Deliverable: `data_cards/*.md`, a go/no-go note.

---

## 3. Phase 1 — Data acquisition and harmonisation (≈2–3 weeks)

### 3.1 Pipeline
```
data/raw/<cohort>/        # untouched downloads (gitignored, never committed)
data/interim/<cohort>/    # long format: one row per (person, visit, feature)
data/processed/           # harmonised tensors + masks + split assignments
```
- Scripts: `scripts/fetch_<cohort>.py`, `src/mwm/data/<cohort>.py` → common long format:
  `person_id, cohort, visit_age (float years), feature, value, unit`.
- Outcomes table: `person_id, age_at_baseline, age_at_death_or_censor, event (0/1), cause (all-cause / CVD)`.
- Respect licences: raw individual-level data never leaves `data/raw`; publish code + pipeline, and only
  publish weights/latents if the licence allows.

### 3.2 Human feature set (v1)
Harmonise a core panel available in ELSA/HRS/NHANES: age, sex, BMI, waist, SBP/DBP, total/HDL/LDL cholesterol,
triglycerides, HbA1c/glucose, CRP, fibrinogen (where present), creatinine/eGFR, haemoglobin, albumin, grip
strength, smoking, diabetes, BP/lipid medication flags.
- Unit harmonisation, log-transform skewed markers (CRP), cohort-wise z-scoring fit on **train only**.
- Keep an explicit **missingness mask**: no imputation in the stored data.

### 3.3 Molecular arm
- Map rodent genes → human orthologs (Ensembl/HGNC), pool to ~300 pathway scores (Reactome/MSigDB Hallmark),
  so the mouse and human molecular spaces share coordinates.
- Time axis = **relative lifespan fraction** (age / species median lifespan) so mouse months and human years
  live on one clock.
- Intervention labels kept as `u` metadata (CR, rapamycin, etc.) for later use; not needed for v1 benchmarks.

### 3.4 Splits (fixed once, saved to disk, never re-drawn)
- Split **by person**, stratified on age band, sex and event status: 75% train / 10% validation / 15% test.
- **External test:** train on ELSA, evaluate unchanged on HRS (cross-cohort generalisation).
- **Temporal test:** within the test people, the target is the *last* observed visit, the input the first.
- TACO/rodent data: all in training (as agreed), with a small held-out set of *studies* only to sanity-check the
  molecular arm, not for the headline comparison.

---

## 4. Phase 2–4 — Models

### 4.1 Shared encoder (Phase 2)
Both models must sit on the **same latent space** so the comparison is fair.
- **v1 encoder: masked tabular encoder** (feature-token embedding + missingness mask + small transformer or
  MLP → `z ∈ R^d`, d≈16–32 for ~30 clinical features). A HetGNN is overkill for a few dozen features; keep it as
  v2 once omics (UKB-PPP, pathway nodes) are present.
- Decoder: reconstruct the masked features; heads: Cox/discrete-time hazard head on `z`, age-regression probe.
- Train as a masked autoencoder (randomly hide observed features, reconstruct them) + hazard loss.
- Realistic size: **1–20M parameters**, not the 300–500M estimated in the chat. Fits a single 3090 trivially.
- Freeze, then encode every visit of every person → `latents.parquet`.

### 4.2 Model A — Neural SDE (Phase 3)
`dz = F_θ(z, age, u) dt + Σ_φ(z, age) dW`, with `F_θ = −∇V_ψ(z) + J_θ(z, age, u)`.
- Library: `torchsde`. Fixed-step Euler–Maruyama or reversible Heun, ~20–50 steps per interval (no adaptive
  stepping end-to-end, see chat Turn 15).
- Loss: for each visit pair `(z₀, age₀) → (z₁, age₁)`, simulate K samples from `z₀` to `age₁`; minimise a
  distributional loss to `z₁` (Gaussian NLL of the sample mean/variance, or energy score), plus hazard loss
  along the path. Small KL/regulariser on Σ so the noise doesn't absorb everything.
- Diagonal, state- and age-dependent Σ, so "entropy grows with age" is learnable, not hard-coded.

### 4.3 Model B — Latent Flow Matching (Phase 4)
Velocity field `v_θ(z, age, u) = −∇V_ψ(z) + g_θ(z, age, u)`.
- **Paired CFM** on visit pairs: `z_s = (1−s)z₀ + s z₁`, `age_s = age₀ + sΔt`,
  target `(z₁ − z₀)/Δt`; loss on the **full** `v_θ` including `−∇V_ψ`. (Fixes the bug in the chat's code where
  the potential was used at inference but never trained.)
- **Snapshot (unpaired) variant** for cross-sectional data (NHANES, rodent RNA-seq): OT-CFM between adjacent
  age-bin marginals with minibatch optimal-transport couplings. Use this to add the cross-sectional cohorts.
- **Stochastic variant (B′)**: add a noise term / stochastic interpolant so B can also express growing
  dispersion. Without it, B can only predict an average trajectory and loses to A on calibration by construction.
- Inference: `torchdiffeq.odeint` (RK4 / dopri5), or Euler–Maruyama for B′.

### 4.4 Training protocol (both)
- Two stages: frozen-encoder training first (fast, clean comparison), then optional end-to-end fine-tuning of
  encoder + dynamics as an ablation.
- Same optimiser budget, same hyperparameter search budget (e.g. 30 Optuna trials each), 3 seeds each.
- Log everything to a tracker (Weights & Biases or MLflow); config via Hydra/YAML; one command per experiment.

---

## 5. Phase 5 — Testing and comparison

### 5.1 Metrics (on held-out people, then on HRS)
| Axis | Metric |
|---|---|
| Forecast accuracy | Latent MSE at follow-up; **decoded biomarker MAE/RMSE per feature** (more interpretable than latent MSE) |
| Uncertainty | CRPS / energy score, 50% and 90% interval coverage, calibration curves |
| Survival | Harrell's / Uno's C-index, time-dependent AUC at 5 and 10 yrs, integrated Brier score |
| Irregular time | Error vs Δt curve (does it degrade gracefully for long gaps?) |
| Cost | Training wall-clock, NFE per 10-year rollout, peak VRAM |

### 5.2 Baselines (must beat these or the fancy models aren't earning their keep)
1. **Last observation carried forward** (z₁ = z₀).
2. **Linear drift**: per-feature linear mixed model on age.
3. **Direct regressor**: MLP/GBM from `(x₀, Δt)` to `x₁`.
4. **Survival:** Cox on baseline features; Pooled Cohort Equations / PREVENT where computable;
   a published biological-age score (e.g. PhenoAge from blood markers).

### 5.3 Ablations
Potential term on/off · diffusion on/off (A vs ODE) · frozen vs end-to-end encoder · with/without
cross-sectional snapshot data · with/without rodent arm · latent dim.

Deliverable: `reports/benchmark.md` with tables, plots, and a plain verdict on A vs B.

---

## 6. Phase 6 — Using the models

- `mwm.predict(person_snapshot, horizon_years, u=None, n_samples=100)` → sampled latent paths, decoded biomarker
  trajectories with intervals, survival curve.
- **Counterfactuals**: `u` from observational medication flags only as an exploratory feature, with a clear
  confounding-by-indication warning (chat Turn 11); rodent interventions in the molecular arm.
- **Biological age readout**: map `z` to the chronological age at which the average trajectory reaches the
  same hazard ("latent age gap").
- Visualiser: 2D/3D projection of the learned field (UMAP/PCA of `z` + streamlines + sample paths): the
  "landscape with a path through it" from chat Turns 23–24, built on real model output.
- Release: model card, data cards, reproducible pipeline; weights only if licences allow.

---

## 7. Phase 7 — Hallmarks of Aging interrogation

**Principle:** chat Turn 19 is right that several hallmark signatures are already visible in the raw data.
So the test is not "does the model show it" but **"does the model reproduce it out-of-sample, and does it
add something the raw data can't show"**. Every test below is written down *before* running it, with a null.

What's testable depends on data: the clinical arm can only speak to systemic hallmarks (inflammation,
dysregulated nutrient sensing via glucose/HbA1c/lipids, loss of resilience); molecular hallmarks
(proteostasis, mitochondria, senescence pathways) need the rodent/GTEx arm or UKB-PPP.

| Hallmark (López-Otín 2023) | Test | Null / control |
|---|---|---|
| **Chronic inflammation (inflammaging)** | Roll a 40-year-old forward to 90; decoded CRP/fibrinogen should show the non-linear late-life rise seen in the raw cross-section, and match held-out older people | Same test on a model trained with age labels shuffled |
| **Loss of homeostasis / entropy** (genomic instability proxy) | Model A: Σ(z, age) should grow with age; Model B′: ensemble dispersion should grow. Compare against observed between-person variance growth in test data | Constant-Σ model; permutation of age |
| **Deregulated nutrient sensing** | Rodent arm: CR / rapamycin `u` should shrink ‖v‖ and shift pathway scores (mTOR, IGF-1, ribosome) the right way. Human arm: metabolic markers' share of the leading drift directions | Random-signature `u` of equal norm |
| **Mitochondrial dysfunction / loss of proteostasis / senescence** | Jacobian eigen-analysis: leading eigenvectors of ∂v/∂z, decoded to pathway space, enriched (GSEA) for OXPHOS, UPR, SASP sets | Eigenvectors of a model trained on shuffled time; random directions |
| **Epigenetic alterations / reversibility** | Rodent arm: reprogramming / embryonic samples should sit "uphill" on V (younger); an OSKM-signature `u` should move old states uphill | Same `u` applied to a model without the potential term |
| **Altered intercellular communication** | Attribution (integrated gradients) of hazard to decoded features over age: does inflammatory/cytokine weight increase with age? | Attribution on baseline-only Cox model |
| **Irreversibility (overall)** | With `u = 0`, simulated paths must never systematically move to younger latent age; V should increase along observed trajectories | Model without the potential decomposition |

Plus two cross-cutting checks:
- **Attractor/basin structure**: find fixed points and basins of `v` (and minima of V); are high-hazard basins
  where deaths in the test set actually end up?
- **Cross-species consistency** (if both arms trained): do the pathway directions of fastest aging agree
  between rodent and human arms beyond chance?

Deliverable: `reports/hallmarks.md`: one section per hallmark, result vs null, honest "not reproduced" where
that's the outcome.

---

## 8. Repository layout

```
mortality-world-model/
  PLAN.md
  previous discussion.md
  data_cards/
  configs/                 # hydra/yaml per experiment
  scripts/                 # fetch_*.py, build_splits.py, run_*.sh
  src/mwm/
    data/                  # cohort loaders, harmonisation, splits
    encoders/              # masked tabular encoder, (v2) hetgnn
    dynamics/              # sde.py (Model A), flow.py (Model B, B′), potential.py
    heads/                 # hazard, decoders
    eval/                  # metrics, baselines, survival
    interrogate/           # jacobians, gsea, rollouts, attribution
    viz/
  tests/                   # unit tests incl. a tiny synthetic cohort with a known drift
  reports/
```
A **synthetic cohort with a known ground-truth SDE** (`tests/`) is used to validate both training loops before
real data: each model must recover the known drift/diffusion, which also sanity-checks the interrogation tools.

---

## 9. Compute and timeline (side-project pace, single RTX 3090)

| Phase | Effort | Compute |
|---|---|---|
| 0 Feasibility | 1 wk | — |
| 1 Data pipeline | 2–3 wks | CPU |
| 2 Encoder | 1 wk | minutes–hours |
| Synthetic-SDE validation | 1 wk | minutes |
| 3 Model A (SDE) | 1–2 wks | hours–days per run |
| 4 Model B (+B′) | 1 wk | minutes–hours per run |
| 5 Benchmark + ablations | 2 wks | ~1–3 GPU-days total |
| 6 Usage / viz / release | 1–2 wks | — |
| 7 Hallmarks | 2–3 wks | hours |

Roughly 3–4 months of evenings/weekends. With ~10–20k people and ~30 features, the 3090 is not the
bottleneck; data access and harmonisation are.

---

## 10. Risks and open questions

- **Too few repeat visits** in open cohorts → mitigated by the Phase 0 gate and the snapshot-flow variant.
- **Cohort shift** (ELSA vs HRS vs NHANES protocols/units) → harmonisation + cohort embedding + external test.
- **Mouse↔human link is weak** by nature → keep the arms separable; report cross-species findings as exploratory.
- **Confounding** in observational medication `u` → no causal claims from human `u` in v1.
- **Licences** may forbid publishing weights trained on individual-level data → check in Phase 0.
- **Open question:** primary endpoint all-cause mortality or ASCVD? Open cohorts give cause-of-death more
  reliably than incident ASCVD; suggest all-cause mortality as primary, CVD death as secondary.
