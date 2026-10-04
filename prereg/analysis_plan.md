# Pre-registered analysis plan: UK Biobank arm

**Version 1.0, dated 2026-10-03.** This plan was written **before anyone on this project had
accessed UK Biobank data.** Changes go into the dated *Amendments* section at the end. Earlier
text is never edited. The code it refers to (`src/mwm/...`) is frozen at the version in the commit
that adds this file. The same tests were checked on synthetic data (positive and negative
controls) in `reports/synthetic_validation.md`.

## 1. Data and cohort

- **Sources:** UK Biobank main dataset (assessment instances 0–3), HES in-patient diagnoses
  (41270/41280), death registry (40000/40001), and the UKB-PPP Olink NPX long table (field 30900,
  coding 143). All analysis runs on the UKB Research Analysis Platform (RAP).
- **Loader:** `mwm.data.ukb.load_ukb`. It is tested on a synthetic file in both the ukbconv
  (`f.<field>.<instance>.<array>`) and RAP (`p<field>_i<instance>_a<array>`) layouts
  (`tests/test_ukb.py`).
- **Inclusion:**
  - a valid instance-0 assessment date;
  - at least 10 of the core-panel features at instance 0;
  - age 37–75 at instance 0.
- **Exclusion:** participants on the current withdrawal list.
- **Core panel (v1 encoder):**
  - demographics and body: sex, BMI, waist;
  - cardiovascular: SBP and DBP (mean of the automated readings), heart rate;
  - function: grip (the larger of left and right);
  - lipids: total cholesterol, HDL, LDL, triglycerides;
  - glycaemia: HbA1c, glucose;
  - inflammation and kidney: CRP, creatinine, cystatin C;
  - blood count and liver: haemoglobin, albumin, WBC, lymphocyte %, MCV, RDW, platelets, ALP, ALT;
  - other: vitamin D, smoking, diabetes, BP medication, lipid medication.
- **Preprocessing:**
  - Log-transform CRP, triglycerides, creatinine, glucose, ALP and WBC.
  - Z-score with statistics from the **train split only**.
  - Keep a missingness mask. No imputation.
- **Omics (v2 encoder only):** Olink NPX for UKB-PPP participants.
- **Time axis:** fractional age at assessment, computed as the field-53 date minus a mid-month birth
  date (fields 34/52).
- **Endpoints:**
  - Primary: all-cause mortality.
  - Secondary: CVD death (primary cause ICD-10 I00–I99, field 40001).
  - Censoring: the registry cut-off date of the release used (recorded in the run log), or the
    loss-to-follow-up date (field 191) if earlier.

## 2. Splits (fixed once, saved, never re-drawn)

- **Main split:** person-level, stratified on 10-year baseline age band × event × sex.
  - Proportions: 75% train / 10% validation / 15% test.
  - Seed `20261003`, made with `mwm.data.common.make_splits`.
  - Saved to `data/processed/splits/ukb.csv`. `save_splits` refuses to overwrite a different split.
- **Temporal test:** test participants with ≥ 2 core-lab visits. The input is the first visit and
  the target is the last.
- **External test:** ELSA (and HRS where the features exist), with the UKB-trained models applied
  unchanged.
- **Test-set access:** the test split is opened **once**, after every model choice has been frozen
  on validation.

## 3. Models (all on the same frozen encoder)

1. **Encoder v1:** masked tabular transformer, d = 16. Trained as a masked autoencoder plus a
   piecewise-exponential hazard head. Latents are whitened on the train split.
2. **Encoder v2 (secondary analysis):** HetGNN with clinical, protein and pathway nodes. Pathways
   come from MSigDB Hallmark (version recorded in the run log).
3. **Model A (Neural SDE):** drift F = −∇V + J; diagonal Σ that depends on state and age.
   - Solver: fixed-step Euler–Maruyama, 32 steps per interval.
   - Loss: Gaussian NLL over K = 32 simulated samples, plus a learned observation noise.
4. **Model B (flow matching):** paired CFM loss on the full velocity −∇V + g.
5. **Model B′ (stochastic flow):** CFM velocity v, plus a denoising score model, plus a fitted
   diffusion g. Drift f = v + ½ g² ∇log p (soft-clamped).
6. **Snapshot OT-CFM:** trained on instance-0 cross-sections only. Reported as an ablation.

**Budget parity:**
- A, B and B′ each get at most 30 hyperparameter trials, selected on validation loss.
- Results are reported over 3 seeds.
- Wall-clock time, NFE per 10-year rollout and peak VRAM are reported for every model.

## 4. Baselines

**Forecasting:**
- LOCF.
- Linear drift: LMM fixed slope with a random intercept, so x₁ = x₀ + slope·Δt.
- Direct MLP and GBM mapping (x₀, age, Δt, u) → x₁.
- Latent-space LOCF, linear drift and MLP.

**Survival:**
- Cox on baseline features plus age.
- Cox on age only.
- PCE/PREVENT where they can be computed.
- PhenoAge (Levine 2018).

## 5. Metrics and decision rules

- **Forecast metrics** (temporal test; decoded clinical features in z-units): MAE, RMSE,
  per-feature MAE, latent MSE, CRPS, energy score, 50% and 90% interval coverage, MAE by Δt bin
  (0–3, 3–6, 6–9, 9+ years).
- **Survival metrics** (baseline visit): Harrell's C, Uno's C, time-dependent AUC at 5 and 10
  years, IBS.
- **G1 (forecasting):** a model beats the baselines if both hold:
  - its test MAE is below the best baseline's MAE in all 3 seeds;
  - the 95% CI of the MAE difference (paired bootstrap, 1000 participant resamples) excludes 0.
- **G2 (survival):** Harrell's C beats Cox on baseline features, with the 95% paired-bootstrap CI
  of the difference excluding 0.
- **G3 (A vs B):** differences in MAE, CRPS and coverage with bootstrap CIs, plus a cost table. The
  verdict is given per axis. There is no composite score.

## 6. Hallmark tests

**Common conditions:**
- Tests use held-out participants with u = 0 unless stated.
- Thresholds are the ones validated on the synthetic positive and negative controls.
- "Reproduced" requires the rule to hold in **≥ 2 of 3 seeds**.
- Code: `mwm.interrogate.*`.

**H1 Inflammaging**
- **Statistic:** acceleration ratio = slope(72–87) / slope(45–60) of the decoded inflammatory
  composite. Rollouts start from test participants aged 40–50.
- **Null / control:**
  - a model trained with shuffled age labels;
  - the raw cross-section ratio;
  - curve agreement with held-out participants aged ≥ 60.
- **Decision rule:** ratio > 1.5 and bootstrap CI lower bound > 1.

**H2 Loss of homeostasis**
- **Statistic:** tr(ΣΣᵀ) at age 80 divided by tr(ΣΣᵀ) at age 45, at fixed held-out states.
- **Null / control:**
  - an age-permutation null;
  - a constant-Σ model (validation NLL).
- **Decision rule:** all of the following hold:
  - ratio > 1.2;
  - CI lower bound > 1;
  - slope > 0 with permutation p < 0.05;
  - the age-dependent Σ has better validation NLL than the constant-Σ model.

**H3 Nutrient sensing**
- **Statistic:**
  - metabolic share of the decoded Δv = v(u=1) − v(u=0);
  - the 10-year rollout effect.
- **Null / control:**
  - random latent directions of equal norm;
  - human u is observational, so this test is exploratory and supports no causal claim.
- **Decision rule:** the share is above the 95th percentile of the null **and** the rollout-effect
  CI upper bound is < 0.

**H4 Mitochondria / proteostasis / SASP (v2 encoder only)**
- **Statistic:** the leading 2-dimensional eigen-subspace of ∂v/∂z, decoded to protein space and
  scored by pathway.
- **Null / control:**
  - random latent subspaces (BH-adjusted);
  - random gene sets.
- **Decision rule:** q_dir < 0.05 and p_gene < 0.05 for OXPHOS, UPR or an inflammatory/SASP set.

**H5 Irreversibility**
- **Statistic:** a max-margin "arrow" direction w (linear SVM through the origin on {f, −f}) is
  fit on half of the held-out states (u = 0). On the other half we compute:
  - the fraction of states with w·f ≤ 0;
  - the fraction of 10-year mean rollouts that move backwards along w.
- **Descriptive only:** the same fractions using a linear latent-age probe. These cannot test
  irreversibility, because the probe loads on mean-reverting dimensions; that was shown on the
  synthetic ground truth.
- **Null / control:** a model without the potential term; V along observed pairs is reported
  descriptively.
- **Decision rule:** state fraction < 0.10 **and** rollout fraction < 0.05.

**H6 Intercellular communication**
- **Statistic:** slope per decade of the inflammatory share of |integrated-gradients attribution|
  of hazard.
- **Null / control:** the same slope from a Cox-on-baseline model.
- **Decision rule:** slope CI lower bound > 0.

**H7 Attractors / basins**
- **Statistic:** number of attractors of the field restricted orthogonal to the max-margin "arrow"
  direction (H5), on a slice at the test-median arrow coordinate, age 70.
- **Null / control:** none on UKB. The test was validated on a synthetic null with a single well.
- **Decision rule:** ≥ 2 attractors (each ≥ 3% of states) **and** deaths over-represented in the
  basin whose members have the highest mean model hazard (one-sided Fisher p < 0.05).

**Multiplicity:** all 7 families are reported whatever their outcome. Within H4, pathways are
BH-controlled.

**Limitations fixed in advance:**
- B's straight-line targets alias dynamics that are faster than the visit gaps.
- The latent-age probe is linear.
- Every human u is confounded by indication.
- Any hallmark test that failed its negative or positive control in the synthetic validation is
  reported as uninformative for that model.

## 7. Stop / re-scope criteria

- If fewer than 5,000 participants have ≥ 2 core-lab visits, forecasting is reported as exploratory
  only.
- If the encoder does not beat LOCF at reconstructing held-out masked features, the dynamics are not
  interpreted.

## Amendments

(none)
