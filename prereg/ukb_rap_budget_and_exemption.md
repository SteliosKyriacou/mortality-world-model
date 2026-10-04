# UK Biobank RAP compute budget and draft GPU download-exemption request

*Draft, 2026-10-03. Every price and policy statement below must be checked against the current
UKB Access Management System (AMS) and RAP rate card before submission. They are working
assumptions, not facts.*

## 1. Workload, scaled from the synthetic benchmark

The measurements come from `reports/synthetic_validation.md`: a single RTX 4070 Ti (12 GB),
20,000 synthetic people, 33k visits, about 10k training visit pairs, and d = 16.

| Step | Synthetic (measured) | UKB scale-up assumption | UKB estimate (1 modern GPU) |
|---|---|---|---|
| Encoder v1 (40 epochs) | ~15 s / 33k visits | ~560k visits (500k at i0 + ~20k i1 + ~60k i2/i3) | ~5 min |
| Model A (80 epochs, K=32, 32 EM steps) | ~2 min / 10k pairs | ~80k visit pairs | ~15–20 min |
| Model B (CFM) | ~10 s | same | ~2 min |
| Model B′ (DSM + diffusion fit) | ~1 min | same | ~8 min |
| Controls (constant Σ, no potential, shuffled age) | ~6 min | same | ~45 min |
| Interrogation + metrics | ~2 min | test split ~75k people | ~15 min |
| **One full seed** | ~15 min | | **~1.5 GPU-h** |
| HetGNN v2 encoder on UKB-PPP (~54k people × ~3k proteins) | not run at scale | graphs of ~3k protein + ~50 pathway nodes per visit | ~2–4 GPU-h per run |

**Total plan:**

| Item | GPU-hours |
|---|---|
| HP search: 30 trials each for A and B/B′ (~0.4 GPU-h average) | ~25 |
| 3 final seeds × full pipeline | ~5 |
| HetGNN: 3 seeds + 10 HP trials | ~40 |
| External test, ablations (PLAN §5.3), re-runs | ~20 |
| **Total** | **~90 GPU-hours** |
| CPU-only work (loader, harmonisation, Cox/GBM baselines, survival metrics) | ~50 CPU-instance-hours |

## 2. Cost estimate (VERIFY current RAP rate card)

| Item | Assumed rate | Quantity | Estimate |
|---|---|---|---|
| Single-GPU instance (e.g. `mem2_ssd1_gpu_x16`/V100- or T4-class) | £0.8–2.5 / h | 90 h | £70–225 |
| CPU instance (`mem1_ssd1_v2_x8`-class) | £0.10–0.30 / h | 50 h | £5–15 |
| Project storage (derived latents, small checkpoints; < 20 GB) | ~£0.015 / GB / month | 20 GB × 12 mo | ~£4 |
| Contingency (×1.5) | | | **total ≈ £120–370** |

Ways to keep it low:
- Checkpoints are small (models are < 2M parameters).
- No raw-data copies.
- Spot/low-priority instances for the hyperparameter search.
- Hard job time-outs.
- The code is CPU/GPU-agnostic, so every pipeline step is debugged on synthetic files in UKB format
  before any billable run.

## 3. Draft justification: exemption to run GPU analysis outside RAP

*(Only needed if RAP GPU capacity or cost turns out to be prohibitive. Default plan: run on RAP.)*

**Request.** Permission to run the GPU training steps on a single, access-controlled
workstation (RTX 4070 Ti, encrypted disk, single named user, no network shares). The only
inputs would be the minimal derived dataset:
- de-identified participant pseudo-IDs;
- about 30 harmonised core-panel variables per visit, with visit ages;
- the death/censoring outcome;
- (v2) the Olink NPX matrix for UKB-PPP participants.

No free text, no dates (ages only), no genetic data and no HES records beyond the derived endpoint.

**Why the exemption is needed.**
1. Model A (Neural SDE) training is many short, iterative GPU jobs: about 30 hyperparameter trials
   per model and about 90 GPU-hours in total. On RAP each job carries instance start-up overhead
   (minutes per job). The interactive debugging needed for a novel model class (SDE solver
   stability, score-correction stiffness, both seen in our synthetic validation) is inefficient
   under per-job billing.
2. The project is unfunded, open-source research. A predictable, near-zero marginal compute cost
   matters for completing the pre-registered analysis (`prereg/analysis_plan.md`) without
   cutting seeds or nulls.

**Safeguards offered.**
- Data minimisation as above. Only derived variables listed in the approved application.
- Storage on an encrypted volume. Access only by the named researcher. Deletion at project end, with
  a deletion certificate.
- No individual-level outputs leave the machine. Only aggregate metrics and model weights leave,
  and weights are released only if UKB policy allows: models are small, and we will run a
  membership-inference check before any release.
- Analysis code is public and pre-registered. The splits are fixed before data access.
- We accept audit and can switch to RAP-only execution at any time.

**Fallback.** If the exemption is not granted, everything runs on RAP with the budget in §2.
The code already supports this: one config per experiment, CSV/JSON logging, and no external
trackers.
