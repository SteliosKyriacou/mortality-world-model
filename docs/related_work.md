# Related work and novelty assessment

Literature search run on 5 October 2026 (web search, every citation checked against a search result).
It assesses Model A as of that date: latent SDE, mortality sink, age-free variant, planted-hallmark
validation and event sources. **Not covered yet: second-order (momentum) / "Navier–Stokes" aging models.**

## Bottom line
Most components are established or close to established:
- The SDE + Fokker–Planck + mortality-sink framing dates to 1977 (Woodbury & Manton).
- Learned stochastic aging models with a mortality head exist (DJIN, 2022).
- Autonomous learned latent aging dynamics exist (Avchaciov 2022, mice).
- UK Biobank-scale latent "world models" appeared in 2025–2026 (Delphi-2M, HealthFlux, HealthFormer, AURORA).

The most defensible contribution found is the **validation methodology**: synthetic worlds with planted
hallmarks and matched hallmark-absent nulls, pre-registered tests, and per-person intervention-versus-truth
checks. No prior work doing this was found. That is a negative search result, not proof of absence.

The second candidate is an **empirical test, on human data, of whether aging is autonomous** (Markovian in a
learned latent state).

## Component ratings

| Component | Rating | Closest prior work |
|---|---|---|
| Encoder (masked transformer / heterogeneous GNN, missing values allowed) | Established | ReMasker (ICLR 2024), GRAPE (NeurIPS 2020), GraphSAGE, DJIN's imputation VAE |
| Latent neural SDE with −∇V + J split, state-dependent noise, forcing | Incremental | DJIN, Li et al. 2020, PRESCIENT, PI-SDE; Pridham & Rutenberg linear mean-reverting models |
| Visit-pair training, simulated likelihood, errors-in-variables | Incremental (engineering) | latent ODE/SDE literature; SPM measurement error |
| Autonomous (age-free) dynamics | Incremental to potentially novel | Avchaciov 2022 (mice, near-linear); Woodbury–Manton (Markov); frailty index as a state |
| Hazard, path-integrated survival, particle Fokker–Planck with killing | Established | Woodbury & Manton 1977; Yashin SPM; Waddington-OT, PRESCIENT, DeepRUOT |
| Planted-hallmark worlds + matched nulls + pre-registered tests | **Potentially novel (strongest)** | Pierson 2019 and PHOENIX (parameter recovery only); DJIN realism classifier |
| Intervention vs truth, per person and trial-style | Incremental (synthetic part potentially novel here) | TE-CDE; HealthFlux and HealthFormer validate against published trials |
| Clinical events as state-dependent jump sources | Established in ML, incremental in aging | Neural Jump SDEs, NJ-ODE, hidden absorbing semi-Markov model, joint and multistate models |
| Whole package | Incremental as architecture; potentially novel as a validated-interpretability framework | DJIN + Avchaciov + Woodbury–Manton + HealthFlux |

Do not claim:
- "a world model of human health with latent continuous-time dynamics on UK Biobank, with mortality and
  virtual trials": HealthFlux, medRxiv, September 2026;
- learned stochastic dynamics with a mortality hazard: DJIN;
- an autonomous latent aging state with an unstable mode: Avchaciov 2022;
- Fokker–Planck with mortality selection: Woodbury & Manton 1977;
- eigenmode and stability analysis of aging dynamics: Pridham & Rutenberg.

## Most defensible claims and what each needs

1. **The first falsifiable benchmark for interpretable learned aging dynamics:** synthetic cohorts with
   planted, mechanistically defined hallmarks and matched nulls, with pre-registered recovery tests. Needs:
   - the benchmark released as code;
   - power and false-positive rate per hallmark over many seeds, at both cohort sizes;
   - robustness to misspecified visit structure, missingness and measurement noise;
   - baselines on the same tests: a linear OU model, the SPM, a DJIN-style observed-space SDE with an age
     input, and an unstructured latent ODE.
2. **A nonlinear latent SDE of human aging with no age input, plus an explicit test of the autonomy
   (Markov) hypothesis on human cohorts.** Needs:
   - on UK Biobank, HRS or CHARLS, a comparison of age-free and age-conditioned variants by held-out
     next-visit likelihood and mortality calibration;
   - residuals not predicted by age;
   - a positive control in a synthetic world with planted age dependence.
3. **The landscape-plus-flux decomposition gives an empirical arrow of aging and attractor structure,
   validated against planted irreversibility and attractors.** Needs:
   - the synthetic proof, with no false positives in the nulls;
   - real-data estimates with uncertainty, compared with Pridham's tipping point near 75 and Fedichev's
     thermodynamic biological age.
4. **Per-person counterfactual intervention accuracy against known truth.** Needs: individual and trial-level
   error, calibration of individual treatment effects, and how it degrades under confounded assignment.
5. **Later: clinical events as state-dependent jump sources.** Needs:
   - synthetic recovery of the planted jump size and rate;
   - on UK Biobank hospital records, a gain in post-event marker prediction and in mortality against a
     joint-model or multistate baseline.

## Verified citations

### Foundations
- Woodbury MA, Manton KG (1977). A random-walk model of human mortality and aging. Theor Popul Biol 11:37–48. doi:10.1016/0040-5809(77)90005-3
- Yashin AI, Arbeev KG, Akushevich I, Kulminski A, Akushevich L, Ukraintseva SV (2007). Stochastic model for analysis of longitudinal data on aging and mortality. Math Biosci 208:538–551. PMID 17300818
- Aalen OO, Gjessing HK (2001). Understanding the shape of the hazard rate: a process point of view. Stat Sci 16(1):1–22
- Lee M-LT, Whitmore GA (2006). Threshold regression for survival analysis. Stat Sci 21(4):501–513. doi:10.1214/088342306000000330
- Weitz JS, Fraser HB (2001). Explaining mortality rate plateaus. PNAS 98(26):15383–15386. doi:10.1073/pnas.261228098
- Strehler BL, Mildvan AS (1960). General theory of mortality and aging. Science 132:14–21. doi:10.1126/science.132.3418.14
- Karin O, et al. (2019). Senescent cell turnover slows with age providing an explanation for the Gompertz law. Nat Commun 10:5495. doi:10.1038/s41467-019-13192-4
- Mello BA (2010). Physiological aging as an infinitesimally ratcheted random walk. Phys Rev E 82:021918
- Flietner V, et al. (2025). A unifying theory of aging and mortality. Sci Rep 15:28766. doi:10.1038/s41598-025-11454-4

### Learned aging dynamics and world models
- Farrell S, Mitnitski A, Rockwood K, Rutenberg AD (2022). Interpretable machine learning for high-dimensional trajectories of aging health (DJIN). PLoS Comput Biol. doi:10.1371/journal.pcbi.1009746
- Avchaciov K, et al. (2022). Unsupervised learning of aging principles from longitudinal data. Nat Commun 13:6529. doi:10.1038/s41467-022-34051-9
- Pyrkov TV, et al. (2021). Longitudinal analysis of blood markers reveals progressive loss of resilience and predicts human lifespan limit. Nat Commun. doi:10.1038/s41467-021-23014-1
- Taneja S, Mitnitski AB, Rockwood K, Rutenberg AD (2016). Dynamical network model for age-related health deficits and mortality. Phys Rev E 93:022309
- Farrell S, et al. (2020). Generating synthetic aging trajectories with a weighted network model using cross-sectional data. Sci Rep 10:19833
- Pridham G, Rutenberg AD (2023). Network dynamical stability analysis reveals key "mallostatic" natural variables. Sci Rep. doi:10.1038/s41598-023-49129-7
- Pridham G, Rutenberg AD (2024). Dynamical network stability analysis of multiple biological ages. J Gerontol A 79(10):glae021
- Pridham G, Rockwood K, Rutenberg A. Aging health dynamics cross a tipping point near age 75. arXiv:2412.07795
- Pierson E, et al. (2019). Inferring multidimensional rates of aging from cross-sectional data. AISTATS, PMLR 89:97–107
- Shmatko A, et al. (2025). Learning the natural history of human disease with generative transformers (Delphi-2M). Nature 647:248–256. doi:10.1038/s41586-025-09529-3
- Wang Z, et al. (2026). A world model simulates the latent dynamics of human health (HealthFlux). medRxiv. doi:10.64898/2026.09.19.26363460
- Lutsker G, et al. (2026). Simulating clinical interventions with a generative multimodal model of human physiology (HealthFormer). arXiv:2604.27899
- Chen J, et al. (2026). A generative AI framework unifies human multi-omics to model aging, metabolic health, and intervention response (AURORA). Cell Metab. doi:10.1016/j.cmet.2026.03.014
- Liu K, et al. (2026). Medical world models. arXiv:2606.16721
- Wu W, et al. (2025). Molecule-dynamic-based aging clock and aging roadmap forecast with Sundial. arXiv:2501.02176
- Tarkhov AE, Denisov KA, Fedichev PO. Aging clocks, entropy, and the limits of age-reversal. bioRxiv doi:10.1101/2022.02.06.479300
- Levine ME, et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan. Aging 10:573–591. doi:10.18632/aging.101414
- Argentieri MA, et al. (2024). Proteomic aging clock predicts mortality and risk of common age-related diseases. Nat Med 30:2450–2460. doi:10.1038/s41591-024-03170-9
- Sun BB, et al. (2023). Plasma proteomic associations with genetics and health in the UK Biobank. Nature 622:329–338. doi:10.1038/s41586-023-06592-6
- Mitnitski AB, Mogilner AJ, Rockwood K (2001). Accumulation of deficits as a proxy measure of aging. ScientificWorldJournal 1:323–336
- Gijzel SMW, et al. (2017). Dynamical resilience indicators in time series of self-rated health correspond to frailty levels. J Gerontol A 72(7):991–996

### Latent ODE/SDE and survival
- Chen RTQ, et al. (2018). Neural ordinary differential equations. arXiv:1806.07366
- Rubanova Y, Chen RTQ, Duvenaud D (2019). Latent ODEs for irregularly-sampled time series. arXiv:1907.03907
- De Brouwer E, et al. (2019). GRU-ODE-Bayes. arXiv:1905.12374
- Kidger P, et al. (2020). Neural controlled differential equations for irregular time series. arXiv:2005.08926
- Li X, et al. (2020). Scalable gradients for stochastic differential equations. arXiv:2001.01328
- Oh Y, Lim D-Y, Kim S (2024). Stable neural SDEs in analyzing irregular time series data. ICLR. arXiv:2402.14989
- Aslanimoghanloo M, ElGazzar A, van Gerven M (2025). Generative modeling of clinical time series via latent SDEs. arXiv:2511.16427
- Moon I, Groha S, Gusev A (2022). SurvLatent ODE. arXiv:2204.09633
- Zeng S, et al. (2025). TrajSurv. arXiv:2508.00657
- Bleistein L, et al. (2024). Dynamical survival analysis with controlled latent states. arXiv:2401.17077
- Cristofoletto A, et al. (2025). Neural diffusion processes for physically interpretable survival prediction. arXiv:2510.00733
- Lee C, Yoon J, van der Schaar M. Dynamic-DeepHit. IEEE Trans Biomed Eng 67:122–133 (DOI not confirmed)
- Seedat N, et al. (2022). Continuous-time modeling of counterfactual outcomes using neural CDEs. arXiv:2206.08311
- Du T, Melis L, Wang T (2024). ReMasker. ICLR. https://openreview.net/forum?id=KI9NqjLVDT
- You J, et al. (2020). Handling missing data with graph representation learning (GRAPE). NeurIPS
- Hamilton WL, Ying R, Leskovec J (2017). GraphSAGE. arXiv:1706.02216

### Population density with growth/death
- Schiebinger G, et al. (2019). Waddington-OT. Cell 176:928–943
- Yeo GHT, Saksena SD, Gifford DK (2021). PRESCIENT. Nat Commun 12:3222. doi:10.1038/s41467-021-23518-w
- Tong A, et al. (2020). TrajectoryNet. ICML
- Huguet G, et al. (2022). MIOFlow. arXiv:2206.14928
- Chizat L, et al. (2018). Unbalanced optimal transport. J Funct Anal 274:3090–3123
- Zhang Z, Li T, Zhou P (2025). Regularized unbalanced optimal transport (DeepRUOT). arXiv:2410.00844
- Jiang Q, Wan L (2024). PI-SDE. Bioinformatics 40(Suppl 2):ii120–ii127

### Interpretation and hallmarks
- Qiu X, et al. (2022). Mapping transcriptomic vector fields of single cells (dynamo). Cell 185:690–711
- Hossain I, et al. (2024). Biologically informed NeuralODEs for genome-wide regulatory dynamics (PHOENIX). Genome Biol 25:127
- López-Otín C, et al. (2023). Hallmarks of aging: an expanding universe. Cell 186:243–278

### Jumps, point processes, joint models
- Jia J, Benson AR (2019). Neural jump SDEs. arXiv:1905.10403
- Herrera C, Krach F, Teichmann J (2021). Neural jump ODEs. arXiv:2006.04727
- Chen RTQ, Amos B, Nickel M (2021). Learning neural event functions for ODEs. arXiv:2011.03902
- Mei H, Eisner J (2017). The neural Hawkes process. arXiv:1612.09328
- Enguehard J, et al. (2020). Neural temporal point processes for modelling EHRs. arXiv:2007.13794
- Alaa AM, van der Schaar M (2018). A hidden absorbing semi-Markov model. JMLR 19. arXiv:1612.06007
- Tsiatis AA, Davidian M (2004). Joint modeling of longitudinal and time-to-event data. Stat Sinica 14:809–834
- Rizopoulos D (2012). Joint Models for Longitudinal and Time-to-Event Data. doi:10.1201/b12208
- Putter H, Fiocco M, Geskus RB (2007). Competing risks and multi-state models. Stat Med 26:2389–2430

## Not confirmed (do not cite as written)
- The journal for the Pridham/Rockwood/Rutenberg tipping-point paper.
- The original 1985 Yashin–Manton–Vaupel SPM paper's authors.
- The DOI for Dynamic-DeepHit; the AgingBio publication of Tarkhov et al.
- Prior planted-hallmark-plus-null benchmarks in aging: none found (absence not proven).
- A fully autonomous nonlinear neural SDE with mortality on human biobank data: none found (nearest: Avchaciov 2022, mice).
