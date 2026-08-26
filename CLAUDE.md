# CLAUDE.md

Guidance for Claude Code when working in this repository. This file is the
cross-machine handoff: it carries everything a fresh session needs. (It is
currently tracked; gitignore it before making the repo public if preferred.)

## What this repository is

The **deployable code for the paper** "Sampling Interventional Outcome
Distributions with Doubly Robust Mean Embeddings and MMD Flow" (Bariscan
Bozkurt et al.). Clean, reviewed code is ported here **chunk by chunk** from a
private debug repository (on the SWC cluster:
`~/Desktop/Interventional-MMD-Gradient-Flow/debug_codes`, gitignored there)
as each experiment is mastered. The debug repo also holds `CODEX.md` (shared
review ledger) and detailed CLAUDE.md files — consult them when on that machine;
this file is self-contained for other machines.

## Working rules (from the user — follow exactly)

- **Never push or commit.** Write files and rerun experiments; the user reviews
  and commits.
- **No riff-raff comments**: no collaborator chat (Hudson/Hugh mentions beyond a
  one-line professional acknowledgment at a module top), no debug-notebook
  references, no port-history chatter. Rigorous NumPy docstrings, math,
  citations (Arbel et al. 2019; Zenati, Bozkurt, Gretton 2025; Dance et al.
  2026; Hertrich et al. 2024).
- **Notebooks are the documentation**: executed top-to-bottom with outputs
  embedded, minimal printing, no spaghetti; hyperparameters live in the
  notebook, reusable logic in `src/`. Section skeleton: data generation (DGP
  math in markdown) -> estimation (formulas) -> flow + visualization -> metric
  evaluation (+ summary for method notebooks).
- **Informative notebook names**: `<DGP>_<estimator>_<kernel>_<flow>_<optimizer>[_<arm/ablation>]`.
- **Port minimally**: only what the current experiment needs; verify each port
  against the debug implementation (bitwise where deterministic) before use.
- Third-party repos go into `src/other_methods/` **whole** (gitignored — not
  publishable without permission), never cherry-picked files; published code
  imports only from our own packages.

## Layout

```
src/
  kernel_utils/      Kernel, LengthScaleKernel (self-locking median heuristic),
                     RBF, RieszKernel, IMQKernel + imq_c_from_median; pairwise
                     distances (eps-guarded sqrt for autograd)
  mmd_flow_utils/    mmd_flow.py: squared_mmd, embedding_squared_distance,
                     witness_gradient, mmd_gradient_flow (Arbel Alg. 1 +
                     weighted-target variant), noise schedules
                     riesz_flow.py: riesz_energy_loss (dense+sliced),
                     adamw_riesz_flow with lr_schedule in {const, cosine},
                     warm-up helpers
  causality_utils/   cpme.py: plug-in / DR / cross-fitted DR weights
                     (cpme_dr_weights_crossfit; reduces to full-sample formula
                     when the training block is everything; degenerate folds
                     raise), select_regularization (returns (lambda, weights)
                     tuple); crossfit.py: make_crossfit_fold_ids; overlap.py:
                     effective_sample_size; policies.py: abstract Policy,
                     GaussianPolicy, context_shift_gaussian, exact
                     importance_weights (continuous actions; pi known by design)
  dens_ratio_utils/  LogisticPropensityModel (fit/predict torch class) +
                     crossfit_propensity(X, A, fold_ids, model_factory);
                     importance_weights.py: cross-fitted continuous-action
                     ratio estimators (classifier vs logged pairs, and
                     linear-Gaussian logging-density fit), raw ratios with
                     loud overflow errors, importance_weight_diagnostics
  data_utils/        decfm.py (2D mixture), colormnist.py (ALL design params as
                     explicit arguments; defaults = the validated config
                     w=10, tau=0.08, digits (1,6)), celeba.py (verified
                     reimplementation of the DecFM CelebA construction; root is
                     a REQUIRED argument), dsprites.py (heart-sprite bank
                     loader + continuous-action DGP of arXiv:2506.02793
                     Sec. 14.1.5; npz path REQUIRED, npz in data/dsprites/)
  autoencoder_utils/ colormnist_ae.py (ConvAE 3x28x28, latent 32),
                     dsprites_ae.py (denoising DspriteAE 1x64x64 latent 32 +
                     snap_to_bank off-manifold score + position_probe),
                     sd_vae.py (frozen stabilityai/sd-vae-ft-mse wrapper:
                     load_sd_vae, encode_images, decode_latents)
  metric_utils/      sliced_wasserstein(x, y, n_projections=1000, p in {1,2},
                     generator/seed) — bitwise-identical to the DecFM paper's
                     evaluation code (verified); accepts array-like
  other_methods/     GITIGNORED. Holds H. Dance's Debiased-Flow-Matching repo
                     whole (metrics verified against it; sampling algorithms
                     may be used later)
NotebookExamples/
  dgp_analysis/      Design-characterisation notebooks, no estimators:
                     DecFM_2D_Mixture, ColorMNIST, CelebA (verified statistics)
  Interventional_MMD_Flow/   Method notebooks (see inventory below)
Simulations/         Empty; multi-seed simulation scripts go here later
tests/               12-test unittest suite (cross-fitted DR weights,
                     continuous-action importance weights); run with
                     `python -m unittest discover -s tests` from the repo root
data/                GITIGNORED. MNIST (auto-downloads), celeba_dgp caches,
                     dsprites/ (the dSprites npz + derived sprite-bank cache)
```

## Environment

Developed on the SWC cluster: `/nfs/ghome/live/bbozkurt/.conda/envs/torchnn`
(Python 3.12, torch 2.10 + CUDA). No requirements file yet — needed packages:
torch, torchvision, diffusers (SD-VAE download on first use), scipy,
scikit-learn, matplotlib, tqdm, pillow. Execute notebooks headlessly with
`jupyter nbconvert --to notebook --execute --inplace` from the notebook's own
directory. CelebA raw data root on SWC: `/nfs/ghome/live/danceh/data/celeba`
(torchvision layout); on other machines download CelebA and pass its root —
caches regenerate automatically via `load_celeba_paper`.

## Experiment inventory and key verified results

All executed clean with embedded outputs; metrics are comparable with the DecFM
paper (its tables use W1; its image experiments use SW2).

**The DecFM paper's image-evaluation protocol** (verified against the paper's
App. B.3 + the vendored doflow code; our CelebA notebooks follow it): per
intervention arm, sliced Wasserstein-2 in FLAT PIXEL space between samples and
n_ref = 2000 held-out real reference images with P_ref(X=1|A=a) = 0.3 (600
blond + 1400 brown per arm, disjoint from the observed sample), reported as
SW2 (base) = the flow's source (the observed arm) vs references and
SW2 (target) = flowed samples vs references, mean +/- SD over 3 training seeds;
1000 unit-normalised Gaussian projections, both sides truncated to the smaller
sample (doflow/evaluation/metrics.py — our metric_utils version is
bitwise-identical given the same projections; ours adds a seed). The paper has
NO other image metric: no FID, no probe, no MMD, no floor row. Our
VAE-round-trip / floor rows are additions beyond the protocol and the notebooks
label them as such. The blondness probe was RETIRED from the notebooks
(user decision 2026-08-23 — performance is measured from the target samples
only); the in-flow diagnostic is now the latent SW2 trajectory of the particles
vs the encoded reference latents, with a reference-split floor line. Our
images are stored in [-1,1] (the paper's range) at 256px; Figure-4-readable
numbers come from the "64 x 64 evaluation" block inside the 256px notebooks
(area-downsampled re-scoring; caveats: n_obs=2048 vs their 20000, single run
vs their 3 seeds).

- **DecFM_2D_Mixture_RBF_MMD_Flow_Euler** (n=4000, m=1000, cross-fitted DR with
  estimated logistic propensity): arm0 SW1 0.421->0.111, arm1 0.437->0.109;
  floors 0.061/0.086 (SW1/SW2).
- **DecFM_2D_Mixture_Riesz_MMD_Flow_AdamW** (same pipeline; cosine schedule,
  lr 5e-4, betas (0.9, 0.999)): beats RBF/Euler on every metric (arm1 SW1
  0.086). KEY: AdamW needs the cosine decay to settle in low dimension; more
  iterations under cosine is WORSE (high-rate phase stretches); dense Riesz
  flows in low-d need float64 (float32 broke Euler).
- **ColorMNIST_DR_IMQ_MMD_Flow_Euler_Arm1 / _Arm0** (matched pair, W=10,
  N=20000, n_DR=4000 support subsample, M=1600, IMQ c=median, step 0.15c^2):
  colour 0.737->0.505(0.281) and 0.292->0.503(0.282) vs oracle 0.497(0.291);
  SW2 vs floor 0.0135/0.0081 and 0.0086/0.0058 (compare each arm to ITS OWN
  floor — digit-1 images have fewer lit pixels).
- **ColorMNIST noise ablation** (`..._Noise_Ablation`, 54 runs): answers whether to use
  noise at all, and which kind. `mmd_gradient_flow` gained
  `noise_mode="injected"|"additive"` (default injected = Arbel Alg. 1, bitwise unchanged).
  Injected perturbs the gradient's evaluation point (mollifies the velocity field,
  minimiser UNCHANGED); additive perturbs the particle (Langevin; biases the stationary
  law by ~beta^2/2gamma unless annealed). RESULTS: where the flow converges, noise never
  helps (best of each ties baseline SW2 0.0137, floor 0.0081); additive is ~100x more
  damaging per unit beta (never share a grid); injected recovers under annealing but
  additive does not (the bias, made visible in the objective); beta must stay BELOW the
  bandwidth or the flow freezes (transport 3.31 -> 0.51); in a true dead zone the roles
  REVERSE (injected cannot bridge a gap, additive can: 0.1081 -> 0.0919). Default stays
  injected + annealing + beta << bandwidth. Full evidence: CODEX.md NOISE-01.
- **CelebA female noise ablation** (`..._Female_Noise_Ablation`, 17 runs): the same
  ablation for the Riesz/AdamW sampler; `adamw_riesz_flow` gained the same
  noise_schedule/noise_mode/generator arguments (no-noise path bitwise unchanged).
  VERDICT: noise is INERT here — best injected 0.0745 / best additive 0.0746 vs
  no-noise 0.0747 (latent SW2), 0.0379 / 0.0377 vs 0.0376 (pixel SW2). Injected is
  far more inert than under Euler/IMQ because the Riesz kernel has NO bandwidth
  (unit-magnitude gradient everywhere → no dead zone) and AdamW normalises each
  coordinate. Additive separates by transport (4.5 → 27.4) as before. KEY: both
  mechanisms show a LOSS-vs-QUALITY INVERSION at large beta (additive beta=0.01:
  loss 62.40 < baseline 63.64 but SW2 0.0820 > 0.0747) — the objective uses an
  ESTIMATED embedding, so a lower objective can mean fitting estimation noise.
  Report distance to held-out references, never the objective. Default: noise off
  for this sampler.
- **ColorMNIST init ablation** (`..._Init_Ablation`): observed init beats
  N(0,I) and N(mu,Sigma) everywhere; the cycle error ||enc(dec(z))-z||
  separates them (1.0 vs ~3.3/3.0) — matching the mean embedding does NOT give
  on-manifold particles.
- **CelebA_DR_Riesz_MMD_Flow_AdamW_Female / _Male** (matched pair; 256px,
  n=2048, n_ref=2000/arm cache celeba_paper_256_n2048_nref2000_seed0.pt,
  SD-VAE 4096-d standardized latents, cross-fitted DR = exact
  post-stratification here, cosine lr 1e-3, 4000 iters, M=min(1024, arm)
  printed — male arm has only 922). Evaluation follows the DecFM protocol
  above. RESULTS: female SW2 (base) 0.052 -> (target) 0.038, at the 1000/1000
  floor 0.040; male 0.056 -> 0.044, floor 0.039 not reached (bigger shift,
  fewer particles). VAE round-trip row: base scores unchanged -> the reduction
  is transport, not decoder smoothing. The blondness probe is RETIRED from
  these notebooks (user decision 2026-08-23); its findings stand in the debug
  repo's CODEX.md as history (honest target = probe on encoded refs 0.416 /
  0.241, not 0.30; at convergence the flow overshot that axis — 0.318 / 0.297;
  no schedule fixed it). The in-notebook diagnostic is now the latent SW2
  trajectory vs encoded reference latents with a reference-split floor —
  over-transport shows as a minimum in that curve, and it DOES on the male arm
  (minimum ~iter 200 then drift away; female flat at floor): "stop at the
  trajectory minimum" is the label-free validation stopping rule. The faces
  figure has three rows: observed / DR-flow sample / held-out reference
  (counterfactual) sample. (Beware small-sample metric reads: 128-sample
  probes showed NO pixel improvement; n_ref=1000 put the base BELOW the split
  floor; n_ref=2000 resolves base > floor >= target.)
- **CelebA_DR_Riesz_MMD_Flow_AdamW_Female_n16384** (observation-count ablation;
  identical config at n=16384, M held at 1024, same 2000 refs; ~16 GB cache;
  Gram cross-fit on GPU float64 with staged frees; metric rows use the first
  2000 of the 9011-image arm = the SW truncation size). ANSWER: flow quality
  unchanged by 8x more observations (SW2 target 0.0373 vs 0.0376; 64px
  re-scoring 0.0369 = 0.0369) — the female arm is reference-limited at this
  evaluation scale. Base and floor rows shift for bookkeeping reasons only
  (truncation size 2000 vs 1126; different disjoint reference draw) — do not
  read those shifts as effects of n. Transport grows 4.5 -> 13.2 (richer
  target support); trajectory converges below the split floor by ~400 iters.
- **Figure-4 comparability lives INSIDE the 256px notebooks**: a "64 x 64
  evaluation" block area-downsamples the same samples + references to 64px
  (the paper's resolution; the [-1,1] range matches natively — the caches
  store [-1,1], celeba.py converts at build) and re-scores SW2/SW1 with the
  base/target/floor rows. A separate 64px sibling pair was tried and
  DISCARDED (2026-08-24, user decision): sd-vae-ft-mse is out of distribution
  on 64px input (PSNR 21.8 dB, warped faces) and must be run at its native
  256 scale on upsampled images (PSNR 38.9 dB, round-trip = base) — at which
  point the "64px experiment" is the 256px pipeline on degraded inputs and
  its numbers are near-identical (verified). NEVER feed the SD-VAE images
  below its training scale; NEVER rescale metric inputs that are already in
  [-1,1] (a 2*Y-1 slip doubled SW once — caught and corrected).
- **dgp_analysis/CelebA**: construction verified EXACTLY vs the reference code
  (pool 61,037 records, identical file sets; cell counts bitwise; conditionals
  0.3000/0.6000/0.1000; blond-male cell caps the design, n_max=22,200 at
  n_ref=2000/arm — derives the paper's "n ~ 20,000").
- **dgp_analysis/ColorMNIST + DecFM**: overlap/ESS tables and
  confounding-strength sweeps; arms are symmetric by construction; single-seed
  arm asymmetries are seed artifacts (measured).
- **dSprites_Continuous_Action_CrossFit_DR_Riesz_MMD_Flow_AdamW** (n=1000,
  continuous 2D action, denoising AE latent 32 trained on the full sprite bank,
  fully cross-fitted DR: estimated linear-Gaussian logging density + per-fold
  CME ridge on shared folds, Riesz+AdamW 2000 iters lr 0.01 warmup 100): the
  flow matches the interventional position law AT the 10-fresh-oracle-draw
  floor on every metric (energy 0.047+/-0.012 vs floor 0.049+/-0.025; SW1
  0.563 vs 0.564; SW2 0.705 vs 0.681; logged base 3.99/5.49/6.32), estimated
  ratios ESS 236 log-RMSE 0.077, off-manifold tail ~4%. Euler baseline is
  step-size fragile (0.05 stalls; 1.0 costs manifold fidelity) — and NOTE:
  large-step Euler's failure MAGNITUDE varies across AE training runs (debug
  run blew up, deploy run only degraded), so claim fragility, not blow-up.
  Ports bitwise-verified vs debug (fold ids, ratios, DGP draws, bank, AE init).

## Recurring gotchas

- Notebook builder scripts must use RAW strings: `\t` in a non-raw string
  silently becomes a tab (label renders wrong), `\m`/`\l` embed SyntaxWarnings
  in committed outputs. After building/executing, grep sources for tabs and
  outputs for "Warning".
- The RBF/LengthScale median heuristic self-locks on the FIRST self-kernel
  call: lock it on the reference data before any cross evaluation.
- `sliced_wasserstein` truncates to the smaller sample (paper convention);
  with 1-d inputs it is the exact W1.
- Keep estimation float64 (Gram solves) and flows float32 on GPU for images —
  EXCEPT low-dimensional Riesz flows, which need float64.
- Cross-fitting: ONE shared fold split for propensity + CME + per-fold ridge.
  Raw weights, no normalisation/clipping (overlap interventions are named
  experiments only).

## Pending

- Add a requirements file before the first real commit (the 12-test suite is
  ported and passing under tests/). This is the last blocking item.
- `Simulations/` is still empty. First concrete job: the 3-seed, n=20000, 64px
  CelebA run at the DecFM paper's exact setting (generate_celeba_confounding's
  defaults ARE those settings) — the only way to a real head-to-head with its
  Figure 4. The dgp_analysis sweeps show large seed variance, so single-seed
  results must not be read as design properties.
- UNVERIFIED: the male n=16384 gain is confounded with M (922 -> 1024, because
  the arm grows past the cap). The matched-M control was launched twice and
  killed both times. The caveat is written into that notebook's summary — run
  the control or keep the caveat, but do not quietly drop it.
- Planned notebook the user named but has not requested yet: a ColorMNIST IPW
  sibling of the DR notebooks. (The noise ablation is DONE, both samplers.)
- Nothing has ever been committed or pushed here — that is the user's call.
