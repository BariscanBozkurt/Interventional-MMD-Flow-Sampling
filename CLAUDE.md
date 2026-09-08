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
                     distances (eps-guarded sqrt for autograd);
                     EncoderPullbackKernel (k(E(y),E(y')) through a frozen
                     differentiable encoder — flows in outcome space)
  mmd_flow_utils/    mmd_flow.py: squared_mmd, embedding_squared_distance,
                     witness_gradient, mmd_gradient_flow (Arbel Alg. 1 +
                     weighted-target variant), noise schedules
                     riesz_flow.py: riesz_energy_loss (dense+sliced),
                     adamw_riesz_flow with lr_schedule in {const, cosine},
                     warm-up helpers
  causality_utils/   herding.py: kernel_herding (Algorithm 2) + evaluate_embedding
                     + herding_diagnostics (unique/multiplicity/coverage — herding
                     selects WITH REPLACEMENT and can quantise);
                     cpme.py: plug-in / DR / cross-fitted DR weights
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
  KernelHerdingSampler/      The selection-based sampler (Algorithm 2 of the CPME
                     paper) on the same targets, for comparison: dSprites and
                     CelebA female, each reusing its flow notebook's estimation
                     cells verbatim and running the flow in-notebook
Simulations/         Multi-seed HPC simulation scripts, one dataset/method folder
                     per experiment (ColorMNIST/dr_mmd_flow, dSprites/dr_mmd_flow,
                     CelebA/dr_mmd_flow); each holds a CLI simulation script + a
                     SLURM array submit script writing one self-describing CSV per
                     task to results/
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
- **ColorMNIST pixel-space ablation**
  (`ColorMNIST_DR_IMQ_MMD_Flow_Euler_Pixel_Space_Ablation`, user-proposed): the
  flow run in PIXEL space through `kernel_utils.EncoderPullbackKernel`
  (k_E(y,y') = k(E(y),E(y')); witness/flow code unchanged, gradient = J_E^T
  latent gradient via autograd; verified bitwise + chain-rule-exact). The
  ESTIMATION half is untouched — the outcome kernel enters only the sampler, so
  the same DR weights are reused and both parametrisations minimise the SAME
  objective, making their values directly comparable.
  THE IMAGE-DOMAIN CONSTRAINT: the decoder ends in a tanh, so the latent flow
  keeps y in [-1,1] for free and the pixel flow does not (it reaches
  [-1.70, 1.04]). Clamping after each step fixes the range but barely helps
  (colour 0.654, SW2 0.0418) and worsens the AE gap to 6.18; REPARAMETRISING
  y = tanh(w) — pull back through the composite encoder E o tanh, no flow code
  changes — is the right fix and is the notebook's MAIN RUN: colour
  0.728 -> 0.578, SW2 0.0504 -> 0.0255, in range by construction.
  RESULT: even so, the flow plateaus. Over a 100x step sweep the objective falls
  3.3e-04 -> 2.4e-06 while colour flattens at 0.576 and SW2 at 0.0253. It
  converges to a point that beats the latent companion on the shared objective
  (4.99e-06 vs 9.60e-06) while 1.9x worse in pixel SW2 (0.0255 vs 0.0137) and
  0.08 short on colour (vs 0.503, oracle 0.494); AE gap 5.02 with AND without the
  constraint (natural 2.86, oracle draw 3.12, decoder-native 1.10) — the
  constraint fixes the RANGE, not the MANIFOLD. ENCODER-FOOLING, visible as
  diffuse structured texture over foreground and background in the amplified
  perturbation grid. KERNEL CHOICE changes the symptom, not the diagnosis: IMQ
  converges-but-fools (canonical, because it isolates the phenomenon); a Riesz
  pullback DIVERGES at every step 0.003-3 (pixels to +/-900, SW2 worsening) —
  measured, recorded in CODEX PULL-01, not kept as a notebook.
  GOTCHA: under y = tanh(w) every kernel argument passes through tanh, THE
  TARGET SUPPORT INCLUDED — pass the target as atanh(y.clamp(+/-0.999)) or the
  flow chases images compressed into [-0.76, 0.76].
  LESSON: the latent sampler works because the DECODER projects moves onto the
  image manifold, not because of the objective — and this now holds under a
  valid image-domain constraint, so it is not a divergence artefact. Evidence:
  CODEX PULL-01.
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
- **CelebA female init ablation** (`..._Female_Init_Ablation`, 4 flows): observed arm vs
  logged marginal P_Y vs N(0,I) vs N(mu,Sigma), pipeline otherwise identical. KEY: all
  four reach the SAME latent SW2 (0.0745-0.0751) and the same final loss — the objective
  is blind to the init — while decoded quality differs 2x: pixel SW2 0.0369 (marginal),
  0.0376 (observed), 0.0441 (N(mu,Sigma)), 0.0759 (N(0,I), WORSE than the 0.0521 of no
  flow at all). The cycle error ||enc(dec(z))-z|| is monotone in that ordering (7.69 /
  9.42 / 11.45 / 13.56; raw observed latents 7.42) — matching a mean embedding does NOT
  put particles on the decoder's manifold. Moment matching is not enough (exact mean AND
  covariance still smears). POSITIVE: P_Y is as good as the arm-conditioned start, so the
  DR target carries arm selection itself. Two design points: the flow works in
  STANDARDISED coordinates so N(0,I) already matches per-coordinate moments (not a
  strawman), and d=4096 > n=2048 makes Sigma_hat singular, so N(mu,Sigma) is sampled
  exactly as mu + C^T g / sqrt(n-1), never via a ridged Cholesky. Evidence: CODEX INIT-01.
- **KernelHerdingSampler/CelebA_DR_Kernel_Herding_Female** (herding vs the flow,
  same run): herding is COMPETITIVE, contrary to expectation — pixel SW2 0.0387 vs
  the flow's 0.0376 (base 0.0521, floor 0.0399); latent SW2 0.0774 vs 0.0747
  (floor 0.0769). NO quantisation here (938/1024 distinct, max multiplicity 2 —
  unlike dSprites' 372/1000 and 35), because the pool is 2x the sample and the
  energy kernel repels strongly in 4096-d. The embedding localises on its own arm
  unprompted: candidates were both arms (45% male), selection is 99.2% female.
  KEY ADVANTAGE OF SELECTION: outputs are real logged photographs with known
  labels, so the intervention is verified EXACTLY without a probe — blond fraction
  0.491 -> 0.349 against the estimator's target 0.300 (~3/4 of the correction).
  Herding produces no face absent from the log; here the target is a reweighting
  of the logged sample, so that suffices. Evidence: CODEX HERD-01.
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
  flow matches the interventional POSITION law at the 10-fresh-oracle-draw
  floor (energy 0.047 vs floor 0.049), estimated ratios ESS 236 log-RMSE
  0.077, off-manifold tail ~4%. CAUTION: those position numbers are computed
  on SNAPPED images (decoded image -> nearest clean bank sprite), which
  discards decoding error. The notebook now ALSO reports the unsnapped PIXEL
  law, where the flow does NOT reach the floor: energy 0.1141 vs floor 0.0352
  (3.2x), SW2 0.0252 vs 0.0138 (1.8x), against a logged base of 0.6240/0.0523
  and an AE round-trip row of 0.6519 (so the decoder itself costs little and
  the transport is real). Quote both; never the snapped number alone. The
  notebook is STRUCTURED this way: Sections 4-7 measure decoded images only
  (pixel-space trajectory, pixel metric table, pixel-scored Euler baseline);
  Section 8 "The Snapped View" holds all snapping (decoded-vs-snapped figure,
  position-law table, position trajectory and scatter). Euler baseline is
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
- `Simulations/` now holds ColorMNIST and dSprites (2026-09-04, user-driven).
  Shared conventions, follow them for new experiments: array ID = SEEDS_PER *
  value_index + data_seed with a generic PARAM/VALUES sweep axis (override via
  --export=ALL); pipeline_seed = 1000 * data_seed + repeat with the reference
  notebook's seed offsets; one CSV per task, REWRITTEN after every repeat
  (crash-safe); requested vs actual sizes both recorded; fail-fast data guards
  (no downloads on compute nodes). ColorMNIST
  (Simulations/ColorMNIST/dr_mmd_flow) mirrors the RefParams_SplitAE notebook,
  runs BOTH arms per repeat on shared nuisances (verified on CPU: the arm-1 row
  is bit-identical whether or not arm 0 runs first). dSprites
  (Simulations/dSprites/dr_mmd_flow) mirrors the CrossFit_DR_Riesz notebook with
  ONE deliberate change: the denoising AE trains on logged images from a
  disjoint split (ae_fraction, default half of n_obs=2000), NOT on the sprite
  bank — the bank is oracle knowledge and survives only in evaluation (probe,
  snapping, clean floor, denoise MSE); denoising targets are the noisy logged
  images themselves. Notebook-parity is by design, not by seed path, for
  dSprites (the AE change breaks bitwise parity); metric scoring runs on the
  GPU (CPU cdist at n_causal=4000 would take ~an hour per repeat). The Euler
  baseline (notebook section 7) is NOT in the script. WEIGHT CLIPPING
  (2026-09, from the first sweep's analysis): the estimated logging-density
  ratio grows SPURIOUS tails on bad-overlap draws (est max weight 192 where
  the oracle weights have ESS 126) — ~10 low-ESS runs of 150 drove the whole
  seed envelope, same seeds bad in every repeat, est ESS < 50 predicted every
  failure. `--weight_clip` (default 5.0, <=0 disables) caps the estimated
  weights pre-DR; searched over {inf,20,10,5,2.5} x failing+healthy seeds:
  5 repairs every failure (seed-9 n=500: 1.89+-2.27 -> 0.30+-0.04), leaves
  healthy seeds unchanged (n=8000 seed 0 at the position floor), keeps the
  position mean unbiased; 2.5 begins to UNDERSHOOT it (truncation bias).
  Unclipped tails also biased the position mean UP (~20.0 vs oracle 18.5).
  The first sweep's CSVs are unclipped; a relaunch overwrites them with the
  clipped method (the executed analysis notebook documents the unclipped
  state).
- CelebA (Simulations/CelebA/dr_mmd_flow, 2026-09-04) follows the shared
  conventions with three specifics: NO AE split (the SD-VAE is external and
  frozen — the user's explicit call); encodings computed once per data seed
  (pipeline-independent); data caches one .pt per
  (image_size, n_obs, n_ref, data_seed), generated on first use from Hugh's raw
  root on NFS and written ATOMICALLY (tmp + os.replace) so concurrent tasks
  sharing a config can't corrupt them, with a legacy-name fallback for the
  pre-existing caches. n_dr subsampling activates only when n_obs > n_dr
  (identity order otherwise — exact notebook path at defaults). Eval on GPU,
  native + 64px tables, latent SW2 with split-half floors. MEMORY: 256px fits
  ~n_obs<=4096 at 24G; n=20000 requires --image_size 64. HF_HUB_OFFLINE=1 in
  the submit script (weights already in the NFS HF cache). The DecFM Figure-4
  head-to-head (3-seed, n=20000, 64px, generate_celeba_confounding's defaults
  ARE those settings) is now one submit line — see the script header — but has
  NOT yet been run. The dgp_analysis sweeps show large seed variance, so
  single-seed results must not be read as design properties.
- DecFM_2D calibration is SETTLED (2026-09-07 search at n=20000, recorded in
  the analysis notebook): lr 5e-4 is the bottom of a sharp U (ratio-to-floor
  3.44 / 2.01 / 1.52 / 2.35 / 4.55 across lr 1e-4..5e-3); more iterations
  overshoot the same way; n_dr 8000 buys ~2% at 2x flow cost. Keep the
  defaults — the ~1.5x-floor plateau is DR-target estimation error, not
  calibration. Also: the 2D floor is a PER-DRAW quantity (varies with seed AND
  n), so raw cross-n SW2 comparisons mislead — the analysis notebook's
  excess/ratio panels are the fair view (a raw "uptick" at n=20000 was exactly
  this artefact).
- UNVERIFIED: the male n=16384 gain is confounded with M (922 -> 1024, because
  the arm grows past the cap). The matched-M control was launched twice and
  killed both times. The caveat is written into that notebook's summary — run
  the control or keep the caveat, but do not quietly drop it.
- UNEXPLAINED: in the init ablation's logged-marginal (P_Y) run, female-origin
  particles have transport 0.02 (they do not move) while the same population
  moves 4.5 from the arm-only start. Hypothesis: AdamW's per-coordinate
  normalisation gives a near-zero NET update to particles near equilibrium
  whose gradient oscillates. A plain-Euler run on the same init would settle
  it. The user has said P_Y's weaker result is acceptable, so this is optional.
- The P_Y subgroup split (female-origin 0.0586 / male-origin 0.0898) was
  measured OUTSIDE the notebook. If it is to be quoted in the paper, it should
  be re-run as a cell inside the init-ablation notebook.
- Planned notebook the user named but has not requested yet: a ColorMNIST IPW
  sibling of the DR notebooks. (The noise ablation is DONE, both samplers.)
- Nothing has ever been committed or pushed here — that is the user's call.
