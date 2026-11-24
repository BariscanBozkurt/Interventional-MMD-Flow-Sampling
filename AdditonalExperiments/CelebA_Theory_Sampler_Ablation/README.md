# CelebA IMQ--Euler versus Riesz--AdamW

This paired ablation tests whether the smooth positive-definite IMQ--Euler
sampler used by the finite-particle analysis performs comparably to the
practical Riesz--AdamW sampler used for the CelebA headline experiment.

## Paired protocol

For every seed and arm, all samplers share the same logged data, projected-
Nyström DR target, initial particles, reference images, and evaluation draws:

1. Riesz energy with AdamW, 100 warm-up updates, cosine decay, peak learning
   rate `0.001`, and 4,000 updates;
2. deterministic normalized-IMQ Euler with median bandwidth, `eta=0.5`, and
   4,000 updates;
3. the same IMQ update evaluated at a noisy query with
   `beta_r = [2 sqrt(d_Z)]^(-1) (r+1)^(-1/4)`.

The configuration uses native 256-pixel CelebA, 20,000 logged observations,
2,000 particles per arm, full-sample nuisance fitting, and 1,024 projected-
Nyström landmarks. Losses belong to different objectives across Riesz and IMQ;
compare SW2, FID, transport, and runtime instead.

The noisy row matches the smooth-kernel Euler form used by the analysis. We do
not claim that the empirically selected `eta=0.5` verifies every conservative
finite-step condition of the theorem.

## Current evidence

The paired three-seed comparison is complete (2026-11-22). All 18
seed/arm/sampler rows pass the completeness and pairing checks. The rerun
Riesz metrics match the archived headline metrics exactly for all six
seed/arm pairs. Average arms within seed, then report mean and sample SD
across seeds:

| Sampler | Pixel SW2 | Legacy FID | Clean FID | Flow seconds/arm |
|---|---:|---:|---:|---:|
| Riesz--AdamW | .02656 ± .00155 | 11.57 ± .08 | 14.53 ± .18 | 349.3 ± .1 |
| IMQ--Euler | .02649 ± .00152 | 11.78 ± .08 | 15.02 ± .18 | 167.9 ± .1 |
| IMQ--Euler + query noise | .02649 ± .00152 | 11.78 ± .08 | 15.02 ± .18 | 168.9 ± .2 |

Times cover particle optimization only, excluding shared fitting, encoding,
decoding and evaluation. The IMQ rows have comparable SW2 and modestly worse
FID. They instantiate the smooth-kernel Euler form without verifying every
theorem condition. The step size was frozen after the seed-0/arm-0 pilot;
arm 1 and seeds 1--2 were not used to retune it. Raw outputs and the analyzer's
report remain private under `results/`.

## Reproduction

- `imq_euler.py` contains the analytic normalized-IMQ witness gradient.
- `celeba_theory_sampler_ablation.py` runs the paired numerical comparison.
- `analyze_sampler_ablation.py` rejects incomplete or unpaired seed/arm grids.
- `celeba_imq_pixel_change_diagnostic.py` reproduces the seed-0 qualitative
  diagnostic for users who have CelebA access.

Run the focused gradient and noise tests with

```bash
python -m unittest -v \
  AdditonalExperiments.CelebA_Theory_Sampler_Ablation.test_imq_euler
```

Use `python <script> --help` for dataset, cache, and output arguments. The selected
[combined initial/final/heatmap PDF](figures/imq_euler_eta0p5/celeba_ranked_pixel_change_top128_both_arms.pdf)
is included for collaborator review. Results, logs, caches, scheduler launchers
and other generated face-image files remain local.
