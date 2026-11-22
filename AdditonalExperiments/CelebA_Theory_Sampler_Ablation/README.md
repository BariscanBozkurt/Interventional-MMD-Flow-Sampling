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

The complete three-seed comparison is not yet reported. In the seed-0 screening
run, deterministic IMQ gives arm-averaged pixel SW2 `0.028070`, versus
`0.028086` for Riesz. Legacy/Clean FID are `11.703/14.901` for IMQ and
`11.488/14.398` for Riesz. These are pilot values only.

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

Use `python <script> --help` for dataset, cache, and output arguments. Results,
logs, caches, scheduler launchers, and generated face images are excluded from
the public repository.
