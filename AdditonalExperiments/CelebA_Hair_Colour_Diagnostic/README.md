# CelebA largest-change diagnostic

## Question

This diagnostic examines whether the CelebA particle flow produces visible
changes in the attribute that defines the population-standardization target.
It ranks the final particles by decoded start-to-final full-image RGB distance
and displays the largest 128 changes in each intervention arm.

This is a qualitative diagnostic of one reported run. It is not a new method,
a separately tuned experiment, a random-sample panel, or an individual-level
counterfactual editing experiment.

## Included results

- [`figures/celeba_ranked_pixel_change_top128_both_arms.pdf`](figures/celeba_ranked_pixel_change_top128_both_arms.pdf)
  contains 16 pages: eight pages for each arm and 16 ranked particles per page.
  Every particle is shown as its decoded initial state, decoded endpoint, and a
  common-scale absolute-difference map.
- [`results/celeba_ranked_pixel_change_top128_metrics.csv`](results/celeba_ranked_pixel_change_top128_metrics.csv)
  records rank, particle index, logged row, source-record index, initial
  `Blond_Hair` label, RGB displacement, uint8 RGB RMSE, and mean luminance
  change.
- [`celeba_top128_change_diagnostic.py`](celeba_top128_change_diagnostic.py)
  reproduces the flow and exports the cache, figures, and metrics.

The 97 MB decoded-image cache and the 16 individual PNG pages are deliberately
not versioned. They can be regenerated with the script.

## Exact configuration

The displayed run is `data_seed=0`, `repeat=0`, one constituent of the
three-seed CelebA headline experiment.

| Component | Value |
| --- | --- |
| Logged observations | 20,000 |
| Particles per arm | 2,000 |
| Image resolution | 256 by 256 |
| Representation | Frozen `stabilityai/sd-vae-ft-mse` posterior mode |
| Latent coordinates | 4 by 32 by 32, coordinatewise standardized |
| Nuisance fitting | Full sample, no splitting |
| CME approximation | Projected Nyström, 1,024 uniform landmarks |
| Ridge | Three-fold CV on a fixed 2,000-row block; selected value 10 |
| Initialization | Logged outcomes from the corresponding arm |
| Particle objective | Full-dimensional Riesz interaction, unsliced |
| Optimizer | AdamW, zero weight decay, no injected noise |
| Updates | 4,000 |
| Peak learning rate | 0.001 |
| Schedule | 100-step warm-up followed by cosine decay |

The diagnostic imports the deployed target-construction and flow functions; it
does not reimplement their numerical updates.

## Provenance checks

The complete 2,000-particle initialization agrees exactly with the production
`*_particles.npz` artifact in both arms. The reproduced run also agrees with
the seed-0 headline CSV:

| Arm | Initial loss | Final loss | Mean latent transport |
| --- | ---: | ---: | ---: |
| 0 | 63.562679 | 63.380585 | 8.616054 |
| 1 | 64.558159 | 64.367035 | 9.138040 |

For ranks 1--64, the regenerated decoded starts and endpoints are byte-identical
to the earlier paper-figure cache, and the decoded-pixel change scores are
identical. The earlier cache retained only 64 pairs, so ranks 65--128 cannot be
compared against a previously saved endpoint array.

The script reads the seed-0 logged record identities from a saved 64-pixel DGP
cache and reloads those same raw CelebA records at 256 pixels. Record sampling
does not depend on output resolution. The flow therefore runs at the reported
256-pixel encoding configuration; it does not use 64-pixel images.

## Findings

| Arm and rank block | Initial label composition | Mean RGB RMSE | Mean luminance change |
| --- | --- | ---: | ---: |
| (a=0), ranks 1--16 | 16/16 `Blond_Hair=1` | 19.015 | -11.658 |
| (a=0), ranks 113--128 | 16/16 `Blond_Hair=1` | 15.442 | -9.575 |
| (a=1), ranks 1--16 | 16/16 `Blond_Hair=0` | 18.269 | +7.296 |
| (a=1), ranks 113--128 | 16/16 `Blond_Hair=0` | 14.412 | +5.956 |

All 128 ranked arm-0 particles start with `Blond_Hair=1`; all 128 ranked arm-1
particles start with `Blond_Hair=0`. This is consistent with the benchmark's
population correction: the logged blond proportions are approximately 0.491
and 0.067 in arms 0 and 1, while both target laws have blond mass 0.3. The
largest arm-0 changes therefore darken blond starting images, whereas the
largest arm-1 changes brighten non-blond starting images.

The difference maps often emphasize hair texture and the head outline, but
they also contain changes in facial tone, clothing, background, and global
illumination. The defensible conclusion is that the largest movements occur in
the subgroup whose mixture mass must change and have the expected arm-dependent
direction. These figures do not establish isolated same-person hair edits or a
calibrated blondness prevalence among the generated endpoints. A frozen
attribute evaluator trained and validated on a disjoint split would be needed
for that quantitative claim.

The 128/128 label composition is established for seed 0 only. The numerical
table in the paper averages the two arms within each of three data seeds.

## Reproduction

Install the repository dependencies and obtain CelebA under its dataset terms.
The diagnostic expects compatible seed-0 DGP metadata containing the logged
`obs_indices`; by default it looks for
`data/celeba_dgp/celeba_paper_64_n20000_seed0.pt`. Then run from the deploy
repository:

```bash
python AdditonalExperiments/CelebA_Hair_Colour_Diagnostic/celeba_top128_change_diagnostic.py \
  --rebuild-cache \
  --celeba-root /path/to/celeba
```

Use `--source-dgp`, `--source-top64`, `--cache`, `--out-dir`, and
`--results-dir` to override the portable defaults. The prior top-64 cache is
used only for validation and is optional. Once the top-128 cache exists, omit
`--rebuild-cache` to regenerate plots without rerunning the flow.

## Data-use note

The PDF contains rendered CelebA examples. It is intended for research review
and collaborator discussion under the CelebA terms. Before placing the PDF in
an unrestricted public repository, confirm that this mode and volume of image
redistribution is compatible with the dataset agreement. The repository does
not include the CelebA dataset or pretrained VAE weights.

