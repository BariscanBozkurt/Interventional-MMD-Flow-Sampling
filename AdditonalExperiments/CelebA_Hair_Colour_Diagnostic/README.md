# CelebA largest-change diagnostic

This diagnostic checks whether the reported CelebA particle flow visibly moves
the hair-colour composition in the required direction. It ranks the 2,000
particles in each arm by decoded start-to-final RGB distance and visualizes the
largest 128 as decoded initializer, decoded endpoint, and absolute pixel-change
heatmap.

This is a post-hoc diagnostic of one reported run. It is not a random-sample
panel, a separately tuned experiment, or an individual-level counterfactual
editing claim.

## Configuration

| Component | Value |
| --- | --- |
| Data seed / repeat | 0 / 0 |
| Logged observations | 20,000 |
| Particles per arm | 2,000 |
| Resolution | 256 by 256 |
| Representation | Frozen `stabilityai/sd-vae-ft-mse` posterior mode |
| Target | Full-sample DR, 1,024 projected-Nyström landmarks |
| Initialization | Logged outcomes from the corresponding arm |
| Flow | Riesz--AdamW, 4,000 updates, 100-step warm-up and cosine decay |

The heatmap compares `D(E(Y_initial))` with the endpoint. This removes fixed
encoder--decoder reconstruction error and isolates flow-induced movement.

## Summary

| Arm and ranks | Initial labels | Mean RGB RMSE | Mean luminance change |
| --- | --- | ---: | ---: |
| `a=0`, 1--16 | 16/16 blond | 19.015 | -11.658 |
| `a=0`, 113--128 | 16/16 blond | 15.442 | -9.575 |
| `a=1`, 1--16 | 16/16 non-blond | 18.269 | +7.296 |
| `a=1`, 113--128 | 16/16 non-blond | 14.412 | +5.956 |

All 128 largest arm-0 changes begin blond, and all 128 largest arm-1 changes
begin non-blond. This agrees with the population correction: arm 0 reduces
blond mass and arm 1 increases it. Changes are not confined to hair; face,
clothing, background, and illumination can also move.

## Reproduction

`celeba_top128_change_diagnostic.py` reconstructs the fixed seed-0 target and
flow, verifies the saved initial particle identities, and exports the ranked
metrics and multipage figure. Run

```bash
python AdditonalExperiments/CelebA_Hair_Colour_Diagnostic/celeba_top128_change_diagnostic.py --help
```

for the required dataset and cache arguments. The selected
[combined initial/final/heatmap PDF](figures/celeba_ranked_pixel_change_top128_both_arms.pdf)
is included for collaborator review. CelebA data, decoded caches, ranked CSVs,
per-page PNGs and duplicate per-arm PDFs remain local regeneration artifacts.
