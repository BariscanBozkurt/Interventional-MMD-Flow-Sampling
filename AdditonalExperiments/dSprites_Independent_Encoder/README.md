# dSprites with an independently trained encoder

This experiment mirrors the fixed-budget ColorMNIST encoder split on the
revised confounded dSprites benchmark. We generate 20,000 logged observations,
use 10,000 only to train the denoising autoencoder, and use the disjoint 10,000
only for propensity estimation, projected-Nyström CME regression, the DR target,
particle initialization, and evaluation.

dSprites has one stochastic continuous-action policy target, rather than two
static intervention arms. Each seed therefore returns 2,000 particles in total.
We use data seeds 0--2 and one pipeline repeat per seed.

All other settings match the headline experiment: the confounded dSprites DGP,
a 32-dimensional latent space, 4,000 autoencoder updates, full-sample nuisance
fitting on the causal half, 1,024 projected-Nyström landmarks, eight target-policy
draws per context, observed-outcome initialization, and 2,000 Riesz--AdamW
updates.

From the deploy-repository root, one seed can be reproduced with

```bash
python Simulations/dSprites/dr_mmd_flow/dsprites_dr_riesz_flow_simulation.py \
  --design confounded --n_obs 20000 \
  --ae_split half --ae_fraction 0.5 --dr_fit nosplit \
  --solver projected_nystrom --n_landmarks 1024 \
  --n_particles 2000 --estimators dr --data_seeds 0 --n_repeats 1 \
  --tag independent_encoder \
  --out_dir AdditonalExperiments/dSprites_Independent_Encoder/runs/manual/results
```

Repeat with seeds 1 and 2, then summarize with

```bash
python AdditonalExperiments/dSprites_Independent_Encoder/summarize_results.py \
  --run-root AdditonalExperiments/dSprites_Independent_Encoder/runs/manual \
  --require-complete
```

## Results

All three seeds completed. Entries are means and sample standard deviations
across seeds 0--2; lower is better.

| Encoder / causal protocol | AE rows | Causal rows | Pixel SW2 x 1,000 | Latent SW2 x 1,000 | FID | Clean-FID |
|---|---:|---:|---:|---:|---:|---:|
| Shared observations | 20,000 | 20,000 | 8.394 ± 0.139 | 39.111 ± 0.521 | 36.486 ± 1.657 | 61.907 ± 5.400 |
| Disjoint 10k / 10k split | 10,000 | 10,000 | 8.827 ± 0.022 | 41.077 ± 1.145 | 35.321 ± 1.807 | 63.287 ± 4.622 |

The disjoint split retains similar final performance despite halving both
training samples. The split runs used H100 GPUs, whereas the displayed
shared-data baseline used L40S GPUs; these are accuracy results, not a runtime
comparison. The learned autoencoders also define different latent coordinate
systems, so pixel SW2 and FID are the primary cross-protocol metrics.

This is an encoder/sample-independence check within a fixed total observation
budget. It also halves both the autoencoder and causal training sample sizes,
so it does not isolate independence from sample-size effects. Different logged
rows may render from the same finite dSprites template bank; the split is over
observations, not over underlying templates.

