# ColorMNIST with an independently trained encoder

This experiment partitions a total budget
of 20,000 logged observations into 10,000 images for encoder training and a
disjoint 10,000 observations for causal estimation. It uses DR–IMMD only, with
2,000 generated particles per arm and data seeds 0, 1 and 2.

## Data-generating process

For each logged observation, draw a colour variable `X ~ Uniform[0,1]` and
`A | X ~ Bernoulli(sigmoid(10 * (X - 0.5)))`. Arm 0 selects MNIST digit 1;
arm 1 selects digit 6. A grayscale template `S` is sampled uniformly with
replacement from that digit's fixed MNIST training bank. With a black
background, the RGB image is

```text
mask = sigmoid(20 * (S - 0.08))
Y = mask * S * (X, 0.2, 1 - X).
```

The intervention fixes the arm while drawing a fresh uniform colour variable
and a fresh template from its digit bank. Oracle images are used only for
evaluation. The renderer and DGP are unchanged from the headline experiment.

## Split and estimation protocol

For data seed `s` there is one pipeline repeat, with `pipeline_seed = 1000*s`.
A random permutation seeded by `pipeline_seed + 2` partitions the 20,000
logged row indices. The first 10,000 train the autoencoder; the remaining
10,000 form the causal sample. These row sets have no intersection.

| Component | Independent-encoder experiment | Original headline baseline |
|---|---:|---:|
| Total logged observations | 20,000 | 20,000 |
| AE training images | 10,000 | 20,000 |
| Causal observations / DR support | 10,000 | 20,000 |
| Overlap between AE and causal rows | 0 | 20,000 |
| Generated particles per arm | 2,000 | 2,000 |
| Scored particles / oracle references per arm | 2,000 / 2,000 | 2,000 / 2,000 |
| Data seeds / pipeline repeats per seed | 3 / 1 | 3 / 1 |

The convolutional autoencoder has a 32-dimensional latent space. It trains
for 3,000 Adam updates, batch size 128 and learning rate 0.001, using only the
AE subset. Its parameters are then frozen. Reconstruction error on the
causal subset is a diagnostic: it does not select or update the encoder.

The propensity and conditional mean embedding use all 10,000 causal
observations, with **full-sample nuisance fitting** (`dr_fit=nosplit`). This
is an encoder split, not nuisance cross-fitting. The target uses projected
Nyström regression with 1,024 landmarks and the original cross-validated
ridge. No additional cap is imposed on the DR support. The context RBF
bandwidth uses up to 4,000 causal rows; ridge selection uses 2,000-row causal
subsets. The recorded five-fold settings concern the existing nuisance
machinery, not five separately trained encoders.

Initial particles come only from the causal subset's observed target arm.
The IMQ kernel has exponent `-0.5`; its scale is the median pairwise latent
distance on up to 5,000 causal rows. The original deterministic Euler flow
runs for 3,000 updates with step size `0.15 * scale**2`. Initial particles
are sampled without replacement, and the completed output must contain
2,000 particles in each arm.

Each arm is evaluated against 2,000 fresh oracle images. A second independent
2,000-image oracle draw supplies the sampling floor and decoder round-trip
diagnostic. These draws use seeds `pipeline_seed + 9` and `pipeline_seed + 10`.
Pixel SW2, legacy FID and Clean-FID follow the existing implementation. Oracle
images do not train the encoder, propensity, CME or sampling kernel.

## Run and summarize

The commands below assume the current directory is the deploy repository
root and the existing MNIST assets are under `data/MNIST/raw`.

```bash
python AdditonalExperiments/ColorMNIST_Independent_Encoder/run_independent_encoder.py \
  --prepare \
  --run-root AdditonalExperiments/ColorMNIST_Independent_Encoder/runs/independent_20261123 \
  --mnist-root data

python AdditonalExperiments/ColorMNIST_Independent_Encoder/run_independent_encoder.py \
  --seed 0 \
  --run-root AdditonalExperiments/ColorMNIST_Independent_Encoder/runs/independent_20261123
```

Repeat the second command with `--seed 1` and `--seed 2`, then summarize:

```bash
python AdditonalExperiments/ColorMNIST_Independent_Encoder/summarize_results.py \
  --run-root AdditonalExperiments/ColorMNIST_Independent_Encoder/runs/independent_20261123 \
  --require-complete
```

The preparation step preserves the source and original three-seed headline
baseline CSVs. The comparison uses their DR rows, rather than the later
twenty-seed rerun. The summary script validates the declared configuration,
averages the two arms within each seed, and then reports the mean and sample
standard deviation across seeds. It writes `per_seed.csv`, `summary.csv`,
`coverage.json` and `comparison.md` under the run's `analysis/` directory.
It rejects invalid configurations and duplicate rows. Missing-arm seeds do
not enter an average. Omit `--require-complete` for explicitly labeled
partial summaries; strict mode writes coverage and reports before failing
if either group lacks any of its three complete seeds.

## Interpretation

The split makes encoder-training observations independent of the causal
observations under the DGP conditional on the fixed MNIST template banks.
Different observations can reuse the same grayscale template, so this is
not a test on held-out handwriting templates.

Compared with the headline, the experiment also halves both training
sample sizes and changes the eligible initial-particle pool. It tests the
practical cost of an independent encoder within a fixed total observation
budget; it does not isolate independence from sample-size effects. The
encoder architecture, training budget, flow rule and evaluation protocol
are retained, but learned latent coordinates and bandwidths can change.
Latent SW2 therefore remains representation dependent. No runtime comparison
is intended.

The AE reconstruction diagnostics use at most 2,000 images from each subset.
In the original shared-data baseline, the CSV field `recon_mse_heldout`
actually uses the same subset as `recon_mse_train`; it is not a held-out
error there. The new split makes the causal-subset diagnostic held out from
encoder training.

This experiment addresses encoder/sample independence only. It does not
establish every assumption of the theoretical guarantees: adaptive kernel
bandwidths, cross-validated ridge, fixed landmark rank and deterministic
Euler flow remain separate qualifications.

## Paired five-fold nuisance comparison

With the same 10,000/10,000 split, we trained one encoder per seed and froze it for both DR fits. The control fits the propensity and projected conditional mean embedding on all 10,000 causal rows. Five-fold fitting trains each nuisance on 8,000 rows and evaluates on the remaining 2,000. Both methods use the same causal-sample bandwidth, starting latents, oracle images and IMQ step size. Each generates 2,000 particles per arm.

| Final metric | Full-sample nuisances | Five-fold nuisances | Paired change |
|---|---:|---:|---:|
| Pixel SW₂ × 1,000 | 9.993 ± 0.620 | 10.100 ± 0.636 | +0.107 ± 0.065 |
| FID | 29.574 ± 0.519 | 29.630 ± 0.566 | +0.056 ± 0.067 |

We averaged arms within each seed and report the mean ± sample SD across three seeds. Cross-fitting gave slightly higher pixel SW₂ on all three. The full-sample row is the control recomputed on the same H100 and encoder as its cross-fitted pair. The earlier independent-encoder run on L40S gave 10.050 ± 0.579 and 29.552 ± 0.518.
