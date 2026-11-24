# ColorMNIST small-batch sampling with cached Fourier features

Compare the original interacting IMQ flow, which produces one batch of 2,000
particles, with a cached field that produces 250 separate batches of eight
images. Evaluate the concatenated 2,000 images using pixel-space sliced
Wasserstein distance (SW2), legacy FID and Clean-FID.

## Implementation

- [rff_imq.py](rff_imq.py): paired sine/cosine features for the IMQ kernel,
  weighted target-mean caching, and exact/RFF interacting Euler flows.
- [reference_field.py](reference_field.py): record the target-minus-source
  feature coefficients, integrate consecutive coefficient blocks, and apply the
  frozen field to independent query points. Includes CUDA graph execution.
- [run_small_batch_comparison.py](run_small_batch_comparison.py): portable
  simulation and evaluation entry point for full and coarsened fields, with
  per-seed results and an aggregate comparison table.

The field uses a fixed DR target and a recorded reference population of 2,000
particles. Full replay uses all 3,000 coefficient vectors. Coarsened replay sums
consecutive blocks of 100 vectors and applies 30 updates with the original step
size. The sum preserves the coefficients' total contribution within a block;
freezing the spatial derivative within that block is an approximation.

More explicitly, for the random-feature mean
$\mu_D(Q)=\mathbb E_{Z\sim Q}[\phi_D(Z)]$, the reference cloud minimizes

$$
F_D(Q)=\frac{1}{2}\left\|
\mu_D(Q)-\widehat\mu_{\mathrm{target}}
\right\|_2^2.
$$

Writing

$$
Q_r^{\mathrm{ref}}
=\frac{1}{M_{\mathrm{ref}}}
 \sum_{j=1}^{M_{\mathrm{ref}}}\delta_{\bar z_j^{(r)}},
\qquad
a_r=\widehat\mu_{\mathrm{target}}-\mu_D(Q_r^{\mathrm{ref}}),
$$

the first variation satisfies

$$
\nabla_z\frac{\delta F_D}{\delta Q}(z)
=J_{\phi_D}(z)^\top
 \left(\mu_D(Q)-\widehat\mu_{\mathrm{target}}\right).
$$

The negative-gradient Euler update of a reference particle is consequently

$$
\bar z_j^{(r+1)}
=\bar z_j^{(r)}
+\eta J_{\phi_D}(\bar z_j^{(r)})^\top a_r.
$$

Thus $a_r$ is precisely the residual that drives the reference MMD flow. The
implementation stores this residual before each update and replays the same
velocity $J_{\phi_D}(z)^\top a_r$ on later query particles. The step size uses
the mean-field flow convention; relative to the ordinary Euclidean gradient of
the concatenated reference-particle vector, its additional
$1/M_{\mathrm{ref}}$ factor is absorbed into the flow-time scaling.

Query particles do not interact. Each request starts from fresh uniform draws
with replacement from the observed arm's latent bank, and the field and decoder
remain fixed. This experiment adds no extra learned network or initializer
noise. It tests sampling from that finite empirical initializer, not recovery of
arbitrary unseen image templates or a general convergence guarantee.

## Reproduce the comparison

Use the deployment repository's Python environment, including its image and FID
dependencies. A CUDA GPU is recommended for the full configuration. Commands
below run from this directory:

```bash
cd AdditonalExperiments/ColorMNIST_RFF_Caching
```

First generate the uncorrupted baseline assets with the existing public
[nuisance experiment](../ColorMNIST_Nuisance_Misspecification/README.md).
Restricting both nuisance factors to one generates only the uncorrupted
plugin/IPW/DR controls, rather than the full misspecification grid:

```bash
python ../ColorMNIST_Nuisance_Misspecification/colormnist_misspecification_severity.py \
  --data-seeds 0 1 2 --arms 0 1 \
  --n-obs 20000 --n-particles 2000 --n-ref 2000 \
  --ae-iterations 3000 --flow-iterations 3000 \
  --cme-cutoffs 1 --weight-factors 1 --mass-modes raw \
  --device cuda --mnist-root ../../data \
  --out-dir results/source_baselines --tag baseline20k
```

These directories contain the frozen autoencoder, logged latent support, raw
signed DR coefficients, initial/final original particles and oracle references.
They are inputs to the comparison; no source asset is overwritten or refitted.

```bash
python run_small_batch_comparison.py \
  --source-dir results/source_baselines/baseline20k_seed0 \
               results/source_baselines/baseline20k_seed1 \
               results/source_baselines/baseline20k_seed2 \
  --output-dir results/comparison --device cuda --allow-fid-download
```

The download option permits obtaining the public Inception weights when they
are not cached. Omit it when the required weights are already available.
Run `python run_small_batch_comparison.py --help` for the available options.

A short CPU execution check uses an existing source directory:

```bash
python run_small_batch_comparison.py \
  --source-dir results/source_baselines/baseline20k_seed0 \
  --output-dir results/smoke --device cpu --smoke --skip-fid
```

The default smoke uses 64/128 features, three flow steps, 32 reference particles,
four requests of eight images and eight SW2 directions. It recomputes the exact
control at 32 particles and three steps, and explicitly labels the outputs as
structural checks. These outputs are not full-budget quality evidence.

## Evaluation protocol

The full configuration uses 20,000 logged observations per data seed, digits 1
and 6, DGP parameters `(w, kappa, tau, alpha) = (10, 20, 0.08, 0.2)`, a frozen
32-dimensional autoencoder, and the original full-sample projected-Nystrom DR
target. The target weights remain signed and unnormalized. Fourier seed is zero;
feature dimensions are 4,096 and 16,384.

Each field generates an actual 250-by-eight pool. Full/coarse fields and both
feature dimensions use the same ordered query starts. FID and SW2 are computed
once per complete 2,000-image pool against the same 2,000 raw oracle images;
metrics are not averaged over eight-image requests. Average the two arms within
each seed, then report the mean and sample standard deviation across data seeds
0, 1 and 2. The original baseline uses its interacting particle cloud, initialized
without replacement, so it is not a sample-by-sample paired query control.

The portable runner saves its SW2 projection directions. By default, it makes
one CPU-seeded draw so that projection generation does not depend on GPU RNG
implementation. `--projection-file` can instead reuse saved directions. The
recorded study below used archived CUDA-generated directions; rerunning the
protocol with a new representation or different directions need not reproduce
every reported decimal.

## Recorded three-seed results

These are the previously executed study's mean and sample SD, after averaging
arms within seed; they are not a new execution of the portable entry point.
Every RFF row used 250 actual requests of eight images. Lower quality scores are
better.

| Method | Updates | SW2 | Legacy FID | Clean-FID | Field cache, MB/arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original exact IMQ, one batch of 2,000 | 3,000 | 0.009402 ± 0.000561 | 30.666 ± 0.867 | 35.053 ± 0.739 | N/A |
| Full RFF, D=4,096 | 3,000 | 0.010403 ± 0.000572 | 31.178 ± 0.259 | 35.113 ± 0.967 | 49.414 |
| Full RFF, D=16,384 | 3,000 | 0.009964 ± 0.000296 | 31.135 ± 0.397 | 35.117 ± 1.031 | 197.657 |
| Coarsened RFF, D=4,096 | 30 | 0.010263 ± 0.000675 | 31.128 ± 0.253 | 35.009 ± 0.986 | 0.754 |
| Coarsened RFF, D=16,384 | 30 | 0.009829 ± 0.000433 | 31.061 ± 0.377 | 34.987 ± 1.043 | 3.015 |

Cache sizes count float32 coefficients and Fourier frequencies per arm in decimal
MB. They exclude the initializer bank, decoder and working memory. The original
direct sampler does not retain a field cache. Coarsening has slightly better mean
quality scores in these runs; this is a descriptive three-seed finding. Update
counts are not measured wall-clock speedup ratios, and field construction is a
separate one-time cost.

## Files kept locally

The [.gitignore](.gitignore) uses an explicit source allowlist. Generated results,
model weights, datasets, execution logs, notebook outputs and development records
are excluded from version control. New files stay excluded until deliberately
added to that allowlist. This keeps local artifacts available while publishing
only the implementation, experiment script and this documentation.
