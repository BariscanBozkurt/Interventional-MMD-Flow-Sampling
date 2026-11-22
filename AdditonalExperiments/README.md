# Additional experiments for the AISTATS rebuttal

This is the short collaborator-facing report. Unless stated otherwise, image
results use 20,000 logged observations, 2,000 returned particles per arm, and
data seeds 0--2. We first average the two intervention arms within each seed and
then report the mean and sample standard deviation across seeds. Lower SW2 and
FID are better. Longer protocols and reproduction details remain in each
experiment's own directory.

## Current status

| Experiment | Status |
| --- | --- |
| ColorMNIST nuisance misspecification | Complete, three seeds |
| ColorMNIST small-batch RFF sampling | Complete, three seeds |
| ColorMNIST repetition study | Complete, 20 seeds |
| Finite-candidate kernel herding | Complete, three seeds |
| Continuous-domain kernel herding | CelebA complete; ColorMNIST seeds 1--2 pending |
| CelebA hair-colour diagnostic | Complete for the reported seed-0 run |
| CelebA IMQ--Euler versus Riesz--AdamW | Three-seed production run in progress |

## 1. Nuisance misspecification on ColorMNIST

We adapt the controlled corruption in the
[DoubleGen experiment of Luedtke and Fukumizu](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1).
Here, `X` is the foreground red fraction. We corrupt the two nuisances
independently:

- **CME corruption:** fit the conditional mean embedding only on rows with
  `X < 0.5` instead of all logged rows;
- **weight corruption:** multiply inverse weights by `1/4` on the red-heavy
  subgroup `X >= 0.5`.

The CME restriction affects only nuisance fitting: all observations remain in
the DR residual and policy-integration terms. We cross these two corruptions and
run plug-in, IPW, and DR targets using 2,000 particles per arm, 1,024 projected-
Nyström landmarks, and 3,000 IMQ--Euler updates. The table reports pixel SW2
times `10^3` over three data seeds.

| CME fit | Inverse weights | Plug-in | IPW | DR |
| --- | --- | ---: | ---: | ---: |
| All `X` | Baseline | 9.413 ± 0.576 | 9.486 ± 0.350 | 9.402 ± 0.561 |
| All `X` | Red-heavy × 1/4 | 9.413 ± 0.576 | 33.420 ± 0.784 | 9.416 ± 0.575 |
| Only `X < 0.5` | Baseline | 27.154 ± 0.895 | 9.486 ± 0.350 | 9.464 ± 0.459 |
| Only `X < 0.5` | Red-heavy × 1/4 | 27.154 ± 0.895 | 33.420 ± 0.784 | 21.579 ± 0.690 |

DR remains close to the baseline when either nuisance alone is corrupted and
degrades when both are corrupted. This is finite-sample evidence, not an oracle
correct-specification experiment: the baseline nuisances are themselves fitted,
and restricting the CME also reduces its training support and sample size.
[Reproduction instructions](ColorMNIST_Nuisance_Misspecification/README.md).

## 2. Small-batch sampling with cached IMQ random features

We approximate the IMQ kernel by a finite feature map
`k(z,z') ≈ phi_D(z)^T phi_D(z')`. For the fixed DR target, cache

```text
mu_target = sum_i v_i phi_D(z_i).
```

A reference cloud of 2,000 particles records, before each Euler update,

```text
a_r = mu_target - (1/M_ref) sum_j phi_D(zbar_j^r).
```

The reference cloud and every later query particle then use the same prescribed
field:

```text
zbar_j^(r+1) = zbar_j^r + eta J_phi(zbar_j^r)^T a_r,
z^(r+1)      = z^r      + eta J_phi(z^r)^T      a_r.
```

Thus a request of eight images does not estimate a noisy eight-particle source
mean and does not interact with other requests. Full replay stores all 3,000
coefficient vectors. The coarsened version sums consecutive blocks of 100 and
uses 30 updates; this freezes the spatial Jacobian within each block and is an
additional numerical approximation. We evaluate 250 actual requests of eight
images as one 2,000-image pool.

| Sampler | Updates | Pixel SW2 | Legacy FID | Clean-FID | Cache MB/arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact IMQ, one interacting batch of 2,000 | 3,000 | 0.009402 ± 0.000561 | 30.666 ± 0.867 | 35.053 ± 0.739 | -- |
| Full RFF, `D=4,096`, 250 × 8 | 3,000 | 0.010403 ± 0.000572 | 31.178 ± 0.259 | 35.113 ± 0.967 | 49.414 |
| Full RFF, `D=16,384`, 250 × 8 | 3,000 | 0.009964 ± 0.000296 | 31.135 ± 0.397 | 35.117 ± 1.031 | 197.657 |
| Coarsened RFF, `D=4,096`, 250 × 8 | 30 | 0.010263 ± 0.000675 | 31.128 ± 0.253 | 35.009 ± 0.986 | 0.754 |
| Coarsened RFF, `D=16,384`, 250 × 8 | 30 | 0.009829 ± 0.000433 | 31.061 ± 0.377 | 34.987 ± 1.043 | 3.015 |

The cached field preserves the large-reference-cloud gradient signal when the
requested batch is small. The finite feature map is not characteristic, and
the coarsened update is not the original 3,000-step trajectory. Cache sizes
exclude the decoder and initializer bank. [Implementation and full evaluation
protocol](ColorMNIST_RFF_Caching/README.md).

## 3. ColorMNIST across 20 data seeds

We reran the unchanged headline DR--IMMD configuration for data seeds 0--19,
with both arms, one pipeline repeat, 20,000 observations, and 2,000 particles
per arm. Seeds 0--2 are reruns rather than copies of the submitted artifacts.

| Runs | Pixel SW2 | Legacy FID | Clean-FID |
| --- | ---: | ---: | ---: |
| Original submitted seeds 0--2 | 0.009304 ± 0.000295 | 30.644 ± 0.809 | 35.146 ± 0.641 |
| Rerun seeds 0--2 | 0.009510 ± 0.000106 | 30.790 ± 0.932 | 35.258 ± 0.451 |
| All rerun seeds 0--19 | 0.009595 ± 0.000499 | 30.345 ± 1.847 | 34.504 ± 2.082 |

For all 20 seeds, the 95% Student-t interval for mean pixel SW2 is
`[0.009362, 0.009829]`. The reruns used the same frozen scientific
configuration but different GPU hardware, so seeds 0--2 are a practical
reproduction check rather than a bitwise replay.
[Reproduction and aggregation protocol](ColorMNIST_20_Seeds/README.md).

## 4. Kernel-herding baselines

At step `p`, kernel herding selects an atom by maximizing the residual target
score

```text
sum_i v_i k(z, z_i) - gamma_p sum_{j=1}^p k(z, z_j).
```

We evaluate two implementations using the same estimated DR embedding as
DR--IMMD Flow:

1. **Finite-candidate herding:** optimize over the logged outcome/latent bank.
   It is a fast retrieval or resampling method. The indexed output is the
   selected logged image; the decoded output passes its selected latent through
   the same frozen decoder used by DR--IMMD. This implementation uses the
   released-code convention `gamma_p=1/p` for `p>0`.
2. **Continuous herding:** optimize each atom over the unrestricted latent
   domain with multistart Adam, then decode it. This is genuine synthesis rather
   than retrieval. It follows the displayed CPME Algorithm-2 convention
   `gamma_p=1/(p+1)`. The inner optimization is approximate.

### Finite-candidate results

| Dataset | Method | Pixel SW2 | Latent SW2 | Legacy FID | Clean-FID | Sampler seconds, two arms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ColorMNIST | DR--IMMD Flow | 0.009389 ± 0.000471 | 0.080120 ± 0.001126 | 30.696 ± 0.838 | 35.102 ± 0.718 | 35.70 |
| ColorMNIST | Herding, indexed | 0.008276 ± 0.000080 | 0.084504 ± 0.001294 | 4.374 ± 0.072 | 4.887 ± 0.065 | 0.80 |
| ColorMNIST | Herding, decoded | 0.009796 ± 0.000249 | 0.084504 ± 0.001294 | 31.486 ± 0.949 | 36.341 ± 0.815 | 0.80 |
| CelebA | DR--IMMD Flow | 0.026562 ± 0.001553 | 0.051289 ± 0.001361 | 11.567 ± 0.085 | 14.528 ± 0.179 | 698.73 |
| CelebA | Herding, indexed | 0.027033 ± 0.001601 | 0.052645 ± 0.001261 | 10.907 ± 0.091 | 13.114 ± 0.076 | 4.93 |
| CelebA | Herding, decoded | 0.027142 ± 0.001610 | 0.052645 ± 0.001261 | 11.447 ± 0.081 | 13.700 ± 0.151 | 4.93 |

The extremely low indexed ColorMNIST FID reflects retrieval of real logged
images and is not a like-for-like generative comparison. After decoding both
methods, performance is close. CelebA herding selects about 1,999 distinct
images out of 2,000, while ColorMNIST uses about 1,567 distinct images.

The two-dimensional benchmark has no encoder or decoder:

| Kernel | DR--IMMD Flow SW2 | Finite herding SW2 |
| --- | ---: | ---: |
| RBF | 0.111993 ± 0.011606 | 0.114597 ± 0.008127 |
| Riesz | 0.126294 ± 0.016524 | 0.110415 ± 0.011188 |

### Continuous-domain CelebA results

| Method | Pixel SW2 | Latent SW2 | Legacy FID | Clean-FID | Optimizer seconds/arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| DR--IMMD Flow | 0.026562 ± 0.001553 | 0.051289 ± 0.001361 | 11.567 ± 0.085 | 14.528 ± 0.179 | 349.32 ± 0.03 |
| Finite herding, decoded | 0.027142 ± 0.001610 | 0.052645 ± 0.001261 | 11.447 ± 0.081 | 13.700 ± 0.151 | 2.479 ± 0.006 |
| Continuous herding | 0.027124 ± 0.001511 | 0.052322 ± 0.001294 | 15.690 ± 0.106 | 20.999 ± 0.049 | 194.35 ± 1.40 |

Continuous herding returns 2,000 distinct off-bank atoms per arm and matches
SW2 reasonably well, but its worse FID indicates that those optimized latents
decode less faithfully. Finite herding is especially strong here because the
target is a reweighting well represented by the 20,000-image logged pool. The
defensible distinction is therefore retrieval versus synthesis, not that
DR--IMMD uniformly dominates herding. The continuous ColorMNIST three-seed run
is still incomplete and is excluded from these tables. [Code, artifacts and
complete aggregation](Kernel_Herding_Baselines/README.md).

## 5. CelebA hair-colour diagnostic

For the exact seed-0 headline Riesz--AdamW run, we rank all 2,000 particles in
each arm by decoded start-to-final RGB distance. The diagnostic displays the
largest 128 per arm as the decoded initializer, decoded endpoint, and a
common-scale absolute pixel-change heatmap. The face images are not
redistributed publicly; users with CelebA access can regenerate them locally.

| Arm and ranks | Initial label composition | Mean RGB RMSE | Mean luminance change |
| --- | --- | ---: | ---: |
| `a=0`, ranks 1--16 | 16/16 blond | 19.015 | -11.658 |
| `a=0`, ranks 113--128 | 16/16 blond | 15.442 | -9.575 |
| `a=1`, ranks 1--16 | 16/16 non-blond | 18.269 | +7.296 |
| `a=1`, ranks 113--128 | 16/16 non-blond | 14.412 | +5.956 |

All 128 largest arm-0 changes begin blond and all 128 largest arm-1 changes
begin non-blond. This matches the required population correction: arm 0 must
reduce blond mass, whereas arm 1 must increase it. Heatmaps often emphasize
hair and the head outline, but also show face, clothing, background, and global
illumination changes. These are post-hoc seed-0 extremes, not random samples or
evidence of isolated same-person hair editing. [Full provenance and caveats](CelebA_Hair_Colour_Diagnostic/README.md).

## 6. IMQ--Euler theory-alignment check on CelebA

The paired production experiment reruns Riesz--AdamW, deterministic IMQ--Euler,
and the theory-aligned noisy IMQ--Euler on the same data, DR target, initial
particles, and evaluation draws. It uses 2,000 particles per arm and 4,000
updates. The three-seed array is still running, so we do not report a final
comparison table yet.

A seed-0 screening result suggests that deterministic IMQ is already
competitive: its arm-averaged pixel SW2 is `0.028070` versus `0.028086` for
Riesz, while legacy/Clean FID are `11.703/14.901` versus `11.488/14.398`.
These are pilot values only. The corresponding largest-change diagnostic can
be regenerated from the public script after obtaining CelebA.
[Experiment status and protocol](CelebA_Theory_Sampler_Ablation/README.md).

## Scope

This report contains the rebuttal-ready results and clearly labels incomplete
campaigns. The subdirectory READMEs provide only the commands needed to
reproduce them. Exploratory CelebA RFF work is retained locally but excluded
from the public payload because it is not used in this rebuttal summary.
