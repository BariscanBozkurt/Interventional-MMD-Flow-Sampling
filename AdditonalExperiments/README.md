# Additional experiments for the AISTATS rebuttal

This is the short collaborator-facing report. Unless stated otherwise, image
results use 20,000 logged observations, 2,000 returned particles per arm, and
data seeds 0--2. We first average the two intervention arms within each seed and
then report the mean and sample standard deviation across seeds. Lower SW2 and
FID are better. Separate experiments use their own matched controls; their
baseline rows should not be pooled. Longer protocols and reproduction details
remain in each experiment's own directory. Status checked on 2026-11-24.

## Current status

| Experiment | Status |
| --- | --- |
| ColorMNIST nuisance misspecification | Complete, three seeds |
| ColorMNIST small-batch RFF sampling | Complete, three seeds |
| ColorMNIST repeated-request timing | Complete, three seeds |
| ColorMNIST 20-data-seed study | Complete |
| [ColorMNIST independent encoder](ColorMNIST_Independent_Encoder/README.md) | Complete, three seeds |
| [dSprites independent encoder](dSprites_Independent_Encoder/README.md) | Complete, three seeds |
| Full-sample versus five-fold nuisance fitting | Complete, three seeds |
| Finite-candidate kernel herding | Complete, three seeds |
| Continuous-domain kernel herding | Complete for ColorMNIST and CelebA, three seeds |
| CelebA hair-colour diagnostic | Complete for the reported seed-0 run |
| CelebA IMQ--Euler versus Riesz--AdamW | Complete, three seeds |

## 1. Nuisance misspecification on ColorMNIST

We adapt the CelebA sensitivity design in Luedtke and Fukumizu's
[DoubleGen, Section M.1.1 ("Nuisance estimation")](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1),
whose comparison is reported in their Table 3. They train the outcome nuisance
only on dark-haired images and divide the inverse-propensity weights of
lighter-haired images by four. We apply the same two-axis idea to ColorMNIST and
add stronger corruption levels.

Here, $X\in[0,1]$ is the foreground red fraction. We first fit an affine
logistic propensity on all 20,000 logged pairs,

$$
\widehat e_1(x)=\operatorname{sigmoid}(\widehat b+\widehat\theta x),
\qquad \widehat e_0(x)=1-\widehat e_1(x),
$$

and form the arm-$a$ inverse weights

$$
w_i^{(a)}=\frac{\mathbf 1\{A_i=a\}}{\widehat e_a(X_i)}.
$$

The simulator uses $e_1(x)=\operatorname{sigmoid}\{10(x-0.5)\}$, but the
reported weights use the fitted $\widehat e$, not this oracle propensity. We do
not clip the fitted propensities or inverse weights.

We then corrupt the nuisances independently. **For the CME support
restriction**, we fit the conditional mean embedding on all rows, or only on
rows satisfying $X<c$ for $c\in\{0.5,0.25,0.1\}$. **For the inverse-weight
corruption**, we keep the fitted propensity fixed and set

$$
h_q(x)=\mathbf 1\{x<0.5\}+q\mathbf 1\{x\geq0.5\},
\qquad
\widetilde w_i^{(a,q)}=h_q(X_i)w_i^{(a)},
$$

where $q\in\{1,1/4,1/20,1/100\}$. Thus $q=1/4$ divides the fitted inverse
weights of red-heavy observations by four, matching the DoubleGen perturbation.
This is applied *after* propensity fitting. The corrupted weights need not be
reciprocals of a valid complementary propensity pair, so we describe this
precisely as inverse-weight corruption rather than a newly fitted propensity
model.

Only CME fitting, its ridge selection, and its landmark selection use the
restricted rows. All 20,000 observations remain in the target-policy average
and DR residual term, and the propensity and autoencoder always use all 20,000
observations. The realized CME training counts were:

| CME support | Total rows, seeds 0 / 1 / 2 | Arm 0 rows | Arm 1 rows |
| --- | ---: | ---: | ---: |
| All $X$ | 20,000 / 20,000 / 20,000 | 10,077 / 10,020 / 9,968 | 9,923 / 9,980 / 10,032 |
| $X<0.5$ | 10,076 / 10,077 / 10,035 | 8,760 / 8,644 / 8,636 | 1,316 / 1,433 / 1,399 |
| $X<0.25$ | 5,058 / 5,026 / 4,989 | 4,928 / 4,870 / 4,852 | 130 / 156 / 137 |
| $X<0.1$ | 2,023 / 2,009 / 1,974 | 2,007 / 1,974 / 1,952 | 16 / 35 / 22 |

We run plug-in, IPW, and DR targets using 2,000 particles per arm, a total
budget of 1,024 projected-Nyström landmarks, and 3,000 IMQ--Euler updates. The
table reports pixel SW2 $\times 10^3$ under the raw weight corruption, after
averaging arms within seed and then taking the mean and sample standard
deviation over seeds 0--2.

| CME support | Red-heavy multiplier $q$ | Plug-in | IPW | DR |
| --- | ---: | ---: | ---: | ---: |
| All $X$ | $1$ | 9.413 ± 0.576 | 9.486 ± 0.350 | 9.402 ± 0.561 |
| All $X$ | $1/4$ | 9.413 ± 0.576 | 33.420 ± 0.784 | 9.416 ± 0.575 |
| All $X$ | $1/20$ | 9.413 ± 0.576 | 41.288 ± 1.207 | 9.422 ± 0.578 |
| All $X$ | $1/100$ | 9.413 ± 0.576 | 42.811 ± 1.288 | 9.423 ± 0.578 |
| $X<0.5$ | $1$ | 27.154 ± 0.895 | 9.486 ± 0.350 | 9.464 ± 0.459 |
| $X<0.5$ | $1/4$ | 27.154 ± 0.895 | 33.420 ± 0.784 | 21.579 ± 0.690 |
| $X<0.5$ | $1/20$ | 27.154 ± 0.895 | 41.288 ± 1.207 | 26.112 ± 0.876 |
| $X<0.5$ | $1/100$ | 27.154 ± 0.895 | 42.811 ± 1.288 | 27.005 ± 0.924 |
| $X<0.25$ | $1$ | 67.196 ± 40.217 | 9.486 ± 0.350 | 9.434 ± 0.474 |
| $X<0.25$ | $1/4$ | 67.196 ± 40.217 | 33.420 ± 0.784 | 46.130 ± 27.043 |
| $X<0.25$ | $1/20$ | 67.196 ± 40.217 | 41.288 ± 1.207 | 54.880 ± 29.788 |
| $X<0.25$ | $1/100$ | 67.196 ± 40.217 | 42.811 ± 1.288 | 56.517 ± 30.231 |
| $X<0.1$ | $1$ | 86.038 ± 57.141 | 9.486 ± 0.350 | 9.581 ± 0.485 |
| $X<0.1$ | $1/4$ | 86.038 ± 57.141 | 33.420 ± 0.784 | 41.199 ± 15.664 |
| $X<0.1$ | $1/20$ | 86.038 ± 57.141 | 41.288 ± 1.207 | 50.020 ± 18.035 |
| $X<0.1$ | $1/100$ | 86.038 ± 57.141 | 42.811 ± 1.288 | 51.653 ± 18.377 |

DR remains near baseline when either nuisance alone is corrupted and degrades
when both are corrupted. The large variability at $X<0.25$ and $X<0.1$ is
consistent with the severe arm imbalance shown above. This is a finite-sample
stress test, not an oracle-correct-specification experiment: the baseline
nuisances are fitted, and the CME restriction changes both support and sample
size. Empirical-mass-preserving weight controls are reported in the detailed
experiment README.

A separate replay with the revised headline encoders gives clean plug-in, IPW
and DR scores of `9.303 +/- 0.375`, `9.396 +/- 0.112` and
`9.292 +/- 0.351`. Under every single CME-support or inverse-weight corruption,
DR remains between `9.304` and `9.372`; with both $X<0.5$ and $q=1/4$, it rises
to `21.428 +/- 0.596`. This matched replay confirms the same single-versus-joint
corruption conclusion without mixing its values into the older severity table.
[Reproduction instructions](ColorMNIST_Nuisance_Misspecification/README.md).

## 2. Small-batch sampling with cached IMQ random features

We approximate the IMQ kernel by a finite feature map,

$$
k(z,z')\approx \phi_D(z)^\top\phi_D(z').
$$

For the fixed DR target, cache

$$
\widehat\mu_{\mathrm{target}}
=\sum_{i=1}^n v_i\phi_D(z_i).
$$

For a latent law $Q$, the finite-feature flow minimizes

$$
F_D(Q)
=\frac{1}{2}\left\|
\mu_D(Q)-\widehat\mu_{\mathrm{target}}
\right\|_2^2,
\qquad
\mu_D(Q)=\mathbb E_{Z\sim Q}[\phi_D(Z)].
$$

This is the random-feature approximation of one half of the squared RKHS
discrepancy used by MMD flow. At iteration $r$, the reference empirical law is

$$
Q_r^{\mathrm{ref}}
=\frac{1}{M_{\mathrm{ref}}}
 \sum_{j=1}^{M_{\mathrm{ref}}}\delta_{\bar z_j^{(r)}}.
$$

Its target-minus-source feature residual is

$$
a_r
=\widehat\mu_{\mathrm{target}}-\mu_D(Q_r^{\mathrm{ref}})
=\widehat\mu_{\mathrm{target}}
-\frac{1}{M_{\mathrm{ref}}}
 \sum_{j=1}^{M_{\mathrm{ref}}}\phi_D(\bar z_j^{(r)}).
$$

Indeed,

$$
\nabla_z\frac{\delta F_D}{\delta Q}(z)
=J_{\phi_D}(z)^\top
 \left(\mu_D(Q)-\widehat\mu_{\mathrm{target}}\right).
$$

Therefore, one Euler step along the negative MMD-flow gradient updates each
reference particle as

$$
\bar z_j^{(r+1)}
=\bar z_j^{(r)}
+\eta J_{\phi_D}(\bar z_j^{(r)})^\top a_r.
$$

This is the mean-field particle velocity; the $1/M_{\mathrm{ref}}$ factor in
the ordinary Euclidean gradient of the concatenated particle vector is absorbed
into the flow-time convention. We record each resulting $a_r$ and apply the
same prescribed field to every later query particle:

$$
z^{(r+1)}
=z^{(r)}+\eta J_{\phi_D}(z^{(r)})^\top a_r.
$$

Thus a request of eight images does not estimate a noisy eight-particle source
mean and does not interact with other requests. Full replay stores all 3,000
coefficient vectors. The coarsened version sums consecutive blocks of 100 and
uses 30 updates; this freezes the spatial Jacobian within each block and is an
additional numerical approximation. We evaluate 250 actual requests of eight
images as one 2,000-image pool.

| Sampler | Updates | Pixel SW2 | Legacy FID | Clean-FID | Cache MB/arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact IMQ, one interacting batch of 2,000 | 3,000 | 0.009402 ± 0.000561 | 30.666 ± 0.867 | 35.053 ± 0.739 | -- |
| Full RFF, $D=4{,}096$, 250 × 8 | 3,000 | 0.010403 ± 0.000572 | 31.178 ± 0.259 | 35.113 ± 0.967 | 49.414 |
| Full RFF, $D=16{,}384$, 250 × 8 | 3,000 | 0.009964 ± 0.000296 | 31.135 ± 0.397 | 35.117 ± 1.031 | 197.657 |
| Coarsened RFF, $D=4{,}096$, 250 × 8 | 30 | 0.010263 ± 0.000675 | 31.128 ± 0.253 | 35.009 ± 0.986 | 0.754 |
| Coarsened RFF, $D=16{,}384$, 250 × 8 | 30 | 0.009829 ± 0.000433 | 31.061 ± 0.377 | 34.987 ± 1.043 | 3.015 |

The cached field preserves the large-reference-cloud gradient signal when the
requested batch is small. The finite feature map is not characteristic, and
the coarsened update is not the original 3,000-step trajectory. Cache sizes
exclude the decoder and initializer bank. [Implementation and full evaluation
protocol](ColorMNIST_RFF_Caching/README.md).

### Paper-matched batch-eight and repeated-request checks

Using the revised headline encoders, we compare one exact 2,000-particle cloud,
250 cached requests of eight particles, and 250 independently interacting
eight-particle groups. The requests are pooled only for distributional scoring.

| Sampler | Pixel SW2 $\times 10^3$ | FID |
| --- | ---: | ---: |
| Exact IMQ, $1\times2{,}000$ | 9.292 ± 0.351 | 30.650 ± 0.788 |
| Cached field, $250\times8$ | 9.733 ± 0.381 | 31.068 ± 0.341 |
| Independent interacting groups, $250\times8$ | 13.874 ± 0.463 | 39.062 ± 0.463 |

For the cached sampler, constructing and coarsening the reference field takes
`4.481 +/- 0.032` seconds per arm. Serving all 250 requests then takes
`0.184 +/- 0.002` seconds with the field and CUDA graph ready, or
`4.709 +/- 0.032` seconds including one-time preparation. These clocks include
fresh starts, decoding and synchronization but exclude common nuisance fitting.

For the original exact sampler, cumulative fitting plus sequential generation
for both arms takes 37.43, 358.20, 393.85 and 714.61 seconds at respectively
2,000, 20,000, 22,000 and 40,000 images per arm. This is component accounting
on the matched GPU, not process-launch latency or a direct DecFM timing.

## 3. ColorMNIST across 20 data seeds

We extended revised Table 1 to data seeds 0–19 on L40S GPUs, using 20,000
observations and 2,000 particles per arm. Seeds 0–2 reused their archived
encoders. Each of the other 17 seeds trained its own encoder. All 40 arm
scores were recomputed. We averaged the two arms within each seed.

| Runs | Pixel SW2 | Legacy FID | Clean-FID |
| --- | ---: | ---: | ---: |
| Revised Table 1, seeds 0–2 | 0.009292 ± 0.000351 | 30.650 ± 0.788 | — |
| Rerun, seeds 0–2 | 0.009292 ± 0.000351 | 30.650 ± 0.788 | 35.087 ± 0.781 |
| Rerun, all 20 seeds | 0.009494 ± 0.000408 | 30.232 ± 1.848 | 34.471 ± 2.122 |

The 95% Student-$t$ interval for the 20-seed mean pixel SW2 is
$[0.009303, 0.009685]$. All six original arm scores reproduced pixel and
latent SW2 exactly. The largest absolute difference in paper-compatible
FID across these arms was 0.00014.
[Reproduction and aggregation protocol](ColorMNIST_20_Seeds/README.md).

## 4. Kernel-herding baselines

Let the fixed estimated target embedding be

$$
\widehat\chi=\sum_{i=1}^n v_i k(z_i,\cdot).
$$

After selecting $p$ atoms $s_1,\ldots,s_p$, kernel herding scores a candidate
$z$ by

$$
S_p(z)
=\left\langle\widehat\chi,k(z,\cdot)\right\rangle_{\mathcal H}
-\gamma_p\sum_{j=1}^p k(s_j,z)
=\sum_{i=1}^n v_i k(z_i,z)
-\gamma_p\sum_{j=1}^p k(s_j,z),
$$

and greedily chooses

$$
s_{p+1}\in\operatorname*{arg\,max}_{z\in\mathcal D}S_p(z).
$$

The first term attracts the new atom toward the estimated interventional
embedding; the second repels it from atoms already selected. The feasible set
$\mathcal D$ determines whether this is retrieval or genuine synthesis.

We evaluate two implementations using the same estimated DR embedding as
DR--IMMD Flow:

1. **Finite-candidate herding:** set
   $\mathcal D=\{E(Y_1),\ldots,E(Y_n)\}$ and evaluate $S_p$ at every logged
   latent, so the finite argmax is exact up to numerical ties. Repeated indices
   are allowed. The indexed output returns the corresponding observed image
   $Y_i$ without modification; the decoded output instead returns
   $D(E(Y_i))$. This is a fast retrieval/resampling baseline, not an
   unrestricted image generator. It uses $\gamma_p=1/p$ for $p>0$.
2. **Continuous-domain herding:** set $\mathcal D=\mathbb R^{d_z}$ and
   approximately maximize $S_p$ by multistart Adam. Logged latents are only
   initial points and fallbacks; the returned atom may lie anywhere in the
   unbounded latent domain and is decoded as $D(s_{p+1})$. This is genuine
   synthesis, but each non-convex argmax is only solved approximately. It uses
   the displayed CPME Algorithm-2 convention $\gamma_p=1/(p+1)$.

[Chen et al. (2026)](https://arxiv.org/abs/2505.20754) likewise distinguish
candidate-set herding from full-domain optimization: restricting the candidates
makes the greedy step tractable but cannot recover an atom outside that pool,
whereas unrestricted herding generally requires a non-convex optimization for
every new atom. Their paper studies probability-target quadrature rather than
signed empirical DR targets or image generation, so we use it to motivate these
two baselines, not to claim that kernel herding is uniformly inferior.

ColorMNIST uses the positive-definite IMQ kernel. CelebA uses the practical
Riesz interaction; that score is not covered by the positive-definite-kernel
RKHS guarantee. The same caveat applies to the two-dimensional Riesz row.

### Retrieval and continuous herding compared

The table reports the completed matched continuous-herding campaigns, each of
which reruns all four methods. All times are mean optimizer seconds **per arm**,
excluding shared representation/target preparation, decoding and evaluation.
The ColorMNIST rows therefore come from one common set of encoder fits and
should not be pooled with the earlier finite-candidate-only campaign.

| Dataset | Method | Pixel SW2 | Latent SW2 | Legacy FID | Clean-FID | Optimizer seconds/arm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ColorMNIST | DR--IMMD Flow | 0.009368 ± 0.000375 | 0.080388 ± 0.001536 | 30.714 ± 0.800 | 35.172 ± 0.713 | 17.87 ± 0.03 |
| ColorMNIST | Herding, retrieved pixels | 0.008286 ± 0.000066 | 0.084662 ± 0.001615 | 4.392 ± 0.038 | 4.897 ± 0.094 | 0.44 ± 0.01 |
| ColorMNIST | Herding, decoded bank | 0.009775 ± 0.000171 | 0.084662 ± 0.001615 | 31.611 ± 0.821 | 36.463 ± 0.908 | 0.44 ± 0.01 |
| ColorMNIST | Continuous herding | 0.010125 ± 0.000463 | 0.080939 ± 0.002540 | 35.903 ± 0.616 | 37.007 ± 1.186 | 105.87 ± 2.75 |
| CelebA | DR--IMMD Flow | 0.026562 ± 0.001553 | 0.051289 ± 0.001361 | 11.567 ± 0.085 | 14.528 ± 0.179 | 349.32 ± 0.03 |
| CelebA | Herding, retrieved pixels | 0.027033 ± 0.001601 | 0.052645 ± 0.001261 | 10.907 ± 0.091 | 13.114 ± 0.076 | 2.48 ± 0.01 |
| CelebA | Herding, decoded bank | 0.027142 ± 0.001610 | 0.052645 ± 0.001261 | 11.447 ± 0.081 | 13.700 ± 0.151 | 2.48 ± 0.01 |
| CelebA | Continuous herding | 0.027124 ± 0.001511 | 0.052322 ± 0.001294 | 15.690 ± 0.106 | 20.999 ± 0.049 | 194.35 ± 1.40 |

The low retrieved-image FID benefits from returning original photographs or
digits without reconstruction. After decoding, finite herding remains
competitive. Continuous herding returns 2,000 distinct ColorMNIST and CelebA
atoms per arm. No ColorMNIST atom matches a logged latent at tolerance
$10^{-6}$; approximately 98.9% of the CelebA atoms are off the logged support
at tolerance $10^{-4}$. This establishes numerical off-bank output, not
semantic novelty. Its local search uses eight starts and 50 Adam steps per atom
for ColorMNIST and 25 for CelebA. Continuous optimizer times exclude the
finite-anchor selection already timed in the finite rows. The differing
repulsion conventions above also mean this is not a pure feasible-domain
ablation. On these runs, DR--IMMD has lower decoded SW2 and FID than continuous
herding, while finite retrieval remains highly competitive. This does not
establish a uniform accuracy or runtime ordering.

The two-dimensional benchmark has no encoder or decoder:

| Kernel | DR--IMMD Flow SW2 | Finite herding SW2 |
| --- | ---: | ---: |
| RBF | 0.111993 ± 0.011606 | 0.114597 ± 0.008127 |
| Riesz | 0.126294 ± 0.016524 | 0.110415 ± 0.011188 |

[Code, artifacts and aggregation protocol](Kernel_Herding_Baselines/README.md).

## 5. CelebA hair-colour diagnostic

For the exact seed-0 headline Riesz--AdamW run, we rank all 2,000 particles in
each arm by decoded start-to-final RGB distance. The diagnostic displays the
largest 128 per arm as the decoded initializer, decoded endpoint, and a
common-scale absolute pixel-change heatmap.

**[Open the Riesz--AdamW initial/final/heatmap PDF: both arms, ranks 1--128](CelebA_Hair_Colour_Diagnostic/figures/celeba_ranked_pixel_change_top128_both_arms.pdf)**

The 16-page PDF shows 16 ranked particles per page. Compare the decoded
initializer with the endpoint, then use the shared heatmap scale to locate
the changes. The starting image is decoded through the same VAE, so the
heatmap isolates flow-induced changes from fixed reconstruction error.

| Arm and ranks | Initial label composition | Mean RGB RMSE | Mean luminance change |
| --- | --- | ---: | ---: |
| $a=0$, ranks 1--16 | 16/16 blond | 19.015 | -11.658 |
| $a=0$, ranks 113--128 | 16/16 blond | 15.442 | -9.575 |
| $a=1$, ranks 1--16 | 16/16 non-blond | 18.269 | +7.296 |
| $a=1$, ranks 113--128 | 16/16 non-blond | 14.412 | +5.956 |

All 128 largest arm-0 changes begin blond and all 128 largest arm-1 changes
begin non-blond. This matches the required population correction: arm 0 must
reduce blond mass, whereas arm 1 must increase it. Heatmaps often emphasize
hair and the head outline, but also show face, clothing, background, and global
illumination changes. After fitting a separate spatially constant gain and
offset to each RGB channel, 55.2% and 72.7% of the pooled squared change remains
in the selected arm-0 and arm-1 panels, respectively; 61.8% remains for the
example highlighted in Figure 4. Thus a global brightness/contrast adjustment
explains only part of the change. These are post-hoc seed-0 extremes, not random
samples or evidence of isolated same-person hair editing.
[Full provenance and caveats](CelebA_Hair_Colour_Diagnostic/README.md).

## 6. IMQ--Euler theory-alignment check on CelebA

The completed paired experiment reruns Riesz--AdamW, deterministic IMQ--Euler,
and noisy IMQ--Euler on the same data, DR target, initial
particles, and evaluation draws. It uses 2,000 particles per arm and 4,000
updates. The table reports the three-seed production results, averaging arms
within seed first; times cover optimization only, per arm.

| Sampler | Pixel SW2 | Legacy FID | Clean-FID | Optimizer seconds/arm |
| --- | ---: | ---: | ---: | ---: |
| Riesz--AdamW | 0.02656 ± 0.00155 | 11.57 ± 0.08 | 14.53 ± 0.18 | 349.3 ± 0.1 |
| IMQ--Euler | 0.02649 ± 0.00152 | 11.78 ± 0.08 | 15.02 ± 0.18 | 167.9 ± 0.1 |
| IMQ--Euler + query noise | 0.02649 ± 0.00152 | 11.78 ± 0.08 | 15.02 ± 0.18 | 168.9 ± 0.2 |

IMQ gives comparable SW2 and modestly worse FID. Its step size was frozen
after a seed-0/arm-0 pilot. The noisy update follows the smooth-kernel Euler
form used in the analysis; this does not verify every theorem condition for
the empirical step size and estimated target.

**[Open the IMQ--Euler initial/final/heatmap PDF: both arms, ranks 1--128](CelebA_Theory_Sampler_Ablation/figures/imq_euler_eta0p5/celeba_ranked_pixel_change_top128_both_arms.pdf)**

This is also a seed-0, largest-change diagnostic; its ranks are selected
independently from the Riesz panel, not matched across the two PDFs.
[Experiment status and protocol](CelebA_Theory_Sampler_Ablation/README.md).

## 7. Independently trained ColorMNIST encoder

The experiment partitions the headline total of 20,000 logged
observations into 10,000 encoder-training images and a disjoint 10,000
causal-estimation observations. It retains 2,000 particles per arm, seeds
0--2, and the headline DR, IMQ--Euler and evaluation settings. Propensity,
CME and particle initialization use only the causal half. Pixel SW2 is
`10.050 +/- 0.579` and FID is `29.552 +/- 0.518`, compared with
`9.292 +/- 0.351` and `30.650 +/- 0.788` in the current shared-data headline.
This tests an independent encoder within a fixed total data budget; it also
halves both training sample sizes, so it does not isolate independence from
sample-size effects. [Protocol and reproduction commands](ColorMNIST_Independent_Encoder/README.md).

## 8. Independently trained dSprites encoder

We applied the same fixed-budget split to the revised confounded dSprites
benchmark: 10,000 observations train the encoder, a disjoint 10,000 fit the DR
target, and each of three seeds returns 2,000 particles for the single
stochastic-policy target. Pixel SW2 is `8.827 +/- 0.022` and FID is
`35.321 +/- 1.807`, compared with `8.394 +/- 0.139` and
`36.486 +/- 1.657` in this campaign's independently rerun shared-data control.
The result is comparable after halving both training samples.
[Protocol and full metrics](dSprites_Independent_Encoder/README.md).

## 9. Full-sample versus five-fold nuisance fitting

We also compare the default full-sample DR target with five-fold fitting of the
propensity and conditional mean embedding. Each fold trains the nuisances on
16,000 observations and evaluates them on the held-out 4,000; the five held-out
blocks together retain all 20,000 observations in the DR target. Within each
benchmark, both variants use the same frozen representation, starting
particles, oracle references and sampler settings.

| Dataset | Metric | Full-sample nuisances | Five-fold nuisances |
| --- | --- | ---: | ---: |
| ColorMNIST | Pixel SW2 $\times 10^3$ | 9.292 ± 0.351 | 9.297 ± 0.339 |
| ColorMNIST | FID | 30.650 ± 0.788 | 30.677 ± 0.812 |
| CelebA | Pixel SW2 $\times 10^3$ | 26.562 ± 1.553 | 26.555 ± 1.559 |
| CelebA | FID | 11.567 ± 0.085 | 11.567 ± 0.086 |

The two protocols give nearly identical three-seed accuracy. This supports
five-fold nuisance fitting as a practical alternative when same-sample
empirical-process control is undesirable; it is not a runtime comparison.

## Scope

This report contains the completed rebuttal-ready results. The subdirectory
READMEs provide detailed protocols and reproduction
commands. Exploratory CelebA RFF work is retained locally but excluded
from the public payload because it is not used in this rebuttal summary.

**Why IPW and DR coincide on CelebA.** The cancellation happens in the
estimated target, before particle optimization. Fix an intervention arm $a$.
Let $\Phi_i=k(z_i,\cdot)$ denote the outcome-kernel function attached to
logged latent $z_i=E(Y_i)$, and let $\widehat m(a,x)$ be its fitted conditional
mean term. The same fitted function is used throughout the full-sample DR
estimator. With $w_i^{(a)}=\mathbf 1\{A_i=a\}/\widehat e_a(X_i)$,

$$
\begin{aligned}
\widehat\chi_{\mathrm{IPW}}^{(a)}
&=\frac1n\sum_{i=1}^n w_i^{(a)}\Phi_i,\\
\widehat\chi_{\mathrm{DR}}^{(a)}
&=\frac1n\sum_{i=1}^n
\left[\widehat m(a,X_i)
+w_i^{(a)}\{\Phi_i-\widehat m(A_i,X_i)\}\right].
\end{aligned}
$$

Thus DR equals IPW plus an average fitted CME, minus an inverse-weighted
average fitted CME. We can compare those two averages exactly. Write
$n_x=\#\{i:X_i=x\}$ and $n_{ax}=\#\{i:A_i=a,X_i=x\}$. Both $X$ and $A$
are binary, and all four cells are nonempty. The saturated full-sample
logistic fit recovers the empirical arm probabilities at its optimum:

$$
\widehat e_a(x)=\frac{n_{ax}}{n_x}.
$$

For a fixed cell $(a,x)$, every evaluation of the common fitted CME returns
the same $\widehat m(a,x)$. Also, $w_i^{(a)}=0$ outside arm $a$. Grouping
the two CME averages by $x\in\{0,1\}$ therefore gives

$$
\begin{aligned}
\sum_i\widehat m(a,X_i)
&=\sum_{x\in\{0,1\}}n_x\widehat m(a,x),\\
\sum_iw_i^{(a)}\widehat m(A_i,X_i)
&=\sum_{x\in\{0,1\}}
\frac{n_{ax}}{\widehat e_a(x)}\widehat m(a,x)
=\sum_{x\in\{0,1\}}n_x\widehat m(a,x).
\end{aligned}
$$

They are the same sum, so

$$
\widehat\chi_{\mathrm{DR}}^{(a)}
-\widehat\chi_{\mathrm{IPW}}^{(a)}
=\frac1n\sum_{x\in\{0,1\}}
\underbrace{\left(n_x-\frac{n_{ax}}{\widehat e_a(x)}\right)}_{=\,0}
\widehat m(a,x)=0.
$$

For example, our 20,000-image sample has 14,000 non-blond and 6,000 blond
observations. Arm 1 contains 8,400 non-blond and 600 blond observations, with
fitted probabilities $0.6$ and $0.1$. Their inverse-weighted counts are
$8{,}400/0.6=14{,}000$ and $600/0.1=6{,}000$, exactly matching the two
full-sample stratum counts. Both estimators consequently give the
post-stratified target

$$
\widehat\chi_{\mathrm{DR}}^{(a)}
=\widehat\chi_{\mathrm{IPW}}^{(a)}
=\sum_{x\in\{0,1\}}\frac{n_x}{n}
\left[\frac1{n_{ax}}\sum_{i:A_i=a,\,X_i=x}\Phi_i\right].
$$

This assigns 70% and 30% of the target weight to the two observed arm-specific
cell means; it does not assert that the approximately transported endpoints
have exactly those measured hair-label proportions.

The **delta kernels are on the nuisance inputs $(X,A)$**, not on the image
latents. They make the cell structure explicit, but empirical stratum
balancing causes the cancellation. It holds for any common fitted
$\widehat m(a,x)$, so changing its ridge penalty or using projected Nyström
does not remove it. The plug-in estimator alone can still differ because its
cell means are ridge-shrunk.

Equality is exact at the empirical propensity optimum; finite optimization
and floating-point arithmetic explain the tiny differences in the recorded
scores. It is specific to this full-sample benchmark, not a general
equivalence between IPW and DR. Cross-fitting or corrupted weights need not
preserve the balancing identity.
