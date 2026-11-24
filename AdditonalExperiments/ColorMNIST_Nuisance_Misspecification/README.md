# ColorMNIST nuisance misspecification

This experiment adapts the controlled nuisance corruption used in the CelebA
study of Luedtke and Fukumizu's
[DoubleGen, Section M.1.1 ("Nuisance estimation")](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1),
whose comparison appears in their Table 3. It tests the expected finite-sample
double-robustness pattern on ColorMNIST; it is not a comparison with the
DoubleGen generator.

## Corruptions

Let $X\in[0,1]$ be the foreground red fraction. The baseline propensity is an
affine logistic model fitted on all 20,000 logged context--action pairs,

$$
\widehat e_1(x)=\operatorname{sigmoid}(\widehat b+\widehat\theta x),
\qquad \widehat e_0(x)=1-\widehat e_1(x),
$$

with arm-specific inverse weights

$$
w_i^{(a)}=\frac{\mathbf 1\{A_i=a\}}{\widehat e_a(X_i)}.
$$

The simulator uses $e_1(x)=\operatorname{sigmoid}\{10(x-0.5)\}$, but the
experiment uses the fitted $\widehat e$ rather than the oracle propensity. No
propensity or inverse-weight clipping is applied.

We vary the two nuisance outputs independently. For the CME, we fit on all
rows or restrict its training support to $X<c$ for
$c\in\{0.5,0.25,0.1\}$. For the weight nuisance, define

$$
h_q(x)=\mathbf 1\{x<0.5\}+q\mathbf 1\{x\geq0.5\},
\qquad
\widetilde w_i^{(a,q)}=h_q(X_i)w_i^{(a)},
$$

for $q\in\{1,1/4,1/20,1/100\}$. The $q=1/4$ setting matches DoubleGen's
factor-of-four perturbation; the remaining values form a severity ladder. We
do not refit the logistic model under a wrong functional form. Instead, we
corrupt its fitted inverse-weight outputs after training. The resulting weights
need not correspond to a valid complementary pair of propensity scores.

The CME restriction affects only its training, ridge-selection, and landmark
rows. All observations remain in the DR residual and policy-integration terms;
the propensity and autoencoder always use all 20,000 observations. Baseline
nuisances are fitted, not oracle quantities.

The realized CME sample counts were:

| CME support | Total rows, seeds 0 / 1 / 2 | Arm 0 rows | Arm 1 rows |
| --- | ---: | ---: | ---: |
| All $X$ | 20,000 / 20,000 / 20,000 | 10,077 / 10,020 / 9,968 | 9,923 / 9,980 / 10,032 |
| $X<0.5$ | 10,076 / 10,077 / 10,035 | 8,760 / 8,644 / 8,636 | 1,316 / 1,433 / 1,399 |
| $X<0.25$ | 5,058 / 5,026 / 4,989 | 4,928 / 4,870 / 4,852 | 130 / 156 / 137 |
| $X<0.1$ | 2,023 / 2,009 / 1,974 | 2,007 / 1,974 / 1,952 | 16 / 35 / 22 |

## Reported configuration

The paper-scale run uses 20,000 logged observations, 2,000 particles per arm,
a total budget of 1,024 projected-Nyström landmarks, full-sample nuisance
fitting, and 3,000 IMQ--Euler updates. Arms are averaged within each data seed,
followed by the mean and sample standard deviation over seeds 0--2. Pixel SW2
is multiplied by $10^3$ below. This table reports the raw weight corruption;
the experiment also records empirical-mass-preserving controls.

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

DR remains close to the baseline when only one nuisance is corrupted and
degrades when both are corrupted. The $X<0.25$ and $X<0.1$ restrictions also
create severe arm imbalance and large between-seed variability, so this is a
controlled finite-sample stress test rather than an oracle specification
theorem.

## Files and reproduction

- `colormnist_doublegen_misspecification.py` implements the original two-by-two
  pilot, including an optional mass-preserving weight-corruption control.
- `colormnist_misspecification_severity.py` implements the paper-scale severity
  grid and the reported baseline/original-corruption comparison.
- `test_misspecification.py` and `test_severity.py` check the corruption,
  target construction, and reuse contracts.
- The two executed analysis notebooks retain aggregate printed outputs. Raw
  result CSVs, model states, particle arrays, logs, and launch records are not
  distributed.

Run the reported grid from the repository root:

```bash
python AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/colormnist_misspecification_severity.py \
  --data-seeds 0 1 2 --arms 0 1 \
  --n-obs 20000 --n-particles 2000 --n-ref 2000 \
  --cme-cutoffs 1 0.5 0.25 0.1 \
  --weight-factors 1 0.25 0.05 0.01 --mass-modes raw preserve \
  --device cuda --mnist-root data
```

Use `--help` for all numerical and output options. Generated outputs are
ignored by Git because they may contain machine provenance.
