# ColorMNIST: DoubleGen-style nuisance misspecification

## Experimental question and source

Does the DR target retain accuracy when either the inverse-weight nuisance or
the conditional outcome embedding is deliberately corrupted, and does that
accuracy survive the fixed particle sampler and decoder?

The primary precedent is Luedtke and Fukumizu, **DoubleGen: Debiased Generative
Modeling of Counterfactuals** (2025), Appendix M.1.1 and Table 3
([paper](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1)). In their CelebA
experiment, the corrupted outcome nuisance is trained only on dark-haired
instances; inverse weights for lighter-haired instances are multiplied by 1/4.
They cross the two interventions and compare DR, plug-in, and IPW methods.

The four-configuration design also follows the classical simulation framework
of Bang and Robins (2005), Section 2.3, Tables 1–2
([paper](https://doi.org/10.1111/j.1541-0420.2005.00377.x)).
Kang and Schafer (2007) provide a complementary example using functional-form
misspecification and explain why finite-sample DR performance can still be poor
([paper](https://doi.org/10.1214/07-STS227)). Bibliographic records are in
`references.bib`.

This is an adaptation of DoubleGen's nuisance-corruption protocol to our
ColorMNIST estimator and sampler. It does not run the DoubleGen generative
model, reproduce its CelebA benchmark, or supply a comparison against DoubleGen.
It targets reviewer fBWo Q3 and VaZw W2 on practical double robustness. Reviewer
XP5B W3's request for DoubleGen as an additional generative baseline remains a
separate experiment.

## Prespecified ColorMNIST adaptation

Keep the existing DGP: `w_conf=10`, `alpha=0.2`, `tau=0.08`, `k_mask=20`,
digits 1 and 6. The covariate `X` is the foreground red fraction; the target
colour distribution is Uniform[0,1] for either intervention arm.

Define the red-heavy subgroup as `X >= 0.5` before examining outcomes.

| Nuisance | Baseline | Corrupted |
| --- | --- | --- |
| Inverse weights | Logistic propensity using continuous X | Multiply the arm-specific inverse weights by 1/4 for X >= 0.5 |
| Conditional embedding | Fit on all eligible nuisance-training rows | Fit only on eligible training rows with X < 0.5 |

Run all four combinations with plug-in, IPW, and DR targets. Baseline means the
usual fitted nuisance, not an oracle-correct finite-sample fit. The corrupted
inverse weights are deliberately distorted estimates; they need not be the
reciprocals of a valid fitted propensity model.

The restriction applies only to CME fitting. All logged observations remain
eligible for the held-out DR residual and target-policy integration. Do not
remove red-heavy observations from the full estimation sample, change the
target intervention, or filter the reference distribution.

## Fixed controls and target construction

The default experiment uses two shared nuisance folds. Each observation's
propensity and CME predictions use the other fold; restricted CME training
intersects that training fold with `X < 0.5`. An optional no-split setting
provides a separate full-sample comparison.

Train one autoencoder on a separately generated logged sample and freeze it.
Share the encoder, logged data, target support, evaluation draws, initial
particles, outcome kernel, and optimizer settings across configurations.
The context bandwidth is fixed using the independent AE sample. Uniform
landmarks and ridge selection depend only on each eligible nuisance-training
block; propensity corruption must not trigger a different CME fit. Ridge CV
uses the first latent coordinate as in the deployed implementation, which is
a tuning proxy rather than a full conditional-distribution loss.

For a fold with training rows T and held-out rows H, accumulate

    v_plugin[T] += fitted_policy_integral_coefficients(T, H) / n
    v_DR[T]     += (policy_integral_coefficients - fitted_residual_coefficients) / n
    v_DR[H]     += w[H] / n
    v_IPW       = w / n

Here n is the complete causal sample size, including observations excluded
from the restricted CME fit. The projected Nyström solve uses the existing
whitened implementation. Matrix ridge is supplied directly; the experiment
does not use the reviewed `lambda_n_half` convenience path.

## Total mass is a separate control

With ideal weights, the raw subgroup corruption changes expected IPW mass
from 1 to 0.625. Multiplying all weights by a positive constant leaves the
minimizers of DoubleGen's IPW risk unchanged (although finite optimization can
still change). Matching an MMD target is sensitive to its total mass, so this
distinction must remain visible.

There is also a known shape distortion: after unit-mass rescaling, the ideal
corrupted IPW colour law has red-heavy probability 0.2 and colour mean 0.35,
compared with 0.5 and 0.5 under the intended target. These values follow by
integrating weights 1 on [0,0.5) and 1/4 on [0.5,1]; they are analytical
controls, not measured experiment results.

- `--corruption-mass raw` applies the original factor-of-four corruption.
- `--corruption-mass preserve` additionally multiplies the corrupted weights
  by `sum(baseline_weights) / sum(corrupted_weights)`. This explicitly declared
  secondary control preserves baseline empirical mass while retaining the
  fourfold relative distortion between subgroups.

The secondary control uses the evaluation sample to calibrate its total mass;
it is a finite-sample sensitivity analysis, not a literal independent-nuisance
theorem experiment. It does not force mass one. Neither mode normalizes or
clips the final plug-in/DR coefficient vectors. Report both mass and shape
diagnostics; do not describe a mass-driven failure as purely a colour error.

## Evaluation and interpretation

Before flow, report coefficient mass, absolute/negative mass, and target
embedding error against independent oracle samples. Weighted colour mean
and second moment have reference values 1/2 and 1/3; they diagnose selected
functionals, not the entire image distribution.

After flow, report pixel and latent SW2 and the foreground-colour W1 distance
to Uniform[0,1]. Include the actual decoded initial particles, an independent
count-matched oracle floor, and a decoder-roundtrip oracle control. All
configurations use the same samples and evaluation randomness. FID is omitted
from this initial experiment to keep the pilot focused and inexpensive.

One seed is a pipeline and mechanism pilot. Rebuttal claims require multiple
independent logged-data seeds. Average intervention arms within each seed
before reporting across-seed means and sample standard deviations. Reused
plug-in/IPW targets across configurations are not additional repetitions.

A successful robustness pattern is DR retaining target accuracy when one
nuisance remains adequate, followed by corresponding sample accuracy. DR is
not guaranteed to win every finite-sample metric. The both-corrupted setting
has no generic consistency guarantee. Fixed landmark rank, ridge CV, and
finite flow iterations remain practical approximation choices.

## Running and validation

The runner is `colormnist_doublegen_misspecification.py`; use `--help` for its
configuration. It uses the existing `torchnn` environment and local MNIST
assets. Outputs go into a fresh experiment-local run directory, with a JSON
configuration/provenance record, CSV metrics, and NumPy artifacts. Existing
run directories are protected from accidental overwrite.

From the deploy repository, the one-seed pilot command is:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -u -B \
  AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/colormnist_doublegen_misspecification.py \
  --device cuda --data-seeds 0 --tag doublegen_raw_pilot
```

Defaults are 5,000 causal observations, 5,000 independent AE-training images,
1,000 particles and reference images per arm, 512 landmarks per training block,
3,000 AE updates, and 3,000 IMQ Euler steps per unique target. Both arms are run.
Each arm has eight unique targets/flows; twelve CSV rows retain the complete
2-by-2-by-3 table. `shared_flow_id` identifies reused plug-in/IPW results.

For the declared mass control, add `--corruption-mass preserve` and choose a
fresh tag. For a larger confirmation experiment, use `--data-seeds 0 1 2
--n-obs 20000 --n-ae 20000 --n-particles 2000 --n-ref 2000 --n-landmarks 1024`
and a fresh tag. These are proposed follow-ups, not executed results.

The focused tests in `test_misspecification.py` check the nuisance restrictions,
raw-weight and mass-control behavior, and agreement with a direct fold-wise
KRR pseudo-outcome construction:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B -m unittest discover \
  -s AdditonalExperiments/ColorMNIST_Nuisance_Misspecification \
  -p 'test_misspecification.py' -v
```

Validation on 2026-11-19: all nine tests passed. A CPU structural smoke run
(`structural_smoke_seed0`) completed all 24 rows with n=120, M=12, two AE/flow
updates, 24 landmarks, fixed matrix ridge 0.1, and 16 projections. That run
checks execution and artifacts only; it is not evidence of statistical accuracy.

## Executed one-seed pilot (2026-11-19)

The default raw-corruption pilot completed on the local RTX 4500 Ada in 275.7
seconds. It produced 24 configuration rows and 16 unique flows. Artifacts are
retained locally in `results/doublegen_raw_pilot_seed0/`; that raw pilot
directory is not part of the compact repository package. The versioned analysis notebook
`ColorMNIST_DoubleGen_Nuisance_Analysis.ipynb` checks provenance and aggregates
the two arms equally within seed 0. Its five code cells executed sequentially
with zero errors and retain the displayed tables and figure as embedded output.

Final pixel SW2 (lower is better; one seed, mean over arms):

| CME | Inverse weights | Plug-in | IPW | DR |
| --- | --- | ---: | ---: | ---: |
| Baseline | Baseline | 0.015561 | 0.012407 | 0.012517 |
| Baseline | Red subgroup × 1/4 | 0.015561 | 0.033401 | 0.013576 |
| Blue-only | Baseline | 0.059001 | 0.012407 | 0.013433 |
| Blue-only | Red subgroup × 1/4 | 0.059001 | 0.033401 | 0.048594 |

The decoded initialization scores 0.043337; the independent count-matched
oracle comparison scores 0.008964 and its decoder-roundtrip comparison scores
0.010833. These comparisons are not mathematical lower bounds.

Before flow, DR target RBF squared error is 0.001247 at baseline, 0.002053 under
CME-only corruption, 0.002722 under weight-only corruption, and 0.269739 with
both corrupted. This supports the intended mechanism in this pilot. It does not
establish robustness across seeds or oracle correctness of either fitted nuisance.
The raw corrupted IPW mass averages 0.625256. The mass-preserving control and
multi-seed confirmation have not been executed. No manuscript results were changed.

## Paper-configuration severity sweep

`colormnist_misspecification_severity.py` extends the controlled corruption
design without changing the pilot runner or its results. The primary sweep
uses data seeds 0, 1, and 2, 20,000 logged observations, and 2,000 particles per
intervention arm. One pipeline repeat is run per seed, as in the paper.

The following are two independently varied axes; run their full factorial,
not only the diagonal of this table:

| Level | CME training rows | Weight multiplier for the fixed subgroup X >= 0.5 |
| --- | --- | ---: |
| Baseline | All logged rows | 1 |
| Original corruption | X < 0.5 | 0.25 |
| Strong | X < 0.25 | 0.05 |
| Extreme | X < 0.1 | 0.01 |

The inverse-weight subgroup threshold stays at 0.5 throughout. Restricting the
CME never removes observations from the DR residual, target integration, or
oracle distribution. The extreme CME has only about 23 arm-1 training rows in
expectation at n=20,000; actual training and landmark counts are recorded, and
missing arm support is an error. These restrictions stress nuisance fitting;
they do not inject arbitrary noise into the final coefficient vector or break
the linear solver. Realized errors need not increase monotonically with severity.

Each nonbaseline weight level is evaluated both raw and after the explicitly
declared rescaling that preserves baseline empirical weight mass. Baseline
weights are shared between the two controls. Thus each arm has four CME fits
and seven distinct weight settings. Plug-in and IPW controls are reused across
irrelevant axes. There are 84 configuration rows but 39 unique flows per arm:
168 rows / 78 flows per seed and 504 rows / 234 flows across the campaign.

The numerical settings match the executed paper configuration: AE fitted on
the same full logged sample, full-sample nuisance fits, projected Nyström with
1,024 uniform landmarks, direct Gram ridge selected by three-fold CV on at most
2,000 eligible training rows using the first latent coordinate, latent dimension
32, 3,000 AE updates, and 3,000 deterministic IMQ Euler steps with step size
`0.15*c_imq**2`. Initialization uses distinct observed target-arm rows. Both
oracle draws have 2,000 images; SW2 uses 1,000 projections. Context bandwidth
uses up to 4,000 random logged contexts; IMQ scale uses up to 5,000 logged
latents. FID is omitted; pixel SW2 is the paper's headline metric.

This differs from the pilot's cross-fitting and independent AE sample. Compare
severity levels within this sweep; a pilot-to-sweep difference is not solely a
sample-size effect. The baseline CME is a fitted nuisance, not an oracle. With
full-sample fitting, a nearly interpolating CME can make the DR correction
small; the correction L1 diagnostic helps expose this. Outside a restricted
CME training support, the DR coefficient remains exactly `w_i/n`.

Run directories are `results/severity_paper_seed{0,1,2}`. Target coefficients
and nuisance diagnostics are saved before sampling, and CSV/progress records
are checkpointed as unique flows finish. Existing run directories cannot be
overwritten. Keep incomplete outputs separate from completed seed summaries.

Validation before submission: ten new severity tests passed; a CPU structural
run completed 46 short flows / 84 rows. Full-grid target construction at the
20k/2k dimensions passed on CUDA for all three seeds with two AE updates and no
flows (`severity_scale_smoke_seed{0,1,2}`). The extreme CME restriction retained
16, 35, and 22 arm-1 training observations, with 8, 18, and 13 arm-1 landmarks,
respectively. These reduced-training runs check execution only.

### Completion and analysis (2026-11-19)

All three NVIDIA L40S simulation tasks completed successfully in approximately
23 minutes 47 seconds each, producing 504 configuration rows and 234 unique
flows. The prepared notebook was subsequently executed locally.

`ColorMNIST_Nuisance_Severity_Analysis.ipynb` now contains five sequential
executed code cells, zero errors, embedded tables and three figures. Summaries
are in `results/severity_paper_summary/`; `cme_training_counts.csv` records
actual eligible training rows, arm counts, landmark counts and selected ridge.
Source/configuration checks pass. An independent recomputation confirms all
84 configurations' means and sample SDs. The plot limits include the full
upper error bars for the severe cases.

The empirical pattern is consistent with DR protection under a single nuisance
corruption, with substantial degradation when both are corrupted. Review
individual seeds before interpreting the strongest CME restrictions: tiny
CV-selected ridge values and large signed coefficient mass produce plug-in
outliers. CME restriction changes both training support and sample size, and
the resulting errors are not monotone across cutoffs. The mass-preserving
control preserves inverse-weight mass, not final DR target mass. These are
three-seed finite-sample findings with fitted baseline nuisances.

The separate acceleration proposal and implementation are documented in
[`../ColorMNIST_RFF_Caching/README.md`](../ColorMNIST_RFF_Caching/README.md).

The complete protocol and audited results are documented in the standalone
[experiment report](EXPERIMENT_REPORT.md),
including DGP equations, exact nuisance corruption, training/CV/landmark counts,
all severity tables, individual-seed instability, and reproduction details.
