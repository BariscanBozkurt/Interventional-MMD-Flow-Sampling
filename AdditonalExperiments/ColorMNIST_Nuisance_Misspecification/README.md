# ColorMNIST nuisance misspecification

This experiment adapts the controlled nuisance corruption used in the
[DoubleGen CelebA study](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1)
to the ColorMNIST interventional-distribution benchmark. It tests the expected
finite-sample double-robustness pattern; it is not a comparison with the
DoubleGen generator.

## Corruptions

Let `X` be the foreground red fraction. We vary the two nuisances
independently:

- **CME corruption:** fit the conditional mean embedding only on rows with
  `X < 0.5` instead of all logged rows;
- **weight corruption:** multiply inverse weights by `1/4` on rows with
  `X >= 0.5`.

The CME restriction affects only nuisance fitting. All observations remain in
the DR residual and policy-integration terms. Baseline nuisances are fitted,
not oracle quantities.

## Reported configuration

The paper-scale run uses 20,000 logged observations, 2,000 particles per arm,
1,024 projected-Nyström landmarks, full-sample nuisance fitting, and 3,000
IMQ--Euler updates. Arms are averaged within each data seed, followed by the
mean and sample standard deviation over seeds 0--2. Pixel SW2 is multiplied by
`10^3` below.

| CME fit | Inverse weights | Plug-in | IPW | DR |
| --- | --- | ---: | ---: | ---: |
| All `X` | Baseline | 9.413 +/- 0.576 | 9.486 +/- 0.350 | 9.402 +/- 0.561 |
| All `X` | Red-heavy x 1/4 | 9.413 +/- 0.576 | 33.420 +/- 0.784 | 9.416 +/- 0.575 |
| Only `X < 0.5` | Baseline | 27.154 +/- 0.895 | 9.486 +/- 0.350 | 9.464 +/- 0.459 |
| Only `X < 0.5` | Red-heavy x 1/4 | 27.154 +/- 0.895 | 33.420 +/- 0.784 | 21.579 +/- 0.690 |

DR remains close to the baseline when only one nuisance is corrupted and
degrades when both are corrupted. CME restriction also reduces nuisance
training support, so the experiment should be interpreted as a controlled
finite-sample stress test rather than an oracle specification theorem.

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
  --cme-cutoffs 1 0.5 --weight-factors 1 0.25 --mass-modes raw \
  --device cuda --mnist-root data
```

Use `--help` for all numerical and output options. Generated outputs are
ignored by Git because they may contain machine provenance.
