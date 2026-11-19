# Additional experiments

New rebuttal experiments live in this directory. The spelling
`AdditonalExperiments` follows the requested directory name.

- [ColorMNIST nuisance misspecification](ColorMNIST_Nuisance_Misspecification/README.md)
  adapts the controlled nuisance-corruption experiment in DoubleGen. Its
  [complete audited report](ColorMNIST_Nuisance_Misspecification/EXPERIMENT_REPORT.md)
  and compact three-seed summaries are included.
- [ColorMNIST cached IMQ Fourier features](ColorMNIST_RFF_Caching/README.md)
  compares exact and approximate interacting flows with a frozen baseline DR target.
- [CelebA hair-colour diagnostic](CelebA_Hair_Colour_Diagnostic/README.md)
  displays the 128 largest decoded particle changes in each arm under the exact
  seed-0 headline configuration and records their provenance and pixel metrics.

Each experiment should keep its configuration, source provenance, results, and
analysis together. Existing `Simulations/` results remain the reference for the
submitted experiments.
