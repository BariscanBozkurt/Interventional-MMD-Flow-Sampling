# Kernel-herding baselines

This directory compares DR--IMMD Flow with finite-candidate kernel herding at
the streamlined paper configuration: 20,000 logged observations, 2,000 returned
samples, both intervention arms, and data seeds 0, 1, and 2.

Kernel herding uses exactly the estimated doubly robust target embedding used by
the matched flow. Its candidate set is the complete logged outcome support. It
is therefore a selection method: every returned candidate is an observed latent
or outcome, and repeated selections are allowed.

## Benchmarks and outputs

- **ColorMNIST:** herding is performed over logged autoencoder latents with the
  headline IMQ kernel. We separately score (i) the original logged image indexed
  by each selected latent and (ii) the same selected latent after decoding.
  DR--IMMD is run in the same process, so all three outputs share one trained
  autoencoder and the same target, initial cloud, references, and metric draws.
- **CelebA:** herding is performed over standardized logged SD-VAE latents with
  the headline Riesz kernel. We separately score the indexed original face and
  its SD-VAE decoded counterpart. The matched DR--IMMD output is again produced
  in the same process.
- **Two-dimensional mixture:** there is no encoder or decoder. Herding directly
  selects logged two-dimensional outcomes. RBF and Riesz variants match the two
  outcome kernels used by the paper's particle-flow comparison. RBF uses the DR
  coefficients verbatim. Because the Riesz energy requires equal mass, its DR
  coefficients are explicitly divided by their total mass; both masses are
  recorded in the CSV.

All candidates come from the logged sample rather than an oracle target-arm
subset. CSV files record the unique-selection count, maximum multiplicity,
candidate coverage, selected-arm fraction, sampler time, and distributional
metrics. Image experiments also retain both raw-indexed and decoded metrics so
that retrieval quality is not confused with decoder quality.

## Continuous latent-space herding

`continuous_latent_herding.py` implements the continuous latent argmax in CPME
Algorithm 2 with multistart Adam ascent. The displayed
Algorithm-2 convention divides the repulsion from the previous `p` atoms by
`p+1`; the finite-candidate baseline follows the appendix/released-code
convention and divides by `p`. Both conventions are written to each comparison
CSV. The logged latent bank supplies only starting points and numerical
fallbacks. Production continuous-herding runs are unbounded: returned atoms can
move anywhere in the learned latent space, matching the feasible domain used by
DR--IMMD Flow. The old padded-box option remains available only to reproduce the
archived pilot.

The continuous ColorMNIST and CelebA runners produce DR--IMMD Flow,
finite-candidate herding, and continuous herding in one process. They record
SW2/FID, exact unique counts, distance to the nearest logged latent,
encoder--decoder cycle error, and improvement of the continuous score over both
its dynamic candidate seed and the matched finite-herding anchor. The full
atoms and optimizer diagnostics are also saved in an NPZ artifact. Each arm
also saves 16 examples selected without replacement by a fixed seed. The PNG
rows are continuous herding, decoded finite herding, and the oracle reference;
an adjacent NPZ retains the exact images and selection indices. CelebA uses
the same standardized 256-pixel frozen SD-VAE coordinates, no-split projected
Nyström DR target with 1,024 landmarks, and reference draws as the matched
headline runner.

The inner maximizations are multistart numerical approximations, not certified
global solutions. ColorMNIST uses the positive-definite IMQ kernel from the
paper configuration. CelebA uses the same practical Riesz interaction as its
matched flow; raw Riesz is not covered by the positive-definite-kernel
convergence result for CPME Algorithm 2. For the implemented unit-mass CelebA
target and the displayed `p+1` denominator, however, the unbounded Riesz score
is coercive: with `p` previous atoms it equals
`-||z||/(p+1) + O(1)` at infinity. The optimizer therefore rejects an unbounded
Riesz run under the alternate `p` denominator, for which this guarantee is not
available.

## Running

Run the desired Python entry point directly, or adapt the command to your own
scheduler. Each image runner accepts a data seed and evaluates both arms. CelebA
additionally requires a user-supplied dataset root.

```bash
python AdditonalExperiments/Kernel_Herding_Baselines/colormnist_kernel_herding.py --help
python AdditonalExperiments/Kernel_Herding_Baselines/colormnist_continuous_latent_herding.py --help
python AdditonalExperiments/Kernel_Herding_Baselines/celeba_kernel_herding.py --help
python AdditonalExperiments/Kernel_Herding_Baselines/celeba_continuous_kernel_herding.py --help
python AdditonalExperiments/Kernel_Herding_Baselines/decfm2d_kernel_herding.py --help
```

`analyze_kernel_herding.py` performs the arm-then-seed aggregation. Generated
CSVs, particle arrays, decoded examples, scheduler launchers, logs, and caches
are excluded from the public repository.
