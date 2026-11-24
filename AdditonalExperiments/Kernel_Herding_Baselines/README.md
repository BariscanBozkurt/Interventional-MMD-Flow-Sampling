# Kernel-herding baselines

This directory compares DR--IMMD Flow with finite-candidate kernel herding at
the streamlined paper configuration: 20,000 logged observations, 2,000 returned
samples, both intervention arms, and data seeds 0, 1, and 2.

Kernel herding uses exactly the estimated doubly robust target embedding used
by the matched flow,

$$
\widehat\chi=\sum_{i=1}^n v_i k(z_i,\cdot).
$$

Given $p$ previously selected atoms $s_1,\ldots,s_p$, it greedily solves

$$
s_{p+1}\in\operatorname*{arg\,max}_{z\in\mathcal D}
\left\{
\sum_{i=1}^n v_i k(z_i,z)
-\gamma_p\sum_{j=1}^p k(s_j,z)
\right\}.
$$

The target term attracts the next atom toward the fixed DR embedding, while the
second term discourages repeatedly representing regions already covered by the
selected atoms. The two implementations differ only in the feasible set and
how this argmax is solved:

- **Finite-candidate herding** uses
  $\mathcal D=\{E(Y_1),\ldots,E(Y_n)\}$ for image experiments and evaluates the
  score at every logged latent. The selected index can either retrieve the
  original observed image $Y_i$ or be decoded as $D(E(Y_i))$. Repeated indices
  are allowed, so this is retrieval/resampling rather than unrestricted
  synthesis. It uses $\gamma_p=1/p$ for $p>0$.
- **Continuous-domain herding** uses $\mathcal D=\mathbb R^{d_z}$ and
  approximately solves each non-convex argmax with multistart Adam. Logged
  latents provide starts and fallbacks but do not constrain the answer. The
  optimized atom is not restricted to the bank and is decoded, so this is
  genuine latent-space synthesis. It uses $\gamma_p=1/(p+1)$.

This distinction follows the general discussion in
[Chen et al. (2026)](https://arxiv.org/abs/2505.20754): candidate restriction
makes the greedy step tractable but cannot recover atoms outside the pool,
whereas full-domain herding generally requires a non-convex optimization for
each atom. That work concerns probability-target quadrature, so it does not by
itself establish guarantees for our signed DR targets or practical Riesz score.

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

`continuous_latent_herding.py` implements the continuous argmax above. Both
denominator conventions are written to each comparison CSV. Production runs
are unbounded, matching the feasible domain used by DR--IMMD Flow. The old
padded-box option remains available only to reproduce the archived pilot.

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
