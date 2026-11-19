# ColorMNIST nuisance-misspecification experiment: complete protocol and results

**Recorded:** 2026-11-19  
**Status:** Completed and audited three-seed experiment; research record for later rebuttal preparation. This is not submitted rebuttal or manuscript text.  
**Primary study:** 20,000 logged observations, 2,000 generated samples per intervention arm, three independent data seeds, one pipeline repeat per seed.  
**Experiment directory:** [`AdditonalExperiments/ColorMNIST_Nuisance_Misspecification`](./) in the deploy repository. The spelling “AdditonalExperiments” is intentional and matches the existing directory.

**Reading guide:** [DGP](#3-data-generating-process-and-causal-target), [corruption definitions](#5-exactly-what-is-corrupted), [training counts](#6-exact-training-counts-and-nuisance-tuning), [target equations](#7-estimated-targets-equations-and-numerical-implementation), [primary results](#10-primary-results-pixel-sw2), [auxiliary metrics](#11-auxiliary-results), [unstable seeds](#12-seed-specific-instability-and-signed-target-diagnostics), and [reproduction and rebuttal caveats](#16-reproduction-commands-and-limitations-for-rebuttal-use).

## 1. Purpose, reviewer connection, and scope

The question is whether an interventional MMD sampler retains sample quality when one of its two nuisance estimates is deliberately corrupted, and how performance changes as those corruptions become more severe. We compare three estimated targets under the **same latent particle sampler**:

- **Plug-in:** uses the fitted conditional mean embedding (CME).
- **IPW:** uses fitted inverse propensity weights.
- **DR:** combines the CME with the inverse-weighted residual correction.

This directly addresses reviewer fBWo Q3 and reviewer VaZw W2. It provides a controlled empirical stress test of the expected double-robustness pattern. It does not resolve VaZw's same-sample empirical-process concern, add repetitions beyond three data seeds, or implement the DoubleGen generative baseline requested by reviewer XP5B.

The principal finding is that DR sample error remains near baseline when either nuisance alone is corrupted. When both are corrupted, error increases. Severe CME restrictions also produce large between-seed variability, associated with sparse arm support, small selected ridge values, and large signed target coefficients. These qualifications are part of the finding.

The study is separate from the [kernel-feature caching and small-batch acceleration study](../ColorMNIST_RFF_Caching/README.md). No acceleration experiment was launched as part of this work.

## 2. Literature precedent and what “misspecification” means here

Luedtke and Fukumizu's **DoubleGen: Debiased Generative Modeling of Counterfactuals** uses a closely related corruption design: its CelebA outcome nuisance is trained only on dark-haired instances, while inverse propensity weights for lighter-haired instances are divided by four. The two corruptions are crossed and plug-in, IPW, and DR methods compared. We adapt that design to ColorMNIST's observed color covariate. [DoubleGen, Appendix M.1.1 and Table 3](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1).

Our adaptation replaces the hair-color subgroup by the prespecified subgroup $X\geq 0.5$, and extends the original factor-of-four perturbation to a severity ladder. We also add a mass-preserving control because embedding matching depends on the scale of the target coefficients.

**Precise terminology:**

1. The weight experiment corrupts the **outputs of a fitted inverse-weight nuisance**. It does not refit a logistic model with an intentionally wrong formula.
2. The CME experiment restricts **where the nuisance is trained**. It uses the same kernel family and fitting algorithm on a smaller, biased training support. It combines extrapolation, fewer observations, and arm imbalance.
3. “Baseline” means the normal fitted nuisance. It does not mean an oracle conditional distribution or an exact finite-sample propensity.
4. “Misspecification” is therefore used operationally as deliberate nuisance corruption/support restriction. This experiment does not establish that a particular nuisance converges to a specified wrong population limit.

Classical background references retained with the experiment are Bang and Robins (2005), *Doubly Robust Estimation in Missing Data and Causal Inference Models*, and Kang and Schafer (2007), *Demystifying Double Robustness*. Their bibliographic records are linked in Section 17; the specific subgroup-corruption recipe here is attributed to DoubleGen.

## 3. Data-generating process and causal target

### 3.1 Variables and treatment assignment

For each logged observation, draw

$$
X_i\sim \mathrm{Uniform}[0,1],\qquad
e_1(x)=\operatorname{sigmoid}\{10(x-0.5)\},\qquad
A_i\mid X_i=x\sim\operatorname{Bernoulli}\{e_1(x)\}.
$$

Here $X$ controls foreground color and is observed by the nuisance estimators. Treatment determines digit identity:

- $A=0$: MNIST digit **1**.
- $A=1$: MNIST digit **6**.

Thus $X$ affects both treatment assignment and image color. Logged digit-1 images tend to be blue-heavy, and digit-6 images tend to be red-heavy. The true propensity is known from the simulator but is **not substituted for the fitted weights** in the reported study. The propensity is strictly between zero and one; this does not prevent finite-sample arm scarcity near the ends of the color range.

### 3.2 Image rendering

Draw a grayscale image $S_i\in[0,1]^{28\times28}$ uniformly with replacement from the MNIST training pool for the selected digit. The local pools contain 6,742 images of digit 1 and 5,918 images of digit 6. Define the soft foreground mask and RGB image by

$$
B_i(u,v)=\operatorname{sigmoid}\{20(S_i(u,v)-0.08)\},
$$

$$
Y_i(u,v)=B_i(u,v)S_i(u,v)\,
       \bigl(X_i,\;0.2,\;1-X_i\bigr).
$$

The image has three channels and size $28\times28$. The parameters are

$$
(w_{\mathrm{conf}},\kappa,\tau,\alpha)=(10,20,0.08,0.2).
$$

Here $\kappa$ is the mask sharpness, $\tau$ its grayscale threshold, and $\alpha$ the green-channel multiplier. In code these are named w_conf, k_mask, tau, and alpha. The background is black, up to the soft-mask construction; no additive observation noise is introduced.

Red and blue intensities share the same foreground template. Consequently the image statistic

$$
C(Y)=\frac{\sum_{u,v}Y_R(u,v)}
           {\sum_{u,v}Y_R(u,v)+\sum_{u,v}Y_B(u,v)+10^{-6}}
$$

recovers $X$ up to the denominator stabilizer for simulator images. Decoded images need not obey the exact rendering formula.

### 3.3 Interventions and oracle references

We target both static interventions $do(A=0)$ and $do(A=1)$. To draw an oracle image under $do(A=a)$:

1. draw a fresh $X\sim\mathrm{Uniform}[0,1]$;
2. fix the digit to 1 if $a=0$, or 6 if $a=1$;
3. draw a fresh grayscale template from that class pool;
4. apply the same rendering formula.

The target foreground-color law is therefore Uniform[0,1] for either arm, with first and second moments $1/2$ and $1/3$. The intervention does not condition on the logged arm's color distribution.

Oracle references are independent simulator draws from the same finite MNIST training banks. They are **not held-out MNIST test-set images**, and template reuse is possible. Independence refers to fresh simulator draws, not disjoint image-template identities.

## 4. Completed study and fixed controls

| Component | Executed configuration |
| --- | --- |
| Logged observations per seed | 20,000 |
| Static target arms | 0 (digit 1), 1 (digit 6) |
| Data seeds / pipeline repeats | 0, 1, 2 / one repeat per seed |
| Generated particles per arm and unique target | 2,000 |
| Oracle images | Two draws of 2,000 per arm |
| AE training observations / latent dimension | 20,000 same-sample outcomes / 32 |
| AE updates / batch size / optimizer | 3,000 / 128 / Adam, learning rate 0.001 |
| Propensity training | All 20,000 rows; affine logistic; 400 Adam updates, learning rate 0.1 |
| Final nuisance fitting | Full-sample; no cross-fitting |
| CME solver / total landmark budget | Projected Nyström / 1,024 |
| Ridge tuning | Three folds, up to 2,000 eligible rows; first latent coordinate |
| Flow | IMQ, exponent −1/2; deterministic Euler; 3,000 updates |
| Euler step | 0.15 times squared arm-specific IMQ scale |
| Initialization | 2,000 observed target-arm indices, without replacement |
| Primary metric | Pixel SW2; float32; 1,000 projections, seed 0 |
| Weight corruption subgroup | X ≥ 0.5, unchanged across CME cutoffs |
| CME cutoffs / weight multipliers | 1, 0.5, 0.25, 0.1 / 1, 1/4, 1/20, 1/100 |
| Nonbaseline weight controls | Raw and empirical-mass preserving |
| Total records / unique flows | 504 / 234 |

For each data seed, one logged dataset, one fitted autoencoder, and one fitted propensity model are shared by the entire severity grid. Within each arm, all methods and corruption settings share the initial particles, oracle references, kernel scale, flow step size, iteration budget, and evaluation projections.

The data seeds are 0, 1, and 2. The corresponding pipeline seeds are 0, 1,000, and 2,000. Different data seeds also change the representation and pipeline randomness; their standard deviations do not isolate only sampling variability in the logged dataset.

**The primary study uses no nuisance cross-fitting and no independent AE sample.** Three-fold CV is used to select a ridge parameter, which is a separate operation from cross-fitting the final nuisance estimator. All-data nuisance fitting was retained to match the deployed paper configuration.

### 4.1 Autoencoder and representation

Images are mapped from $[0,1]$ to $[-1,1]$ before AE training. A convolutional autoencoder with 32-dimensional latent code is trained on all 20,000 logged images for 3,000 minibatch updates, using batch size 128 and Adam learning rate $10^{-3}$. The encoder and decoder are then frozen and reused for every corruption and estimator within that seed.

The encoder uses convolutions with channels $3\to32\to64\to128$ and spatial sizes $28\to14\to7\to4$. The first two convolutions have kernel/stride/padding $(4,2,1)$; the third uses $(3,2,1)$. Each is followed by GroupNorm with eight groups and SiLU. A linear layer maps $128\times4\times4=2,048$ features to the 32-dimensional code.

The decoder maps 32 back to 2,048, reshapes to $128\times4\times4$, and uses transposed convolutions $128\to64\to32\to3$ with spatial sizes $4\to7\to14\to28$. Kernel/stride/padding are $(3,2,1)$ with output padding zero, then $(4,2,1)$ twice. Hidden stages use GroupNorm and SiLU; the output uses tanh. Training minimizes pixel reconstruction MSE. Minibatch indices are uniform draws with replacement. There is no denoising augmentation or held-out early stopping. Encoding is performed in batches of 512.

Let $Z_i=E(Y_i)$ denote the resulting logged latent support. Nuisance linear algebra uses float64; the existing AE and particle-flow path use float32. Target coefficients are constructed and archived in float64 and cast to the latent dtype when passed to the sampler. Decoded outputs are mapped back to $[0,1]$ for image evaluation.

The reported reconstruction MSE is measured on the first 2,000 logged model-range images. It is an in-sample diagnostic, not validation error.

### 4.2 Baseline propensity and inverse weights

Fit a scalar logistic regression with intercept,

$$
\widehat e_1(x)=\operatorname{sigmoid}(\widehat b+\widehat\theta x),
\qquad \widehat e_0(x)=1-\widehat e_1(x).
$$

The fit uses all 20,000 observed pairs $(X_i,A_i)$, float64 CPU tensors, zero initialization, and 400 full-batch Adam updates at learning rate 0.1. The logistic family contains the simulator propensity, but the fitted values are not oracle values.

For target arm $a$ define

$$
w_i^{(a)}=\frac{\mathbf 1\{A_i=a\}}{\widehat e_a(X_i)}.
$$

Weights are zero outside the selected treatment arm. There is no propensity clipping or weight clipping. Nonfinite or boundary-invalid predictions trigger an error.

### 4.3 CME kernel, fitting, landmarks, and ridge selection

The input kernel is the product of an RBF context kernel and a delta treatment kernel:

$$
K\bigl((x,a),(x',a')\bigr)=k_X(x,x')\mathbf 1\{a=a'\}.
$$

The context length scale is fixed once per seed using up to 4,000 uniformly selected logged contexts with index-selection seed 0. The same length scale is used for every CME restriction and weight setting. This bandwidth uses the full logged context distribution; the restriction affects the fitted nuisance's training rows, CV rows, and landmarks.

The RBF convention is
$$
k_X(x,x')=\exp\!\left[-\frac{\|x-x'\|^2}{2\ell_X^2}\right],
\qquad
\ell_X=\sqrt{\frac12\operatorname{median}_{i<j}\|X_i-X_j\|^2}
$$
on the selected contexts. The diagnostic latent RBF uses the same convention. Distances are computed from squared norms and matrix products, with negative numerical roundoff clamped to zero.

Within each eligible training support:

- select 1,024 uniform landmarks without replacement, using the pipeline seed;
- select up to 2,000 eligible rows for ridge CV, using seed 0;
- select the unnormalized Gram-matrix ridge from
  $\{10,1,0.1,0.01,0.001,0.0001\}$;
- use three-fold CV, seed 0, squared prediction error for the **first latent coordinate** as the tuning proxy;
- refit the projected Nyström estimator using all eligible training rows and the selected ridge.

The first latent coordinate is a tuning proxy, not a full conditional-distribution loss. CV is not stratified by treatment. The same fit is reused for all inverse-weight perturbations. The stored arm-0 and arm-1 fit metadata agree: the product kernel, landmarks, and ridge choice are shared, while the target intervention enters the right-hand side.

The CV permutation is split into three interlaced folds, taking positions $f,f+3,f+6,\ldots$ for fold $f$. The criterion averages the three held-out fold MSEs equally. CV uses dense KRR on its selected subset; the final fit uses projected Nyström on the full eligible support.

The ridge $\rho$ is inserted directly into the matrix system. It is not silently multiplied or divided by the eligible training size. If compared with a normalized KRR convention, the corresponding $\lambda$ is $\rho/|T|$. Selecting $\rho$ on a CV subset and using it directly for the larger fit is a practical convention of this study, not an asymptotic ridge schedule.

### 4.4 End-to-end sequence for each seed

1. Generate 20,000 logged triples from the fixed DGP.
2. Train the AE on those outcomes, freeze it, and encode the full sample.
3. Fit the baseline propensity on all logged context/action pairs.
4. Fix the context and diagnostic RBF bandwidths and save the logged latent support.
5. For each arm, construct baseline inverse weights and the six corrupted variants.
6. For each CME cutoff, select eligible rows, landmarks, and ridge; solve for plug-in and every DR correction using all 20,000 integration/residual rows.
7. Save nuisance indices, coefficients, and weight vectors before sampling.
8. Select one initial particle cloud, fix the arm's IMQ scale, and generate two oracle references.
9. Evaluate the estimated targets and shared initialization/oracle controls.
10. Run and save each unique 3,000-step particle flow; decode and evaluate its final cloud.
11. Reuse identical plug-in/IPW flows across irrelevant factorial axes and checkpoint rows/progress after each unique flow.
12. After all seeds finish, verify provenance and complete grids, average arms within seed, and calculate across-seed means and sample SDs.

## 5. Exactly what is corrupted

### 5.1 CME training-support restriction

For cutoff $c$, define

$$
T_c=
\begin{cases}
\{1,\ldots,n\}, & c=1,\\
\{i:X_i<c\}, & c\in\{0.5,0.25,0.1\}.
\end{cases}
$$

The four settings are:

| Label | Eligible CME training rows | Intended stress |
|---|---|---|
| Full | All logged rows | Baseline fitted CME |
| Original corruption | $X<0.5$ | Exclude the red-heavy half |
| Strong restriction | $X<0.25$ | Train on the bluest quarter |
| Extreme restriction | $X<0.1$ | Train on the bluest tenth |

Training rows, landmarks, and ridge-CV rows all obey the selected restriction. The same input kernel family and solver are retained. No random noise is added to the final coefficient vector, and the linear solver is not deliberately damaged.

**What is not restricted:** all $n=20,000$ logged contexts remain in the empirical target-policy integration, all observations remain available to the DR residual correction, the AE and propensity fits retain all 20,000 observations, and the oracle distribution remains unchanged. The target is always the full-population intervention, never the intervention restricted to $X<c$.

### 5.2 Inverse-weight corruption

Independently of the CME cutoff, keep the corruption subgroup fixed at $X\geq0.5$. For multiplier $q$, define

$$
h_q(x)=\mathbf 1\{x<0.5\}+q\mathbf 1\{x\geq0.5\},\qquad
w_{i,\mathrm{raw}}^{(a,q)}=h_q(X_i)w_i^{(a)},
$$

with $q\in\{1,1/4,1/20,1/100\}$.

The baseline is $q=1$. The other settings reduce red-heavy inverse weights by factors of 4, 20, and 100. No observations are removed from the propensity training data. This perturbation is applied **after** fitting the baseline propensity.

The corruption boundary does not move to 0.25 or 0.1 when the CME restriction changes. For example, at $c=0.1$, observations with $0.1\leq X<0.5$ are excluded from CME fitting but retain baseline inverse weights.

These corrupted nuisance outputs need not be the reciprocals of a valid complementary pair of treatment probabilities. They must not be described as a newly fitted propensity model.

### 5.3 Explicit empirical-mass-preserving control

For each nonbaseline $q$, we also construct, separately within each seed and arm,

$$
s_{a,q}=\frac{\sum_i w_i^{(a)}}{\sum_i w_{i,\mathrm{raw}}^{(a,q)}},
\qquad
w_{i,\mathrm{preserve}}^{(a,q)}
=s_{a,q}w_{i,\mathrm{raw}}^{(a,q)}.
$$

This preserves the **baseline fitted empirical weight mass**, not mass one and not necessarily mass $n$ before dividing by $n$. It leaves the relative red/blue weight distortion equal to $q$. It is an explicit sensitivity control calibrated on the full observed sample.

The final plug-in and DR coefficient vectors are neither normalized nor clipped. In particular, preserving inverse-weight mass does not imply preserving DR target mass. Linearity gives

$$
v_{\mathrm{DR,preserve}}
=v_{\mathrm{PI}}+
s_{a,q}\bigl(v_{\mathrm{DR,raw}}-v_{\mathrm{PI}}\bigr).
$$

Only the residual correction is scaled; the plug-in component is unchanged. Positive rescaling leaves the inverse-weight ESS unchanged, but changes the MMD target and can change sampling performance.

### 5.4 Analytic mass and shape distortion under ideal weights

For intuition only, suppose the original weights were exact. Under the target $X\sim U[0,1]$, the raw corrupted mass is $(1+q)/2$. If that distorted law were normalized, its red-heavy probability and mean would be

$$
P_q(X\geq0.5)=\frac{q}{1+q},\qquad
E_q[X]=\frac{1+3q}{4(1+q)}.
$$

| Multiplier q | Ideal raw mass | Normalized red-heavy probability | Normalized color mean |
| --- | --- | --- | --- |
| 1 | 1.000000 | 0.500000 | 0.500000 |
| 1/4 | 0.625000 | 0.200000 | 0.350000 |
| 1/20 | 0.525000 | 0.047619 | 0.273810 |
| 1/100 | 0.505000 | 0.009901 | 0.254950 |

These are analytic benchmarks, not the realized fitted-weight masses and not a prediction that the finite MMD flow must return exactly that normalized law. They show why mass preservation cannot remove the subgroup shape distortion.

### 5.5 Full factorial and reused controls

We cross all four CME supports with all seven distinct weight settings:

- one baseline, $q=1$;
- three nonbaseline multipliers, each with raw and mass-preserving versions.

For each combination, we evaluate plug-in, IPW, and DR. Per arm this gives $4\times7\times3=84$ configuration rows. Plug-in is invariant to the weight setting, and IPW is invariant to the CME cutoff, so only

$$
4\ \text{plug-in}+7\ \text{IPW}+28\ \text{DR}=39
$$

unique flows are needed per arm. Totals are **168 rows / 78 flows per seed**, and **504 rows / 234 flows across three seeds**. Repeated plug-in/IPW entries are shared controls, not extra independent runs.

## 6. Exact training counts and nuisance tuning

### 6.1 Observations available to each CME

The counts below come from the saved nuisance-fit artifacts. Each row describes the joint CME training set; the two treatment blocks are separated by the delta kernel.

| CME support | Seed | Total CME rows | Arm 0 (digit 1) | Arm 1 (digit 6) |
| --- | --- | --- | --- | --- |
| All | 0 | 20,000 | 10,077 | 9,923 |
| All | 1 | 20,000 | 10,020 | 9,980 |
| All | 2 | 20,000 | 9,968 | 10,032 |
| $X<0.5$ | 0 | 10,076 | 8,760 | 1,316 |
| $X<0.5$ | 1 | 10,077 | 8,644 | 1,433 |
| $X<0.5$ | 2 | 10,035 | 8,636 | 1,399 |
| $X<0.25$ | 0 | 5,058 | 4,928 | 130 |
| $X<0.25$ | 1 | 5,026 | 4,870 | 156 |
| $X<0.25$ | 2 | 4,989 | 4,852 | 137 |
| $X<0.1$ | 0 | 2,023 | 2,007 | 16 |
| $X<0.1$ | 1 | 2,009 | 1,974 | 35 |
| $X<0.1$ | 2 | 1,974 | 1,952 | 22 |

These counts apply to **every weight-corruption level and both mass controls** at that cutoff. The propensity and AE always use all 20,000 observations. The target-policy integration and DR correction always use all 20,000 observations. Counts do not become 2,000 merely because the generated cloud has 2,000 particles.

The most severe arm-1 fits have only **16, 35, and 22 training outcomes**, respectively. This is a strong extrapolation and imbalance stress test. It is not a clean equal-sample-size comparison between two function classes.

### 6.2 Landmarks, CV rows, and selected ridge

| CME support | Seed | Landmarks arm 0 / arm 1 | CV total | CV arm 0 / arm 1 | Gram ridge ρ |
| --- | --- | --- | --- | --- | --- |
| All | 0 | 503 / 521 | 2,000 | 976 / 1,024 | 1 |
| All | 1 | 506 / 518 | 2,000 | 1,011 / 989 | 0.1 |
| All | 2 | 496 / 528 | 2,000 | 978 / 1,022 | 1 |
| $X<0.5$ | 0 | 881 / 143 | 2,000 | 1,727 / 273 | 10 |
| $X<0.5$ | 1 | 873 / 151 | 2,000 | 1,692 / 308 | 1 |
| $X<0.5$ | 2 | 870 / 154 | 2,000 | 1,737 / 263 | 1 |
| $X<0.25$ | 0 | 995 / 29 | 2,000 | 1,943 / 57 | 1 |
| $X<0.25$ | 1 | 996 / 28 | 2,000 | 1,934 / 66 | 0.001 |
| $X<0.25$ | 2 | 996 / 28 | 2,000 | 1,944 / 56 | 1 |
| $X<0.1$ | 0 | 1,016 / 8 | 2,000 | 1,984 / 16 | 1 |
| $X<0.1$ | 1 | 1,006 / 18 | 2,000 | 1,966 / 34 | 1 |
| $X<0.1$ | 2 | 1,011 / 13 | 1,974 | 1,952 / 22 | 0.0001 |

The landmark budget is 1,024 across both arms, not 1,024 per arm. Uniform selection is not balanced by treatment, so the extreme arm-1 landmark counts are **8, 18, and 13**. The spectral cutoff may reduce the retained numerical rank further; the actual retained rank is not separately archived in these metadata.

## 7. Estimated targets: equations and numerical implementation

Write $\phi(z)=k(z,\cdot)$ for an outcome feature map and $\widehat m_c(x,a)$ for the fitted CME using training support $T_c$. The conceptual estimators are

$$
\widehat\chi_{\mathrm{PI},c}^{(a)}
=\frac1n\sum_{i=1}^n \widehat m_c(X_i,a),
$$

$$
\widehat\chi_{\mathrm{IPW},q}^{(a)}
=\frac1n\sum_{i=1}^n w_i^{(a,q)}\phi(Z_i),
$$

$$
\widehat\chi_{\mathrm{DR},c,q}^{(a)}
=\frac1n\sum_{i=1}^n
\left[
\widehat m_c(X_i,a)+
w_i^{(a,q)}
\{\phi(Z_i)-\widehat m_c(X_i,A_i)\}
\right].
$$

The same formulas apply to either raw or mass-preserved weights. The denominator is always the **complete causal sample size $n=20,000$**, not the restricted CME count or arm count.

The implementation represents each target as $\sum_i v_i\phi(Z_i)$ over the full logged support. To state the deployed projected Nyström construction, let $T=T_c$, let $L\subset T$ be the landmarks, and define

$$
U=K_{T,L},\quad W=K_{L,L},\quad
W=V D V^\top.
$$

Keep eigenvalues $d_j>10^{-8}d_{\max}$, form $G=V_rD_r^{-1/2}$ and $\Phi=UG$, and define the linear map

$$
B(r)=\Phi(\Phi^\top\Phi+\rho I)^{-1}G^\top r.
$$

For each landmark $\ell$, the policy and residual right-hand sides are

$$
r_{\mathrm{PI},\ell}
=\mathbf 1\{A_\ell=a\}\sum_{i=1}^n k_X(X_\ell,X_i),
\qquad
r_{\mathrm{res}}=-K_{L,\{1,\ldots,n\}}w.
$$

With $\operatorname{embed}_T$ placing a vector on the training indices and zeros elsewhere,

$$
v_{\mathrm{PI}}=\frac{\operatorname{embed}_T B(r_{\mathrm{PI}})}n,
\qquad v_{\mathrm{IPW}}=\frac wn,
$$

$$
v_{\mathrm{DR}}=
v_{\mathrm{PI}}+\frac{\operatorname{embed}_T B(r_{\mathrm{res}})}n+\frac wn.
$$

The landmark right-hand sides are accumulated over **all** observations in blocks of 2,048. Policy and all seven residual right-hand sides are solved together for a given CME fit. The parameter called jitter acts as a relative spectral cutoff in this solver; it should not be described as adding $10^{-8}I$ to every system.

A useful exact invariant is

$$
i\notin T_c\quad\Longrightarrow\quad
v_{\mathrm{PI},i}=0,\qquad
v_{\mathrm{DR},i}=w_i/n.
$$

Thus excluded CME-training observations keep their own weighted outcome contribution. The residual does not disappear outside the training region.

Signed coefficients, nonunit total mass, and an unattainable probability embedding are all possible. No positive projection, clipping, or normalization is imposed. The full-CME DR target can be very close to plug-in when fitted residuals are small; empirical success in that case should not be attributed entirely to a large DR correction.

## 8. Fixed sampler, initialization, and paired randomness

For each arm, select 2,000 distinct observed indices from that arm without replacement. Initialize particles at their encoded outcomes. The indices are distinct; this is not a claim that image templates or latent coordinates cannot repeat. There is no initial jitter or resampling from the inverse weights.

The outcome kernel is

$$
k(z,z')=(c_{\mathrm{IMQ}}^2+\|z-z'\|^2)^{-1/2}.
$$

Its scale is the median pairwise latent distance estimated from up to 5,000 logged latents. The helper is called separately for each arm and uses the evolving global random state, so the two arm scales can differ slightly. The scale and step size are fixed across every corruption and estimator **within an arm**.

Run 3,000 deterministic Euler updates,

$$
z_j^{t+1}
=z_j^t-\eta\left[
 \frac1M\sum_{\ell=1}^M\nabla_1 k(z_j^t,z_\ell^t)
 -\sum_{i=1}^n v_i\nabla_1 k(z_j^t,Z_i)
\right],
\qquad \eta=0.15c_{\mathrm{IMQ}}^2.
$$

The target coefficients and target support are frozen throughout each flow. There is no injected noise, additive noise, adaptive stopping, or AdamW update. No Deep-MMD discriminator or random Fourier feature approximation is used. Decode the returned particles once for final image metrics.

| Seed | Arm | IMQ c | Euler step | Context RBF ℓ | Diagnostic RBF ℓ | AE reconstruction MSE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 13.48166084 | 27.26327686 | 0.20765351 | 9.51651669 | 0.002426671 |
| 0 | 1 | 13.52962780 | 27.45762426 | 0.20765351 | 9.51651669 | 0.002426671 |
| 1 | 0 | 12.85377789 | 24.78294089 | 0.20908707 | 9.08541012 | 0.002427972 |
| 1 | 1 | 12.83878422 | 24.72515703 | 0.20908707 | 9.08541012 | 0.002427972 |
| 2 | 0 | 13.11882210 | 25.81552398 | 0.20897059 | 9.27520752 | 0.002479424 |
| 2 | 1 | 13.13669300 | 25.88590545 | 0.20897059 | 9.27520752 | 0.002479424 |

The saved step sizes are large-looking because the IMQ kernel is unnormalized; they must be interpreted together with its amplitude and scale.

Randomness is paired as follows:

| Operation | Seed/rule |
|---|---|
| Logged DGP on CUDA | Data seed $s\in\{0,1,2\}$ |
| AE initialization and minibatch generator | Pipeline seed $p=1000s$ |
| Propensity setup | Global torch seed reset to $p$; zero-initialized deterministic CPU fit |
| Context bandwidth index subsample | CPU generator seed 0 |
| CME landmarks | CPU generator seed $p$, restarted for each eligible support |
| CV row subsample and fold assignment | Seed 0 |
| Initial observed-arm particle selection | CPU generator seed $p+1$, restarted for each arm |
| Oracle reference 1 | CUDA generator seed $p+9$ |
| Oracle reference 2 | CUDA generator seed $p+10$ |
| SW2 directions | Seed 0 for each call |
| IMQ bandwidth subsample | Global torch RNG at each arm's setup |

Each arm uses the same two reference-generator seed values, so random draws across arms are coupled; the arms are averaged before computing uncertainty. The two reference streams are separate from each other and from logged-data generation.

## 9. Evaluation protocol and aggregation

### 9.1 Metrics

**Pixel SW2 (primary).** Decode to $[0,1]$, flatten each image to 2,352 coordinates, cast to float32, and use 1,000 unit-normalized Gaussian projection directions with seed 0. Sort the projected values and compute

$$
\mathrm{SW2}=
\left[
\frac1{LM}\sum_{\ell=1}^L\sum_{j=1}^M
\left(\langle\theta_\ell,Y\rangle_{(j)}
-\langle\theta_\ell,Y^{\mathrm{ref}}\rangle_{(j)}\right)^2
\right]^{1/2}.
$$

Here $L=1,000$ and $M=2,000$. The square root is taken after averaging squared differences over both samples and directions. There is no additional division by image dimension. The two compared sample counts are equal.

**Latent SW2.** The same projection protocol on 32-dimensional generated and oracle latents. This measures the encoded representation, not the full image law. Each seed trains a different encoder, so latent scales can differ between seeds; pixel SW2 is the primary comparison.

**Color W1.** Compute $C(Y)$ from each decoded image and evaluate the exact one-dimensional $W_1$ distance between that empirical distribution and Uniform[0,1]. The implementation integrates the empirical quantile function analytically; it does not approximate the target with another random uniform sample.

**Pre-flow target RBF squared distance.** With a separate RBF diagnostic kernel, compute

$$
\left\|\sum_i v_i k_{\mathrm{diag}}(Z_i,\cdot)
-\frac1{n_{\mathrm{ref}}}\sum_j k_{\mathrm{diag}}(Z_j^{\mathrm{ref}},\cdot)\right\|^2.
$$

The RBF bandwidth is fixed using the first 2,000 logged latents; computation uses float64 and blocks of 256. This diagnoses the **estimated target before sampling**. It is not final-particle MMD and is not the IMQ optimization objective. Raw coefficients include their own mass; no renormalization is applied.

**Additional recorded diagnostics.** The raw CSVs preserve decoded color mean and second moment; coefficient mass, absolute mass, negative mass and count; raw and mass-divided weighted covariate moments; inverse-weight mass, maximum and ESS; DR correction L1; first-update and final IMQ objectives; and timing. See Section 15 for column definitions.

FID and kernel-herding comparisons were not computed in this experiment.

### 9.2 Shared evaluation controls

For each arm, two fresh oracle draws of 2,000 images are generated. The first is the reference for all sample comparisons. The controls are:

- **Initialization:** decode the actual selected initial latent particles and compare with reference 1. This is the sampler's iteration-zero decoded cloud, not all logged images.
- **Count-matched oracle comparison:** compare raw reference 2 with raw reference 1.
- **Oracle encoder–decoder roundtrip:** encode and decode reference 2, then compare with raw reference 1.
- **Target-RBF oracle comparison:** compare the two oracle empirical latent embeddings.

The “floor” names in the artifacts refer to empirical reference comparisons, not mathematical lower bounds. Their uncertainty is retained.

| Control | Pixel SW2 | Latent SW2 | Color W1 | Color mean | Color second moment |
| --- | --- | --- | --- | --- | --- |
| Decoded initialization | 0.043365 ± 0.000080 | 0.495919 ± 0.020645 | 0.222877 ± 0.003244 | 0.505009 ± 0.004484 | 0.338770 ± 0.006926 |
| Raw oracle comparison | 0.006303 ± 0.000323 | 0.071834 ± 0.004171 | 0.006298 ± 0.001915 | 0.504384 ± 0.003116 | 0.338191 ± 0.003315 |
| Oracle encoder–decoder roundtrip | 0.008192 ± 0.000887 | 0.071834 ± 0.004171 | 0.014868 ± 0.001924 | 0.509914 ± 0.003358 | 0.341844 ± 0.003808 |

The target-RBF oracle comparison is 0.000421 ± 0.000130. The stored latent roundtrip score uses the original encoded reference latents, so it equals the latent oracle score; it does not re-encode the decoded images.

### 9.3 Averaging and uncertainty

For metric $d_{s,a}$, compute

$$
\bar d_s=(d_{s,0}+d_{s,1})/2,\qquad
\bar d=\frac13\sum_{s=0}^2\bar d_s,
$$

$$
\mathrm{SD}=
\sqrt{\frac1{3-1}\sum_{s=0}^2(\bar d_s-\bar d)^2}.
$$

All reported “$\pm$” values are **sample standard deviations across three seed-level arm averages**, not standard errors or confidence intervals. There is one pipeline repeat per data seed. Arm rows and reused plug-in/IPW results are not counted as independent repetitions. The baseline is shown in both mass-control tables for readability but is only estimated once.

## 10. Primary results: pixel SW2

All values below are **unscaled SW2**, mean $\pm$ sample SD. Lower is better. The tables include every tested CME/weight combination; repeated plug-in/IPW values reflect shared controls.

### 10.1 Raw inverse-weight corruption

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.009413 ± 0.000576 | 0.009486 ± 0.000350 | 0.009402 ± 0.000561 |
| All | 1/4 | 0.009413 ± 0.000576 | 0.033420 ± 0.000784 | 0.009416 ± 0.000575 |
| All | 1/20 | 0.009413 ± 0.000576 | 0.041288 ± 0.001207 | 0.009422 ± 0.000578 |
| All | 1/100 | 0.009413 ± 0.000576 | 0.042811 ± 0.001288 | 0.009423 ± 0.000578 |
| $X<0.5$ | 1 | 0.027154 ± 0.000895 | 0.009486 ± 0.000350 | 0.009464 ± 0.000459 |
| $X<0.5$ | 1/4 | 0.027154 ± 0.000895 | 0.033420 ± 0.000784 | 0.021579 ± 0.000690 |
| $X<0.5$ | 1/20 | 0.027154 ± 0.000895 | 0.041288 ± 0.001207 | 0.026112 ± 0.000876 |
| $X<0.5$ | 1/100 | 0.027154 ± 0.000895 | 0.042811 ± 0.001288 | 0.027005 ± 0.000924 |
| $X<0.25$ | 1 | 0.067196 ± 0.040217 | 0.009486 ± 0.000350 | 0.009434 ± 0.000474 |
| $X<0.25$ | 1/4 | 0.067196 ± 0.040217 | 0.033420 ± 0.000784 | 0.046130 ± 0.027043 |
| $X<0.25$ | 1/20 | 0.067196 ± 0.040217 | 0.041288 ± 0.001207 | 0.054880 ± 0.029788 |
| $X<0.25$ | 1/100 | 0.067196 ± 0.040217 | 0.042811 ± 0.001288 | 0.056517 ± 0.030231 |
| $X<0.1$ | 1 | 0.086038 ± 0.057141 | 0.009486 ± 0.000350 | 0.009581 ± 0.000485 |
| $X<0.1$ | 1/4 | 0.086038 ± 0.057141 | 0.033420 ± 0.000784 | 0.041199 ± 0.015664 |
| $X<0.1$ | 1/20 | 0.086038 ± 0.057141 | 0.041288 ± 0.001207 | 0.050020 ± 0.018035 |
| $X<0.1$ | 1/100 | 0.086038 ± 0.057141 | 0.042811 ± 0.001288 | 0.051653 ± 0.018377 |

### 10.2 Empirical-mass-preserving weight corruption

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.009413 ± 0.000576 | 0.009486 ± 0.000350 | 0.009402 ± 0.000561 |
| All | 1/4 | 0.009413 ± 0.000576 | 0.028982 ± 0.001461 | 0.009429 ± 0.000566 |
| All | 1/20 | 0.009413 ± 0.000576 | 0.043214 ± 0.000928 | 0.009449 ± 0.000565 |
| All | 1/100 | 0.009413 ± 0.000576 | 0.046735 ± 0.000781 | 0.009455 ± 0.000564 |
| $X<0.5$ | 1 | 0.027154 ± 0.000895 | 0.009486 ± 0.000350 | 0.009464 ± 0.000459 |
| $X<0.5$ | 1/4 | 0.027154 ± 0.000895 | 0.028982 ± 0.001461 | 0.018214 ± 0.000696 |
| $X<0.5$ | 1/20 | 0.027154 ± 0.000895 | 0.043214 ± 0.000928 | 0.025206 ± 0.000916 |
| $X<0.5$ | 1/100 | 0.027154 ± 0.000895 | 0.046735 ± 0.000781 | 0.026928 ± 0.001042 |
| $X<0.25$ | 1 | 0.067196 ± 0.040217 | 0.009486 ± 0.000350 | 0.009434 ± 0.000474 |
| $X<0.25$ | 1/4 | 0.067196 ± 0.040217 | 0.028982 ± 0.001461 | 0.031981 ± 0.016294 |
| $X<0.25$ | 1/20 | 0.067196 ± 0.040217 | 0.043214 ± 0.000928 | 0.043888 ± 0.019774 |
| $X<0.25$ | 1/100 | 0.067196 ± 0.040217 | 0.046735 ± 0.000781 | 0.046550 ± 0.020311 |
| $X<0.1$ | 1 | 0.086038 ± 0.057141 | 0.009486 ± 0.000350 | 0.009581 ± 0.000485 |
| $X<0.1$ | 1/4 | 0.086038 ± 0.057141 | 0.028982 ± 0.001461 | 0.033945 ± 0.021071 |
| $X<0.1$ | 1/20 | 0.086038 ± 0.057141 | 0.043214 ± 0.000928 | 0.047380 ± 0.029467 |
| $X<0.1$ | 1/100 | 0.086038 ± 0.057141 | 0.046735 ± 0.000781 | 0.050351 ± 0.031094 |

### 10.3 Supported interpretation

The [full comparison figure](results/severity_paper_summary/pixel_sw2_flow_severity.png) displays both mass controls, with error bars showing the sample SD across three seed-level arm averages.

1. **Weight-only corruption:** with full CME, raw IPW worsens from approximately 0.00949 to 0.04281, while DR stays approximately 0.00942. The mass-preserving version shows the same protection of DR.
2. **CME-only corruption:** with baseline weights, plug-in worsens from approximately 0.00941 to 0.08604 at the strongest restriction, while DR remains approximately 0.0094–0.0096.
3. **Both corrupted:** DR loses this protection. At $c=0.5$, it retains better SW2 than either singly robust alternative over the tested corruption levels. At $c=0.25$ and $c=0.1$, raw DR has higher mean SW2 than IPW; this reversal is driven by an unstable seed at each cutoff, not by uniformly worse performance in every seed.
4. **Severity pattern:** at a fixed restricted CME support, increasing weight corruption worsens DR performance. Shrinking CME support does not yield a monotone error ordering across all cutoffs.
5. **Mass control:** preserving baseline inverse-weight mass does not remove the effect. It is a diagnostic intervention, not a uniformly beneficial correction.
6. **Initialization:** the strongest both-corrupted mean SW2 can exceed the decoded initialization's error. The sampler can fit a poor estimated target and produce a worse interventional approximation.

The saved three-seed grid also shows monotone increases in DR SW2, color W1, and target-RBF error along $q=1\to1/4\to1/20\to1/100$ for each restricted cutoff and mass mode, separately within each arm-averaged seed. This is an observed grid property, not a universal guarantee.

## 11. Auxiliary results

### 11.1 Decoded color W1 to Uniform[0,1]

These are sample-level color errors, not coefficient moments.

**Raw weights**

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.017020 ± 0.003250 | 0.020318 ± 0.004140 | 0.016235 ± 0.003381 |
| All | 1/4 | 0.017020 ± 0.003250 | 0.071493 ± 0.007599 | 0.016347 ± 0.003331 |
| All | 1/20 | 0.017020 ± 0.003250 | 0.084528 ± 0.009173 | 0.016383 ± 0.003318 |
| All | 1/100 | 0.017020 ± 0.003250 | 0.086313 ± 0.009481 | 0.016391 ± 0.003315 |
| $X<0.5$ | 1 | 0.092467 ± 0.001907 | 0.020318 ± 0.004140 | 0.016794 ± 0.002414 |
| $X<0.5$ | 1/4 | 0.092467 ± 0.001907 | 0.071493 ± 0.007599 | 0.071823 ± 0.000251 |
| $X<0.5$ | 1/20 | 0.092467 ± 0.001907 | 0.084528 ± 0.009173 | 0.090047 ± 0.001226 |
| $X<0.5$ | 1/100 | 0.092467 ± 0.001907 | 0.086313 ± 0.009481 | 0.093627 ± 0.001385 |
| $X<0.25$ | 1 | 0.163647 ± 0.112419 | 0.020318 ± 0.004140 | 0.016314 ± 0.002853 |
| $X<0.25$ | 1/4 | 0.163647 ± 0.112419 | 0.071493 ± 0.007599 | 0.119589 ± 0.064585 |
| $X<0.25$ | 1/20 | 0.163647 ± 0.112419 | 0.084528 ± 0.009173 | 0.145538 ± 0.078315 |
| $X<0.25$ | 1/100 | 0.163647 ± 0.112419 | 0.086313 ± 0.009481 | 0.149930 ± 0.080750 |
| $X<0.1$ | 1 | 0.172949 ± 0.195791 | 0.020318 ± 0.004140 | 0.016454 ± 0.003342 |
| $X<0.1$ | 1/4 | 0.172949 ± 0.195791 | 0.071493 ± 0.007599 | 0.099384 ± 0.042699 |
| $X<0.1$ | 1/20 | 0.172949 ± 0.195791 | 0.084528 ± 0.009173 | 0.121800 ± 0.057958 |
| $X<0.1$ | 1/100 | 0.172949 ± 0.195791 | 0.086313 ± 0.009481 | 0.125399 ± 0.060847 |

**Mass-preserving weights**

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.017020 ± 0.003250 | 0.020318 ± 0.004140 | 0.016235 ± 0.003381 |
| All | 1/4 | 0.017020 ± 0.003250 | 0.137416 ± 0.006110 | 0.016070 ± 0.003355 |
| All | 1/20 | 0.017020 ± 0.003250 | 0.212119 ± 0.005115 | 0.016025 ± 0.003331 |
| All | 1/100 | 0.017020 ± 0.003250 | 0.230484 ± 0.004972 | 0.016017 ± 0.003326 |
| $X<0.5$ | 1 | 0.092467 ± 0.001907 | 0.020318 ± 0.004140 | 0.016794 ± 0.002414 |
| $X<0.5$ | 1/4 | 0.092467 ± 0.001907 | 0.137416 ± 0.006110 | 0.059177 ± 0.000901 |
| $X<0.5$ | 1/20 | 0.092467 ± 0.001907 | 0.212119 ± 0.005115 | 0.088034 ± 0.003565 |
| $X<0.5$ | 1/100 | 0.092467 ± 0.001907 | 0.230484 ± 0.004972 | 0.095112 ± 0.004235 |
| $X<0.25$ | 1 | 0.163647 ± 0.112419 | 0.020318 ± 0.004140 | 0.016314 ± 0.002853 |
| $X<0.25$ | 1/4 | 0.163647 ± 0.112419 | 0.137416 ± 0.006110 | 0.080256 ± 0.024000 |
| $X<0.25$ | 1/20 | 0.163647 ± 0.112419 | 0.212119 ± 0.005115 | 0.121698 ± 0.039288 |
| $X<0.25$ | 1/100 | 0.163647 ± 0.112419 | 0.230484 ± 0.004972 | 0.130765 ± 0.042395 |
| $X<0.1$ | 1 | 0.172949 ± 0.195791 | 0.020318 ± 0.004140 | 0.016454 ± 0.003342 |
| $X<0.1$ | 1/4 | 0.172949 ± 0.195791 | 0.137416 ± 0.006110 | 0.103567 ± 0.051192 |
| $X<0.1$ | 1/20 | 0.172949 ± 0.195791 | 0.212119 ± 0.005115 | 0.152943 ± 0.076970 |
| $X<0.1$ | 1/100 | 0.172949 ± 0.195791 | 0.230484 ± 0.004972 | 0.163345 ± 0.081211 |

Pixel SW2 and color recovery need not rank methods identically. At $c=0.5$, $q=0.01$, raw DR has better pixel SW2 than IPW but slightly worse color W1 (approximately 0.09363 versus 0.08631). At $c=0.1$, $q=0.01$, preserving mass improves DR SW2 slightly but worsens color W1 (approximately 0.12540 to 0.16334).

At $q=0.01$, mass-preserved IPW has color W1 approximately 0.23048 compared with 0.08631 for raw IPW. This illustrates why mass preservation is not itself a distributional correction.

### 11.2 Pre-flow target RBF squared distance

These values diagnose the target-estimation stage against a finite oracle reference. They are squared RKHS distances and use a different kernel from the flow.

**Raw weights**

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.000539 ± 0.000060 | 0.000700 ± 0.000133 | 0.000547 ± 0.000055 |
| All | 1/4 | 0.000539 ± 0.000060 | 0.077060 ± 0.003034 | 0.000559 ± 0.000061 |
| All | 1/20 | 0.000539 ± 0.000060 | 0.122015 ± 0.004932 | 0.000564 ± 0.000062 |
| All | 1/100 | 0.000539 ± 0.000060 | 0.132249 ± 0.005371 | 0.000565 ± 0.000063 |
| $X<0.5$ | 1 | 0.038995 ± 0.006689 | 0.000700 ± 0.000133 | 0.000573 ± 0.000045 |
| $X<0.5$ | 1/4 | 0.038995 ± 0.006689 | 0.077060 ± 0.003034 | 0.021864 ± 0.002794 |
| $X<0.5$ | 1/20 | 0.038995 ± 0.006689 | 0.122015 ± 0.004932 | 0.034627 ± 0.005040 |
| $X<0.5$ | 1/100 | 0.038995 ± 0.006689 | 0.132249 ± 0.005371 | 0.037538 ± 0.005566 |
| $X<0.25$ | 1 | 0.912140 ± 1.342359 | 0.000700 ± 0.000133 | 0.000590 ± 0.000042 |
| $X<0.25$ | 1/4 | 0.912140 ± 1.342359 | 0.077060 ± 0.003034 | 0.298620 ± 0.413714 |
| $X<0.25$ | 1/20 | 0.912140 ± 1.342359 | 0.122015 ± 0.004932 | 0.478366 ± 0.663043 |
| $X<0.25$ | 1/100 | 0.912140 ± 1.342359 | 0.132249 ± 0.005371 | 0.519382 ± 0.719933 |
| $X<0.1$ | 1 | 1.922187 ± 2.924024 | 0.000700 ± 0.000133 | 0.000836 ± 0.000422 |
| $X<0.1$ | 1/4 | 1.922187 ± 2.924024 | 0.077060 ± 0.003034 | 0.149513 ± 0.135550 |
| $X<0.1$ | 1/20 | 1.922187 ± 2.924024 | 0.122015 ± 0.004932 | 0.236916 ± 0.213404 |
| $X<0.1$ | 1/100 | 1.922187 ± 2.924024 | 0.132249 ± 0.005371 | 0.256809 ± 0.231088 |

**Mass-preserving weights**

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.000539 ± 0.000060 | 0.000700 ± 0.000133 | 0.000547 ± 0.000055 |
| All | 1/4 | 0.000539 ± 0.000060 | 0.018235 ± 0.002035 | 0.000580 ± 0.000058 |
| All | 1/20 | 0.000539 ± 0.000060 | 0.040216 ± 0.002500 | 0.000603 ± 0.000060 |
| All | 1/100 | 0.000539 ± 0.000060 | 0.047031 ± 0.002514 | 0.000609 ± 0.000061 |
| $X<0.5$ | 1 | 0.038995 ± 0.006689 | 0.000700 ± 0.000133 | 0.000573 ± 0.000045 |
| $X<0.5$ | 1/4 | 0.038995 ± 0.006689 | 0.018235 ± 0.002035 | 0.014000 ± 0.001143 |
| $X<0.5$ | 1/20 | 0.038995 ± 0.006689 | 0.040216 ± 0.002500 | 0.030974 ± 0.003753 |
| $X<0.5$ | 1/100 | 0.038995 ± 0.006689 | 0.047031 ± 0.002514 | 0.036243 ± 0.004652 |
| $X<0.25$ | 1 | 0.912140 ± 1.342359 | 0.000700 ± 0.000133 | 0.000590 ± 0.000042 |
| $X<0.25$ | 1/4 | 0.912140 ± 1.342359 | 0.018235 ± 0.002035 | 0.098168 ± 0.119960 |
| $X<0.25$ | 1/20 | 0.912140 ± 1.342359 | 0.040216 ± 0.002500 | 0.221005 ± 0.269537 |
| $X<0.25$ | 1/100 | 0.912140 ± 1.342359 | 0.047031 ± 0.002514 | 0.258960 ± 0.315580 |
| $X<0.1$ | 1 | 1.922187 ± 2.924024 | 0.000700 ± 0.000133 | 0.000836 ± 0.000422 |
| $X<0.1$ | 1/4 | 1.922187 ± 2.924024 | 0.018235 ± 0.002035 | 0.150637 ± 0.223768 |
| $X<0.1$ | 1/20 | 1.922187 ± 2.924024 | 0.040216 ± 0.002500 | 0.352093 ± 0.526970 |
| $X<0.1$ | 1/100 | 1.922187 ± 2.924024 | 0.047031 ± 0.002514 | 0.415322 ± 0.622297 |

### 11.3 Latent SW2

The following summaries are computed directly from the completed per-seed CSVs using the same arm-then-seed aggregation. They are supplementary to the three metrics displayed in the executed analysis notebook.

**Raw weights**

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.081411 ± 0.001356 | 0.093306 ± 0.011458 | 0.080173 ± 0.000941 |
| All | 1/4 | 0.081411 ± 0.001356 | 1.581747 ± 0.073156 | 0.081210 ± 0.001229 |
| All | 1/20 | 0.081411 ± 0.001356 | 2.174276 ± 0.095628 | 0.081571 ± 0.001365 |
| All | 1/100 | 0.081411 ± 0.001356 | 2.299869 ± 0.100891 | 0.081648 ± 0.001397 |
| $X<0.5$ | 1 | 0.913905 ± 0.162693 | 0.093306 ± 0.011458 | 0.081949 ± 0.000388 |
| $X<0.5$ | 1/4 | 0.913905 ± 0.162693 | 1.581747 ± 0.073156 | 0.617140 ± 0.081441 |
| $X<0.5$ | 1/20 | 0.913905 ± 0.162693 | 2.174276 ± 0.095628 | 0.832120 ± 0.122295 |
| $X<0.5$ | 1/100 | 0.913905 ± 0.162693 | 2.299869 ± 0.100891 | 0.877930 ± 0.131102 |
| $X<0.25$ | 1 | 2.136380 ± 0.295211 | 0.093306 ± 0.011458 | 0.082101 ± 0.001108 |
| $X<0.25$ | 1/4 | 2.136380 ± 0.295211 | 1.581747 ± 0.073156 | 1.333147 ± 0.062988 |
| $X<0.25$ | 1/20 | 2.136380 ± 0.295211 | 2.174276 ± 0.095628 | 1.726734 ± 0.113577 |
| $X<0.25$ | 1/100 | 2.136380 ± 0.295211 | 2.299869 ± 0.100891 | 1.806973 ± 0.155681 |
| $X<0.1$ | 1 | 3.136854 ± 0.622132 | 0.093306 ± 0.011458 | 0.089000 ± 0.011827 |
| $X<0.1$ | 1/4 | 3.136854 ± 0.622132 | 1.581747 ± 0.073156 | 1.486229 ± 0.029719 |
| $X<0.1$ | 1/20 | 3.136854 ± 0.622132 | 2.174276 ± 0.095628 | 2.017665 ± 0.068624 |
| $X<0.1$ | 1/100 | 3.136854 ± 0.622132 | 2.299869 ± 0.100891 | 2.128018 ± 0.085623 |

**Mass-preserving weights**

| CME training support | Red-subgroup multiplier | Plug-in | IPW | DR |
| --- | --- | --- | --- | --- |
| All | 1 | 0.081411 ± 0.001356 | 0.093306 ± 0.011458 | 0.080173 ± 0.000941 |
| All | 1/4 | 0.081411 ± 0.001356 | 0.340953 ± 0.008630 | 0.081494 ± 0.001344 |
| All | 1/20 | 0.081411 ± 0.001356 | 0.502089 ± 0.003114 | 0.082405 ± 0.001653 |
| All | 1/100 | 0.081411 ± 0.001356 | 0.541429 ± 0.004007 | 0.082655 ± 0.001747 |
| $X<0.5$ | 1 | 0.913905 ± 0.162693 | 0.093306 ± 0.011458 | 0.081949 ± 0.000388 |
| $X<0.5$ | 1/4 | 0.913905 ± 0.162693 | 0.340953 ± 0.008630 | 0.460581 ± 0.038500 |
| $X<0.5$ | 1/20 | 0.913905 ± 0.162693 | 0.502089 ± 0.003114 | 0.760082 ± 0.086029 |
| $X<0.5$ | 1/100 | 0.913905 ± 0.162693 | 0.541429 ± 0.004007 | 0.842867 ± 0.100119 |
| $X<0.25$ | 1 | 2.136380 ± 0.295211 | 0.093306 ± 0.011458 | 0.082101 ± 0.001108 |
| $X<0.25$ | 1/4 | 2.136380 ± 0.295211 | 0.340953 ± 0.008630 | 0.862574 ± 0.144512 |
| $X<0.25$ | 1/20 | 2.136380 ± 0.295211 | 0.502089 ± 0.003114 | 1.384196 ± 0.052272 |
| $X<0.25$ | 1/100 | 2.136380 ± 0.295211 | 0.541429 ± 0.004007 | 1.519802 ± 0.014138 |
| $X<0.1$ | 1 | 3.136854 ± 0.622132 | 0.093306 ± 0.011458 | 0.089000 ± 0.011827 |
| $X<0.1$ | 1/4 | 3.136854 ± 0.622132 | 0.340953 ± 0.008630 | 0.766838 ± 0.294157 |
| $X<0.1$ | 1/20 | 3.136854 ± 0.622132 | 0.502089 ± 0.003114 | 1.185914 ± 0.310643 |
| $X<0.1$ | 1/100 | 3.136854 ± 0.622132 | 0.541429 ± 0.004007 | 1.293561 ± 0.301680 |

## 12. Seed-specific instability and signed-target diagnostics

### 12.1 Severe CME plug-in fits

| CME support | Seed | Selected ridge | Plug-in SW2 (arm mean) | DR SW2 with baseline weights (arm mean) |
| --- | --- | --- | --- | --- |
| $X<0.25$ | 0 | 1 | 0.043359 | 0.009125 |
| $X<0.25$ | 1 | 0.001 | 0.113630 | 0.009980 |
| $X<0.25$ | 2 | 1 | 0.044600 | 0.009197 |
| $X<0.1$ | 0 | 1 | 0.054513 | 0.009021 |
| $X<0.1$ | 1 | 1 | 0.051604 | 0.009869 |
| $X<0.1$ | 2 | 0.0001 | 0.151998 | 0.009852 |

Small CV-selected ridge values coincide with the large plug-in outliers: $c=0.25$ in seed 1 and $c=0.1$ in seed 2. This is an association, not a controlled demonstration that ridge selection alone causes the failure.

The extreme $c=0.1$, seed-2, arm-0 plug-in target has absolute coefficient mass about 162.45 and negative mass about 81.00, while its total mass is only about 0.446. Large cancellation makes this target numerically and statistically sensitive. In the same seed, the arm-1 plug-in sample error is also large.

| CME support | Seed | Arm | Plug-in total mass | Absolute mass | Negative mass | Pixel SW2 |
| --- | --- | --- | --- | --- | --- | --- |
| $X<0.25$ | 1 | 0 | 0.627973 | 126.689316 | 63.030671 | 0.087021 |
| $X<0.25$ | 1 | 1 | 0.576738 | 28.106680 | 13.764971 | 0.140239 |
| $X<0.1$ | 2 | 0 | 0.446462 | 162.452191 | 81.002864 | 0.115487 |
| $X<0.1$ | 2 | 1 | 0.442959 | 14.646320 | 7.101681 | 0.188509 |

### 12.2 DR versus IPW within seeds

For the strongest raw corruption at $c=0.1$, the arm-averaged DR-minus-IPW pixel-SW2 differences by seed are approximately

$$
(-0.001486,\;-0.000891,\;+0.028903).
$$

DR is slightly better in two seeds and much worse in the third, producing higher mean error and larger variability. Do not translate the mean table into a claim that DR is uniformly worse than IPW under both corruptions, or into a significance claim.

### 12.3 How mass and correction diagnostics should be read

| CME support | Multiplier | Weight mode | DR total mass | DR absolute mass | DR negative mass | DR correction L1 |
| --- | --- | --- | --- | --- | --- | --- |
| All | 1 | baseline | 0.9995 ± 0.0004 | 0.9995 ± 0.0004 | 0.0000 ± 0.0000 | 0.0591 ± 0.0234 |
| All | 1/100 | raw | 0.9979 ± 0.0014 | 0.9979 ± 0.0014 | 0.0000 ± 0.0000 | 0.0759 ± 0.0133 |
| All | 1/100 | preserve | 0.9995 ± 0.0006 | 1.0002 ± 0.0002 | 0.0004 ± 0.0004 | 0.1518 ± 0.0285 |
| $X<0.5$ | 1 | baseline | 0.9998 ± 0.0052 | 0.9998 ± 0.0052 | 0.0000 ± 0.0000 | 3.2159 ± 1.2951 |
| $X<0.5$ | 1/100 | raw | 0.7535 ± 0.0261 | 2.8196 ± 1.2974 | 1.0330 ± 0.6357 | 0.0518 ± 0.0032 |
| $X<0.5$ | 1/100 | preserve | 0.7618 ± 0.0191 | 2.8050 ± 1.2871 | 1.0216 ± 0.6340 | 0.1037 ± 0.0087 |
| $X<0.25$ | 1 | baseline | 0.9995 ± 0.0072 | 1.1308 ± 0.2246 | 0.0656 ± 0.1137 | 28.2788 ± 42.8121 |
| $X<0.25$ | 1/100 | raw | 0.5862 ± 0.0354 | 21.3368 ± 33.8404 | 10.3753 ± 16.9025 | 7.0755 ± 9.4693 |
| $X<0.25$ | 1/100 | preserve | 0.6387 ± 0.0108 | 15.6452 ± 25.0559 | 7.5033 ± 12.5226 | 13.8601 ± 18.4743 |
| $X<0.1$ | 1 | baseline | 0.9984 ± 0.0067 | 1.0423 ± 0.0733 | 0.0219 ± 0.0380 | 31.4645 ± 50.0244 |
| $X<0.1$ | 1/100 | raw | 0.5257 ± 0.0134 | 9.8971 ± 15.8669 | 4.6857 ± 7.9267 | 21.5862 ± 34.2719 |
| $X<0.1$ | 1/100 | preserve | 0.6823 ± 0.0409 | 11.5995 ± 16.8525 | 5.4586 ± 8.4462 | 42.2378 ± 66.9947 |

This compact diagnostic table reports baseline weights and the strongest multiplier for each CME support. Complete coefficient vectors and all intermediate levels are preserved in the linked run artifacts.

Raw target moments $\sum_i v_i X_i$ and $\sum_i v_i X_i^2$ are not automatically probability moments when $\sum_i v_i\ne1$ or coefficients are signed. Mass-divided moments are diagnostic ratios only; they do not change the estimator passed to the flow.

The fixed-target IMQ loss can improve even when oracle sample quality worsens. It measures optimization against the fitted target and must not replace evaluation against the interventional reference.

## 13. Earlier pilot and why it is kept separate

Before the paper-configuration sweep, a one-seed pilot implemented the original four nuisance combinations at $c=0.5$ and $q=1/4$.

| Setting | Initial pilot | Completed severity study |
|---|---|---|
| Logged sample | 5,000 | 20,000 |
| Generated/reference count per arm | 1,000 | 2,000 |
| Data seeds | 0 only | 0, 1, 2 |
| AE training | Separate independent logged sample of 5,000 | All 20,000 causal-sample outcomes |
| Nuisance fitting | Two-fold cross-fitting | Full-sample/no splitting |
| Context bandwidth | Independent AE-sample contexts | Logged contexts |
| Landmark budget | 512 per training block | 1,024 per eligible support |
| Ridge CV row cap | 1,000 | 2,000 |
| AE/flow updates | 3,000 / 3,000 | 3,000 / 3,000 |
| SW2 numeric dtype | float64 | float32, matching the deployed paper protocol |
| Corruption grid | Original 2-by-2, raw weights | Four CME supports, seven weight settings |
| Execution | Local RTX 4500 Ada | Three parallel L40S tasks |

The pilot's two full CME training folds each contained 2,500 observations. Restricting those training folds to $X<0.5$ left 1,260 and 1,281 observations; each evaluation fold still contained 2,500 observations. The pilot produced 24 configuration rows and 16 unique flows in 275.7 seconds. Its equal-arm pixel-SW2 means were:

| CME | Weights | Plug-in | IPW | DR |
|---|---|---:|---:|---:|
| Full | Baseline | 0.015561 | 0.012407 | 0.012517 |
| Full | Red subgroup $\times1/4$ | 0.015561 | 0.033401 | 0.013576 |
| $X<0.5$ | Baseline | 0.059001 | 0.012407 | 0.013433 |
| $X<0.5$ | Red subgroup $\times1/4$ | 0.059001 | 0.033401 | 0.048594 |

These are one-seed pilot values without a seed-level uncertainty estimate. The pilot and severity sweep must not be pooled or interpreted as a pure sample-size ablation. The pilot's cross-fitting does not carry over to the full-sample severity results.

The pilot's decoded initialization, matched oracle comparison, and oracle encoder–decoder roundtrip scored 0.043337, 0.008964, and 0.010833, respectively, in pixel SW2.

The pilot artifacts are retained locally, and its [executed analysis notebook](ColorMNIST_DoubleGen_Nuisance_Analysis.ipynb) is included with the experiment code.

## 14. Validation, execution, and timing

### 14.1 Completed checks

- Nine focused pilot tests passed, covering nuisance restrictions, retained residual observations, weight controls, and cross-fitted/full-sample algebra.
- Ten severity tests passed, including all 84 target vectors on a small synthetic fixture evaluated for one target arm against an independent dense-KRR construction. Both observed treatment arms are present in the fixture.
- The dense equivalence fixture uses all eligible rows as landmarks and zero spectral cutoff. It validates algebra; it does not prove that the deployed 1,024-landmark approximation equals full KRR at $n=20,000$.
- A CPU pilot structural run completed 24 rows with deliberately tiny training/flow budgets.
- A severity CPU structural run completed 46 short flows / 84 rows.
- Full-grid CUDA target-construction smoke checks passed for seeds 0/1/2 at the 20k/2k dimensions, using two AE updates and no particle flows. Those smoke runs are execution checks, not accuracy results.
- The full simulation array completed all 504 rows and 234 unique flows. All ten recorded source hashes agree across seeds and with the source files audited for this report.
- The executed severity analysis notebook has five sequential code cells, execution counts 1–5, and no error outputs.
- Independent recomputation of all 84 configurations' seven notebook-summary metrics agrees with the saved means and sample SDs to a maximum difference of $3.6\times10^{-15}$.
- A plotting issue that froze shared y-axis limits after the first panel was corrected. All three exported figures were regenerated with limits covering the largest mean-plus-SD, so the severe results are not clipped.

No new simulations were run to prepare this document. The report is based on completed artifacts, with additional read-only aggregation for the supplementary tables.

### 14.2 Cluster execution and local analysis

The severity study used three concurrent NVIDIA L40S tasks, one per data seed.
Each requested one GPU, four CPU cores, and 24 GB host memory. All three tasks
completed successfully in approximately 23 minutes 47 seconds.

| Seed | Pipeline seconds | Representation/propensity seconds | Sum of 78 flow timers | Mean flow seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1403.696 | 19.524 | 1376.524 | 17.648 |
| 1 | 1404.106 | 19.645 | 1377.088 | 17.655 |
| 2 | 1404.143 | 19.664 | 1377.260 | 17.657 |

The flow timer ends before decoding and sample-metric evaluation. It includes the flow routine's endpoint IMQ objective evaluations. Setup costs repeat on many CSV rows, and a shared flow's time repeats across reused factorial entries. Deduplicate by seed/arm or shared_flow_id before summing times. These timings are not a controlled acceleration benchmark.

The analysis notebook was subsequently executed locally. The saved run
environment records Python 3.12, PyTorch 2.10.0+cu126, NumPy 2.3.5, CUDA
execution, and the NVIDIA L40S model. A complete environment lockfile and MNIST
data hashes were not recorded, so the source hash manifest is not a full
bitwise-reproducibility guarantee.

## 15. Artifact map and data dictionary

### 15.1 Source and analysis files

All experiment-specific paths below are relative to the deploy repository.

| File | Role |
|---|---|
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/colormnist_misspecification_severity.py | Full-sample severity runner and target construction |
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/colormnist_doublegen_misspecification.py | Original pilot; reused weight-corruption, metric, and diagnostic helpers |
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/test_severity.py | Ten focused severity tests |
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/test_misspecification.py | Nine original pilot tests |
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/ColorMNIST_Nuisance_Severity_Analysis.ipynb | Executed full-study analysis |
| src/data_utils/colormnist.py | DGP, MNIST pools, color statistic, model/image range conversion |
| src/autoencoder_utils/colormnist_ae.py | AE architecture, training, batched encoding |
| src/dens_ratio_utils/propensity.py | Fitted logistic propensity |
| src/causality_utils/cpme.py | Ridge CV and projected Nyström solver |
| src/kernel_utils/kernels.py and linear_algebra.py | Kernels and distance computations |
| src/mmd_flow_utils/mmd_flow.py | Weighted-target deterministic Euler flow |
| src/metric_utils/metrics.py | SW2 implementation |

Cluster-specific launch records and scheduler scripts are intentionally omitted
from the portable repository package. The commands in Section 16 describe the
same experiment without embedding private paths, host names, job identifiers,
or scheduler settings.

### 15.2 Per-seed outputs

The compact repository record retains the row-level tables for [seed 0](results/severity_paper_seed0/results.csv), [seed 1](results/severity_paper_seed1/results.csv), and [seed 2](results/severity_paper_seed2/results.csv). The table below documents the complete local run directory. Of these files, only `results.csv` is versioned; large per-flow arrays, provenance records with cluster metadata, and trained checkpoints remain local regeneration artifacts.

| Artifact | Contents |
|---|---|
| provenance.json | Completion state, configuration, data seed, source hashes, software/hardware, timing |
| progress.json | Completed/expected rows and unique flows, last shared flow, elapsed time |
| results.csv | 168 factorial rows with metrics and diagnostics |
| logged_support.npz | Logged X, A, encoded Z, fitted e1 |
| autoencoder.pt | Frozen AE state dictionary and latent dimension |
| nuisance_fits_arm0.json, nuisance_fits_arm1.json | Exact training, CV and landmark indices; counts; ridge; CV errors |
| targets_arm0.npz, targets_arm1.npz | Every unique float64 coefficient vector and all inverse-weight variants |
| evaluation_arm0.npz, evaluation_arm1.npz | Initial indices/latents and the two oracle image draws |
| particles_arm*_*.npz | Final latent particles for each unique flow |

No decoded image array is required for reproducing a stored flow's sample metrics: its particles can be decoded with the saved AE. The raw logged RGB image array is not separately saved in these run directories; the DGP and seeds plus the local MNIST assets define its regeneration.

### 15.3 Summary outputs

The [summary directory](results/severity_paper_summary/) contains:

- arm_averages_by_seed.csv: 252 seed/configuration rows after averaging arms;
- severity_mean_sd.csv: 84 estimator/configuration summaries;
- severity_plot_mean_sd.csv: 96 displayed rows after reusing baseline in both mass panels;
- cme_training_counts.csv: 24 seed/arm/cutoff rows;
- inverse_weight_diagnostics.csv: 42 seed/arm/weight-setting rows;
- shared_controls_by_seed.csv and shared_controls_mean_sd.csv;
- analysis_provenance.json: common configuration, source runs/hashes, aggregation convention;
- pixel_sw2_flow_severity.png, colour_w1_uniform_flow_severity.png, and target_rbf_squared_distance_severity.png.

### 15.4 Raw CSV fields and interpretation

| Fields | Meaning |
|---|---|
| data_seed, pipeline_seed, repeat, arm | Replication and intervention identifiers; repeat is always zero |
| cme_cutoff, weight_factor, corruption_mass, weight_threshold | Experimental cells; baseline/raw/preserve labels |
| n_obs, n_ae, n_particles, n_ref | Logged/AE/generated/reference counts |
| dr_fit, ae_split, solver | nosplit, none, projected_nystrom |
| n_landmarks, gram_ridge | Total requested landmark budget and direct Gram ridge |
| n_cme_train, n_cme_train_arm, n_landmarks_arm | Eligible total rows, target-arm rows, target-arm landmarks |
| weight_mass, baseline_weight_mass | Mean of current and baseline inverse weights over all n observations |
| weight_ess | $(\sum_i w_i)^2/\sum_i w_i^2$; zero weights outside the arm are retained |
| weight_max | Maximum current inverse weight |
| corruption_factor | Current weight sum divided by unrescaled corrupted sum; 1 for raw/baseline, $s_{a,q}$ for preserve |
| target_rbf_squared_distance | Pre-flow signed target versus empirical oracle under diagnostic RBF |
| correction_l1 | $\sum_i\lvert v_{\mathrm{DR},i}-v_{\mathrm{PI},i}\rvert$; blank for other estimators; coefficient difference, not CME prediction error |
| c_imq, step_size | Arm-specific flow kernel scale and Euler step |
| context_lengthscale, diagnostic_lengthscale | Fixed context and diagnostic RBF scales |
| reconstruction_mse | AE MSE on first 2,000 logged model-range images |
| sum_v, sum_abs_v | Total and absolute target coefficient mass |
| negative_mass, negative_count | $-\sum_{v_i<0}v_i$ and number of negative coefficients; tiny numerical negatives are counted |
| target_colour_mean, target_colour_second_moment | Raw $\sum_i v_iX_i$ and $\sum_i v_iX_i^2$; names do not imply a normalized probability law |
| target_colour_mean_divided_by_mass, target_colour_second_moment_divided_by_mass | Above divided by sum_v when positive; diagnostics only |
| pixel_sw2_*, latent_sw2_*, colour_w1_uniform_*, colour_mean_*, colour_second_moment_* | Sample metrics for suffixes initial, floor, roundtrip_floor, and flow |
| target_rbf_oracle_floor | RBF squared distance between independent oracle latent embeddings |
| objective_after_first_update | IMQ squared embedding discrepancy after update 1, not at initialization |
| objective_final | IMQ squared embedding discrepancy after update 3,000 against that fitted signed target |
| seconds_representation_and_propensity | Shared data/representation/propensity setup time within a seed |
| seconds_arm_setup | Shared target/reference/diagnostic setup time within an arm |
| seconds_unique_flow | Timed flow routine, excluding subsequent decoding and sample metrics |
| shared_flow_id | Deduplication key for reused plug-in/IPW results |

### 15.5 Source fingerprints

The following SHA256 values were recorded identically in all three completed-run provenance files. They identify the reviewed source used for the runs; other imported package helpers and data files are not exhaustively covered by this manifest.

| Source path (deploy-relative) | SHA256 |
| --- | --- |
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/colormnist_misspecification_severity.py | 16a45a144d3563a45e83fe96856b3fd02dbb46f4516a1813783bdbbedd086b6c |
| AdditonalExperiments/ColorMNIST_Nuisance_Misspecification/colormnist_doublegen_misspecification.py | fb1090c38d13b21c7d73c3117794ca8c0b9d857a120b713efd52e78e8bac6687 |
| src/causality_utils/cpme.py | 32cd22b967f95fbc8e70b0299ea57721606db3e483155a056e2ef8adbae73295 |
| src/data_utils/colormnist.py | c0ec80ae6fde61719bb46a87013ab8e10f822ea5edfcebacd41a6b5e066224f6 |
| src/autoencoder_utils/colormnist_ae.py | 1930167cfd1deb53aa7a0b08c4187900fc102218e1503dd951ce69a04c5a55c5 |
| src/dens_ratio_utils/propensity.py | dd7b585ac8f9a8852c235ca3ce71f24f88ec8baed334a297e6f20d4319577c64 |
| src/kernel_utils/kernels.py | 5184b556266312963c755c637fba6d40273315d7ce86bb537fdb359ed59c44b8 |
| src/kernel_utils/linear_algebra.py | 0f3f7c71a66afadf69ef0ff8fdf44720a5e7bdf5e45d90d8811dcdf86d2efc48 |
| src/metric_utils/metrics.py | bdfacacb672a257246fdaa1182453e44409582aa05c299e454651983ef60bf64 |
| src/mmd_flow_utils/mmd_flow.py | 117962f78e33242a988f4d07c3284a18c906f428b19d267fe048a46200816247 |

## 16. Reproduction commands and limitations for rebuttal use

### 16.1 Run a fresh replication without overwriting completed outputs

From the deploy experiment directory, using the existing torchnn environment and local MNIST files:

~~~bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -u -B \
  colormnist_misspecification_severity.py \
  --device cuda --data-seeds 0 1 2 \
  --n-obs 20000 --n-particles 2000 --n-ref 2000 \
  --latent-dim 32 --n-landmarks 1024 --cv-max-n 2000 \
  --ae-iterations 3000 --flow-iterations 3000 \
  --cme-cutoffs 1.0 0.5 0.25 0.1 \
  --weight-factors 1.0 0.25 0.05 0.01 \
  --weight-threshold 0.5 --mass-modes raw preserve \
  --rhs-block-size 2048 --metric-block-size 256 \
  --n-projections 1000 --tag severity_replication
~~~

This example runs seeds sequentially on one GPU. The original array launcher ran them concurrently. The runner refuses an existing run directory. Choose a fresh tag for any later replication; do not overwrite the archived severity_paper runs.

The prepared analysis notebook explicitly reads severity_paper_seed0/1/2. Point a separate copy to a new tag before analyzing a new campaign. To re-execute the existing completed analysis:

~~~bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -m nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=python3 \
  ColorMNIST_Nuisance_Severity_Analysis.ipynb
~~~

Focused tests from the deploy repository:

~~~bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -m unittest discover \
  -s AdditonalExperiments/ColorMNIST_Nuisance_Misspecification \
  -p 'test_*.py' -v
~~~

There are 19 focused tests across the pilot and severity test files. Historical test success and completed-run evidence are recorded above; writing this report did not trigger another simulation campaign.

### 16.2 Claims supported by this experiment

A defensible empirical statement is:

> Under a controlled ColorMNIST nuisance-corruption protocol adapted from DoubleGen, the DR sampler retained near-baseline accuracy when either the conditional embedding or inverse-weight nuisance alone was corrupted. Corrupting both increased target and sample error. The strongest training-support restrictions produced substantial between-seed variability, which was retained in the reported uncertainty.

This is suggested wording for discussion, not accepted manuscript text. Any eventual rebuttal should state the 20k/2k configuration, three-seed arm-then-seed aggregation, actual corruption definitions, and the severe-case variability.

### 16.3 Claims not established and unresolved qualifications

- No oracle correctness is established for either fitted baseline nuisance.
- No same-sample empirical-process theorem is validated by this simulation.
- The AE is fitted on the same logged outcomes used for estimation.
- Fixed landmark rank, the relative eigenspace cutoff, ridge CV, finite precision, and finite flow iterations remain practical approximation choices.
- CME restrictions change support, sample size, and arm balance simultaneously; realized error need not be monotone in the cutoff.
- Very sparse arm-1 fits and a CV proxy based on one latent coordinate can amplify instability. A matched-sample-size ablation or ridge sensitivity study would be separate work.
- Signed empirical targets need not be probability embeddings or attainable by equal-weight particles.
- Preserving inverse-weight mass does not preserve DR target mass, eliminate shape distortion, or ensure improvement in all metrics.
- Three independent seed averages give limited precision. No confidence interval, significance claim, or universal ranking follows from these tables.
- Pixel SW2 and color W1 measure different aspects of quality; neither alone establishes recovery of the full interventional image law.
- Shared MNIST training banks do not test generalization to unseen digit templates.
- “Oracle floor” refers to a matched empirical comparison, not a strict lower bound.
- This is not a DoubleGen-model comparison, kernel-herding comparison, FID study, or computational acceleration study.
- No manuscript or reviewer-response file was revised as part of documenting these results.

## 17. References and related records

1. Alex Luedtke and Kenji Fukumizu (2025). **DoubleGen: Debiased Generative Modeling of Counterfactuals.** [arXiv:2509.16842v1](https://arxiv.org/abs/2509.16842v1). The specific nuisance-corruption precedent is in [Appendix M.1.1](https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1).
2. Heejung Bang and James M. Robins (2005). **Doubly Robust Estimation in Missing Data and Causal Inference Models.** *Biometrics*, 61(4), 962–973. [DOI](https://doi.org/10.1111/j.1541-0420.2005.00377.x).
3. Joseph D. Y. Kang and Joseph L. Schafer (2007). **Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data.** *Statistical Science*, 22(4), 523–539. [DOI](https://doi.org/10.1214/07-STS227).
4. [Experiment bibliography](references.bib), [experiment README](README.md), and [executed severity analysis notebook](ColorMNIST_Nuisance_Severity_Analysis.ipynb).
5. The separate [acceleration research note and implementation](../ColorMNIST_RFF_Caching/README.md).

