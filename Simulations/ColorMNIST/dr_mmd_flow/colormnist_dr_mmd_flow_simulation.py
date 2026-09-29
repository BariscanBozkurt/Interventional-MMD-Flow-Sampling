"""
ColorMNIST simulation: cross-fitted DR-CPME target + IMQ-kernel MMD flow (forward Euler).

Script version of
`NotebookExamples/Interventional_MMD_Flow/ColorMNIST_DR_IMQ_MMD_Flow_Euler_Arm1_RefParams_SplitAE.ipynb`,
step for step: the logged sample is split once into an autoencoder half and a causal
half, the frozen encoder defines the latent space, the cross-fitted DR weights define
the target embedding, and an IMQ MMD flow transports observed arm-a latents onto it.

Each repeat runs BOTH intervention arms by default (the evaluation protocol averages
over a in {0, 1}), sharing every arm-independent stage — the logged sample, the
split, the autoencoder, the DR subsample with its encoded support, the cross-fitted
propensity and the per-fold ridge lambdas — so the two arms are paired on identical
nuisances and only the importance weights, DR weight vector, flow and evaluation are
arm-specific.

One CSV row per (data_seed, repeat, arm), holding the DGP constants, split sizes,
nuisance diagnostics (on-arm ESS, sum of DR weights, per-fold lambdas), flow
diagnostics (bandwidth, step, objective at start and end) and the pixel-space
evaluation against fresh oracle draws (colour-ratio law, latent SW2, pixel SW1, SW2,
RBF MMD^2) for
the initial particles, the flowed particles, and the oracle-vs-oracle floor. Every
cheap distributional floor also gets a `*_floor_matched` twin computed on both oracle
draws subsampled to the count the row's own sample was scored at, because those metrics
are sample-size biased and a particle-count sweep would otherwise move the rows while
the floor stood still.

Evaluation protocol (2026-09-15, matched to Dance et al.): no autoencoder split
(--ae_split none: the AE trains on the whole logged sample and the DR support is
every row), Nystrom solver at every n, --n_particles 2000 per arm scored on
--eval_particles 2000 against a FIXED --n_ref 2000 oracle draw (seed offset +9),
floor = a second 2000-draw (+10) against the first, plus the decoder round-trip of
that second draw; FID with both scorers of `metric_utils.fid`; 3 data seeds, no
repeats. Every row is therefore a 2000-vs-2000 number at every n_obs.

Seed scheme, mirroring the notebook's SEED-offset convention:
`data_seed` fixes the logged sample; `pipeline_seed = 1000 * data_seed + repeat`
fixes everything downstream (the split, the autoencoder, the DR subsample, the
particle pick and the oracle evaluation draws). The arm-1 rows at the default
arguments follow the notebook's seed path exactly.

Launched by `submit_colormnist_dr_mmd_flow_parallel.sh`; single-run example:

    python colormnist_dr_mmd_flow_simulation.py --n_obs 20000 --data_seeds 0 --n_repeats 3
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

from data_utils import (color_ratio, load_digit_pools, sample_interventional,
                        sample_observed, to_image_range, to_model_range)
from autoencoder_utils import encode_in_batches, train_conv_ae
from kernel_utils import IMQKernel, RBF, imq_c_from_median
from causality_utils import (cpme_dr_weights, cpme_dr_weights_crossfit,
                             cpme_dr_weights_crossfit_nystrom, cpme_dr_weights_nystrom,
                             cpme_dr_weights_crossfit_projected_nystrom,
                             cpme_dr_weights_projected_nystrom,
                             lambda_n_half,
                             make_crossfit_fold_ids, select_regularization)
from dens_ratio_utils import LogisticPropensityModel, crossfit_propensity
from mmd_flow_utils import mmd_gradient_flow, squared_mmd
from metric_utils import CleanFIDScorer, LegacyFIDScorer, sliced_wasserstein

COLUMNS = [
    "tag", "device", "init", "ae_split", "solver", "dr_fit", "lambda_rule", "n_landmarks",
    "data_seed", "repeat", "pipeline_seed", "target_a",
    "n_obs", "w_conf", "tau", "alpha", "k_mask",
    "n_ae", "n_causal", "n_arm_causal", "n_dr", "n_folds",
    "n_particles_requested", "n_particles", "n_flow_eval", "n_ref", "latent_dim",
    "ae_iterations", "flow_iterations",
    "recon_mse_train", "recon_mse_heldout",
    "ess_on_arm", "n_on_arm", "max_weight", "sum_v", "sum_abs_v", "fold_lambdas",
    "c_imq", "step_size", "objective_initial", "objective_final",
    "latent_sw2_initial", "latent_sw2_flow", "latent_sw2_floor",
    "colour_mean_initial", "colour_std_initial", "colour_w2_initial",
    "sw1_initial", "sw2_initial", "mmd2_initial",
    "colour_mean_flow", "colour_std_flow", "colour_w2_flow", "sw1_flow", "sw2_flow", "mmd2_flow",
    "colour_mean_floor", "colour_std_floor", "colour_w2_floor",
    "sw1_floor", "sw2_floor", "mmd2_floor",
    "colour_mean_floor_matched", "colour_std_floor_matched", "colour_w2_floor_matched",
    "sw1_floor_matched", "sw2_floor_matched", "mmd2_floor_matched",
    "latent_sw2_floor_matched", "n_floor_matched", "n_floor_matched_latent",
    "colour_mean_roundtrip", "colour_std_roundtrip", "colour_w2_roundtrip",
    "sw1_roundtrip", "sw2_roundtrip", "mmd2_roundtrip",
    "colour_mean_oracle", "colour_std_oracle",
    "fid_legacy_initial", "fid_legacy_flow", "fid_legacy_floor", "fid_legacy_roundtrip",
    "fid_clean_initial", "fid_clean_flow", "fid_clean_floor", "fid_clean_roundtrip",
    "seconds_ae", "seconds_nuisance", "seconds_flow", "seconds_fid", "seconds_repeat",
]


def write_rows(out_path: Path, rows: list) -> None:
    """Atomically rewrite the task CSV (tmp + rename).

    A plain open(out_path, "w") truncates the previous good version BEFORE writing,
    so a failure mid-write (disk quota, kill) destroys already-completed rows — this
    happened in production with EDQUOT. Writing to a task-unique tmp and renaming
    makes every failure leave the last complete CSV untouched.
    """
    tmp = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise



PARTICLE_INDICES: dict = {}


def particle_generator(args, seed: int, offset: int = 0) -> "torch.Generator":
    """CPU generator for the starting-particle draw; see --particle_seed."""
    base = seed if args.particle_seed is None else args.particle_seed + seed
    return torch.Generator().manual_seed(base + offset)


def write_particle_indices(out_path: Path) -> None:
    """Save the logged-sample rows used as starting particles, for cross-implementation reuse."""
    if PARTICLE_INDICES:
        np.savez(out_path.with_name(out_path.stem + "_particles.npz"), **PARTICLE_INDICES)

def resolve_solver(args) -> str:
    """Which ridge solver this configuration uses.

    'auto' keys off n_obs rather than support size: a split run and a no-split run at
    the same n_obs then agree on the solver, so comparing them never mixes a
    modelling change with a numerical one.
    """
    if args.solver != "auto":
        return args.solver
    return "exact" if args.n_obs <= args.exact_max_n_obs else "nystrom"


def gaussian_init(reference, n_particles: int, mode: str,
                  generator) -> "torch.Tensor":
    """Particles from N(0, I) or the reference sample's empirical N(mu, Sigma).

    The moment-matched draw uses the CENTRED DATA MATRIX rather than a D x D
    Cholesky: for centred Z (n, D) and g ~ N(0, I_n), Z^T g / sqrt(n - 1) has
    exactly the sample covariance. That costs O(n D), needs no jitter, and stays
    well defined when n < D (the empirical Gaussian is then degenerate on the
    data span, which is the correct object). `reference` is the flow's own target
    support, so both draws live in the coordinates the flow operates in.
    """
    n, dim = reference.shape
    if mode == "normal":
        return torch.randn(n_particles, dim, generator=generator,
                           device=reference.device, dtype=reference.dtype)
    mean = reference.mean(dim=0, keepdim=True)
    weights = torch.randn(n_particles, n, generator=generator,
                          device=reference.device, dtype=reference.dtype)
    return mean + weights @ (reference - mean) / (n - 1) ** 0.5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=20000,
                        help="logged sample size n")
    parser.add_argument("--n_particles", type=int, default=2000,
                        help="flow particles M (capped at the on-arm causal-split size)")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="size of the fixed oracle reference draw every row is scored "
                             "against, and of the second draw that is the floor; "
                             "independent of M and of n_obs")
    parser.add_argument("--eval_particles", type=int, default=2000,
                        help="score the decoded initial/flow rows on a fixed random subset "
                             "of this many particles (the whole set when M <= this); keeps "
                             "the evaluated count at the reference count when M is larger")
    parser.add_argument("--n_dr", type=int, default=None,
                        help="optional cap on the DR support, kept for reproducing the "
                             "pre-2026-09 runs (--solver exact --n_dr 4000). Unset, the "
                             "support is every eligible row.")
    parser.add_argument("--ae_fraction", type=float, default=0.5,
                        help="fraction of the logged sample used to train the autoencoder")
    parser.add_argument("--target_arms", type=int, nargs="+", default=[0, 1],
                        choices=(0, 1), help="intervention arms run in each repeat")
    parser.add_argument("--w_conf", type=float, default=10.0,
                        help="confounding strength w in e(1|x) = sigmoid(w (x - 1/2))")
    parser.add_argument("--tau", type=float, default=0.08, help="foreground mask threshold")
    parser.add_argument("--alpha", type=float, default=0.2, help="fixed green channel")
    parser.add_argument("--k_mask", type=float, default=20.0, help="foreground mask sharpness")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="pipeline repeats per data seed")
    parser.add_argument("--fid", type=str, default="both",
                        choices=("none", "legacy", "clean", "both"),
                        help="FID of the decoded initial / flowed particles and of a second "
                             "oracle draw (floor) against the oracle draw, with the scorers "
                             "of metric_utils.fid: 'legacy' reproduces the FID column of the "
                             "Dance et al. (2026) tables (Frechet distance on the GPU, their "
                             "ColorMNIST convention), 'clean' is standard Clean-FID. "
                             "'clean' needs the weights cached in ~/.cache/torch/clean-fid.")
    parser.add_argument("--fid_workers", type=int, default=8,
                        help="threads for Clean-FID's per-image resize (results identical)")
    parser.add_argument("--particle_seed", type=int, default=None,
                        help="base seed for the STARTING PARTICLE draw, kept independent of the "
                             "rest of the pipeline so another implementation can reproduce the "
                             "same initial cloud. The draw uses seed particle_seed + 1000 * "
                             "data_seed + repeat on a CPU generator (CPU so the stream does not "
                             "depend on the device). Default None keeps the historical derivation "
                             "from the pipeline seed. The chosen rows are also written verbatim "
                             "to <csv stem>_particles.npz, which is the exact way to share them: "
                             "it needs no RNG agreement at all. CAVEAT: those indices address "
                             "THIS run's logged sample, so they are portable only to a run that "
                             "shares the data draw. ColorMNIST draws its data with a CUDA "
                             "generator and CelebA from a cached file, so a different device or "
                             "cache yields a different sample; to hand starting particles to an "
                             "implementation that generates its own data, export the particle "
                             "IMAGES rather than these indices.")
    parser.add_argument("--init", type=str, default="observed",
                        choices=("observed", "normal", "moment"),
                        help="initial particles: 'observed' = the logged arm (default), "
                             "'normal' = N(0, I), 'moment' = the empirical N(mu, Sigma) "
                             "of the flow's target support. Both Gaussians are drawn in "
                             "the coordinates the flow operates in. NOTE: the *_initial "
                             "metric columns describe the run's actual starting cloud, so "
                             "under a Gaussian init they are not the observed-arm baseline "
                             "-- take that from the matching --init observed run.")
    parser.add_argument("--ae_split", type=str, default="none", choices=("half", "none"),
                        help="'half' trains the autoencoder on a disjoint half of the "
                             "logged sample (the reference design); 'none' trains it on "
                             "every row and lets the causal stage reuse them, trading the "
                             "feature map's independence for twice the eligible data.")
    parser.add_argument("--solver", type=str, default="projected_nystrom",
                        choices=("auto", "exact", "nystrom", "projected_nystrom"),
                        help="how each fold's CME ridge system is solved. 'auto' (default) "
                             "solves exactly while that stays affordable and switches to "
                             "Nystrom above --exact_max_n_obs. The DR support is EVERY "
                             "eligible row either way, so the solver is a tractability "
                             "detail, not a method variant, and should not appear as a "
                             "factor in results tables. The threshold is keyed on n_obs "
                             "rather than support size, so split and no-split runs at the "
                             "same n_obs always share a solver and their comparison is "
                             "never confounded by it.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly; above it a "
                             "fold would be a (4/5 n_obs)^2 float64 system")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold")
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"),
                        help="how the DR nuisances are fitted. DEFAULT CHANGED 2026-09-28 to 'nosplit', which fits the CME and the propensity on ALL rows; 'crossfit' (the previous default) fits both out of fold. The same-sample notes license reusing observations for the NUISANCES. NOTE the filename token is deliberately still keyed on 'crossfit': a 'nosplit' run appends _drnosplit and a 'crossfit' run appends nothing, exactly as before the default flipped, so every historical filename is still reproduced byte for byte and no new run can overwrite an archived one. The dr_fit CSV column is the authoritative record. "
                             "historical behaviour) fits the CME and the propensity out of "
                             "fold; 'nosplit' fits both on ALL rows. NOTE this is a DIFFERENT "
                             "axis from --ae_split, which controls whether the autoencoder "
                             "sees the causal rows. The same-sample notes license reusing "
                             "observations for NUISANCES but explicitly not for an encoder "
                             "selected on those rows, so --ae_split half --dr_fit nosplit is "
                             "the combination their theory covers.")
    parser.add_argument("--lambda_rule", type=str, default="cv",
                        choices=("cv", "n_half"),
                        help="ridge selection. 'cv' (default, historical) cross-validates per "
                             "training block; 'n_half' uses lambda = c * n^-1/2, the rate the "
                             "same-sample theory gives as sufficient, and keeps the penalty "
                             "matched between --dr_fit arms.")
    parser.add_argument("--lambda_constant", type=float, default=1.0,
                        help="the constant c in --lambda_rule n_half")
    parser.add_argument("--snapshot_every", type=int, default=None,
                        help="store the particle cloud every this many iterations so one run "
                             "yields a whole iteration curve. This sampler uses a FIXED Euler "
                             "step, so a snapshot at step s is exactly a run of s steps -- "
                             "unlike the cosine-scheduled Riesz simulations.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--ae_iterations", type=int, default=3000)
    parser.add_argument("--flow_iterations", type=int, default=3000)
    parser.add_argument("--tag", type=str, default="refparams")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--mnist_root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available")
    return parser.parse_args()


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    suffix += "_nosplit" if args.ae_split == "none" else ""
    # Tokens appended only off-default so historical filenames are reproduced exactly.
    # `_dr*` is deliberately distinct from the `_nosplit` token just above: --ae_split
    # and --dr_fit are different ablations and their filenames must not collide.
    if args.dr_fit != "crossfit":
        suffix += f"_dr{args.dr_fit}"
    if args.lambda_rule != "cv":
        suffix += f"_lam{args.lambda_rule}"
    # One token per explicitly requested solver, so an --solver sweep cannot have two
    # cells write the same filename. Keyed on args.solver (not the RESOLVED solver), so
    # historical runs that passed --solver auto keep their exact filenames.
    if args.solver == "projected_nystrom":
        suffix += "_projnys"
    elif args.solver == "exact":
        suffix += "_exact"
    elif args.solver == "nystrom":
        suffix += "_nys"
    if args.flow_iterations != 3000:
        suffix += f"_it{args.flow_iterations}"
    name = (f"colormnist_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
            f"_w{args.w_conf:g}_a{args.alpha:g}_k{args.k_mask:g}_tau{args.tau:g}"
            f"_arms{arms}_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def evaluate_images(images_01: torch.Tensor, reference_01: torch.Tensor,
                    kernel_pixel: RBF) -> dict:
    """Colour law plus flat-pixel SW1 / SW2 / unbiased RBF MMD^2 against the reference.

    `colour_w2` is the one-dimensional Wasserstein-2 distance between the colour-ratio
    laws (exact for equal counts through the sorted-projection estimator), the
    "Color W2" column of the Dance et al. (2026) ColorMNIST table."""
    flatten = lambda images: images.reshape(images.shape[0], -1)
    ratio = color_ratio(images_01)
    ratio_reference = color_ratio(reference_01)
    return {
        "colour_mean": float(ratio.mean()), "colour_std": float(ratio.std()),
        "colour_w2": sliced_wasserstein(ratio[:, None], ratio_reference[:, None], p=2,
                                        n_projections=1, seed=0).item(),
        "sw1": sliced_wasserstein(flatten(images_01), flatten(reference_01), p=1, seed=0).item(),
        "sw2": sliced_wasserstein(flatten(images_01), flatten(reference_01), p=2, seed=0).item(),
        "mmd2": squared_mmd(kernel_pixel, flatten(images_01), flatten(reference_01),
                            unbiased=True).item(),
    }


def subsample_rows(sample: torch.Tensor, k: int,
                   generator: "torch.Generator") -> torch.Tensor:
    """A random k-row subset of `sample`, or the whole sample when it has at most k rows.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, ...)
    k : int
        Target row count.
    generator : torch.Generator
        CPU generator; the permutation is drawn on the CPU so the subset does not
        depend on the device.

    Returns
    -------
    torch.Tensor of shape (min(n, k), ...)
    """
    if sample.shape[0] <= k:
        return sample
    pick = torch.randperm(sample.shape[0], generator=generator)[:k]
    return sample[pick.to(sample.device)]


def matched_floor(images_01: torch.Tensor, reference_01: torch.Tensor,
                  kernel_pixel: RBF, k: int, seed: int) -> dict:
    """The oracle-vs-oracle floor recomputed at the count this row's flow was scored at.

    Sliced Wasserstein, the colour-ratio W2 and the unbiased MMD^2 are all SAMPLE-SIZE
    BIASED: two independent draws of the SAME law score further apart at small counts
    than at large ones. `sliced_wasserstein` truncates to the smaller of its two inputs,
    so a flow row is effectively scored at k = min(n_scored, n_ref) while the plain
    floor is scored at the full n_ref. Sweeping the particle count then moves the flow
    rows while the floor stays put, which reads as a spurious "more particles is
    better". Subsampling BOTH floor inputs to the same k removes that gap; the subsets
    come from a fixed seed derived from the pipeline seed, so the columns are
    reproducible.

    Parameters
    ----------
    images_01, reference_01 : torch.Tensor of shape (n, 3, 28, 28)
        The two independent oracle draws, in [0, 1].
    kernel_pixel : RBF
        Pixel kernel with its bandwidth already locked on the reference draw.
    k : int
        Effective count at which this row's flow sample was scored.
    seed : int
        Seed of the CPU generator drawing both subsets.

    Returns
    -------
    dict
        The keys of `evaluate_images`, to be suffixed by the caller.
    """
    generator = torch.Generator().manual_seed(seed)
    return evaluate_images(subsample_rows(images_01, k, generator),
                           subsample_rows(reference_01, k, generator), kernel_pixel)


def run_arm(target_a: int, ae, pools: list, data: torch.Tensor,
            causal_idx: torch.Tensor, A_obs: torch.Tensor, A_s: torch.Tensor,
            Z_s: torch.Tensor, e1: torch.Tensor, K_x, K_a, gram, gram_x, solver: str,
            fold_ids: np.ndarray, fold_lambdas: dict, args: argparse.Namespace,
            seed: int, device: torch.device, fid_scorers: dict = None,
            nosplit_lambda: float = None) -> dict:
    """Arm-specific half of one repeat: DR weights, flow, and oracle evaluation."""
    n_dr = len(A_s)
    e_target = e1 if target_a == 1 else 1.0 - e1
    w_hat = (A_s.cpu() == target_a).double() / e_target
    on_arm = w_hat[A_s.cpu() == target_a]
    ess = float(on_arm.sum() ** 2 / (on_arm ** 2).sum())
    if solver in ("nystrom", "projected_nystrom"):
        # K_pi[i, j] = K_X[i, j] * 1{a_i == a}: the target action is deterministic
        on_arm_flag = (A_s.cpu() == target_a).double()
        gram_pi = lambda rows, cols: gram_x(rows, cols) * on_arm_flag[rows][:, None]
        # `projected_nystrom` restricts the CME ridge regression to the landmark-spanned
        # subspace and builds the DR right-hand side only at the landmarks. That is the
        # estimator the imported consistency rate is stated for, and it touches an
        # m x |H| cross-kernel instead of a |T| x |H| one. `nystrom` applies an
        # approximate Gram inverse to the exact full right-hand side -- a different
        # estimator, kept as the baseline that produced every earlier table.
        projected = solver == "projected_nystrom"
        if args.dr_fit == "nosplit":
            solve = (cpme_dr_weights_projected_nystrom if projected
                     else cpme_dr_weights_nystrom)
            v_dr = solve(gram, gram_pi, w_hat, nosplit_lambda,
                         n_landmarks=args.n_landmarks,
                         generator=torch.Generator().manual_seed(seed)).float().to(device)
        else:
            solve = (cpme_dr_weights_crossfit_projected_nystrom if projected
                     else cpme_dr_weights_crossfit_nystrom)
            v_dr = solve(gram, gram_pi, w_hat, fold_ids, fold_lambdas,
                         n_landmarks=args.n_landmarks,
                         generator=torch.Generator().manual_seed(seed)).float().to(device)
    else:
        K_a_target = (A_s == target_a).double().cpu()[:, None].expand(n_dr, n_dr).contiguous()
        if args.dr_fit == "nosplit":
            v_dr = cpme_dr_weights(K_x, K_a, K_a_target, w_hat,
                                   nosplit_lambda).float().to(device)
        else:
            v_dr = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_hat, fold_ids,
                                            fold_lambdas).float().to(device)

    t0 = time.time()
    arm_causal = causal_idx[A_obs[causal_idx] == target_a]
    Z_arm = encode_in_batches(ae.encode, data[arm_causal])
    n_particles = min(args.n_particles, Z_arm.shape[0])
    # CPU generator: the CUDA stream is device-dependent, so a GPU draw cannot be
    # reproduced by another implementation (or on another card).
    pick = torch.randperm(Z_arm.shape[0],
                          generator=particle_generator(args, seed, 1))[:n_particles].to(device)
    PARTICLE_INDICES[f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"] = (
        arm_causal[pick].cpu().numpy())
    if args.init == "observed":
        initial_particles = Z_arm[pick].clone()
    else:
        initial_particles = gaussian_init(
            Z_s, n_particles, args.init,
            torch.Generator(device=device).manual_seed(seed + 3))

    c_imq = imq_c_from_median(Z_s)
    kernel = IMQKernel(c=c_imq, beta=-0.5)
    step_size = 0.15 * c_imq ** 2
    particles, history = mmd_gradient_flow(kernel, target_samples=Z_s,
                                           initial_particles=initial_particles,
                                           step_size=step_size,
                                           n_iterations=args.flow_iterations,
                                           target_weights=v_dr,
                                           snapshot_every=args.snapshot_every,
                                           log_every=500, verbose=False)
    seconds_flow = time.time() - t0

    eval_idx = torch.arange(n_particles, device=particles.device)
    if args.eval_particles is not None and n_particles > args.eval_particles:
        eval_idx = torch.randperm(n_particles, generator=torch.Generator(device=device).manual_seed(seed + 11),
                                  device=device)[:args.eval_particles]
    with torch.no_grad():
        decoded_initial = to_image_range(ae.decode(initial_particles[eval_idx]))
        decoded_flowed = to_image_range(ae.decode(particles[eval_idx]))
    _, Y_oracle = sample_interventional(pools, args.n_ref, a=target_a, tau=args.tau,
                                        alpha=args.alpha, k_mask=args.k_mask,
                                        generator=torch.Generator(device=device).manual_seed(seed + 9))
    _, Y_oracle_2 = sample_interventional(pools, args.n_ref, a=target_a, tau=args.tau,
                                          alpha=args.alpha, k_mask=args.k_mask,
                                          generator=torch.Generator(device=device).manual_seed(seed + 10))
    # Decoder ceiling: the second oracle draw passed through the autoencoder. Its
    # distance to the first draw is what a PERFECT flow could reach through this
    # decoder; the floor row (raw second draw) is what a perfect sampler could reach.
    with torch.no_grad():
        decoded_oracle = to_image_range(ae.decode(encode_in_batches(
            ae.encode, to_model_range(Y_oracle_2))))
    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(Y_oracle.reshape(len(Y_oracle), -1))  # lock the bandwidth on the reference

    # Latent SW2: particles against the ENCODED oracle draw, in the raw AE latent the
    # flow operates in (ColorMNIST does not standardise its latents). The floor encodes
    # the second oracle draw, so it is the same quantity computed on two independent
    # samples of the true interventional law.
    with torch.no_grad():
        Z_oracle = encode_in_batches(ae.encode, to_model_range(Y_oracle))
        Z_oracle_2 = encode_in_batches(ae.encode, to_model_range(Y_oracle_2))
    latent_sw2 = {
        "initial": sliced_wasserstein(initial_particles, Z_oracle, p=2, seed=0).item(),
        "flow": sliced_wasserstein(particles, Z_oracle, p=2, seed=0).item(),
        "floor": sliced_wasserstein(Z_oracle_2, Z_oracle, p=2, seed=0).item(),
    }
    # Count-matched floors (see `matched_floor`). The latent metrics score EVERY
    # particle while the pixel metrics score the --eval_particles subset, so the two
    # effective counts differ whenever M exceeds --eval_particles and each floor takes
    # its own k.
    k_pixel = min(len(eval_idx), len(Y_oracle))
    k_latent = min(n_particles, len(Z_oracle))
    latent_generator = torch.Generator().manual_seed(seed + 13)
    latent_sw2["floor_matched"] = sliced_wasserstein(
        subsample_rows(Z_oracle_2, k_latent, latent_generator),
        subsample_rows(Z_oracle, k_latent, latent_generator), p=2, seed=0).item()

    ratio_oracle = color_ratio(Y_oracle)
    row = {
        "target_a": target_a,
        "n_arm_causal": int(len(arm_causal)),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "n_flow_eval": int(len(eval_idx)), "n_ref": int(len(Y_oracle)),
        "ess_on_arm": ess, "n_on_arm": int(len(on_arm)), "max_weight": float(w_hat.max()),
        "sum_v": float(v_dr.sum()), "sum_abs_v": float(v_dr.abs().sum()),
        "c_imq": float(c_imq), "step_size": float(step_size),
        "objective_initial": float(history["squared_mmd"][0]),
        "objective_final": float(history["squared_mmd"][-1]),
        "latent_sw2_initial": latent_sw2["initial"],
        "latent_sw2_flow": latent_sw2["flow"],
        "latent_sw2_floor": latent_sw2["floor"],
        "latent_sw2_floor_matched": latent_sw2["floor_matched"],
        "n_floor_matched": int(k_pixel), "n_floor_matched_latent": int(k_latent),
        "colour_mean_oracle": float(ratio_oracle.mean()),
        "colour_std_oracle": float(ratio_oracle.std()),
        "seconds_flow": round(seconds_flow, 1),
    }
    for name, images in (("initial", decoded_initial), ("flow", decoded_flowed),
                         ("floor", Y_oracle_2), ("roundtrip", decoded_oracle)):
        metrics = evaluate_images(images, Y_oracle, kernel_pixel)
        row.update({f"{key}_{name}": value for key, value in metrics.items()})
    matched = matched_floor(Y_oracle_2, Y_oracle, kernel_pixel, k_pixel, seed + 12)
    row.update({f"{key}_floor_matched": value for key, value in matched.items()})
    t0 = time.time()
    for impl, scorer in (fid_scorers or {}).items():
        oracle_features = scorer.features(Y_oracle, image_range="zero_one")
        for name, images in (("initial", decoded_initial), ("flow", decoded_flowed),
                             ("floor", Y_oracle_2), ("roundtrip", decoded_oracle)):
            row[f"fid_{impl}_{name}"] = scorer.distance_from_features(
                scorer.features(images, image_range="zero_one"), oracle_features)
    row["seconds_fid"] = round(time.time() - t0, 1)
    return row


def run_repeat(pools: list, data: torch.Tensor, X_obs: torch.Tensor, A_obs: torch.Tensor,
               args: argparse.Namespace, data_seed: int, repeat: int,
               device: torch.device, fid_scorers: dict = None) -> list:
    """One pipeline pass on a fixed logged sample; returns one CSV row per arm."""
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- split: autoencoder half / causal half (disjoint) ------------------------------
    if args.ae_split == "none":
        ae_idx = causal_idx = torch.arange(args.n_obs, device=device)
        n_ae = args.n_obs
    else:
        n_ae = int(round(args.ae_fraction * args.n_obs))
        split = torch.randperm(args.n_obs,
                               generator=torch.Generator(device=device).manual_seed(seed + 2),
                               device=device)
        ae_idx, causal_idx = split[:n_ae], split[n_ae:]

    # -- autoencoder feature map on the AE half only -----------------------------------
    t0 = time.time()
    ae, _ = train_conv_ae(data[ae_idx], latent_dim=args.latent_dim,
                          n_iterations=args.ae_iterations, batch_size=128,
                          learning_rate=1e-3, device=device,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          log_every=max(1, args.ae_iterations // 2), verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        mse_train = float((ae(data[ae_idx][:2000]) - data[ae_idx][:2000]).pow(2).mean())
        mse_heldout = float((ae(data[causal_idx][:2000]) - data[causal_idx][:2000]).pow(2).mean())
    seconds_ae = time.time() - t0

    # -- shared nuisances on a causal-half subsample -----------------------------------
    t0 = time.time()
    solver = resolve_solver(args)
    # The support is every eligible row; --n_dr caps it only when asked for explicitly.
    if args.n_dr is not None and args.n_dr < len(causal_idx):
        pick_dr = torch.randperm(len(causal_idx),
                                 generator=torch.Generator(device=device).manual_seed(seed + 5),
                                 device=device)[:args.n_dr]
        subsample = causal_idx[pick_dr]
    else:
        subsample = causal_idx
    n_dr = len(subsample)
    X_s, A_s = X_obs[subsample], A_obs[subsample]
    Z_s = encode_in_batches(ae.encode, data[subsample])

    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    torch.manual_seed(seed)
    if args.dr_fit == "nosplit":
        # Fit once on all rows and predict on them. NOT crossfit_propensity with one
        # fold: that silently returns a constant 0.5, which would leave the two arms
        # differing only in the CME nuisance.
        model = LogisticPropensityModel()
        model.fit(X_s.unsqueeze(1).double().cpu(), A_s.double().cpu())
        e1 = model.predict(X_s.unsqueeze(1).double().cpu()).reshape(-1).double()
    else:
        e1 = crossfit_propensity(X_s.unsqueeze(1).double().cpu(), A_s.double().cpu(), fold_ids)

    # The RBF lengthscale is locked once on a bounded subsample; the median over the
    # full n_dr^2 pairs is both unaffordable and unnecessary at n_dr = 20000.
    kernel_x = RBF(use_length_scale_heuristic=True)
    X_cpu = X_s.unsqueeze(1).double().cpu()
    A_cpu = A_s.cpu()
    anchor = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:4000]
    _ = kernel_x(X_cpu[anchor])

    def gram_x(rows, cols):
        "K_X block alone; K_pi = K_X * 1{a_i == a} needs it without the action factor."
        return kernel_x(X_cpu[rows], X_cpu[cols])

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_dr^2 Gram is never formed."
        return gram_x(rows, cols) * (A_cpu[rows][:, None] == A_cpu[cols][None, :]).double()

    # lambda does not need the full training block, which would itself be an n^3 solve
    if args.lambda_rule == "n_half":
        nosplit_lambda = lambda_n_half(n_dr, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    else:
        fold_lambdas = {}
        for fold in np.unique(fold_ids):
            train = np.where(fold_ids != fold)[0]
            pick = train[torch.randperm(len(train),
                                        generator=torch.Generator().manual_seed(0))[:2000].numpy()]
            fold_lambdas[int(fold)], _ = select_regularization(gram(pick, pick),
                                                               Z_s[pick, 0].double().cpu(), seed=0)
        nosplit_lambda = None
        if args.dr_fit == "nosplit":
            pick = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:2000].numpy()
            nosplit_lambda, _ = select_regularization(gram(pick, pick),
                                                      Z_s[pick, 0].double().cpu(), seed=0)
    K_x = kernel_x(X_cpu) if solver == "exact" else None
    K_a = (A_cpu[:, None] == A_cpu[None, :]).double() if solver == "exact" else None
    seconds_nuisance = time.time() - t0

    # -- per-arm: DR weights, flow, evaluation -----------------------------------------
    shared = {
        "tag": args.tag, "device": str(device), "init": args.init,
        "ae_split": args.ae_split,
        "solver": solver, "dr_fit": args.dr_fit, "lambda_rule": args.lambda_rule,
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "n_obs": args.n_obs, "w_conf": args.w_conf,
        "tau": args.tau, "alpha": args.alpha, "k_mask": args.k_mask,
        "n_ae": int(n_ae), "n_causal": int(len(causal_idx)),
        "n_dr": int(n_dr), "n_folds": args.n_folds, "latent_dim": args.latent_dim,
        "ae_iterations": args.ae_iterations, "flow_iterations": args.flow_iterations,
        "recon_mse_train": mse_train, "recon_mse_heldout": mse_heldout,
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        "seconds_ae": round(seconds_ae, 1), "seconds_nuisance": round(seconds_nuisance, 1),
    }
    rows = []
    for target_a in args.target_arms:
        row = dict(shared)
        row.update(run_arm(target_a, ae, pools, data, causal_idx, A_obs, A_s, Z_s,
                           e1, K_x, K_a, gram, gram_x, solver, fold_ids,
                           fold_lambdas, args, seed, device, fid_scorers,
                           nosplit_lambda))
        rows.append(row)
    seconds_repeat = round(time.time() - t_start, 1)
    for row in rows:
        row["seconds_repeat"] = seconds_repeat
    return rows


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = csv_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   output: {out_path}", flush=True)

    # Read MNIST straight from the repo's data/ folder; torchvision only downloads
    # when the files are absent, and compute nodes may have no internet, so fail
    # loudly here rather than let a missing folder turn into a download attempt.
    if not (Path(args.mnist_root) / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
        sys.exit(f"MNIST not found under {args.mnist_root}/MNIST/raw — "
                 "copy the repo's data/ folder there (no download on compute nodes).")
    pools = load_digit_pools(root=args.mnist_root, device=device)
    fid_scorers = {}
    if args.fid in ("legacy", "both"):
        fid_scorers["legacy"] = LegacyFIDScorer(device=device, batch_size=64,
                                                distance_device=device)
    if args.fid in ("clean", "both"):
        fid_scorers["clean"] = CleanFIDScorer(device=device, batch_size=64,
                                              num_workers=args.fid_workers, allow_download=False)

    rows = []
    for data_seed in args.data_seeds:
        generator = torch.Generator(device=device).manual_seed(data_seed)
        X_obs, A_obs, Y_obs = sample_observed(pools, args.n_obs, w_conf=args.w_conf,
                                              tau=args.tau, alpha=args.alpha,
                                              k_mask=args.k_mask, generator=generator)
        data = to_model_range(Y_obs)
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(pools, data, X_obs, A_obs, args, data_seed, repeat, device,
                                  fid_scorers)
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            write_particle_indices(out_path)
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']}] "
                      f"colour flow {row['colour_mean_flow']:.3f} "
                      f"(initial {row['colour_mean_initial']:.3f}, oracle {row['colour_mean_oracle']:.3f})   "
                      f"SW2 flow {row['sw2_flow']:.4f} (floor {row['sw2_floor']:.4f})   "
                      + (f"FID legacy flow {row['fid_legacy_flow']:.1f} "
                         f"(initial {row['fid_legacy_initial']:.1f}, floor {row['fid_legacy_floor']:.1f})   "
                         if "fid_legacy_flow" in row else "")
                      + f"ESS {row['ess_on_arm']:.0f}/{row['n_on_arm']}   "
                      f"{row['seconds_repeat']:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
