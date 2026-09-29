"""
dSprites simulation: continuous actions, cross-fitted DR-CPME target, Riesz MMD flow (AdamW).

Script version of
`NotebookExamples/Interventional_MMD_Flow/dSprites_Continuous_Action_CrossFit_DR_Riesz_MMD_Flow_AdamW.ipynb`
with one deliberate change: the denoising autoencoder is trained on LOGGED IMAGES from
a split disjoint from the causal estimate, not on the sprite bank. The bank is the
entire noise-free outcome manifold, so training on it hands the feature map oracle
knowledge; here the logged sample is split once into an autoencoder half and a causal
half, the AE sees only the first (its denoising targets are the noisy logged images
themselves), and the nuisances, DR weights, flow particles and evaluation all live on
the second. The bank remains on the EVALUATION side only: denoise-MSE and probe
diagnostics, off-manifold distances, the snapped position law, and the clean-oracle
floor.

Everything else follows the notebook: estimated continuous-action importance weights
(known target policy over a cross-fitted logging-density fit), RBF product kernel on
(X, A) with a resampled target-action kernel mean, cross-fitted one-step DR weights,
and a dense Riesz flow under AdamW with linear warm-up on standardized latents.

One CSV row per (data_seed, repeat). Metrics (energy distance, SW1, SW2 in flat pixel
space, mean +/- std over --n_ref fresh interventional reference draws of matched size)
for: the logged images, their AE round-trip, the decoded flow output, the noisy-oracle
floor, and the clean-sprite floor (the appropriate one for a denoising decoder).
The snapped position law (energy/SW1/SW2 on grid positions) is reported alongside,
with position means and off-manifold diagnostics.

Seed scheme: `data_seed` fixes the logged sample; `pipeline_seed = 1000 * data_seed +
repeat` fixes the split, the autoencoder and the flow; reference draws use seeds
100000 + 1000 * pipeline_seed + j, disjoint from every data seed.

Launched by `submit_dsprites_dr_riesz_flow_parallel.sh`; single-run example:

    python dsprites_dr_riesz_flow_simulation.py --n_obs 2000 --data_seeds 0 --n_repeats 3
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

from causality_utils import (cpme_dr_weights, cpme_dr_weights_crossfit,
                             cpme_dr_weights_crossfit_nystrom,
                             cpme_dr_weights_crossfit_projected_nystrom,
                             cpme_dr_weights_nystrom, cpme_dr_weights_projected_nystrom,
                             lambda_n_half, make_crossfit_fold_ids, select_regularization)
from data_utils import load_dsprite_bank, sample_cpme_dsprite_data
from dens_ratio_utils import (crossfit_logging_policy_log_prob,
                              fit_linear_gaussian_policy,
                              importance_weight_diagnostics,
                              importance_weights_from_logging_log_prob)
from autoencoder_utils import (encode_in_batches, position_probe, snap_to_bank,
                               train_dsprite_ae)
from kernel_utils import RBF, RieszKernel
from mmd_flow_utils import adamw_riesz_flow, squared_mmd
from metric_utils import sliced_wasserstein

PIXEL_ROWS = ("logged", "roundtrip", "flow", "floor_noisy", "floor_clean")
POSITION_ROWS = ("logged", "flow", "floor")
METRICS = ("energy", "sw1", "sw2")
# Floor rows that also get a COUNT-MATCHED twin; see `score_matched`.
PIXEL_FLOOR_ROWS = ("floor_noisy", "floor_clean")

COLUMNS = (
    ["tag", "device", "init", "data_seed", "repeat", "pipeline_seed",
     "n_obs", "sigma_policy", "policy_shift", "pos_gain", "pixel_noise",
     "n_ae", "n_causal", "n_folds", "n_resample", "n_ref",
     "latent_dim", "ae_iterations", "flow_iterations", "learning_rate", "warmup_steps",
     "ae_train_mse", "denoise_mse_logged", "probe_r2_x", "probe_r2_y",
     "weight_clip", "solver", "ae_split", "dr_fit", "lambda_rule", "n_particles",
     "n_landmarks",
     "oracle_w_ess", "est_w_ess", "est_w_max", "est_w_log_rmse",
     "lengthscale_x", "lengthscale_a", "fold_lambdas", "sum_v", "sum_abs_v",
     "reweighted_pos_x", "reweighted_pos_y",
     "loss_initial", "loss_final", "mean_transport",
     "offmani_roundtrip_median", "offmani_roundtrip_frac3",
     "offmani_flow_median", "offmani_flow_frac3"]
    + [f"{metric}_{row}_{stat}" for row in PIXEL_ROWS for metric in METRICS
       for stat in ("mean", "std")]
    + [f"pos_{metric}_{row}_{stat}" for row in POSITION_ROWS for metric in METRICS
       for stat in ("mean", "std")]
    + ["pos_mean_logged_x", "pos_mean_logged_y", "pos_mean_flow_x", "pos_mean_flow_y",
       "pos_mean_oracle_x", "pos_mean_oracle_y"]
    + [f"{metric}_{row}_matched_{stat}" for row in PIXEL_FLOOR_ROWS for metric in METRICS
       for stat in ("mean", "std")]
    + [f"pos_{metric}_floor_matched_{stat}" for metric in METRICS
       for stat in ("mean", "std")]
    + ["n_floor_matched",
       "seconds_ae", "seconds_nuisance", "seconds_flow", "seconds_decode",
       "seconds_eval", "seconds_repeat"]
)


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


def resolve_solver(args) -> str:
    """Which ridge solver this configuration uses; see --solver."""
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

def particle_generator(args, seed: int, offset: int = 0) -> "torch.Generator":
    """CPU generator for the starting-particle draw, matching the sibling simulations.

    A CPU generator keeps the draw device-independent. `--particle_seed` is not a
    flag of this script; `getattr` keeps the signature identical to the siblings so
    the option can be added later without touching the call sites.
    """
    particle_seed = getattr(args, "particle_seed", None)
    base = seed if particle_seed is None else particle_seed + seed
    return torch.Generator().manual_seed(base + offset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=2000,
                        help="logged sample size (split between the AE and causal halves)")
    parser.add_argument("--ae_split", type=str, default="half", choices=("half", "none"),
                        help="'half' (default, the reference design) trains the autoencoder "
                             "on a disjoint --ae_fraction of the logged sample, so the "
                             "feature map is independent of the causal stage. 'none' trains "
                             "it on EVERY row and lets the causal stage reuse them, matching "
                             "the ColorMNIST flag of the same name. NOTE the same-sample "
                             "notes license reusing observations for NUISANCES but explicitly "
                             "not for an encoder selected on those rows, so 'none' is an "
                             "empirical setting, not one their theory covers -- the licensed "
                             "combination is --ae_split half --dr_fit nosplit.")
    parser.add_argument("--ae_fraction", type=float, default=0.5,
                        help="fraction of the logged sample used to train the autoencoder; "
                             "ignored when --ae_split none")
    parser.add_argument("--sigma_policy", type=float, default=0.5,
                        help="policy noise scale (logging and target)")
    parser.add_argument("--policy_shift", type=float, default=0.2,
                        help="mean shift of the target policy against the logging policy")
    parser.add_argument("--pos_gain", type=float, default=3.0,
                        help="action-to-position gain of the outcome model")
    parser.add_argument("--pixel_noise", type=float, default=0.05,
                        help="observation noise on the sprite images")
    parser.add_argument("--weight_clip", type=float, default=5.0,
                        help="cap on the estimated importance weights before the DR stage "
                             "(<= 0 disables). The estimated logging-density ratio grows "
                             "spurious tails on bad-overlap draws (observed: est max weight "
                             "192 where the oracle weights have ESS 126), and those runs "
                             "dominate the seed envelope. Default 5.0 from a 2026-09 search "
                             "over {inf, 20, 10, 5, 2.5} x failing+healthy seeds: 5 repairs "
                             "every failure, leaves healthy seeds unchanged, and keeps the "
                             "position mean unbiased; 2.5 begins to undershoot it.")
    parser.add_argument("--init", type=str, default="observed",
                        choices=("observed", "normal", "moment"),
                        help="initial particles: 'observed' = the logged arm (default), "
                             "'normal' = N(0, I), 'moment' = the empirical N(mu, Sigma) "
                             "of the flow's target support. Both Gaussians are drawn in "
                             "the coordinates the flow operates in. NOTE: the *_initial "
                             "metric columns describe the run's actual starting cloud, so "
                             "under a Gaussian init they are not the observed-arm baseline "
                             "-- take that from the matching --init observed run.")
    parser.add_argument("--solver", type=str, default="projected_nystrom",
                        choices=("auto", "exact", "nystrom", "projected_nystrom"),
                        help="how each fold's CME ridge system is solved. 'auto' (default) "
                             "solves exactly while affordable and switches to Nystrom above "
                             "--exact_max_n_obs. The DR support is the whole causal split "
                             "either way, so the solver is a tractability detail, not a "
                             "method variant. Keyed on n_obs so runs compared at the same "
                             "n_obs always share a solver. NOTE the causal split is "
                             "n_obs/2 here, so the current grid never leaves the exact "
                             "regime at the default ae_fraction. 'projected_nystrom' is a "
                             "DIFFERENT estimator of the CME nuisance from 'nystrom', not a "
                             "cheaper route to the same number -- see the solver branch in "
                             "`run_repeat` -- and 'auto' never selects it.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold")
    parser.add_argument("--n_particles", type=int, default=None,
                        help="flow particles M, capped at the causal-split size. Default "
                             "None uses EVERY causal row, which is the historical behaviour "
                             "(the particle cloud was welded to the target support). A "
                             "measure-transport sampler with M particles standing in for N "
                             "target atoms pulls each particle toward a centroid of about "
                             "N/M of them, so M is a per-sample fidelity knob, not merely a "
                             "sample count.")
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"),
                        help="how the DR nuisances are fitted. DEFAULT CHANGED 2026-09-28 to 'nosplit', which fits the CME and the propensity on ALL rows; 'crossfit' (the previous default) fits both out of fold. The same-sample notes license reusing observations for the NUISANCES. NOTE the filename token is deliberately still keyed on 'crossfit': a 'nosplit' run appends _drnosplit and a 'crossfit' run appends nothing, exactly as before the default flipped, so every historical filename is still reproduced byte for byte and no new run can overwrite an archived one. The dr_fit CSV column is the authoritative record. "
                             "historical behaviour) fits the CME and the logging density out "
                             "of fold; 'nosplit' fits both on ALL rows. NOTE the same-sample "
                             "notes require RKHS-norm control on the ratio itself, which "
                             "--weight_clip does NOT provide: range clipping bounds the "
                             "values but establishes neither norm control nor prediction "
                             "consistency for the true ratio.")
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
                             "yields a whole iteration curve. This sampler uses a CONSTANT "
                             "learning rate, so a snapshot at step s is exactly a run of s "
                             "steps.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_resample", type=int, default=8,
                        help="target-policy draws averaged into the target action kernel")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--ae_noise_std", type=float, default=0.05,
                        help="denoising corruption added during AE training")
    parser.add_argument("--ae_iterations", type=int, default=4000)
    parser.add_argument("--flow_iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--n_ref", type=int, default=10,
                        help="fresh interventional reference draws per metric")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="pipeline repeats per data seed")
    parser.add_argument("--tag", type=str, default="splitae")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--dsprites_npz", type=str,
                        default=str(REPO_ROOT / "data" / "dsprites"
                                    / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available")
    return parser.parse_args()


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    # Tokens appended only off-default so historical filenames are reproduced exactly.
    # `_nosplit` is the AUTOENCODER axis (same token as ColorMNIST); `_dr*` is the
    # separate nuisance axis. They are different ablations and must stay distinguishable.
    suffix += "_nosplit" if args.ae_split == "none" else ""
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
    if args.n_particles is not None:
        suffix += f"_M{args.n_particles}"
    if args.flow_iterations != 2000:
        suffix += f"_it{args.flow_iterations}"
    name = (f"dsprites_{args.tag}_nobs{args.n_obs}_shift{args.policy_shift:g}"
            f"_sig{args.sigma_policy:g}_gain{args.pos_gain:g}_noise{args.pixel_noise:g}"
            f"_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def score(sample: torch.Tensor, references: list, device: torch.device) -> dict:
    """Energy distance / SW1 / SW2 against each reference; mean and std over references.

    Scored on `device`: the O(n^2) pairwise distances behind the energy distance are
    prohibitive on CPU at the larger sweep sizes.
    """
    energy_kernel = RieszKernel()
    sample = sample.double().to(device)
    out = {}
    for metric, fn in (("energy", lambda a, b: float(squared_mmd(energy_kernel, a, b))),
                       ("sw1", lambda a, b: float(sliced_wasserstein(a, b, p=1, seed=0))),
                       ("sw2", lambda a, b: float(sliced_wasserstein(a, b, p=2, seed=0)))):
        values = [fn(sample, reference.double().to(device)) for reference in references]
        out[f"{metric}_mean"], out[f"{metric}_std"] = float(np.mean(values)), float(np.std(values))
    return out


def subsample(sample: torch.Tensor, k: int, generator: "torch.Generator") -> torch.Tensor:
    """A uniform random k-row subsample; the sample itself when it has at most k rows."""
    if sample.shape[0] <= k:
        return sample
    pick = torch.randperm(sample.shape[0], generator=generator)[:k]
    return sample[pick.to(sample.device)]


def score_matched(sample: torch.Tensor, references: list, device: torch.device,
                  k: int, seed: int) -> dict:
    """`score` with BOTH sides randomly subsampled to k rows.

    The energy distance and the sliced Wasserstein distances are strongly
    SAMPLE-SIZE BIASED: on this design the SW2 between two independent draws of the
    same law falls from 0.027 at n = 250 to 0.007 at n = 4000. The references here
    are built at n_causal while the flow row carries only `n_particles` atoms (and
    `sliced_wasserstein` truncates to the smaller side), so a plain floor -- scored
    at the full reference size -- sits at a different point of that bias curve than
    the row it is supposed to bound. Re-scoring the floor at k = the flow's own
    effective count makes the two comparable, which is what a --n_particles sweep
    needs: otherwise the flow rows move with M for a purely statistical reason while
    the floor stays put.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, d)
    references : list of torch.Tensor
    device : torch.device
        Where the metrics are evaluated; see `score`.
    k : int
        Effective count of the row this floor is matched to.
    seed : int
        Fixed seed of the subsampling draw, derived from the pipeline seed.

    Returns
    -------
    dict
        The keys of `score`.
    """
    generator = torch.Generator().manual_seed(seed)
    return score(subsample(sample, k, generator),
                 [subsample(reference, k, generator) for reference in references],
                 device)


def run_repeat(sprite_bank: np.ndarray, bank_flat: torch.Tensor,
               bank_positions: torch.Tensor, data: dict, args: argparse.Namespace,
               data_seed: int, repeat: int, device: torch.device) -> dict:
    """One full pipeline pass on a fixed logged sample; returns one CSV row."""
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- split: autoencoder half / causal half (disjoint), or shared ---------------------
    if args.ae_split == "none":
        # every row trains the autoencoder AND carries the causal estimate
        ae_idx = causal_idx = torch.arange(args.n_obs)
        n_ae = args.n_obs
    else:
        n_ae = int(round(args.ae_fraction * args.n_obs))
        split = torch.randperm(args.n_obs, generator=torch.Generator().manual_seed(seed + 2))
        ae_idx, causal_idx = split[:n_ae], split[n_ae:]
    n_causal = len(causal_idx)

    Y_ae_img = data["Y_logged"][ae_idx].float().reshape(-1, 1, 64, 64).to(device)
    Y_log_img = data["Y_logged"][causal_idx].float().reshape(-1, 1, 64, 64).to(device)
    Y_target = data["Y_target"][causal_idx].float()
    pos_logged = data["pos_logged"][causal_idx].double()
    pos_target = data["pos_target"][causal_idx].double()
    X_np = data["X"][causal_idx].numpy()
    A_np = data["A_logged"][causal_idx].numpy()

    # -- denoising autoencoder on the AE half's logged images only ---------------------
    # The denoising targets are the noisy logged images themselves (no clean sprites,
    # no bank): honest for a logged-bandit application, at the price of the decoder
    # partially reproducing the observation noise instead of stripping it.
    t0 = time.time()
    ae, history = train_dsprite_ae(Y_ae_img, latent_dim=args.latent_dim,
                                   noise_std=args.ae_noise_std,
                                   n_iterations=args.ae_iterations, device=device,
                                   verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    seconds_ae = time.time() - t0

    with torch.no_grad():
        recon = torch.cat([ae(Y_log_img[i:i + 256]) for i in range(0, n_causal, 256)])
    denoise_mse = float(((recon.reshape(n_causal, -1)
                          - bank_flat[pos_logged[:, 0].long() * 32
                                      + pos_logged[:, 1].long()]) ** 2).mean())
    bank_t = bank_flat.reshape(1024, 1, 64, 64)
    Z_bank = encode_in_batches(ae.encode,
                               bank_t + args.pixel_noise * torch.randn(bank_t.shape,
                                                                       device=device))
    probe = position_probe(Z_bank, bank_positions)

    # -- cross-fitted continuous-action nuisances and DR weights -----------------------
    t0 = time.time()
    fold_ids = make_crossfit_fold_ids(n_causal, n_folds=args.n_folds, seed=0)
    if args.dr_fit == "nosplit":
        # Fit the logging density once on all rows and score those same rows. The
        # cross-fitted helper has no single-fold path, and a degenerate fold split
        # would not be the full-sample fit the same-sample argument refers to.
        logging_hat = fit_linear_gaussian_policy(X_np, A_np)
        log_pi0_hat = logging_hat.log_prob(X_np, A_np)
    else:
        log_pi0_hat = crossfit_logging_policy_log_prob(X_np, A_np, fold_ids=fold_ids)
    w_hat = importance_weights_from_logging_log_prob(data["target_policy"], X_np, A_np,
                                                     log_pi0_hat)
    if args.weight_clip is not None and args.weight_clip > 0:
        w_hat = np.minimum(w_hat, args.weight_clip)
    oracle_diag = importance_weight_diagnostics(
        data["importance_weights"][causal_idx].numpy())
    diag = importance_weight_diagnostics(w_hat,
                                         data["importance_weights"][causal_idx].numpy())

    Z_log = encode_in_batches(ae.encode, Y_log_img).cpu()
    solver = resolve_solver(args)
    kernel_x = RBF(use_length_scale_heuristic=True)
    kernel_a = RBF(use_length_scale_heuristic=True)
    X_t = data["X"][causal_idx].double()
    A_t = data["A_logged"][causal_idx].double()

    # Both lengthscales are locked ONCE on a bounded, SEEDED anchor subsample, as the
    # ColorMNIST simulation does. Two reasons. The median over all n_causal^2 pairs is
    # unaffordable at the sweep sizes; and `calculate_and_fix_length_scale` subsamples
    # with the GLOBAL torch RNG above its own 5000-row limit, so the lengthscale would
    # otherwise depend on how much randomness the pipeline consumed earlier -- a seeded
    # anchor is reproducible whatever else changes upstream. Unlike ColorMNIST/CelebA,
    # where the action factor is an equality indicator, the action here is CONTINUOUS
    # and 2-d, so kernel_a is a genuine RBF and needs a lock of its own.
    anchor = torch.randperm(n_causal, generator=torch.Generator().manual_seed(0))[:4000]
    kernel_x.calculate_and_fix_length_scale(X_t[anchor])
    kernel_a.calculate_and_fix_length_scale(A_t[anchor])

    # The target-policy action draws are stored once (n_resample x n_causal x action
    # dim, trivial) so the target action kernel can be evaluated blockwise instead of
    # accumulated into a full n_causal x n_causal matrix. `resample_rng` is seeded and
    # consumed in the same order as before, so the draws themselves are unchanged.
    resample_rng = np.random.default_rng(0)
    A_resamples = [torch.as_tensor(data["target_policy"].sample(X_np, resample_rng),
                                   dtype=torch.float64)
                   for _ in range(args.n_resample)]

    def gram_x(rows, cols):
        "K_X block alone; the target-action factor needs it without the logged one."
        return kernel_x(X_t[rows], X_t[cols])

    def gram_a(rows, cols):
        "K_A block over the LOGGED actions; a genuine RBF here, the action is continuous."
        return kernel_a(A_t[rows], A_t[cols])

    def gram_a_target(rows, cols):
        """mean_b k_A(a_i, a~_j^b) block.

        Note the ORIENTATION: `rows` index the LOGGED actions and `cols` the target
        policy DRAWS, matching K_a_target[i, j] = mean_b k_A(A_logged[i], A_draw_b[j]).
        Transposing it would silently estimate a different object.
        """
        block = kernel_a(A_t[rows], A_resamples[0][cols])
        for A_resample in A_resamples[1:]:
            block = block + kernel_a(A_t[rows], A_resample[cols])
        return block / len(A_resamples)

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_causal^2 Gram is never formed."
        return gram_x(rows, cols) * gram_a(rows, cols)

    def gram_pi(rows, cols):
        "K_pi = K_X * K_A_target on demand."
        return gram_x(rows, cols) * gram_a_target(rows, cols)

    # The exact solvers take whole matrices, so they -- and only they -- pay the O(n^2).
    if solver == "exact":
        K_x = kernel_x(X_t)
        K_a = kernel_a(A_t)
        K_a_target = torch.zeros(n_causal, n_causal, dtype=torch.float64)
        for A_resample in A_resamples:
            K_a_target += kernel_a(A_t, A_resample)
        K_a_target /= args.n_resample

    if args.lambda_rule == "n_half":
        nosplit_lambda = lambda_n_half(n_causal, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    else:
        fold_lambdas = {}
        for fold in np.unique(fold_ids):
            train = np.where(fold_ids != fold)[0]
            # lambda does not need the full training block, which is itself an n^3 solve
            pick = train[torch.randperm(len(train),
                                        generator=torch.Generator().manual_seed(0))[:2000].numpy()]
            fold_lambdas[int(fold)], _ = select_regularization(gram(pick, pick),
                                                               Z_log[pick, 0].double(), seed=0)
        nosplit_lambda = None
        if args.dr_fit == "nosplit":
            pick = torch.randperm(n_causal,
                                  generator=torch.Generator().manual_seed(0))[:2000].numpy()
            nosplit_lambda, _ = select_regularization(gram(pick, pick),
                                                      Z_log[pick, 0].double(), seed=0)
    w_t = torch.as_tensor(w_hat, dtype=torch.float64)
    if solver in ("nystrom", "projected_nystrom"):
        # `projected_nystrom` restricts the CME ridge regression to the landmark-spanned
        # subspace and builds the DR right-hand side only at the landmarks -- the
        # estimator the imported consistency rate is stated for, touching an m x |H|
        # cross-kernel instead of a |T| x |H| one. `nystrom` applies an approximate Gram
        # inverse to the exact full right-hand side: a different estimator, kept as the
        # baseline that produced every earlier table.
        projected = solver == "projected_nystrom"
        if args.dr_fit == "nosplit":
            solve = (cpme_dr_weights_projected_nystrom if projected
                     else cpme_dr_weights_nystrom)
            v_dr = solve(gram, gram_pi, w_t, nosplit_lambda,
                         n_landmarks=args.n_landmarks,
                         generator=torch.Generator().manual_seed(seed))
        else:
            solve = (cpme_dr_weights_crossfit_projected_nystrom if projected
                     else cpme_dr_weights_crossfit_nystrom)
            v_dr = solve(gram, gram_pi, w_t, fold_ids, fold_lambdas,
                         n_landmarks=args.n_landmarks,
                         generator=torch.Generator().manual_seed(seed))
    else:
        if args.dr_fit == "nosplit":
            v_dr = cpme_dr_weights(K_x, K_a, K_a_target, w_t, nosplit_lambda)
        else:
            v_dr = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_t, fold_ids, fold_lambdas)
    seconds_nuisance = time.time() - t0

    # -- Riesz MMD flow (AdamW, warm-up) on standardized latents -----------------------
    t0 = time.time()
    latent_mean = Z_log.mean(dim=0, keepdim=True)
    latent_std = Z_log.std(dim=0, keepdim=True).clamp_min(1e-3)
    to_flow = lambda z: ((z - latent_mean) / latent_std).float()
    from_flow = lambda z: z.cpu().float() * latent_std + latent_mean

    flow_target = to_flow(Z_log).to(device)      # the logged latents: always the target
    # The particle cloud is drawn from the target support but is NOT the same object:
    # decoupling them is what makes a particle sweep possible without also changing the
    # DR support (which would confound M with the estimator).
    n_particles = (flow_target.shape[0] if args.n_particles is None
                   else min(args.n_particles, flow_target.shape[0]))
    if args.init == "observed":
        if n_particles == flow_target.shape[0]:
            initial_particles = flow_target.clone()   # historical path, bitwise unchanged
        else:
            pick = torch.randperm(flow_target.shape[0],
                                  generator=particle_generator(args, seed, 1))[:n_particles]
            initial_particles = flow_target[pick.to(device)].clone()
    else:
        initial_particles = gaussian_init(
            flow_target, n_particles, args.init,
            torch.Generator(device=device).manual_seed(seed + 3))
    flow_particles, flow_info = adamw_riesz_flow(
        initial_particles, flow_target,
        target_weights=v_dr.float().to(device), sliced=False,
        n_iterations=args.flow_iterations, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, snapshot_every=args.snapshot_every,
        log_every=500, verbose=False)
    # `seconds_flow` covers the particle init and the flow call ONLY. Until
    # 2026-09-28 it also contained the decode below, which is itself O(n_particles)
    # and so contaminated exactly the --n_particles axis; runs archived before that
    # date are not comparable with this column.
    seconds_flow = time.time() - t0
    mean_transport = float((flow_particles.cpu() - initial_particles.cpu())
                           .norm(dim=1).mean())

    def decode(particles):
        with torch.no_grad():
            return torch.cat([ae.decode(from_flow(particles[i:i + 256]).to(device))
                              for i in range(0, particles.shape[0], 256)]
                             ).reshape(particles.shape[0], -1)

    t0 = time.time()
    flow_images = decode(flow_particles)
    seconds_decode = time.time() - t0

    # -- evaluation against fresh matched-size interventional references ---------------
    t0 = time.time()
    fresh = [sample_cpme_dsprite_data(sprite_bank, n_samples=n_causal,
                                      sigma_policy=args.sigma_policy,
                                      policy_shift=args.policy_shift,
                                      pos_gain=args.pos_gain,
                                      pixel_noise=args.pixel_noise,
                                      seed=100000 + 1000 * seed + j)
             for j in range(args.n_ref)]
    fresh_img = [d["Y_target"].float() for d in fresh]
    fresh_pos = [d["pos_target"].double() for d in fresh]

    obs_decode = recon.reshape(n_causal, -1).cpu()
    clean_oracle = bank_flat.cpu()[(pos_target.long()[:, 0] * 32
                                    + pos_target.long()[:, 1])].float()
    pixel_samples = {"logged": data["Y_logged"][causal_idx].float(),
                     "roundtrip": obs_decode,
                     "flow": flow_images.cpu(),
                     "floor_noisy": Y_target,
                     "floor_clean": clean_oracle}

    flow_pos, flow_offmani = snap_to_bank(flow_images, bank_flat)
    flow_pos = flow_pos.cpu().float()
    _, obs_offmani = snap_to_bank(obs_decode.to(device), bank_flat)
    position_samples = {"logged": pos_logged,
                        "flow": flow_pos.double(),
                        "floor": pos_target}

    row = {
        "tag": args.tag, "device": str(device), "init": args.init,
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "n_obs": args.n_obs, "sigma_policy": args.sigma_policy,
        "policy_shift": args.policy_shift, "pos_gain": args.pos_gain,
        "pixel_noise": args.pixel_noise,
        "n_ae": int(n_ae), "n_causal": int(n_causal), "ae_split": args.ae_split,
        "n_folds": args.n_folds,
        "n_resample": args.n_resample, "n_ref": args.n_ref,
        "latent_dim": args.latent_dim, "ae_iterations": args.ae_iterations,
        "flow_iterations": args.flow_iterations, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "ae_train_mse": float(history[-1][1]), "denoise_mse_logged": denoise_mse,
        "probe_r2_x": float(probe["posX_r2"]), "probe_r2_y": float(probe["posY_r2"]),
        "weight_clip": (args.weight_clip if args.weight_clip is not None
                        and args.weight_clip > 0 else float("inf")),
        "solver": solver, "dr_fit": args.dr_fit, "lambda_rule": args.lambda_rule,
        "n_particles": int(n_particles),
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "oracle_w_ess": float(oracle_diag["ess"]), "est_w_ess": float(diag["ess"]),
        "est_w_max": float(diag["max"]), "est_w_log_rmse": float(diag["log_rmse"]),
        "lengthscale_x": float(kernel_x.length_scale),
        "lengthscale_a": float(kernel_a.length_scale),
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        "sum_v": float(v_dr.sum()), "sum_abs_v": float(v_dr.abs().sum()),
        "reweighted_pos_x": float((v_dr * pos_logged[:, 0]).sum()),
        "reweighted_pos_y": float((v_dr * pos_logged[:, 1]).sum()),
        "loss_initial": float(flow_info["loss"][0]),
        "loss_final": float(flow_info["loss"][-1]),
        "mean_transport": mean_transport,
        "offmani_roundtrip_median": float(obs_offmani.median()),
        "offmani_roundtrip_frac3": float((obs_offmani > 3).float().mean()),
        "offmani_flow_median": float(flow_offmani.median()),
        "offmani_flow_frac3": float((flow_offmani > 3).float().mean()),
        "pos_mean_logged_x": float(pos_logged[:, 0].mean()),
        "pos_mean_logged_y": float(pos_logged[:, 1].mean()),
        "pos_mean_flow_x": float(flow_pos[:, 0].mean()),
        "pos_mean_flow_y": float(flow_pos[:, 1].mean()),
        "pos_mean_oracle_x": float(pos_target[:, 0].mean()),
        "pos_mean_oracle_y": float(pos_target[:, 1].mean()),
    }
    for name, sample in pixel_samples.items():
        metrics = score(sample, fresh_img, device)
        row.update({f"{key.split('_')[0]}_{name}_{key.split('_')[1]}": value
                    for key, value in metrics.items()})
    for name, sample in position_samples.items():
        metrics = score(sample, fresh_pos, device)
        row.update({f"pos_{key.split('_')[0]}_{name}_{key.split('_')[1]}": value
                    for key, value in metrics.items()})

    # Count-matched floors: the same quantities at the flow row's effective count.
    n_matched = int(n_particles)
    for offset, name in enumerate(PIXEL_FLOOR_ROWS):
        metrics = score_matched(pixel_samples[name], fresh_img, device, n_matched,
                                500000 + 10 * seed + offset)
        row.update({f"{key.split('_')[0]}_{name}_matched_{key.split('_')[1]}": value
                    for key, value in metrics.items()})
    metrics = score_matched(position_samples["floor"], fresh_pos, device, n_matched,
                            500000 + 10 * seed + len(PIXEL_FLOOR_ROWS))
    row.update({f"pos_{key.split('_')[0]}_floor_matched_{key.split('_')[1]}": value
                for key, value in metrics.items()})
    row["n_floor_matched"] = n_matched

    row.update({"seconds_ae": round(seconds_ae, 1),
                "seconds_nuisance": round(seconds_nuisance, 1),
                "seconds_flow": round(seconds_flow, 1),
                "seconds_decode": round(seconds_decode, 1),
                "seconds_eval": round(time.time() - t0, 1),
                "seconds_repeat": round(time.time() - t_start, 1)})
    return row


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = csv_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   output: {out_path}", flush=True)

    npz = Path(args.dsprites_npz)
    cache = npz.parent / "heart_sprite_bank_32x32.npy"
    if not npz.exists() and not cache.exists():
        sys.exit(f"dSprites data not found: neither {npz} nor {cache} exists — "
                 "copy the repo's data/dsprites folder (no download on compute nodes).")
    sprite_bank = load_dsprite_bank(str(npz), cache_path=str(cache))
    bank_flat = torch.as_tensor(sprite_bank, dtype=torch.float32).reshape(1024, -1).to(device)
    bank_positions = torch.stack(torch.meshgrid(torch.arange(32), torch.arange(32),
                                                indexing="ij"), dim=-1).reshape(1024, 2)

    rows = []
    for data_seed in args.data_seeds:
        data = sample_cpme_dsprite_data(sprite_bank, n_samples=args.n_obs,
                                        sigma_policy=args.sigma_policy,
                                        policy_shift=args.policy_shift,
                                        pos_gain=args.pos_gain,
                                        pixel_noise=args.pixel_noise, seed=data_seed)
        for repeat in range(args.n_repeats):
            row = run_repeat(sprite_bank, bank_flat, bank_positions, data, args,
                             data_seed, repeat, device)
            rows.append(row)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            print(f"[data_seed {data_seed} repeat {repeat}] "
                  f"pixel energy flow {row['energy_flow_mean']:.4f} "
                  f"(logged {row['energy_logged_mean']:.4f}, "
                  f"clean floor {row['energy_floor_clean_mean']:.4f})   "
                  f"pos mean flow ({row['pos_mean_flow_x']:.1f}, {row['pos_mean_flow_y']:.1f}) "
                  f"oracle ({row['pos_mean_oracle_x']:.1f}, {row['pos_mean_oracle_y']:.1f})   "
                  f"ESS {row['est_w_ess']:.0f}/{row['n_causal']}   "
                  f"{row['seconds_repeat']:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
