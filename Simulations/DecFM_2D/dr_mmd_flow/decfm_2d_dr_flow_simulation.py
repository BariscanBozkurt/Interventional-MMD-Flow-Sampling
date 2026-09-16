"""
DecFM 2D mixture simulation: cross-fitted DR-CPME target, RBF-Euler and Riesz-AdamW flows.

Script version of the twin notebooks
`NotebookExamples/Interventional_MMD_Flow/DecFM_2D_Mixture_RBF_MMD_Flow_Euler.ipynb` and
`DecFM_2D_Mixture_Riesz_MMD_Flow_AdamW.ipynb`, which share the design (Dance et al.,
2026, Appendix B.2.2: Y = X + mixture noise, so both interventional laws equal the
marginal P_Y), the cross-fitted logistic propensity, and the DR weight fitting, and
differ only in the sampler. Each repeat therefore fits the nuisances once and runs
BOTH samplers on BOTH arms from the SAME per-arm initial particles — a paired
comparison of forward-Euler RBF flow against AdamW Riesz flow at every configuration.

Everything runs on CPU in float64 — there is no GPU code in this experiment, and the
script has been validated with CUDA hidden.

One CSV row per (data_seed, repeat, arm, sampler): DGP constants, weight diagnostics
(on-arm ESS, sum of DR weights, per-fold lambdas, kernel lengthscales), the sampler's
own step/learning-rate, mean transport, and SW1 / SW2 / unbiased RBF MMD^2 against the
oracle marginal for the observed arm and the flowed particles, with the
oracle-vs-oracle floor alongside.

Evaluation protocol (2026-09-15, matched to Dance et al.): the DR support is EVERY
logged row (--n_dr caps it only on request) solved with the Nystrom solver at every n;
--n_particles 2000 particles are generated per arm and scored against a FIXED
--n_ref 2000 counterfactual reference draw (seed 100000 + data_seed, independent of
n_obs), with the floor a second independent 2000-draw against the first, so every
row is a 2000-vs-2000 number regardless of n_obs. The outcome kernel median stays
locked on the FULL logged sample, as in the notebooks.

Seed scheme: `data_seed` fixes the draw; `pipeline_seed = 1000 * data_seed + repeat`
fixes the propensity fit, the DR subsample, the particle picks and the Riesz flow.

Launched by `submit_decfm_2d_dr_flow_parallel.sh`; single-run example:

    python decfm_2d_dr_flow_simulation.py --n_obs 4000 --data_seeds 0 --n_repeats 3
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

from kernel_utils import RBF
from data_utils import sample_decfm_2d_data
from causality_utils import (cpme_dr_weights_crossfit,
                             cpme_dr_weights_crossfit_nystrom,
                             make_crossfit_fold_ids, select_regularization)
from dens_ratio_utils import LogisticPropensityModel, crossfit_propensity
from mmd_flow_utils import adamw_riesz_flow, mmd_gradient_flow, squared_mmd
from metric_utils import sliced_wasserstein

COLUMNS = [
    "tag", "init", "solver", "n_landmarks",
    "data_seed", "repeat", "pipeline_seed", "sampler", "target_a",
    "n_obs", "n_dr", "n_ref", "n_arm", "n_particles_requested", "n_particles", "n_floor",
    "confounding_strength", "mixture_delta", "noise_std", "n_folds",
    "ess_on_arm", "n_on_arm", "sum_v", "sum_abs_v", "fold_lambdas",
    "lengthscale_x", "lengthscale_y",
    "flow_iterations", "step_or_lr", "mean_transport",
    "sw1_initial", "sw2_initial", "mmd2_initial",
    "sw1_flow", "sw2_flow", "mmd2_flow",
    "sw1_floor", "sw2_floor", "mmd2_floor",
    "seconds_nuisance", "seconds_flow", "seconds_repeat",
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
    parser.add_argument("--n_obs", type=int, default=4000,
                        help="logged sample size n")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="size of the fixed counterfactual reference draw used for "
                             "every metric (and of the second draw used for the floor)")
    parser.add_argument("--n_particles", type=int, default=2000,
                        help="flow particles m per arm (capped at the arm size)")
    parser.add_argument("--solver", type=str, default="nystrom",
                        choices=("auto", "exact", "nystrom"),
                        help="how each fold's CME ridge system is solved. 'auto' (default) "
                             "solves exactly while affordable and switches to Nystrom above "
                             "--exact_max_n_obs. The DR support is EVERY logged row either "
                             "way, so the solver is a tractability detail, not a method "
                             "variant, and should not appear as a factor in results tables.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly; above it a fold "
                             "would be a (4/5 n_obs)^2 float64 system and the Gram alone "
                             "3.2 GB at n_obs = 20000")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold")
    parser.add_argument("--n_dr", type=int, default=None,
                        help="optional cap on the DR support, kept for reproducing the "
                             "pre-2026-09 runs (--solver exact --n_dr 4000). Unset, the "
                             "support is every logged row.")
    parser.add_argument("--samplers", type=str, nargs="+",
                        default=["rbf_euler", "riesz_adamw"],
                        choices=("rbf_euler", "riesz_adamw"))
    parser.add_argument("--confounding_strength", type=float, default=3.0,
                        help="gamma in the logistic propensity")
    parser.add_argument("--mixture_delta", type=float, default=2.0,
                        help="separation of the two outcome-noise components")
    parser.add_argument("--noise_std", type=float, default=0.2,
                        help="within-component outcome noise")
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
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--step_factor", type=float, default=0.05,
                        help="Euler step = step_factor * (outcome lengthscale)^2")
    parser.add_argument("--euler_iterations", type=int, default=3000)
    parser.add_argument("--riesz_iterations", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="AdamW peak learning rate (Riesz sampler)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="pipeline repeats per data seed")
    parser.add_argument("--tag", type=str, default="twins")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    return parser.parse_args()



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
    """Which ridge solver this configuration uses; see --solver."""
    if args.solver != "auto":
        return args.solver
    return "exact" if args.n_obs <= args.exact_max_n_obs else "nystrom"


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    name = (f"decfm2d_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
            f"_gamma{args.confounding_strength:g}_delta{args.mixture_delta:g}"
            f"_sig{args.noise_std:g}_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def run_repeat(data: dict, args: argparse.Namespace, data_seed: int, repeat: int) -> list:
    """One pipeline pass on a fixed logged sample; one row per (arm, sampler)."""
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, A_logged, Y_logged = data["X"], data["A_logged"], data["Y_logged"]
    Y_oracle, Y_oracle_2 = data["Y_ref"], data["Y_ref_2"]
    A_flat = A_logged.ravel()
    n = len(A_flat)

    # -- shared nuisances on the DR support --------------------------------------------
    t0 = time.time()
    solver = resolve_solver(args)
    # The support is every logged row; --n_dr caps it only when asked for explicitly.
    if args.n_dr is not None and args.n_dr < n:
        pick_dr = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 5))[:args.n_dr]
    else:
        pick_dr = torch.arange(n)
    X_s, A_s = X[pick_dr], A_logged[pick_dr]
    Y_s = Y_logged[pick_dr]
    A_s_flat = A_s.ravel()
    n_dr = len(pick_dr)

    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    propensity = crossfit_propensity(X_s, A_s_flat, fold_ids,
                                     model_factory=LogisticPropensityModel)

    kernel_x = RBF(use_length_scale_heuristic=True)
    kernel_a = RBF(length_scale=0.5)
    kernel_y = RBF(use_length_scale_heuristic=True)
    # Lock the context lengthscale on a bounded subsample: the median over all
    # n_dr^2 pairs is neither affordable nor necessary at n_dr = 20000.
    anchor = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:4000]
    _ = kernel_x(X_s[anchor])
    _ = kernel_y(Y_logged)   # median locked on the FULL logged sample, as in the notebooks

    def gram_x(rows, cols):
        "K_X block alone; K_pi needs it without the action factor."
        return kernel_x(X_s[rows], X_s[cols])

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_dr^2 Gram is never formed."
        return gram_x(rows, cols) * kernel_a(A_s[rows], A_s[cols])

    K_x = kernel_x(X_s) if solver == "exact" else None
    K_a = kernel_a(A_s) if solver == "exact" else None

    fold_lambdas = {}
    for fold in np.unique(fold_ids):
        train = np.where(fold_ids != fold)[0]
        # lambda does not need the full training block, which is itself an n^3 solve
        pick = train[torch.randperm(len(train),
                                    generator=torch.Generator().manual_seed(0))[:2000].numpy()]
        fold_lambdas[int(fold)], _ = select_regularization(gram(pick, pick),
                                                           Y_s[pick, 0], seed=0)
    seconds_nuisance = time.time() - t0

    base = {
        "tag": args.tag, "init": args.init, "solver": solver,
        "n_landmarks": args.n_landmarks if solver == "nystrom" else 0,
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "n_obs": args.n_obs, "n_dr": int(n_dr), "n_ref": int(len(Y_oracle)),
        "n_particles_requested": args.n_particles,
        "confounding_strength": args.confounding_strength,
        "mixture_delta": args.mixture_delta, "noise_std": args.noise_std,
        "n_folds": args.n_folds,
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        "lengthscale_x": float(kernel_x.length_scale),
        "lengthscale_y": float(kernel_y.length_scale),
        "seconds_nuisance": round(seconds_nuisance, 1),
    }

    def evaluate(cloud):
        return {
            "sw1": sliced_wasserstein(cloud, Y_oracle, p=1, seed=0).item(),
            "sw2": sliced_wasserstein(cloud, Y_oracle, p=2, seed=0).item(),
            "mmd2": squared_mmd(kernel_y, cloud, Y_oracle, unbiased=True).item(),
        }

    n_floor = len(Y_oracle_2)
    floor = evaluate(Y_oracle_2)      # second independent reference draw vs the first

    rows = []
    for target_a in (0, 1):
        e_a = propensity if target_a == 1 else 1.0 - propensity
        w_hat = (A_s_flat == target_a).double() / e_a
        on_arm = w_hat[A_s_flat == target_a]
        if solver == "nystrom":
            # K_A_pi[i, j] = k_A(a_i, a) is the same for every column j, so K_pi's
            # blocks are K_X blocks scaled row-wise.
            target_column = kernel_a(A_s, torch.full((1, 1), float(target_a))).reshape(-1)
            gram_pi = lambda rows, cols: (gram_x(rows, cols)
                                          * target_column[rows][:, None])
            v_dr = cpme_dr_weights_crossfit_nystrom(
                gram, gram_pi, w_hat, fold_ids, fold_lambdas,
                n_landmarks=args.n_landmarks,
                generator=torch.Generator().manual_seed(seed))
        else:
            K_a_target = kernel_a(A_s, torch.full((n_dr, 1), float(target_a)))
            v_dr = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_hat, fold_ids,
                                            fold_lambdas)

        arm_idx = torch.nonzero(A_flat == target_a).ravel()
        n_particles = min(args.n_particles, len(arm_idx))
        generator = particle_generator(args, seed)
        if args.init == "observed":
            pick = arm_idx[torch.randperm(len(arm_idx), generator=generator)[:n_particles]]
            PARTICLE_INDICES[f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"] = pick.cpu().numpy()
            init = Y_logged[pick].clone()
        else:
            init = gaussian_init(Y_s, n_particles, args.init,
                                 torch.Generator().manual_seed(seed + 3))
        initial_metrics = evaluate(Y_logged[arm_idx])

        for sampler in args.samplers:
            t0 = time.time()
            if sampler == "rbf_euler":
                step = args.step_factor * kernel_y.length_scale.item() ** 2
                flowed, _ = mmd_gradient_flow(kernel_y, Y_s, init.clone(),
                                              step_size=step,
                                              n_iterations=args.euler_iterations,
                                              target_weights=v_dr, verbose=False)
                iterations, step_or_lr = args.euler_iterations, step
            else:
                torch.manual_seed(seed + 1)
                flowed, _ = adamw_riesz_flow(init.clone(), Y_s, target_weights=v_dr,
                                             n_iterations=args.riesz_iterations,
                                             learning_rate=args.learning_rate,
                                             warmup_steps=args.warmup_steps,
                                             lr_schedule="cosine", betas=(0.9, 0.999),
                                             verbose=False)
                iterations, step_or_lr = args.riesz_iterations, args.learning_rate

            row = dict(base)
            row.update({
                "sampler": sampler, "target_a": target_a,
                "n_arm": int(len(arm_idx)), "n_particles": int(n_particles),
                "n_floor": int(n_floor),
                "ess_on_arm": float(on_arm.sum() ** 2 / (on_arm ** 2).sum()),
                "n_on_arm": int(len(on_arm)),
                "sum_v": float(v_dr.sum()), "sum_abs_v": float(v_dr.abs().sum()),
                "flow_iterations": iterations, "step_or_lr": float(step_or_lr),
                "mean_transport": float((flowed - init).norm(dim=1).mean()),
                "seconds_flow": round(time.time() - t0, 1),
            })
            row.update({f"{key}_initial": value for key, value in initial_metrics.items()})
            row.update({f"{key}_flow": value for key, value in evaluate(flowed).items()})
            row.update({f"{key}_floor": value for key, value in floor.items()})
            rows.append(row)

    seconds_repeat = round(time.time() - t_start, 1)
    for row in rows:
        row["seconds_repeat"] = seconds_repeat
    return rows


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    out_path = csv_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_path}", flush=True)

    rows = []
    for data_seed in args.data_seeds:
        data = sample_decfm_2d_data(n_samples=args.n_obs,
                                    confounding_strength=args.confounding_strength,
                                    mixture_delta=args.mixture_delta,
                                    noise_std=args.noise_std, seed=data_seed)
        # Fixed reference draw, independent of n_obs: first half scores every row, the
        # second half is the floor's sample. Seeds are disjoint from every data seed.
        reference = sample_decfm_2d_data(n_samples=2 * args.n_ref,
                                         confounding_strength=args.confounding_strength,
                                         mixture_delta=args.mixture_delta,
                                         noise_std=args.noise_std,
                                         seed=100000 + data_seed)["Y_oracle"]
        data["Y_ref"], data["Y_ref_2"] = reference[:args.n_ref], reference[args.n_ref:]
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(data, args, data_seed, repeat)
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            write_particle_indices(out_path)
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']} "
                      f"{row['sampler']:11s}] SW2 {row['sw2_initial']:.3f} -> "
                      f"{row['sw2_flow']:.3f} (floor {row['sw2_floor']:.3f})   "
                      f"sum_v {row['sum_v']:.3f}   {row['seconds_flow']:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
