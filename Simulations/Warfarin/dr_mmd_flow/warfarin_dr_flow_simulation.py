"""
Warfarin semi-synthetic: sampling an interventional outcome law under a
STOCHASTIC continuous-action policy.

Every other simulation in this repository intervenes deterministically -- a
single arm, or a fixed shift of a continuous action. This one uses a genuinely
stochastic target policy on real patient covariates, and in its headline setting
a MEAN-PRESERVING one: the target action law is a 50/50 mixture at -delta and
+delta around the logging mean, so E[A | X] is unchanged while the law becomes
bimodal with conditional variance larger by delta^2.

That choice is the point. The intervention is invisible to any summary that
tracks the policy through its mean action, yet the mis-dosing outcome
max(|a - TherDose(X)| - tol * TherDose(X), 0) is a kinked function of the
action, so the interventional OUTCOME law moves substantially -- at the
calibrated default the probability of severe mis-dosing rises by about 70% while
the mean action moves under 1% of an action standard deviation. Recovering that
requires sampling the law, not estimating a scalar functional of it.

Pipeline, matching the other simulations: cross-fitted (or no-split) one-step DR
weights over a projected Nystrom CME, then a particle flow against the resulting
target embedding. The outcome is scalar, so the flow runs directly in outcome
space with no autoencoder and the oracle law is available exactly by drawing
A ~ pi(. | X) and evaluating the dosing cost.

Design: data_utils.warfarin. Estimator: causality_utils.cpme. Flow:
mmd_flow_utils. Data: data/warfarin (IWPC/PharmGKB; see its SOURCE.txt).
"""

import argparse
import csv
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
                            cpme_dr_weights_nystrom,
                            cpme_dr_weights_projected_nystrom,
                            make_crossfit_fold_ids, select_regularization)
from data_utils import (interventional_cdf, load_warfarin_cohort,
                        sample_warfarin_data)
from dens_ratio_utils import (crossfit_logging_policy_log_prob,
                             fit_linear_gaussian_policy,
                             importance_weight_diagnostics)
from kernel_utils import RBF, RieszKernel
from metric_utils import sliced_wasserstein
from mmd_flow_utils import adamw_riesz_flow, squared_mmd

COLUMNS = [
    "tag", "device", "target_policy", "init", "solver", "dr_fit", "lambda_rule",
    "data_seed", "repeat", "pipeline_seed",
    "n_obs", "n_cohort", "n_particles_requested", "n_particles", "n_ref",
    "n_landmarks", "n_folds", "theta", "delta_scale", "delta", "tolerance",
    "outcome_noise", "flow_iterations", "learning_rate", "warmup_steps",
    "weight_clip",
    "oracle_w_ess", "est_w_ess", "est_w_max", "est_w_log_rmse",
    "sum_v", "sum_abs_v",
    "action_mean_logged", "action_mean_target", "action_sd_logged",
    "action_sd_target",
    "loss_initial", "loss_final", "mean_transport",
]
#: Distributional metrics are recorded for four samples, so the table carries its
#: own references: `logged` is the no-intervention baseline, `floor` an
#: independent oracle draw (the best any sampler could do at this count), and
#: `initial` the particle cloud before the flow.
METRIC_ROWS = ("logged", "initial", "flow", "floor")
METRIC_NAMES = ("w1", "w2", "mmd2", "ks", "mean_err", "sd_err",
                "q05_err", "q10_err", "q50_err", "q90_err", "q95_err",
                "tail_err", "cvar95_err")
#: The same two distances taken against the EXACT interventional law rather than
#: a finite draw of it; see `exact_law_metrics`.
EXACT_METRIC_NAMES = ("ks_exact", "w1_exact")
COLUMNS += [f"{name}_{row}" for row in METRIC_ROWS
            for name in METRIC_NAMES + EXACT_METRIC_NAMES]
COLUMNS += ["seconds_nuisance", "seconds_flow", "seconds_eval", "seconds_repeat"]

#: Outcome value above which a patient counts as severely mis-dosed, in mg/week
#: of dosing error. Used for the tail-probability and CVaR columns.
SEVERE_MISDOSE = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=20000,
                        help="logged observations; patients are resampled with "
                             "replacement above the cohort size")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="size of the oracle reference draw every row is scored "
                             "against, and of the second draw that is the floor")
    parser.add_argument("--n_particles", type=int, default=2000)
    parser.add_argument("--target_policy", type=str, default="mean_preserving",
                        choices=("mean_preserving", "shift", "shifted_mixture"),
                        help="'mean_preserving' is the headline setting: E[A|X] is "
                             "unchanged and only the shape of the action law moves")
    parser.add_argument("--delta_scale", type=float, default=1.0,
                        help="target shift as a multiple of the logging policy's "
                             "conditional scale. 1.0 is calibrated: sqrt(2) times the "
                             "conditional action spread at about 70% effective sample "
                             "size. 2.0 collapses the overlap")
    parser.add_argument("--theta", type=float, default=0.5,
                        help="fraction of the logged dose's variance explained by BMI, "
                             "i.e. the strength of the confounding")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="half-width of the acceptable dosing band, as a fraction "
                             "of the therapeutic dose")
    parser.add_argument("--outcome_noise", type=float, default=0.1)
    parser.add_argument("--init", type=str, default="observed",
                        choices=("observed", "normal", "moment"),
                        help="initial particles: the logged outcomes, N(0, I), or the "
                             "empirical N(mu, Sigma) of the flow's target support")
    parser.add_argument("--solver", type=str, default="projected_nystrom",
                        choices=("auto", "exact", "nystrom", "projected_nystrom"))
    parser.add_argument("--exact_max_n_obs", type=int, default=10000)
    parser.add_argument("--n_landmarks", type=int, default=1024)
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"))
    parser.add_argument("--lambda_rule", type=str, default="cv", choices=("cv", "n_half"))
    parser.add_argument("--lambda_constant", type=float, default=1.0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--weight_clip", type=float, default=None,
                        help="cap on the estimated weights; None disables. The default "
                             "design has healthy overlap and needs none")
    parser.add_argument("--flow_iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--tag", type=str, default="warfarin")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--warfarin_csv", type=str,
                        default=str(REPO_ROOT / "data" / "warfarin"
                                    / "iwpc_warfarin_subject_data.csv"))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_solver(args: argparse.Namespace) -> str:
    """Which ridge solver this configuration uses; see --solver."""
    if args.solver != "auto":
        return args.solver
    return "exact" if args.n_obs <= args.exact_max_n_obs else "nystrom"


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    if args.dr_fit != "crossfit":
        suffix += f"_dr{args.dr_fit}"
    if args.lambda_rule != "cv":
        suffix += f"_lam{args.lambda_rule}"
    if args.solver == "projected_nystrom":
        suffix += "_projnys"
    elif args.solver == "exact":
        suffix += "_exact"
    elif args.solver == "nystrom":
        suffix += "_nys"
    if args.flow_iterations != 2000:
        suffix += f"_it{args.flow_iterations}"
    name = (f"warfarin_{args.tag}_{args.target_policy}_nobs{args.n_obs}"
            f"_M{args.n_particles}_d{args.delta_scale:g}_th{args.theta:g}"
            f"_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def gaussian_init(reference: torch.Tensor, n_particles: int, mode: str,
                  generator: torch.Generator) -> torch.Tensor:
    """Particles from N(0, I) or the reference sample's empirical N(mu, Sigma).

    The moment-matched draw uses the centred data matrix rather than a Cholesky
    factor, which is exact, needs no jitter, and stays defined when n < d.
    """
    n, dim = reference.shape
    if mode == "normal":
        return torch.randn(n_particles, dim, generator=generator,
                           dtype=reference.dtype)
    mean = reference.mean(dim=0, keepdim=True)
    noise = torch.randn(n_particles, n, generator=generator, dtype=reference.dtype)
    return mean + noise @ (reference - mean) / (n - 1) ** 0.5


def distribution_metrics(sample: torch.Tensor, reference: torch.Tensor,
                         kernel) -> dict:
    """
    Distance between two scalar samples, reported several ways.

    A one-dimensional outcome makes the optimal-transport distances exact: the
    sorted-difference formula is W_p, not an approximation, so `w1`/`w2` need no
    projections. The quantile, tail and CVaR entries are reported because the
    mean-preserving intervention moves the body and the tail of the law by very
    different amounts, and a single aggregate would hide that.
    """
    a = np.sort(np.asarray(sample, dtype=float).reshape(-1))
    b = np.sort(np.asarray(reference, dtype=float).reshape(-1))
    size = min(len(a), len(b))
    grid = (np.arange(size) + 0.5) / size
    qa, qb = np.quantile(a, grid), np.quantile(b, grid)

    tail_a, tail_b = np.mean(a > SEVERE_MISDOSE), np.mean(b > SEVERE_MISDOSE)
    cvar = lambda v: v[v >= np.quantile(v, 0.95)].mean()
    out = {"w1": float(np.abs(qa - qb).mean()),
           "w2": float(np.sqrt(((qa - qb) ** 2).mean())),
           "ks": float(np.abs(np.searchsorted(b, a, "right") / len(b)
                              - (np.arange(len(a)) + 1) / len(a)).max()),
           "mean_err": float(a.mean() - b.mean()),
           "sd_err": float(a.std() - b.std()),
           "tail_err": float(tail_a - tail_b),
           "cvar95_err": float(cvar(a) - cvar(b))}
    for q in (0.05, 0.10, 0.50, 0.90, 0.95):
        out[f"q{int(100 * q):02d}_err"] = float(np.quantile(a, q) - np.quantile(b, q))
    x = torch.as_tensor(a, dtype=torch.float64)[:, None]
    y = torch.as_tensor(b, dtype=torch.float64)[:, None]
    out["mmd2"] = float(squared_mmd(kernel, x, y))
    return out


def exact_cdf_grid(values: np.ndarray, outcome_noise: float,
                   n_body: int = 1200, n_atom: int = 241) -> np.ndarray:
    """
    Points at which to evaluate the exact CDF: uniform over the observed
    support, plus a dense patch around zero.

    The cost has an atom at zero -- every correctly dosed patient scores exactly
    zero -- which the observation noise smears over a width of a few
    `outcome_noise`. That feature is far narrower than the rest of the law
    (tens of mg/week), so a single uniform grid would either miss it or be
    needlessly large. The grid is padded past the largest value so the upper
    tail the samples do not reach still contributes to W1.
    """
    lo = min(float(np.min(values)), 0.0) - 1.0
    hi = float(np.max(values)) * 1.25 + 5.0
    pad = max(6.0 * outcome_noise, 1e-3)
    return np.unique(np.concatenate([np.linspace(lo, hi, n_body),
                                     np.linspace(-pad, pad + 1.0, n_atom)]))


def exact_law_metrics(sample: np.ndarray, grid: np.ndarray,
                      exact_cdf: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov and Wasserstein-1 distance of a scalar sample from the
    EXACT interventional CDF.

    Scored against a finite oracle draw, a sampler can come out ahead of the
    reference purely through the reference's own Monte Carlo error, which makes
    the comparison meaningless. The exact law has no such error. In one
    dimension W1 is the integral of the absolute CDF difference, so integrating
    it over `grid` is exact up to the quadrature; KS is evaluated at the sample
    points, where the empirical CDF attains its extremes, with the exact CDF
    interpolated onto them.
    """
    x = np.sort(np.asarray(sample, dtype=float).reshape(-1))
    size = len(x)
    at_sample = np.interp(x, grid, exact_cdf)
    steps = np.arange(1, size + 1) / size
    ks = max(np.abs(steps - at_sample).max(),
             np.abs(steps - 1.0 / size - at_sample).max())
    empirical = np.searchsorted(x, grid, "right") / size
    return {"ks_exact": float(ks),
            "w1_exact": float(np.trapezoid(np.abs(empirical - exact_cdf), grid))}


def run_repeat(cohort: dict, args: argparse.Namespace, data_seed: int,
               repeat: int, device: torch.device) -> dict:
    """One pipeline pass on a fresh logged sample; one row."""
    seed = 1000 * data_seed + repeat
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()

    data = sample_warfarin_data(cohort, n_samples=args.n_obs, theta=args.theta,
                                delta_scale=args.delta_scale,
                                target=args.target_policy,
                                tolerance=args.tolerance,
                                outcome_noise=args.outcome_noise, seed=seed)
    X = data["X"].numpy()
    A = data["A_logged"].numpy()
    Y = data["Y_logged"].double()
    rows = data["row_index"].numpy()[:, None].astype(float)

    # -- nuisances: the logging density is fitted, the target policy is known ---------
    t0 = time.time()
    fold_ids = make_crossfit_fold_ids(args.n_obs, n_folds=args.n_folds, seed=0)
    if args.dr_fit == "nosplit":
        logging_hat = fit_linear_gaussian_policy(X, A)
        log_pi0 = logging_hat.log_prob(X, A)
    else:
        log_pi0 = crossfit_logging_policy_log_prob(X, A, fold_ids=fold_ids)
    log_ratio = data["target_policy"].log_prob(rows, A) - log_pi0
    w_hat = np.exp(log_ratio)
    if args.weight_clip is not None and args.weight_clip > 0:
        w_hat = np.minimum(w_hat, args.weight_clip)
    oracle_w = data["importance_weights"].numpy()
    diagnostics = importance_weight_diagnostics(w_hat, oracle_w)
    oracle_diagnostics = importance_weight_diagnostics(oracle_w)

    # -- DR target embedding over the logged outcomes ---------------------------------
    solver = resolve_solver(args)
    kernel_x = RBF(use_length_scale_heuristic=True)
    kernel_a = RBF(use_length_scale_heuristic=True)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    A_t = torch.as_tensor(A, dtype=torch.float64)
    anchor = torch.randperm(args.n_obs, generator=torch.Generator().manual_seed(0))[:4000]
    kernel_x.calculate_and_fix_length_scale(X_t[anchor])
    kernel_a.calculate_and_fix_length_scale(A_t[anchor])

    # The target policy is stochastic, so the target action kernel is the policy
    # average E_{a ~ pi(.|x_j)} k_A(a_i, a); approximated by the same fixed set of
    # draws for every block so the Gram is consistent across calls.
    rng = np.random.default_rng(seed + 11)
    draws = [data["target_policy"].sample(rows, rng) for _ in range(8)]
    draws = [torch.as_tensor(d, dtype=torch.float64) for d in draws]

    gram_x = lambda r, c: kernel_x(X_t[r], X_t[c])
    gram = lambda r, c: gram_x(r, c) * kernel_a(A_t[r], A_t[c])
    def gram_pi(r, c):
        total = sum(kernel_a(A_t[r], draw[c]) for draw in draws) / len(draws)
        return gram_x(r, c) * total

    if args.lambda_rule == "n_half":
        ridge = args.lambda_constant * args.n_obs ** -0.5
        fold_lambdas = {int(f): ridge for f in np.unique(fold_ids)}
    else:
        pick = torch.randperm(args.n_obs,
                              generator=torch.Generator().manual_seed(0))[:2000].numpy()
        ridge, _ = select_regularization(gram(pick, pick), Y[pick, 0], seed=0)
        fold_lambdas = {int(f): ridge for f in np.unique(fold_ids)}

    weights = torch.as_tensor(w_hat, dtype=torch.float64)
    landmark_generator = torch.Generator().manual_seed(seed)
    if solver in ("nystrom", "projected_nystrom"):
        projected = solver == "projected_nystrom"
        if args.dr_fit == "nosplit":
            solve = (cpme_dr_weights_projected_nystrom if projected
                     else cpme_dr_weights_nystrom)
            v_dr = solve(gram, gram_pi, weights, ridge,
                         n_landmarks=args.n_landmarks, generator=landmark_generator)
        else:
            solve = (cpme_dr_weights_crossfit_projected_nystrom if projected
                     else cpme_dr_weights_crossfit_nystrom)
            v_dr = solve(gram, gram_pi, weights, fold_ids, fold_lambdas,
                         n_landmarks=args.n_landmarks, generator=landmark_generator)
    else:
        index = np.arange(args.n_obs)
        K_x, K_a, K_pi = gram_x(index, index), kernel_a(A_t, A_t), gram_pi(index, index)
        K_a_target = K_pi / K_x.clamp_min(1e-300)
        if args.dr_fit == "nosplit":
            v_dr = cpme_dr_weights(K_x, K_a, K_a_target, weights, ridge)
        else:
            v_dr = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, weights,
                                            fold_ids, fold_lambdas)
    seconds_nuisance = time.time() - t0

    # -- particle flow in outcome space -----------------------------------------------
    t0 = time.time()
    flow_target = Y.to(device)
    n_particles = min(args.n_particles, args.n_obs)
    generator = torch.Generator().manual_seed(seed + 3)
    if args.init == "observed":
        pick = torch.randperm(args.n_obs, generator=generator)[:n_particles]
        initial = flow_target[pick.to(device)].clone()
    else:
        initial = gaussian_init(Y, n_particles, args.init, generator).to(device)

    torch.manual_seed(seed + 1)
    flowed, info = adamw_riesz_flow(
        initial.clone(), flow_target, target_weights=v_dr.to(device), sliced=False,
        n_iterations=args.flow_iterations, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, lr_schedule="cosine", verbose=False)
    seconds_flow = time.time() - t0

    # -- evaluation against fresh oracle draws ----------------------------------------
    t0 = time.time()
    reference = sample_warfarin_data(cohort, n_samples=args.n_ref, theta=args.theta,
                                     delta_scale=args.delta_scale,
                                     target=args.target_policy,
                                     tolerance=args.tolerance,
                                     outcome_noise=args.outcome_noise,
                                     seed=100000 + seed)["Y_target"].numpy().ravel()
    floor_draw = sample_warfarin_data(cohort, n_samples=args.n_ref, theta=args.theta,
                                      delta_scale=args.delta_scale,
                                      target=args.target_policy,
                                      tolerance=args.tolerance,
                                      outcome_noise=args.outcome_noise,
                                      seed=200000 + seed)["Y_target"].numpy().ravel()
    metric_kernel = RBF(use_length_scale_heuristic=True)
    metric_kernel.calculate_and_fix_length_scale(
        torch.as_tensor(reference, dtype=torch.float64)[:, None])
    samples = {"logged": Y.numpy().ravel()[:n_particles],
               "initial": initial.detach().cpu().numpy().ravel(),
               "flow": flowed.detach().cpu().numpy().ravel(),
               "floor": floor_draw}
    metrics = {f"{name}_{row}": value
               for row, sample in samples.items()
               for name, value in distribution_metrics(sample, reference,
                                                       metric_kernel).items()}

    # The same rows scored against the closed-form law, which carries no Monte
    # Carlo error of its own. One CDF evaluation is shared by every row.
    grid = exact_cdf_grid(np.concatenate(list(samples.values()) + [reference]),
                          args.outcome_noise)
    exact_cdf = interventional_cdf(cohort, grid, target=args.target_policy,
                                   theta=args.theta, delta_scale=args.delta_scale,
                                   tolerance=args.tolerance,
                                   outcome_noise=args.outcome_noise)
    metrics.update({f"{name}_{row}": value
                    for row, sample in samples.items()
                    for name, value in exact_law_metrics(sample, grid,
                                                         exact_cdf).items()})
    seconds_eval = time.time() - t0

    action_logged = A.ravel()
    action_target = data["A_target"].numpy().ravel()
    row = {
        "tag": args.tag, "device": str(device), "target_policy": args.target_policy,
        "init": args.init, "solver": solver, "dr_fit": args.dr_fit,
        "lambda_rule": args.lambda_rule, "data_seed": data_seed, "repeat": repeat,
        "pipeline_seed": seed, "n_obs": args.n_obs,
        "n_cohort": len(cohort["ther_dose"]),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "n_ref": args.n_ref,
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "n_folds": args.n_folds, "theta": args.theta,
        "delta_scale": args.delta_scale, "delta": data["policy_info"]["delta"],
        "tolerance": args.tolerance, "outcome_noise": args.outcome_noise,
        "flow_iterations": args.flow_iterations, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_clip": args.weight_clip if args.weight_clip is not None else 0.0,
        "oracle_w_ess": oracle_diagnostics["ess"], "est_w_ess": diagnostics["ess"],
        "est_w_max": diagnostics["max"], "est_w_log_rmse": diagnostics.get("log_rmse"),
        "sum_v": float(v_dr.sum()), "sum_abs_v": float(v_dr.abs().sum()),
        "action_mean_logged": float(action_logged.mean()),
        "action_mean_target": float(action_target.mean()),
        "action_sd_logged": float(action_logged.std()),
        "action_sd_target": float(action_target.std()),
        "loss_initial": float(info["loss"][0]), "loss_final": float(info["loss"][-1]),
        "mean_transport": float((flowed - initial).norm(dim=1).mean()),
        "seconds_nuisance": round(seconds_nuisance, 1),
        "seconds_flow": round(seconds_flow, 1),
        "seconds_eval": round(seconds_eval, 1),
        "seconds_repeat": round(time.time() - started, 1),
    }
    row.update(metrics)
    return row


def write_rows(path: Path, rows: list) -> None:
    """Atomic-ish rewrite after every repeat, so a timeout keeps partial results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    csv_file = Path(args.warfarin_csv)
    if not csv_file.exists():
        sys.exit(f"warfarin table not found at {csv_file}; see data/warfarin/SOURCE.txt")
    cohort = load_warfarin_cohort(str(csv_file))
    out_path = csv_path(args)
    print(f"device: {device}   cohort: {len(cohort['ther_dose'])} patients, "
          f"{cohort['X'].shape[1]} covariates", flush=True)
    print(f"output: {out_path}", flush=True)

    rows = []
    for data_seed in args.data_seeds:
        for repeat in range(args.n_repeats):
            row = run_repeat(cohort, args, data_seed, repeat, device)
            rows.append(row)
            write_rows(out_path, rows)
            print(f"[seed {data_seed} repeat {repeat}] "
                  f"W1 logged {row['w1_logged']:.3f} -> flow {row['w1_flow']:.3f} "
                  f"(floor {row['w1_floor']:.3f})   "
                  f"exact W1 {row['w1_exact_logged']:.3f} -> "
                  f"{row['w1_exact_flow']:.3f} (floor {row['w1_exact_floor']:.3f})   "
                  f"tail err {row['tail_err_flow']:+.4f}   "
                  f"{row['seconds_repeat']:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
