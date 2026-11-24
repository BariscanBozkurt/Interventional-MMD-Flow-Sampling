"""
ColorMNIST solver timing benchmark: exact full ridge against the two Nystrom solvers.

WHY THIS EXISTS. The solver timings that can be recovered from the headline CSVs are
confounded, in five ways that this script removes by construction.

    1. `seconds_cme` is not the solve. It bundles the ridge SELECTION (a cross-validation
       over a fixed 2000-row subsample), the Gram PREPARATION and the solve into one
       window. For `solver=exact` the preparation includes forming the dense n x n blocks
       `K_x` and `K_a` -- 3.2 GB in float64 at n = 20000 -- which the Nystrom paths never
       form. At n = 20000 that construction, not the solve, is most of the 49.3 s.
    2. The two sides were never paired. `exact` rows exist only at `dr_fit=nosplit` and
       only at n in {5000, 10000, 20000}; the Nystrom rows span four sample sizes, both
       fitting schemes and two GPUs. Solver is therefore entangled with `dr_fit`, with
       sample size and with hardware.
    3. The rows were pooled from launcher tasks that ran at different times on a shared
       GPU, with unequal repeat counts (6, 12 and 30 rows per cell), so contention enters
       the mean unevenly.
    4. Nothing was warmed up. The first CUDA kernel of a process pays lazy initialisation
       and autotuning, which lands on whichever solver happened to run first.
    5. The sampling half was never separated from the training half, although the solver
       changes BOTH: a Nystrom target expansion carries fewer support points, so its flow
       is genuinely cheaper, and that effect was tangled with the contention noise.

WHAT THIS SCRIPT DOES INSTEAD. For each (sample size, data seed, repeat) it builds the
logged sample, trains the autoencoder, fits the propensity and selects the ridge ONCE,
then runs every solver against those identical inputs inside one process on one device.
Each solver is warmed up before it is timed, and the solver ORDER is rotated across
repeats so no solver permanently occupies the cold slot. The phases are timed separately:

    seconds_gram        forming the solver's kernel representation. The dense n x n
                        blocks for `exact`; zero for the Nystrom solvers, which address
                        the kernel through closures and never materialise it.
    seconds_solve       the target-weight solve itself, at a FIXED ridge. This is the
                        clean algorithmic comparison.
    seconds_cv          selecting that ridge. Measured once per repeat and written onto
                        every solver row of it, because the selection runs on a fixed
                        2000-row subsample through the same closures for every solver:
                        it is the SAME work regardless of solver and does not grow with
                        n. It is reported so that "training with tuning" can be formed as
                        seconds_gram + seconds_solve + seconds_cv, and so that the reader
                        can see it is not what separates the solvers.
    seconds_training    seconds_gram + seconds_solve + seconds_cv.
    seconds_flow        the MMD flow.
    seconds_decode      decoding the flowed particles.
    seconds_sampling    seconds_flow + seconds_decode.

Excluded from both totals, recorded for completeness: `seconds_ae` (autoencoder training
and the encoder passes) and `seconds_ratio` (the propensity half), which are shared by
every solver and identical within a repeat by construction.

The exact solver is allowed to fail. It forms THREE dense n x n float64 blocks (K_x,
K_a and the expanded K_a_target), 3.2 GB each at n = 20000, so beyond some sample size
it will not fit; such a cell is written with `status=oom` and NaN timings rather than
crashing the ladder, and the run continues. The ladder stops at n = 20000 because the
ColorMNIST digit pools hold 12660 images in total: every larger sample is drawn with
replacement, as the headline runs at n = 20000 already are, and a timing ladder built
on increasingly duplicated images measures less and less.

`latent_sw2_flow` is recorded on every row so that a timing win can be read against what
it costs statistically: a faster solver that samples a worse distribution is not faster.

Launched by the SLURM template in `slurm_templates/`; single-run example:

    python colormnist_solver_timing_simulation.py --n_obs_ladder 5000 10000 \
        --solvers exact projected_nystrom --data_seeds 0 --n_repeats 2
"""

import argparse
import csv
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

from data_utils import (load_digit_pools, sample_interventional, sample_observed,
                        to_image_range, to_model_range)
from autoencoder_utils import encode_in_batches, train_conv_ae
from kernel_utils import IMQKernel, RBF, imq_c_from_median
from causality_utils import (cpme_dr_weights, cpme_dr_weights_nystrom,
                             cpme_dr_weights_projected_nystrom,
                             make_crossfit_fold_ids, select_regularization)
from dens_ratio_utils import LogisticPropensityModel
from mmd_flow_utils import mmd_gradient_flow
from metric_utils import sliced_wasserstein

#: Written in this order. One row per (sample size, data seed, repeat, solver).
COLUMNS = [
    "tag", "device", "gpu_name", "hostname", "status",
    "n_obs", "n_dr", "data_seed", "repeat", "solver", "order_index",
    "n_landmarks", "dr_fit", "target_a", "n_particles", "flow_iterations",
    "seconds_gram", "seconds_solve", "seconds_cv", "seconds_training",
    "seconds_flow", "seconds_decode", "seconds_sampling",
    "seconds_ae", "seconds_ratio",
    "support_size", "sum_v", "sum_abs_v", "objective_final", "latent_sw2_flow",
]

SOLVERS = ("exact", "nystrom", "projected_nystrom")


def sync_device(device) -> None:
    """Block until the device's queued work has finished, so a wall clock is honest."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def hardware_provenance(device: torch.device) -> dict:
    """Which machine produced a row, so placement is auditable after the fact."""
    name = (torch.cuda.get_device_name(device) if torch.device(device).type == "cuda"
            else "cpu")
    return {"device": str(device), "gpu_name": name, "hostname": socket.gethostname()}


def is_oom(error: BaseException) -> bool:
    """Whether `error` is an out-of-memory failure rather than a genuine bug."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower())


def parse_args() -> argparse.Namespace:
    """Command line. Defaults match the ColorMNIST headline configuration."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_obs_ladder", type=int, nargs="+",
                   default=[5000, 10000, 15000, 20000],
                   help="sample sizes to time, in the order given")
    p.add_argument("--solvers", type=str, nargs="+", default=list(SOLVERS),
                   choices=SOLVERS)
    p.add_argument("--data_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--target_a", type=int, default=0,
                   help="one arm is enough for a timing comparison; the solve is per-arm")
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed passes of every solver before the timed repeats, per "
                        "sample size. 0 disables, which leaves the first solver cold")
    p.add_argument("--exact_max_n_obs", type=int, default=0,
                   help="skip `exact` above this sample size. 0 (default) attempts every "
                        "rung and records status=oom where it does not fit")
    # --- held at the headline configuration ------------------------------------------
    p.add_argument("--n_particles", type=int, default=2000)
    p.add_argument("--n_ref", type=int, default=2000)
    p.add_argument("--n_landmarks", type=int, default=1024)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--dr_fit", type=str, default="nosplit", choices=("nosplit",),
                   help="cross-fitting multiplies the solve by the fold count without "
                        "changing which solver is faster, so it is held out")
    p.add_argument("--w_conf", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.08)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--k_mask", type=float, default=20.0)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--ae_iterations", type=int, default=3000)
    p.add_argument("--flow_iterations", type=int, default=3000)
    p.add_argument("--mnist_root", type=str, default=str(REPO_ROOT / "data"))
    p.add_argument("--out_dir", type=str,
                   default=str(Path(__file__).resolve().parent / "results_timing"))
    p.add_argument("--tag", type=str, default="solvertiming")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def shared_inputs(pools, args, n_obs: int, data_seed: int, repeat: int,
                  device: torch.device) -> dict:
    """Everything every solver reads, built once so the solvers are exactly paired.

    Mirrors the headline simulation's own sequence: the logged draw, the autoencoder on
    the whole sample, the encoder passes, the propensity and the ridge selection.

    Parameters
    ----------
    pools : list
        MNIST digit pools from `load_digit_pools`.
    args : argparse.Namespace
    n_obs, data_seed, repeat : int
        This cell's sample size and seeds; the pipeline seed is 1000 * data_seed + repeat.
    device : torch.device

    Returns
    -------
    dict
        The Gram closures, the importance weights, the selected ridge, the DR support's
        latents, the starting particles, the flow kernel and step, the oracle latents,
        and the shared `seconds_ae` / `seconds_ratio` / `seconds_cv`.
    """
    seed = 1000 * data_seed + repeat
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = torch.Generator(device=device).manual_seed(data_seed)
    X_obs, A_obs, Y_obs = sample_observed(pools, n_obs, w_conf=args.w_conf, tau=args.tau,
                                          alpha=args.alpha, k_mask=args.k_mask,
                                          generator=generator)
    data = to_model_range(Y_obs)

    # -- autoencoder: shared, excluded from both totals -------------------------------
    t0 = time.time()
    ae, _ = train_conv_ae(data, latent_dim=args.latent_dim,
                          n_iterations=args.ae_iterations, batch_size=128,
                          learning_rate=1e-3, device=device,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          log_every=max(1, args.ae_iterations // 2), verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    Z_s = encode_in_batches(ae.encode, data)
    sync_device(device)
    seconds_ae = time.time() - t0

    # -- propensity half: shared, excluded ---------------------------------------------
    t0 = time.time()
    n_dr = n_obs
    X_s, A_s = X_obs, A_obs
    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    model = LogisticPropensityModel()
    model.fit(X_s.unsqueeze(1).double().cpu(), A_s.double().cpu())
    e1 = model.predict(X_s.unsqueeze(1).double().cpu()).reshape(-1).double().to(device)
    e_target = e1 if args.target_a == 1 else 1.0 - e1
    w_hat = ((A_s == args.target_a).double() / e_target).to(device)
    sync_device(device)
    seconds_ratio = time.time() - t0

    # -- kernel closures: the representation every solver addresses --------------------
    kernel_x = RBF(use_length_scale_heuristic=True)
    X_dr = X_s.unsqueeze(1).double().to(device)
    A_dr = A_s.to(device)
    anchor = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:4000]
    _ = kernel_x(X_dr[anchor])            # lock the lengthscale on a bounded subsample

    def gram_x(rows, cols):
        "K_X block alone; K_pi = K_X * 1{a_i == a} needs it without the action factor."
        return kernel_x(X_dr[rows], X_dr[cols])

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_dr^2 Gram is never formed."
        return gram_x(rows, cols) * (A_dr[rows][:, None] == A_dr[cols][None, :]).double()

    # -- ridge selection: identical work for every solver, so timed once ---------------
    t0 = time.time()
    pick = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:2000].numpy()
    nosplit_lambda, _ = select_regularization(gram(pick, pick),
                                              Z_s[pick, 0].double().to(device), seed=0)
    sync_device(device)
    seconds_cv = time.time() - t0

    # -- sampling-side inputs ----------------------------------------------------------
    arm = torch.where(A_obs == args.target_a)[0]
    Z_arm = Z_s[arm]
    n_particles = min(args.n_particles, Z_arm.shape[0])
    pick_particles = torch.randperm(
        Z_arm.shape[0], generator=torch.Generator().manual_seed(seed + 1))[:n_particles]
    initial_particles = Z_arm[pick_particles.to(device)].clone()
    c_imq = imq_c_from_median(Z_s)
    _, Y_oracle = sample_interventional(
        pools, args.n_ref, a=args.target_a, tau=args.tau, alpha=args.alpha,
        k_mask=args.k_mask, generator=torch.Generator(device=device).manual_seed(seed + 9))
    with torch.no_grad():
        Z_oracle = encode_in_batches(ae.encode, to_model_range(Y_oracle))

    return {"ae": ae, "A_s": A_s, "Z_s": Z_s, "w_hat": w_hat, "gram": gram,
            "gram_x": gram_x, "X_dr": X_dr, "A_dr": A_dr, "n_dr": n_dr,
            "fold_ids": fold_ids, "nosplit_lambda": nosplit_lambda,
            "initial_particles": initial_particles, "kernel": IMQKernel(c=c_imq, beta=-0.5),
            "step_size": 0.15 * c_imq ** 2, "Z_oracle": Z_oracle, "seed": seed,
            "seconds_ae": seconds_ae, "seconds_ratio": seconds_ratio,
            "seconds_cv": seconds_cv}


def solve_target_weights(solver: str, shared: dict, args, device):
    """One solver's target weights, with its Gram construction timed separately.

    Returns
    -------
    (torch.Tensor, float, float)
        The DR weight vector, `seconds_gram` and `seconds_solve`.
    """
    A_s, w_hat, n_dr = shared["A_s"], shared["w_hat"], shared["n_dr"]
    lam = shared["nosplit_lambda"]

    t0 = time.time()
    if solver == "exact":
        # The dense blocks the exact ridge needs. O(n^2) in memory, and the reason the
        # exact path stops fitting: 3.2 GB in float64 at n = 20000, 12.8 GB at n = 40000.
        K_x = shared["gram_x"](slice(None), slice(None))
        K_a = (shared["A_dr"][:, None] == shared["A_dr"][None, :]).double()
        K_a_target = ((A_s == args.target_a).double().to(device)[:, None]
                      .expand(n_dr, n_dr).contiguous())
    sync_device(device)
    seconds_gram = time.time() - t0

    t0 = time.time()
    if solver == "exact":
        v = cpme_dr_weights(K_x, K_a, K_a_target, w_hat, lam)
    else:
        on_arm_flag = (A_s == args.target_a).double().to(device)
        gram_pi = lambda rows, cols: shared["gram_x"](rows, cols) * on_arm_flag[rows][:, None]
        solve = (cpme_dr_weights_projected_nystrom if solver == "projected_nystrom"
                 else cpme_dr_weights_nystrom)
        v = solve(shared["gram"], gram_pi, w_hat, lam,
                  n_landmarks=args.n_landmarks,
                  generator=torch.Generator().manual_seed(shared["seed"]))
    sync_device(device)
    seconds_solve = time.time() - t0

    if solver == "exact":
        del K_x, K_a, K_a_target
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return v.float().to(device), seconds_gram, seconds_solve


def flow_and_time(v_target, shared: dict, args, device):
    """This solver's sampling half: the flow and the decode, timed apart.

    Returns
    -------
    dict
        `seconds_flow`, `seconds_decode`, `objective_final` and `latent_sw2_flow`.
    """
    t0 = time.time()
    particles, history = mmd_gradient_flow(
        shared["kernel"], target_samples=shared["Z_s"],
        initial_particles=shared["initial_particles"], step_size=shared["step_size"],
        n_iterations=args.flow_iterations, target_weights=v_target,
        snapshot_every=None, log_every=500, verbose=False)
    sync_device(device)
    seconds_flow = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        _ = to_image_range(shared["ae"].decode(particles))
    sync_device(device)
    seconds_decode = time.time() - t0

    return {"seconds_flow": seconds_flow, "seconds_decode": seconds_decode,
            "objective_final": float(history["squared_mmd"][-1]),
            "latent_sw2_flow": sliced_wasserstein(particles, shared["Z_oracle"],
                                                  p=2, seed=0).item()}


def time_one_solver(solver: str, shared: dict, args, device, order_index: int,
                    n_obs: int, data_seed: int, repeat: int, provenance: dict) -> dict:
    """One CSV row: this solver's training and sampling cost on the shared inputs."""
    row = {c: "" for c in COLUMNS}
    row.update(provenance)
    row.update({"tag": args.tag, "n_obs": n_obs, "n_dr": shared["n_dr"],
                "data_seed": data_seed, "repeat": repeat, "solver": solver,
                "order_index": order_index, "n_landmarks": args.n_landmarks,
                "dr_fit": args.dr_fit, "target_a": args.target_a,
                "n_particles": int(shared["initial_particles"].shape[0]),
                "flow_iterations": args.flow_iterations,
                "seconds_ae": round(shared["seconds_ae"], 2),
                "seconds_ratio": round(shared["seconds_ratio"], 2),
                "seconds_cv": round(shared["seconds_cv"], 2)})
    try:
        v, seconds_gram, seconds_solve = solve_target_weights(solver, shared, args, device)
    except Exception as error:                       # noqa: BLE001 - re-raised unless OOM
        if not is_oom(error):
            raise
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        print(f"      {solver}: out of memory at n_obs={n_obs}", flush=True)
        row["status"] = "oom"
        return row

    sampling = flow_and_time(v, shared, args, device)
    training = seconds_gram + seconds_solve + shared["seconds_cv"]
    row.update({
        "status": "ok",
        "seconds_gram": round(seconds_gram, 2),
        "seconds_solve": round(seconds_solve, 2),
        "seconds_training": round(training, 2),
        "seconds_flow": round(sampling["seconds_flow"], 2),
        "seconds_decode": round(sampling["seconds_decode"], 2),
        "seconds_sampling": round(sampling["seconds_flow"] + sampling["seconds_decode"], 2),
        "support_size": int((v != 0).sum()),
        "sum_v": round(float(v.sum()), 6),
        "sum_abs_v": round(float(v.abs().sum()), 6),
        "objective_final": sampling["objective_final"],
        "latent_sw2_flow": round(sampling["latent_sw2_flow"], 6)})
    return row


def solver_order(solvers: list, repeat: int) -> list:
    """The solvers rotated by `repeat`, so none permanently holds the cold first slot."""
    k = repeat % len(solvers)
    return solvers[k:] + solvers[:k]


def write_rows(out_path: Path, rows: list) -> None:
    """Rewrite the CSV from scratch, so a resumed run cannot half-append."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    provenance = hardware_provenance(device)
    if not (Path(args.mnist_root) / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
        sys.exit(f"MNIST not found under {args.mnist_root}/MNIST/raw — "
                 "copy the repo's data/ folder there (no download on compute nodes).")

    seeds = "-".join(str(s) for s in args.data_seeds)
    ladder = "-".join(str(n) for n in args.n_obs_ladder)
    out_path = Path(args.out_dir) / f"colormnist_{args.tag}_n{ladder}_ds{seeds}.csv"
    print(f"device: {device} ({provenance['gpu_name']})   output: {out_path}", flush=True)

    pools = load_digit_pools(root=args.mnist_root, device=device)

    rows = []
    for n_obs in args.n_obs_ladder:
        solvers = [s for s in args.solvers
                   if not (s == "exact" and 0 < args.exact_max_n_obs < n_obs)]
        for data_seed in args.data_seeds:
            for repeat in range(args.n_repeats):
                print(f"  n_obs={n_obs} seed={data_seed} repeat={repeat}", flush=True)
                shared = shared_inputs(pools, args, n_obs, data_seed, repeat, device)
                if repeat == 0:
                    for _ in range(args.warmup):
                        for solver in solvers:
                            time_one_solver(solver, shared, args, device, -1,
                                            n_obs, data_seed, repeat, provenance)
                for order_index, solver in enumerate(solver_order(solvers, repeat)):
                    row = time_one_solver(solver, shared, args, device, order_index,
                                          n_obs, data_seed, repeat, provenance)
                    rows.append(row)
                    print(f"      {solver:18s} gram {row['seconds_gram']:>7} "
                          f"solve {row['seconds_solve']:>7} flow {row['seconds_flow']:>7} "
                          f"({row['status']})", flush=True)
                    write_rows(out_path, rows)
                del shared
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print(f"\n{len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
