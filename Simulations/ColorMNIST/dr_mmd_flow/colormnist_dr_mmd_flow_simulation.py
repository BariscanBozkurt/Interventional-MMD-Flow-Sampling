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
evaluation against fresh oracle draws (colour-ratio law, SW1, SW2, RBF MMD^2) for
the initial particles, the flowed particles, and the oracle-vs-oracle floor.

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
from causality_utils import (cpme_dr_weights_crossfit, make_crossfit_fold_ids,
                             select_regularization)
from dens_ratio_utils import crossfit_propensity
from mmd_flow_utils import mmd_gradient_flow, squared_mmd
from metric_utils import sliced_wasserstein

COLUMNS = [
    "tag", "device", "data_seed", "repeat", "pipeline_seed", "target_a",
    "n_obs", "w_conf", "tau", "alpha", "k_mask",
    "n_ae", "n_causal", "n_arm_causal", "n_dr", "n_folds",
    "n_particles_requested", "n_particles", "latent_dim",
    "ae_iterations", "flow_iterations",
    "recon_mse_train", "recon_mse_heldout",
    "ess_on_arm", "n_on_arm", "max_weight", "sum_v", "sum_abs_v", "fold_lambdas",
    "c_imq", "step_size", "objective_initial", "objective_final",
    "colour_mean_initial", "colour_std_initial", "sw1_initial", "sw2_initial", "mmd2_initial",
    "colour_mean_flow", "colour_std_flow", "sw1_flow", "sw2_flow", "mmd2_flow",
    "colour_mean_floor", "colour_std_floor", "sw1_floor", "sw2_floor", "mmd2_floor",
    "colour_mean_oracle", "colour_std_oracle",
    "seconds_ae", "seconds_nuisance", "seconds_flow", "seconds_repeat",
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=20000,
                        help="logged sample size n")
    parser.add_argument("--n_particles", type=int, default=1600,
                        help="flow particles M (capped at the on-arm causal-split size)")
    parser.add_argument("--n_dr", type=int, default=4000,
                        help="DR subsample size (capped at the causal-split size)")
    parser.add_argument("--ae_fraction", type=float, default=0.5,
                        help="fraction of the logged sample used to train the autoencoder")
    parser.add_argument("--target_arms", type=int, nargs="+", default=[0, 1],
                        choices=(0, 1), help="intervention arms run in each repeat")
    parser.add_argument("--w_conf", type=float, default=10.0,
                        help="confounding strength w in e(1|x) = sigmoid(w (x - 1/2))")
    parser.add_argument("--tau", type=float, default=0.08, help="foreground mask threshold")
    parser.add_argument("--alpha", type=float, default=0.0, help="fixed green channel")
    parser.add_argument("--k_mask", type=float, default=10.0, help="foreground mask sharpness")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="pipeline repeats per data seed")
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
    name = (f"colormnist_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
            f"_w{args.w_conf:g}_a{args.alpha:g}_k{args.k_mask:g}_tau{args.tau:g}"
            f"_arms{arms}_ds{seeds}.csv")
    return Path(args.out_dir) / name


def evaluate_images(images_01: torch.Tensor, reference_01: torch.Tensor,
                    kernel_pixel: RBF) -> dict:
    """Colour law plus flat-pixel SW1 / SW2 / unbiased RBF MMD^2 against the reference."""
    flatten = lambda images: images.reshape(images.shape[0], -1)
    ratio = color_ratio(images_01)
    return {
        "colour_mean": float(ratio.mean()), "colour_std": float(ratio.std()),
        "sw1": sliced_wasserstein(flatten(images_01), flatten(reference_01), p=1, seed=0).item(),
        "sw2": sliced_wasserstein(flatten(images_01), flatten(reference_01), p=2, seed=0).item(),
        "mmd2": squared_mmd(kernel_pixel, flatten(images_01), flatten(reference_01),
                            unbiased=True).item(),
    }


def run_arm(target_a: int, ae, pools: list, data: torch.Tensor,
            causal_idx: torch.Tensor, A_obs: torch.Tensor, A_s: torch.Tensor,
            Z_s: torch.Tensor, e1: torch.Tensor, K_x: torch.Tensor, K_a: torch.Tensor,
            fold_ids: np.ndarray, fold_lambdas: dict, args: argparse.Namespace,
            seed: int, device: torch.device) -> dict:
    """Arm-specific half of one repeat: DR weights, flow, and oracle evaluation."""
    n_dr = len(A_s)
    e_target = e1 if target_a == 1 else 1.0 - e1
    w_hat = (A_s.cpu() == target_a).double() / e_target
    on_arm = w_hat[A_s.cpu() == target_a]
    ess = float(on_arm.sum() ** 2 / (on_arm ** 2).sum())
    K_a_target = (A_s == target_a).double().cpu()[:, None].expand(n_dr, n_dr).contiguous()
    v_dr = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_hat, fold_ids,
                                    fold_lambdas).float().to(device)

    t0 = time.time()
    arm_causal = causal_idx[A_obs[causal_idx] == target_a]
    Z_arm = encode_in_batches(ae.encode, data[arm_causal])
    n_particles = min(args.n_particles, Z_arm.shape[0])
    pick = torch.randperm(Z_arm.shape[0],
                          generator=torch.Generator(device=device).manual_seed(seed + 1),
                          device=device)[:n_particles]
    initial_particles = Z_arm[pick].clone()

    c_imq = imq_c_from_median(Z_s)
    kernel = IMQKernel(c=c_imq, beta=-0.5)
    step_size = 0.15 * c_imq ** 2
    particles, history = mmd_gradient_flow(kernel, target_samples=Z_s,
                                           initial_particles=initial_particles,
                                           step_size=step_size,
                                           n_iterations=args.flow_iterations,
                                           target_weights=v_dr, log_every=500, verbose=False)
    seconds_flow = time.time() - t0

    with torch.no_grad():
        decoded_initial = to_image_range(ae.decode(initial_particles))
        decoded_flowed = to_image_range(ae.decode(particles))
    _, Y_oracle = sample_interventional(pools, n_particles, a=target_a, tau=args.tau,
                                        alpha=args.alpha, k_mask=args.k_mask,
                                        generator=torch.Generator(device=device).manual_seed(seed + 9))
    _, Y_oracle_2 = sample_interventional(pools, n_particles, a=target_a, tau=args.tau,
                                          alpha=args.alpha, k_mask=args.k_mask,
                                          generator=torch.Generator(device=device).manual_seed(seed + 10))
    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(Y_oracle.reshape(n_particles, -1))  # lock the bandwidth on the reference

    ratio_oracle = color_ratio(Y_oracle)
    row = {
        "target_a": target_a,
        "n_arm_causal": int(len(arm_causal)),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "ess_on_arm": ess, "n_on_arm": int(len(on_arm)), "max_weight": float(w_hat.max()),
        "sum_v": float(v_dr.sum()), "sum_abs_v": float(v_dr.abs().sum()),
        "c_imq": float(c_imq), "step_size": float(step_size),
        "objective_initial": float(history["squared_mmd"][0]),
        "objective_final": float(history["squared_mmd"][-1]),
        "colour_mean_oracle": float(ratio_oracle.mean()),
        "colour_std_oracle": float(ratio_oracle.std()),
        "seconds_flow": round(seconds_flow, 1),
    }
    for name, images in (("initial", decoded_initial), ("flow", decoded_flowed),
                         ("floor", Y_oracle_2)):
        metrics = evaluate_images(images, Y_oracle, kernel_pixel)
        row.update({f"{key}_{name}": value for key, value in metrics.items()})
    return row


def run_repeat(pools: list, data: torch.Tensor, X_obs: torch.Tensor, A_obs: torch.Tensor,
               args: argparse.Namespace, data_seed: int, repeat: int,
               device: torch.device) -> list:
    """One pipeline pass on a fixed logged sample; returns one CSV row per arm."""
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- split: autoencoder half / causal half (disjoint) ------------------------------
    n_ae = int(round(args.ae_fraction * args.n_obs))
    split = torch.randperm(args.n_obs, generator=torch.Generator(device=device).manual_seed(seed + 2),
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
    n_dr = min(args.n_dr, len(causal_idx))
    pick_dr = torch.randperm(len(causal_idx),
                             generator=torch.Generator(device=device).manual_seed(seed + 5),
                             device=device)[:n_dr]
    subsample = causal_idx[pick_dr]
    X_s, A_s = X_obs[subsample], A_obs[subsample]
    Z_s = encode_in_batches(ae.encode, data[subsample])

    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    torch.manual_seed(seed)
    e1 = crossfit_propensity(X_s.unsqueeze(1).double().cpu(), A_s.double().cpu(), fold_ids)

    kernel_x = RBF(use_length_scale_heuristic=True)
    K_x = kernel_x(X_s.unsqueeze(1).double().cpu())
    K_a = (A_s[:, None] == A_s[None, :]).double().cpu()
    K_pair = K_x * K_a
    fold_lambdas = {}
    for fold in np.unique(fold_ids):
        train = np.where(fold_ids != fold)[0]
        fold_lambdas[int(fold)], _ = select_regularization(K_pair[train][:, train],
                                                           Z_s[train, 0].double().cpu(), seed=0)
    seconds_nuisance = time.time() - t0

    # -- per-arm: DR weights, flow, evaluation -----------------------------------------
    shared = {
        "tag": args.tag, "device": str(device),
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
                           e1, K_x, K_a, fold_ids, fold_lambdas, args, seed, device))
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

    rows = []
    for data_seed in args.data_seeds:
        generator = torch.Generator(device=device).manual_seed(data_seed)
        X_obs, A_obs, Y_obs = sample_observed(pools, args.n_obs, w_conf=args.w_conf,
                                              tau=args.tau, alpha=args.alpha,
                                              k_mask=args.k_mask, generator=generator)
        data = to_model_range(Y_obs)
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(pools, data, X_obs, A_obs, args, data_seed, repeat, device)
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']}] "
                      f"colour flow {row['colour_mean_flow']:.3f} "
                      f"(initial {row['colour_mean_initial']:.3f}, oracle {row['colour_mean_oracle']:.3f})   "
                      f"SW2 flow {row['sw2_flow']:.4f} (floor {row['sw2_floor']:.4f})   "
                      f"ESS {row['ess_on_arm']:.0f}/{row['n_on_arm']}   "
                      f"{row['seconds_repeat']:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
