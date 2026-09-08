"""
CelebA simulation: cross-fitted DR-CPME target + Riesz MMD flow (AdamW, warm-up + cosine).

Script version of
`NotebookExamples/Interventional_MMD_Flow/CelebA_DR_Riesz_MMD_Flow_AdamW_Female.ipynb`
(and its Male twin): the DecFM CelebA confounding design (binary A = Male, binary
X = Blond_Hair, cells sampled to P(X=1) = 0.3, P(A=1|X=0) = 0.6, P(A=1|X=1) = 0.1),
frozen pretrained SD-VAE latents, cross-fitted DR weights over the exact binary
kernels, and a dense Riesz flow under AdamW with linear warm-up and cosine decay on
standardized latents. No autoencoder split is needed: the encoder-decoder is an
external pretrained model that never sees this data at training time.

Each repeat runs BOTH intervention arms by default, sharing every arm-independent
stage (the cached data draw, the SD-VAE encodings, the cross-fitted propensity and
the per-fold ridge lambdas); only the importance weights, DR weight vector, particle
pick, flow and evaluation are arm-specific. One CSV row per (data_seed, repeat, arm).

Metrics per row: latent SW2 of the flowed particles against the encoded references
(with the reference split-half floor), and pixel-space SW2 / SW1 / RBF MMD^2 against
the arm's held-out interventional references for the observed arm, its VAE
round-trip, and the flow output — at the native resolution and at 64 x 64 (the
resolution of Dance et al., 2026, Figure 4).

Data caches: one .pt per (image_size, n_obs, n_ref, data_seed), generated on first
use from the raw CelebA root (readable NFS path) and written atomically so that
concurrent array tasks sharing a configuration cannot corrupt it. Generation
materialises every image; at 256 px keep n_obs modest (memory grows as
n_obs x 3 x S^2 floats — n_obs = 20000 at 256 px does NOT fit a 12 GB node; use
--image_size 64 for the DecFM-scale runs, whose defaults ARE the paper's settings).

Seed scheme: `data_seed` fixes the data draw (observed sample and both reference
sets); `pipeline_seed = 1000 * data_seed + repeat` fixes the propensity fit, the DR
subsample (active only when n_dr < n_obs), the particle pick and the flow. At the
default arguments the arm-0 row of (data_seed 0, repeat 0) follows the Female
notebook's seed path exactly.

Launched by `submit_celeba_dr_riesz_flow_parallel.sh`; single-run example:

    python celeba_dr_riesz_flow_simulation.py --n_obs 2048 --data_seeds 0 --n_repeats 3
"""

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

from data_utils import binary_stats, load_celeba_paper
from autoencoder_utils import decode_latents, encode_images, load_sd_vae
from causality_utils import (cpme_dr_weights_crossfit, make_crossfit_fold_ids,
                             select_regularization)
from dens_ratio_utils import crossfit_propensity
from mmd_flow_utils import adamw_riesz_flow, squared_mmd
from metric_utils import sliced_wasserstein
from kernel_utils import RBF

PIXEL_ROWS = ("observed", "roundtrip", "flow")

COLUMNS = (
    ["tag", "device", "data_seed", "repeat", "pipeline_seed", "target_a",
     "image_size", "n_obs", "n_ref", "n_dr", "n_folds", "latent_dim",
     "n_arm", "n_particles_requested", "n_particles", "n_eval",
     "flow_iterations", "learning_rate", "warmup_steps", "lr_schedule",
     "n_a0x0", "n_a0x1", "n_a1x0", "n_a1x1", "obs_blond_frac_arm",
     "ess_on_arm", "n_on_arm", "max_weight", "sum_v", "sum_abs_v", "sum_vX",
     "fold_lambdas",
     "loss_initial", "loss_final", "mean_transport",
     "latent_sw2_initial", "latent_sw2_flow", "latent_sw2_floor"]
    + [f"{metric}_{row}" for row in PIXEL_ROWS for metric in ("sw2", "sw1", "mmd2")]
    + ["sw2_floor", "sw1_floor", "mmd2_floor"]
    + [f"{metric}_64_{row}" for row in PIXEL_ROWS for metric in ("sw2", "sw1")]
    + ["sw2_64_floor", "sw1_64_floor",
       "seconds_encode", "seconds_nuisance", "seconds_flow", "seconds_eval",
       "seconds_repeat"]
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=2048,
                        help="observed sample size")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="held-out interventional references per arm")
    parser.add_argument("--image_size", type=int, default=256,
                        help="image side length (64 = the DecFM paper's resolution)")
    parser.add_argument("--n_particles", type=int, default=1024,
                        help="flow particles M (capped at the observed arm size)")
    parser.add_argument("--n_dr", type=int, default=4096,
                        help="DR support size; rows are subsampled only when n_obs exceeds this "
                             "(keeps the n^2 kernel solves bounded on large-n sweeps)")
    parser.add_argument("--target_arms", type=int, nargs="+", default=[0, 1],
                        choices=(0, 1), help="intervention arms run in each repeat")
    parser.add_argument("--eval_max", type=int, default=None,
                        help="cap on observed/round-trip evaluation rows (None = the full "
                             "arm, the notebook protocol); set on large-n runs to bound "
                             "decode time and GPU memory, e.g. 2048 at n_obs = 16384")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--flow_iterations", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=("const", "cosine"))
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="data-draw seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="pipeline repeats per data seed")
    parser.add_argument("--tag", type=str, default="riesz")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--cache_dir", type=str,
                        default=str(REPO_ROOT / "data" / "celeba_dgp"))
    parser.add_argument("--celeba_root", type=str,
                        default="/nfs/ghome/live/danceh/data/celeba",
                        help="raw CelebA root; needed only when a cache must be generated")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available")
    return parser.parse_args()


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    name = (f"celeba_{args.tag}_sz{args.image_size}_nobs{args.n_obs}"
            f"_M{args.n_particles}_arms{arms}_ds{seeds}.csv")
    return Path(args.out_dir) / name


def ensure_cache(args: argparse.Namespace, data_seed: int) -> Path:
    """Return the cache path for this configuration, generating it atomically if absent."""
    cache_dir = Path(args.cache_dir)
    canonical = (cache_dir / f"celeba_paper_{args.image_size}_n{args.n_obs}"
                             f"_nref{args.n_ref}_seed{data_seed}.pt")
    if canonical.exists():
        return canonical
    legacy = cache_dir / f"celeba_paper_{args.image_size}_n{args.n_obs}_seed{data_seed}.pt"
    if args.n_ref == 2000 and legacy.exists():   # older caches omit the (then-default) n_ref
        return legacy
    if not Path(args.celeba_root).exists():
        sys.exit(f"cache {canonical.name} is missing and the raw CelebA root "
                 f"{args.celeba_root} is not readable — generate the cache on a machine "
                 "that can see the root, or fix --celeba_root.")
    # Generate under a task-unique name, then rename: os.replace is atomic on one
    # filesystem, so concurrent array tasks sharing this configuration can never
    # observe a half-written cache (the losing writer just overwrites with identical
    # bytes, since the draw is fully determined by the seed).
    expected_gb = (args.n_obs + 2 * args.n_ref) * 3 * args.image_size ** 2 * 4 / 1e9
    free_gb = shutil.disk_usage(cache_dir).free / 1e9
    if free_gb < 1.3 * expected_gb:   # statvfs cannot see NFS quotas, but it does
        sys.exit(f"cache dir {cache_dir} has {free_gb:.1f} GB free; generating "
                 f"{canonical.name} needs ~{expected_gb:.1f} GB (plus headroom). "
                 "Use a larger --cache_dir / CACHE_DIR.")   # catch small local scratch
    tmp = canonical.with_name(canonical.name + f".tmp{os.getpid()}")
    try:
        load_celeba_paper(tmp, root=args.celeba_root, seed=data_seed,
                          image_size=args.image_size, n_obs=args.n_obs, n_ref=args.n_ref,
                          verbose=False)
        os.replace(tmp, canonical)
    except BaseException as error:
        tmp.unlink(missing_ok=True)   # never leave a partial multi-GB tmp behind
        raise SystemExit(
            f"cache generation failed for {canonical.name} (~{expected_gb:.1f} GB needed "
            f"in {cache_dir}): {error!r}. If this is EDQUOT/ENOSPC, free space or point "
            "--cache_dir (CACHE_DIR in the submit script) at a filesystem with room."
        ) from error
    return canonical


def run_arm(target_a: int, shared: dict, args: argparse.Namespace, seed: int,
            device: torch.device) -> dict:
    """Arm-specific half of one repeat: DR weights, flow, and pixel evaluation."""
    A, X, Y = shared["A"], shared["X"], shared["Y"]
    A_s, X_s = shared["A_s"], shared["X_s"]
    Z_std, Z_std_s = shared["Z_std"], shared["Z_std_s"]
    e1, fold_ids, fold_lambdas = shared["e1"], shared["fold_ids"], shared["fold_lambdas"]
    n_dr = len(A_s)

    w_hat = (A_s == target_a).double() / (e1 if target_a == 1 else 1.0 - e1)
    on_arm = w_hat[A_s == target_a]
    K_x = (X_s[:, None] == X_s[None, :]).double()
    K_a = (A_s[:, None] == A_s[None, :]).double()
    K_a_target = (A_s == target_a).double()[:, None].expand(n_dr, n_dr).contiguous()
    v_dr = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_hat, fold_ids,
                                    fold_lambdas).float().to(device)

    t0 = time.time()
    arm_indices = torch.nonzero(A == target_a).ravel()
    n_particles = min(args.n_particles, len(arm_indices))
    pick = arm_indices[torch.randperm(len(arm_indices),
                                      generator=torch.Generator().manual_seed(seed))[:n_particles]]
    initial_particles = Z_std[pick].clone()

    torch.manual_seed(seed + 1)
    flow_particles, history = adamw_riesz_flow(
        initial_particles.clone(), Z_std_s, target_weights=v_dr, sliced=False,
        n_iterations=args.flow_iterations, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, lr_schedule=args.lr_schedule,
        log_every=500, verbose=False)
    seconds_flow = time.time() - t0

    # -- evaluation --------------------------------------------------------------------
    t0 = time.time()
    vae = shared["vae"]
    Y_ref = shared["Y_ref"][target_a]
    Z_ref_std = shared["Z_ref_std"][target_a]
    half = len(Y_ref) // 2
    latent_sw2 = {
        "initial": sliced_wasserstein(initial_particles, Z_ref_std, p=2, seed=0).item(),
        "flow": sliced_wasserstein(flow_particles, Z_ref_std, p=2, seed=0).item(),
        "floor": sliced_wasserstein(Z_ref_std[:half], Z_ref_std[half:], p=2, seed=0).item(),
    }

    eval_indices = arm_indices
    if args.eval_max is not None and len(arm_indices) > args.eval_max:
        pick_eval = torch.randperm(len(arm_indices),
                                   generator=torch.Generator().manual_seed(seed + 7))[:args.eval_max]
        eval_indices = arm_indices[pick_eval]
    with torch.no_grad():
        flow_images = decode_latents(vae, shared["from_flow"](flow_particles),
                                     args.image_size)
        observed_images = Y[eval_indices]
        roundtrip_images = decode_latents(vae, encode_images(vae, observed_images),
                                          args.image_size)

    flatten = lambda images: images.reshape(images.shape[0], -1).to(device)
    reference = flatten(Y_ref)
    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(reference)   # lock the bandwidth on the reference
    to_64 = lambda images: F.interpolate(images, size=64, mode="area").clamp(-1.0, 1.0)
    reference_64 = flatten(to_64(Y_ref))

    row = {
        "target_a": target_a, "n_arm": int(len(arm_indices)),
        "n_eval": int(len(eval_indices)),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "obs_blond_frac_arm": float(X[arm_indices].float().mean()),
        "ess_on_arm": float(on_arm.sum() ** 2 / (on_arm ** 2).sum()),
        "n_on_arm": int(len(on_arm)), "max_weight": float(w_hat.max()),
        "sum_v": float(v_dr.sum()), "sum_abs_v": float(v_dr.abs().sum()),
        "sum_vX": float((v_dr.cpu() * X_s.float()).sum()),
        "loss_initial": float(history["loss"][0]),
        "loss_final": float(history["loss"][-1]),
        "mean_transport": float((flow_particles - initial_particles).norm(dim=1).mean()),
        "latent_sw2_initial": latent_sw2["initial"],
        "latent_sw2_flow": latent_sw2["flow"],
        "latent_sw2_floor": latent_sw2["floor"],
        "seconds_flow": round(seconds_flow, 1),
    }
    for name, images in (("observed", observed_images), ("roundtrip", roundtrip_images),
                         ("flow", flow_images)):
        rows_px = flatten(images)
        row[f"sw2_{name}"] = sliced_wasserstein(rows_px, reference, p=2, seed=0).item()
        row[f"sw1_{name}"] = sliced_wasserstein(rows_px, reference, p=1, seed=0).item()
        row[f"mmd2_{name}"] = squared_mmd(kernel_pixel, rows_px, reference,
                                          unbiased=True).item()
        rows_64 = flatten(to_64(images))
        row[f"sw2_64_{name}"] = sliced_wasserstein(rows_64, reference_64, p=2, seed=0).item()
        row[f"sw1_64_{name}"] = sliced_wasserstein(rows_64, reference_64, p=1, seed=0).item()
    row["sw2_floor"] = sliced_wasserstein(reference[:half], reference[half:], p=2, seed=0).item()
    row["sw1_floor"] = sliced_wasserstein(reference[:half], reference[half:], p=1, seed=0).item()
    row["mmd2_floor"] = squared_mmd(kernel_pixel, reference[:half], reference[half:],
                                    unbiased=True).item()
    row["sw2_64_floor"] = sliced_wasserstein(reference_64[:half], reference_64[half:],
                                             p=2, seed=0).item()
    row["sw1_64_floor"] = sliced_wasserstein(reference_64[:half], reference_64[half:],
                                             p=1, seed=0).item()
    row["seconds_eval"] = round(time.time() - t0, 1)
    return row


def run_repeat(shared: dict, args: argparse.Namespace, data_seed: int, repeat: int,
               device: torch.device) -> list:
    """One pipeline pass on a fixed data draw; returns one CSV row per arm."""
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- shared nuisances: propensity + per-fold lambdas on the DR support -------------
    t0 = time.time()
    n = len(shared["A"])
    if args.n_dr < n:
        pick_dr = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 5))[:args.n_dr]
    else:
        pick_dr = torch.arange(n)   # identity: exact notebook path at the defaults
    A_s, X_s = shared["A"][pick_dr], shared["X"][pick_dr]
    Z_std_s = shared["Z_std"][pick_dr]
    n_dr = len(pick_dr)

    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    torch.manual_seed(seed)
    e1 = crossfit_propensity(X_s.double().unsqueeze(1), A_s.double(), fold_ids)

    K_pair = ((X_s[:, None] == X_s[None, :]) & (A_s[:, None] == A_s[None, :])).double()
    fold_lambdas = {}
    for fold in np.unique(fold_ids):
        train = np.where(fold_ids != fold)[0]
        fold_lambdas[int(fold)], _ = select_regularization(K_pair[train][:, train],
                                                           Z_std_s[train, 0].double().cpu(),
                                                           seed=0)
    seconds_nuisance = time.time() - t0

    shared_repeat = dict(shared, A_s=A_s, X_s=X_s, Z_std_s=Z_std_s, e1=e1,
                         fold_ids=fold_ids, fold_lambdas=fold_lambdas)
    stats = binary_stats(shared["A"], shared["X"])
    base = {
        "tag": args.tag, "device": str(device),
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "image_size": args.image_size, "n_obs": args.n_obs, "n_ref": args.n_ref,
        "n_dr": int(n_dr), "n_folds": args.n_folds,
        "latent_dim": int(shared["Z_std"].shape[1]),
        "flow_iterations": args.flow_iterations, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps, "lr_schedule": args.lr_schedule,
        "n_a0x0": stats["cells"][(0, 0)], "n_a0x1": stats["cells"][(0, 1)],
        "n_a1x0": stats["cells"][(1, 0)], "n_a1x1": stats["cells"][(1, 1)],
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        "seconds_encode": shared["seconds_encode"],
        "seconds_nuisance": round(seconds_nuisance, 1),
    }
    rows = []
    for target_a in args.target_arms:
        row = dict(base)
        row.update(run_arm(target_a, shared_repeat, args, seed, device))
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

    vae = load_sd_vae(device)
    rows = []
    for data_seed in args.data_seeds:
        cache = ensure_cache(args, data_seed)
        data = load_celeba_paper(cache, root=args.celeba_root, seed=data_seed,
                                 image_size=args.image_size, n_obs=args.n_obs,
                                 n_ref=args.n_ref, verbose=False)
        A = data["A"].view(-1).long()
        X = data["X"].view(-1).long()
        Y = data["Y"]

        # Encodings are pipeline-independent (frozen VAE, fixed draw): once per data seed.
        t0 = time.time()
        Z = encode_images(vae, Y)
        latent_mean = Z.mean(dim=0, keepdim=True)
        latent_std = Z.std(dim=0, keepdim=True).clamp_min(1e-3)
        Z_std = (Z - latent_mean) / latent_std
        Y_ref, Z_ref_std = {}, {}
        for arm in args.target_arms:
            Y_ref[arm] = data[f"Y{arm}_ref"]
            Z_ref_std[arm] = (encode_images(vae, Y_ref[arm]) - latent_mean) / latent_std
        seconds_encode = round(time.time() - t0, 1)

        shared = {"vae": vae, "A": A, "X": X, "Y": Y, "Z_std": Z_std,
                  "Y_ref": Y_ref, "Z_ref_std": Z_ref_std,
                  "from_flow": lambda p: p * latent_std + latent_mean,
                  "seconds_encode": seconds_encode}
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(shared, args, data_seed, repeat, device)
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']}] "
                      f"latent SW2 flow {row['latent_sw2_flow']:.4f} "
                      f"(initial {row['latent_sw2_initial']:.4f}, floor {row['latent_sw2_floor']:.4f})   "
                      f"pixel SW2 flow {row['sw2_flow']:.4f} (floor {row['sw2_floor']:.4f})   "
                      f"sum_v {row['sum_v']:.3f}   {row['seconds_repeat']:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
