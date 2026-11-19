"""Controlled nuisance corruption for ColorMNIST interventional MMD flow.

Adapted from Luedtke and Fukumizu, DoubleGen (arXiv:2509.16842v1),
Appendix M.1.1. The red subgroup X >= threshold replaces their lighter-haired
subgroup: its inverse weights are reduced, and the corrupted conditional
embedding is fitted only on blue-subgroup observations. All logged outcomes
remain available to the DR residual correction. This is an adaptation of their
experimental design, not an implementation or benchmark of DoubleGen itself.

The four configurations share an independently trained frozen autoencoder,
logged data, evaluation folds, initial particles and oracle references. Targets
remain raw signed coefficient vectors. ``--corruption-mass preserve`` is an
explicit additional control on the corrupted inverse weights only.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autoencoder_utils import encode_in_batches, train_conv_ae
from causality_utils import make_crossfit_fold_ids, select_regularization
from causality_utils.cpme import _projected_nystrom_coefficients
from data_utils import (color_ratio, load_digit_pools, sample_interventional,
                        sample_observed, to_image_range, to_model_range)
from dens_ratio_utils import LogisticPropensityModel, crossfit_propensity
from kernel_utils import IMQKernel, RBF, imq_c_from_median
from metric_utils import sliced_wasserstein
from mmd_flow_utils import mmd_gradient_flow, squared_mmd


def corrupt_importance_weights(weights, X, threshold=0.5, downweight=0.25,
                              mass_mode="raw"):
    """Reduce red-subgroup inverse weights, optionally preserving baseline mass.

    Parameters
    ----------
    weights : torch.Tensor of shape (n,)
        Raw importance weights for one intervention arm, including zero weights.
    X : torch.Tensor of shape (n,) or (n, 1)
        Observed colour confounder. X >= threshold defines the red subgroup.
    mass_mode : {"raw", "preserve"}
        ``preserve`` multiplies the corrupted vector by the ratio of baseline
        mass to corrupted mass. It is a declared sensitivity intervention.
    """
    if mass_mode not in ("raw", "preserve"):
        raise ValueError("mass_mode must be 'raw' or 'preserve'")
    if not 0 < downweight <= 1 or not 0 < threshold < 1:
        raise ValueError("require 0 < downweight <= 1 and 0 < threshold < 1")
    x = torch.as_tensor(X, device=weights.device).reshape(-1)
    if weights.ndim != 1 or x.shape != weights.shape:
        raise ValueError("weights and X must have one entry per observation")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("importance weights must be finite and nonnegative")
    if not bool(torch.isfinite(x).all()) or not float(weights.sum()) > 0:
        raise ValueError("X must be finite and the arm must have positive mass")
    multiplier = torch.where(x >= threshold, downweight, 1.0).to(weights)
    corrupted = weights * multiplier
    if mass_mode == "preserve":
        corrupted = corrupted * (weights.sum() / corrupted.sum())
    return corrupted


def build_target_weights(X, A, Z, baseline_weights, fold_ids, kernel_x, *, arm,
                         fit_mode="crossfit", n_landmarks=512, gram_ridge=None,
                         seed=0, threshold=0.5, downweight=0.25,
                         corruption_mass="raw", cv_max_n=1000, jitter=1e-8):
    """Construct the four paired nuisance configurations over the full support.

    For each held-out block H, the full CME fits on the other folds T; the
    corrupted CME fits on T intersect {X < threshold}. Its projected Nyström
    plug-in and two residual terms are solved together from the same landmarks.
    Every observation still contributes its own w_i phi(Y_i) residual term.

    Returns
    -------
    targets : dict
        Float64 coefficient vectors keyed by (cme_mode, weight_mode, estimator),
        where modes are full/blue_only and baseline/red_downweighted, and
        estimators are plugin/ipw/dr. All vectors retain the full logged support.
    metadata : list of dict
        Training, evaluation and landmark indices and the matrix ridge for each
        CME fit. No target coefficient normalization is applied.
    """
    if fit_mode not in ("crossfit", "nosplit"):
        raise ValueError("fit_mode must be 'crossfit' or 'nosplit'")
    X = torch.as_tensor(X, dtype=torch.float64).reshape(-1, 1)
    A = torch.as_tensor(A, device=X.device).reshape(-1)
    Z = torch.as_tensor(Z, dtype=torch.float64, device=X.device)
    w = torch.as_tensor(baseline_weights, dtype=torch.float64,
                        device=X.device).reshape(-1)
    n = len(X)
    if n < 4 or len(A) != n or len(Z) != n or len(w) != n:
        raise ValueError("X, A, Z and weights must share at least four rows")
    if not bool(torch.isfinite(X).all() and torch.isfinite(Z).all()):
        raise ValueError("X and Z must be finite")
    if n_landmarks < 1 or cv_max_n < 3:
        raise ValueError("n_landmarks >= 1 and cv_max_n >= 3 are required")
    if gram_ridge is not None and (not math.isfinite(gram_ridge) or gram_ridge <= 0):
        raise ValueError("gram_ridge must be finite and positive")
    corrupted = corrupt_importance_weights(w, X, threshold, downweight,
                                          corruption_mass)
    importance = torch.stack((w, corrupted), dim=1)
    rows = torch.arange(n, device=X.device)
    if fit_mode == "crossfit":
        folds = torch.as_tensor(fold_ids, device=X.device).reshape(-1)
        if len(folds) != n or len(folds.unique()) < 2:
            raise ValueError("crossfit requires at least two folds covering all rows")
        splits = [(rows[folds != f], rows[folds == f]) for f in folds.unique()]
    else:
        splits = [(rows, rows)]

    def gram(left, right):
        return kernel_x(X[left], X[right]) * (A[left, None] == A[right][None, :])

    targets, metadata = {}, []
    for cme_mode in ("full", "blue_only"):
        v_plugin = torch.zeros(n, dtype=torch.float64, device=X.device)
        v_residual = torch.zeros(n, 2, dtype=torch.float64, device=X.device)
        for fold_number, (train_all, heldout) in enumerate(splits):
            train = (train_all if cme_mode == "full"
                     else train_all[X[train_all, 0] < threshold])
            if len(train) < 3 or not bool((A[train] == arm).any()):
                raise ValueError(f"insufficient arm-{arm} training data in {cme_mode}")
            generator = torch.Generator().manual_seed(seed + 37 * fold_number)
            order = torch.randperm(len(train), generator=generator).to(X.device)
            landmarks = train[order[:min(n_landmarks, len(train))]]
            rho = gram_ridge
            if rho is None:
                pick = train[order[:min(cv_max_n, len(train))]]
                rho, _ = select_regularization(gram(pick, pick), Z[pick, 0], seed=seed)
            cross = gram(landmarks, heldout)
            rhs_plugin = (kernel_x(X[landmarks], X[heldout]).sum(dim=1)
                          * (A[landmarks] == arm))
            rhs = torch.cat((rhs_plugin[:, None],
                             -cross @ importance[heldout]), dim=1)
            coefficients = _projected_nystrom_coefficients(
                gram(train, landmarks), gram(landmarks, landmarks), rhs,
                float(rho), jitter)
            v_plugin[train] += coefficients[:, 0] / n
            v_residual[train] += coefficients[:, 1:] / n
            metadata.append({
                "cme_mode": cme_mode, "fold": fold_number,
                "train_indices": train.cpu().tolist(),
                "heldout_indices": heldout.cpu().tolist(),
                "landmarks": landmarks.cpu().tolist(), "gram_ridge": float(rho),
            })
        for column, weight_mode in enumerate(("baseline", "red_downweighted")):
            v_ipw = importance[:, column] / n
            targets[cme_mode, weight_mode, "plugin"] = v_plugin.clone()
            targets[cme_mode, weight_mode, "ipw"] = v_ipw.clone()
            targets[cme_mode, weight_mode, "dr"] = (
                v_plugin + v_residual[:, column] + v_ipw)
    if not all(bool(torch.isfinite(v).all()) for v in targets.values()):
        raise FloatingPointError("nonfinite target coefficients")
    return targets, metadata


@torch.no_grad()
def uniform_w1(values):
    """Exact W1 between a finite scalar empirical law and Uniform[0, 1]."""
    x = torch.as_tensor(values, dtype=torch.float64).reshape(-1).sort().values
    if len(x) == 0 or not bool(torch.isfinite(x).all()):
        raise ValueError("values must be nonempty and finite")
    n = len(x)
    lower = torch.arange(n, dtype=x.dtype, device=x.device) / n
    upper = lower + 1.0 / n
    clipped = torch.maximum(lower, torch.minimum(x, upper))
    integral = ((clipped - lower).square() + (upper - clipped).square()) / 2
    integral += (x - clipped).abs() / n
    return float(integral.sum())


@torch.no_grad()
def target_distances(kernel, support, targets, reference, block_size=256):
    """Squared RKHS distance to a fixed oracle empirical embedding, in blocks."""
    keys = list(targets)
    weights = torch.stack([targets[key] for key in keys], dim=1).double()
    support, reference = support.double(), reference.double()
    quadratic = weights.new_zeros(len(keys))
    cross = weights.new_zeros(len(keys))
    for start in range(0, len(support), block_size):
        stop = min(start + block_size, len(support))
        block = support[start:stop]
        quadratic += (weights[start:stop] * (kernel(block, support) @ weights)).sum(0)
        cross += weights[start:stop].T @ kernel(block, reference).mean(1)
    reference_norm = weights.new_zeros(())
    for start in range(0, len(reference), block_size):
        reference_norm += kernel(reference[start:start + block_size], reference).sum()
    values = quadratic - 2 * cross + reference_norm / len(reference) ** 2
    return {key: float(value) for key, value in zip(keys, values)}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generator(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(int(seed))


@torch.no_grad()
def sample_metrics(images, latents, reference_images, reference_latents, projections):
    """Matched-count image/latent scores and the exact known colour-law error."""
    colour = color_ratio(images)
    return {
        "pixel_sw2": float(sliced_wasserstein(
            images.flatten(1).double(), reference_images.flatten(1).double(),
            p=2, seed=0, n_projections=projections)),
        "latent_sw2": float(sliced_wasserstein(
            latents.double(), reference_latents.double(), p=2, seed=0,
            n_projections=projections)),
        "colour_w1_uniform": uniform_w1(colour),
        "colour_mean": float(colour.mean()),
        "colour_second_moment": float(colour.square().mean()),
    }


def weight_diagnostics(v, x):
    mass = float(v.sum())
    first = float(v @ x.double())
    second = float(v @ x.double().square())
    return {
        "sum_v": mass, "sum_abs_v": float(v.abs().sum()),
        "negative_mass": float(-v[v < 0].sum()),
        "negative_count": int((v < 0).sum()),
        "target_colour_mean": first,
        "target_colour_second_moment": second,
        "target_colour_mean_divided_by_mass": first / mass if mass > 0 else None,
        "target_colour_second_moment_divided_by_mass": second / mass if mass > 0 else None,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_seed(args, data_seed, run_dir, device):
    """Train one independent representation, then run all paired configurations."""
    start = time.perf_counter()
    seed = 100000 * data_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    pools = load_digit_pools(args.mnist_root, digits=(1, 6), device=torch.device("cpu"))
    dgp = {"w_conf": 10.0, "alpha": 0.2, "tau": 0.08, "k_mask": 20.0}
    x_cpu, a_cpu, y_cpu = sample_observed(
        pools, args.n_obs, generator=generator(seed + 1), **dgp)
    x_ae, _, y_ae = sample_observed(
        pools, args.n_ae, generator=generator(seed + 2), **dgp)
    print(f"seed {data_seed}: train independent AE on {args.n_ae} images", flush=True)
    ae, _ = train_conv_ae(
        to_model_range(y_ae), latent_dim=args.latent_dim,
        n_iterations=args.ae_iterations, device=device,
        generator=generator(seed + 3, device), verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    torch.save({"state_dict": {k: v.cpu() for k, v in ae.state_dict().items()},
                "latent_dim": args.latent_dim}, run_dir / "autoencoder.pt")
    Z = encode_in_batches(ae.encode, to_model_range(y_cpu).to(device))
    Z_ae = encode_in_batches(ae.encode, to_model_range(y_ae).to(device))
    X, A = x_cpu.double().to(device), a_cpu.to(device)
    kernel_x = RBF(use_length_scale_heuristic=True).to(device)
    kernel_x.calculate_and_fix_length_scale(x_ae[:2000, None].double().to(device))
    diagnostic_kernel = RBF(use_length_scale_heuristic=True).to(device)
    diagnostic_kernel.calculate_and_fix_length_scale(Z_ae[:2000].double())
    c_imq = imq_c_from_median(Z_ae[:2000])
    if not math.isfinite(c_imq) or c_imq <= 0:
        raise FloatingPointError("degenerate latent IMQ bandwidth")
    flow_kernel = IMQKernel(c=c_imq, beta=-0.5)
    folds = make_crossfit_fold_ids(args.n_obs, args.n_folds, seed=seed + 4)
    if args.dr_fit == "crossfit":
        e1 = crossfit_propensity(x_cpu[:, None].double(), a_cpu.double(), folds)
    else:
        fitted = LogisticPropensityModel().fit(x_cpu[:, None].double(), a_cpu.double())
        e1 = fitted.predict(x_cpu[:, None].double())
    e1 = e1.double().to(device)
    if not bool(torch.isfinite(e1).all() and (e1 > 0).all() and (e1 < 1).all()):
        raise FloatingPointError("nonfinite or boundary fitted propensity; no clipping applied")
    sync(device)
    seconds_representation_and_propensity = time.perf_counter() - start
    arrays = {"X": x_cpu.numpy(), "A": a_cpu.numpy(), "Z": Z.cpu().numpy(),
              "e1": e1.cpu().numpy(), "fold_ids": folds}
    rows, all_fit_metadata = [], {}
    for arm in args.arms:
        arm_start = time.perf_counter()
        probability = e1 if arm == 1 else 1 - e1
        baseline_w = (A == arm).double() / probability
        corrupted_w = corrupt_importance_weights(
            baseline_w, X, args.threshold, args.downweight, args.corruption_mass)
        unscaled_w = corrupt_importance_weights(
            baseline_w, X, args.threshold, args.downweight, "raw")
        corruption_factor = float(corrupted_w.sum() / unscaled_w.sum())
        print(f"seed {data_seed}, arm {arm}: build paired targets", flush=True)
        targets, fit_metadata = build_target_weights(
            X, A, Z, baseline_w, folds, kernel_x, arm=arm, fit_mode=args.dr_fit,
            n_landmarks=args.n_landmarks, gram_ridge=args.gram_ridge,
            seed=seed + 5, threshold=args.threshold, downweight=args.downweight,
            corruption_mass=args.corruption_mass, cv_max_n=args.cv_max_n)
        all_fit_metadata[str(arm)] = fit_metadata
        candidates = torch.where(a_cpu == arm)[0]
        if len(candidates) < args.n_particles:
            raise ValueError(f"arm {arm} has {len(candidates)} initial candidates, "
                             f"fewer than requested M={args.n_particles}")
        initial_indices = candidates[torch.randperm(
            len(candidates), generator=generator(seed + 6 + arm))[:args.n_particles]]
        initial = Z[initial_indices.to(device)].clone()
        reference_kwargs = {k: value for k, value in dgp.items() if k != "w_conf"}
        _, ref_cpu = sample_interventional(
            pools, args.n_ref, a=arm, generator=generator(seed + 10 + 2 * arm),
            **reference_kwargs)
        _, ref2_cpu = sample_interventional(
            pools, args.n_ref, a=arm, generator=generator(seed + 11 + 2 * arm),
            **reference_kwargs)
        reference, ref2 = ref_cpu.to(device), ref2_cpu.to(device)
        Z_reference = encode_in_batches(ae.encode, to_model_range(reference))
        Z_ref2 = encode_in_batches(ae.encode, to_model_range(ref2))
        discrepancy = target_distances(
            diagnostic_kernel, Z, targets, Z_reference, args.metric_block_size)
        k = args.n_particles
        with torch.no_grad():
            initial_images = to_image_range(ae.decode(initial))
            roundtrip = to_image_range(ae.decode(Z_ref2[:k]))
        shared = {}
        for label, images, latents in (
                ("initial", initial_images, initial),
                ("floor", ref2[:k], Z_ref2[:k]),
                ("roundtrip_floor", roundtrip, Z_ref2[:k])):
            scores = sample_metrics(images, latents, reference[:k], Z_reference[:k],
                                    args.n_projections)
            shared.update({f"{name}_{label}": value for name, value in scores.items()})
        shared["target_rbf_oracle_floor"] = float(squared_mmd(
            diagnostic_kernel, Z_ref2.double(), Z_reference.double()))
        sync(device)
        shared["seconds_arm_setup"] = time.perf_counter() - arm_start
        arrays[f"arm{arm}_initial_indices"] = initial_indices.numpy()
        arrays[f"arm{arm}_reference_images"] = ref_cpu.numpy()
        arrays[f"arm{arm}_reference2_images"] = ref2_cpu.numpy()
        arrays[f"arm{arm}_baseline_weights"] = baseline_w.cpu().numpy()
        arrays[f"arm{arm}_corrupted_weights"] = corrupted_w.cpu().numpy()
        completed_flows = {}
        for (cme_mode, weight_mode, estimator), v in targets.items():
            flow_key = (f"plugin_{cme_mode}" if estimator == "plugin" else
                        f"ipw_{weight_mode}" if estimator == "ipw" else
                        f"dr_{cme_mode}_{weight_mode}")
            row = {
                "data_seed": data_seed, "arm": arm, "cme_mode": cme_mode,
                "weight_mode": weight_mode, "estimator": estimator,
                "dr_fit": args.dr_fit, "n_obs": args.n_obs, "n_ae": args.n_ae,
                "n_particles": k, "n_ref": args.n_ref,
                "corruption_mass": args.corruption_mass,
                "corruption_factor": corruption_factor,
                "threshold": args.threshold, "downweight": args.downweight,
                "sum_w_baseline": float(baseline_w.sum()),
                "sum_w_corrupted": float(corrupted_w.sum()),
                "ess_w_baseline": float(baseline_w.sum().square() / baseline_w.square().sum()),
                "ess_w_corrupted": float(corrupted_w.sum().square() / corrupted_w.square().sum()),
                "target_rbf_squared_distance": discrepancy[cme_mode, weight_mode, estimator],
                "seconds_representation_and_propensity": seconds_representation_and_propensity,
                "c_imq": c_imq, "step_size": 0.15 * c_imq ** 2,
                "context_lengthscale": float(kernel_x.length_scale),
                "diagnostic_lengthscale": float(diagnostic_kernel.length_scale),
                **weight_diagnostics(v, X), **shared,
            }
            arrays[f"arm{arm}_target_{cme_mode}_{weight_mode}_{estimator}"] = v.cpu().numpy()
            if not args.targets_only:
                if flow_key not in completed_flows:
                    print(f"seed {data_seed}, arm {arm}: {flow_key}", flush=True)
                    flow_start = time.perf_counter()
                    final, history = mmd_gradient_flow(
                        flow_kernel, Z, initial, step_size=row["step_size"],
                        n_iterations=args.flow_iterations, target_weights=v.to(Z),
                        log_every=args.flow_iterations, verbose=False)
                    sync(device)
                    seconds_flow = time.perf_counter() - flow_start
                    if not bool(torch.isfinite(final).all()):
                        raise FloatingPointError(f"nonfinite flow for {flow_key}")
                    with torch.no_grad():
                        decoded = to_image_range(ae.decode(final))
                    metrics = sample_metrics(decoded, final, reference[:k],
                                             Z_reference[:k], args.n_projections)
                    completed_flows[flow_key] = {
                        **{f"{name}_flow": value for name, value in metrics.items()},
                        "objective_after_first_update": history["squared_mmd"][0],
                        "objective_final": history["squared_mmd"][-1],
                        "seconds_unique_flow": seconds_flow,
                    }
                    arrays[f"arm{arm}_particles_{flow_key}"] = final.cpu().numpy()
                row.update(completed_flows[flow_key])
            row["shared_flow_id"] = f"seed{data_seed}_arm{arm}_{flow_key}"
            rows.append(row)
        write_csv(run_dir / "results.csv", rows)
        write_json(run_dir / "nuisance_fits.json", all_fit_metadata)
        np.savez_compressed(run_dir / "artifacts.npz", **arrays)
    return {"rows": len(rows), "seconds_total": time.perf_counter() - start}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--arms", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--n-obs", type=int, default=5000)
    parser.add_argument("--n-ae", type=int, default=5000)
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--n-ref", type=int, default=1000)
    parser.add_argument("--ae-iterations", type=int, default=3000)
    parser.add_argument("--flow-iterations", type=int, default=3000)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dr-fit", choices=("crossfit", "nosplit"), default="crossfit")
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--n-landmarks", type=int, default=512)
    parser.add_argument("--gram-ridge", type=float, default=None,
                        help="fixed positive matrix ridge; otherwise training-only CV")
    parser.add_argument("--cv-max-n", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--downweight", type=float, default=0.25)
    parser.add_argument("--corruption-mass", choices=("raw", "preserve"), default="raw")
    parser.add_argument("--n-projections", type=int, default=1000)
    parser.add_argument("--metric-block-size", type=int, default=256)
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--mnist-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--tag", default="pilot")
    args = parser.parse_args(argv)
    for name in ("n_obs", "n_ae", "n_particles", "n_ref", "ae_iterations",
                 "flow_iterations", "latent_dim", "n_landmarks", "n_projections",
                 "metric_block_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_ref < args.n_particles:
        parser.error("--n-ref must be at least --n-particles for matched-count scores")
    if args.n_folds < 2 or args.n_folds >= args.n_obs:
        parser.error("require 2 <= n-folds < n-obs")
    if args.cv_max_n < 3 or not 0 < args.threshold < 1 or not 0 < args.downweight <= 1:
        parser.error("invalid CV size, subgroup threshold or downweight")
    if args.gram_ridge is not None and (not math.isfinite(args.gram_ridge) or args.gram_ridge <= 0):
        parser.error("--gram-ridge must be finite and positive")
    if Path(args.tag).name != args.tag or args.tag in ("", ".", ".."):
        parser.error("--tag must be a single nonempty directory name")
    if len(set(args.data_seeds)) != len(args.data_seeds) or len(set(args.arms)) != len(args.arms):
        parser.error("duplicate seeds or arms are not allowed")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this execution context")
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    source_paths = [Path(__file__), REPO_ROOT / "src/causality_utils/cpme.py",
                    REPO_ROOT / "src/causality_utils/crossfit.py",
                    REPO_ROOT / "src/data_utils/colormnist.py",
                    REPO_ROOT / "src/autoencoder_utils/colormnist_ae.py",
                    REPO_ROOT / "src/dens_ratio_utils/propensity.py",
                    REPO_ROOT / "src/kernel_utils/kernels.py",
                    REPO_ROOT / "src/kernel_utils/linear_algebra.py",
                    REPO_ROOT / "src/metric_utils/metrics.py",
                    REPO_ROOT / "src/mmd_flow_utils/mmd_flow.py"]
    hashes = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_paths}
    for data_seed in args.data_seeds:
        run_dir = args.out_dir / f"{args.tag}_seed{data_seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        provenance = {
            "status": "running", "config": configuration, "data_seed": data_seed,
            "source_sha256": hashes, "device": str(device),
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "reference": "https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1",
            "timing_note": "shared costs repeat on rows; deduplicate shared_flow_id for sampler times",
            "interpretation": "baseline fitted nuisances are not oracle-correct; no FID computed",
        }
        write_json(run_dir / "provenance.json", provenance)
        print(f"device={device}; output={run_dir}", flush=True)
        try:
            provenance.update(run_seed(args, data_seed, run_dir, device))
            provenance["status"] = "complete"
        except Exception as error:
            provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(run_dir / "provenance.json", provenance)
            raise
        write_json(run_dir / "provenance.json", provenance)
        print(f"completed {provenance['rows']} rows in {provenance['seconds_total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
