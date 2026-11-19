"""Paper-scale ColorMNIST nuisance-corruption severity sweep.

The protocol extends DoubleGen Appendix M.1.1 (arXiv:2509.16842v1):
restrict conditional-embedding training support and downweight a fixed red
subgroup. The CME cutoff and weighting subgroup are independent parameters.
All observations remain in the DR residual and policy integration. Full-sample
nuisances, the logged-data AE, and IMQ Euler settings follow the paper runner.
The separate pilot script remains unchanged.
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
from causality_utils import select_regularization
from causality_utils.cpme import _projected_nystrom_coefficients
from data_utils import (color_ratio, load_digit_pools, sample_interventional, sample_observed,
                        to_image_range, to_model_range)
from dens_ratio_utils import LogisticPropensityModel
from kernel_utils import IMQKernel, RBF, imq_c_from_median
from metric_utils import sliced_wasserstein
from mmd_flow_utils import mmd_gradient_flow, squared_mmd
from colormnist_doublegen_misspecification import (
    corrupt_importance_weights, generator, sync,
    target_distances, uniform_w1, weight_diagnostics,
)


def make_weight_settings(weights, X, factors=(1.0, 0.25, 0.05, 0.01),
                         mass_modes=("raw", "preserve"), threshold=0.5):
    """Return unique inverse-weight settings, with one uncorrupted baseline."""
    factors = tuple(float(factor) for factor in factors)
    if len(set(factors)) != len(factors) or any(not 0 < f <= 1 for f in factors):
        raise ValueError("weight factors must be unique and in (0, 1]")
    if not mass_modes or any(m not in ("raw", "preserve") for m in mass_modes):
        raise ValueError("mass_modes must contain raw and/or preserve")
    if len(set(mass_modes)) != len(mass_modes):
        raise ValueError("duplicate mass modes")
    result = {}
    for factor in factors:
        modes = ("baseline",) if factor == 1 else mass_modes
        for mode in modes:
            result[factor, mode] = corrupt_importance_weights(
                weights, X, threshold=threshold, downweight=factor,
                mass_mode="raw" if mode == "baseline" else mode)
    return result


def build_severity_targets(X, A, Z, baseline_weights, kernel_x, *, arm,
                          cme_cutoffs=(1.0, 0.5, 0.25, 0.1),
                          weight_factors=(1.0, 0.25, 0.05, 0.01),
                          mass_modes=("raw", "preserve"), weight_threshold=0.5,
                          n_landmarks=1024, gram_ridge=None, seed=0,
                          cv_max_n=2000, jitter=1e-8, block_size=2048):
    """Build raw full-sample targets while fitting each CME only once.

    Parameters
    ----------
    cme_cutoffs : sequence of float
        A cutoff of 1 fits every logged row. Otherwise the CME training rows
        satisfy X < cutoff. This never changes the evaluation population.
    weight_threshold : float
        Fixed subgroup boundary for inverse-weight corruption, independent of
        the CME cutoffs.
    gram_ridge : float, optional
        Positive ridge of the unnormalised Gram system. Otherwise selected on
        at most cv_max_n eligible training rows using the paper's CV rule.

    Returns
    -------
    targets : dict
        Keys are (cme_cutoff, weight_factor, mass_mode, estimator), where
        estimator is plugin/ipw/dr. Factor 1 occurs only with mode baseline.
    metadata : list of dict
        Eligible training/CV/landmark indices, counts, and selected ridge.
    weight_settings : dict
        Raw inverse-weight tensors keyed by (factor, mass_mode).
    """
    X = torch.as_tensor(X, dtype=torch.float64).reshape(-1, 1)
    A = torch.as_tensor(A, device=X.device).reshape(-1)
    Z = torch.as_tensor(Z, dtype=torch.float64, device=X.device)
    w = torch.as_tensor(baseline_weights, dtype=torch.float64,
                        device=X.device).reshape(-1)
    n = len(X)
    if n < 4 or any(len(values) != n for values in (A, Z, w)):
        raise ValueError("X, A, Z and weights must share at least four rows")
    if not bool(torch.isfinite(X).all() and torch.isfinite(Z).all()):
        raise ValueError("X and Z must be finite")
    if not bool(((A == 0) | (A == 1)).all()) or arm not in (0, 1):
        raise ValueError("binary actions and an arm in {0, 1} are required")
    cutoffs = tuple(float(c) for c in cme_cutoffs)
    if not cutoffs or len(set(cutoffs)) != len(cutoffs) or any(not 0 < c <= 1 for c in cutoffs):
        raise ValueError("CME cutoffs must be unique and in (0, 1]")
    if n_landmarks < 1 or cv_max_n < 3 or block_size < 1:
        raise ValueError("invalid landmark, CV or block size")
    if gram_ridge is not None and (not math.isfinite(gram_ridge) or gram_ridge <= 0):
        raise ValueError("gram_ridge must be finite and positive")
    settings = make_weight_settings(w, X, weight_factors, mass_modes, weight_threshold)
    weight_keys = list(settings)
    importance = torch.stack(list(settings.values()), dim=1)
    all_rows = torch.arange(n, device=X.device)

    def gram(left, right):
        return kernel_x(X[left], X[right]) * (A[left, None] == A[right][None, :])

    targets, metadata = {}, []
    for cutoff in cutoffs:
        train = all_rows if cutoff == 1 else all_rows[X[:, 0] < cutoff]
        if len(train) < 3 or not bool((A[train] == arm).any()):
            raise ValueError(f"insufficient arm-{arm} CME training data at cutoff {cutoff}")
        order = torch.randperm(len(train), generator=generator(seed)).to(X.device)
        landmarks = train[order[:min(n_landmarks, len(train))]]
        if not bool((A[landmarks] == arm).any()):
            raise ValueError(f"uniform landmarks omit arm {arm} at cutoff {cutoff}")
        cv_order = torch.randperm(len(train), generator=generator(0)).to(X.device)
        pick = train[cv_order[:min(cv_max_n, len(train))]]
        rho = gram_ridge
        cv_errors = None
        if rho is None:
            rho, errors = select_regularization(gram(pick, pick), Z[pick, 0], seed=0)
            cv_errors = errors.cpu().tolist()
        rhs = X.new_zeros((len(landmarks), 1 + len(weight_keys)))
        for start in range(0, n, block_size):
            heldout = all_rows[start:start + block_size]
            kx = kernel_x(X[landmarks], X[heldout])
            rhs[:, 0] += kx.sum(1) * (A[landmarks] == arm)
            cross = kx * (A[landmarks, None] == A[heldout][None, :])
            rhs[:, 1:] -= cross @ importance[heldout]
        coefficients = _projected_nystrom_coefficients(
            gram(train, landmarks), gram(landmarks, landmarks), rhs, float(rho), jitter)
        v_plugin = X.new_zeros(n)
        v_plugin[train] = coefficients[:, 0] / n
        for column, (factor, mass_mode) in enumerate(weight_keys):
            v_ipw = importance[:, column] / n
            v_dr = v_plugin + v_ipw
            v_dr[train] += coefficients[:, column + 1] / n
            targets[cutoff, factor, mass_mode, "plugin"] = v_plugin
            targets[cutoff, factor, mass_mode, "ipw"] = v_ipw
            targets[cutoff, factor, mass_mode, "dr"] = v_dr
        metadata.append({
            "cme_cutoff": cutoff, "train_indices": train.cpu().tolist(),
            "landmarks": landmarks.cpu().tolist(),
            "cv_indices": pick.cpu().tolist() if gram_ridge is None else [],
            "gram_ridge": float(rho), "cv_errors": cv_errors,
            "n_train": len(train), "n_train_arm0": int((A[train] == 0).sum()),
            "n_train_arm1": int((A[train] == 1).sum()),
            "n_landmarks_arm0": int((A[landmarks] == 0).sum()),
            "n_landmarks_arm1": int((A[landmarks] == 1).sum()),
        })
    if not all(bool(torch.isfinite(v).all()) for v in targets.values()):
        raise FloatingPointError("nonfinite target coefficients")
    return targets, metadata, settings


def number_token(value):
    return f"{value:g}".replace(".", "p")


def target_flow_id(key):
    """Identify numerically shared plug-in/IPW flows across factorial rows."""
    cutoff, factor, mass_mode, estimator = key
    cme = f"c{number_token(cutoff)}"
    weight = f"f{number_token(factor)}_{mass_mode}"
    if estimator == "plugin":
        return f"plugin_{cme}"
    if estimator == "ipw":
        return f"ipw_{weight}"
    if estimator == "dr":
        return f"dr_{cme}_{weight}"
    raise ValueError("unknown estimator")


@torch.no_grad()
def sample_metrics(images, latents, reference_images, reference_latents, projections):
    """Use the paper's float32 SW2 projection protocol and exact colour W1."""
    colour = color_ratio(images)
    return {
        "pixel_sw2": float(sliced_wasserstein(
            images.flatten(1).float(), reference_images.flatten(1).float(),
            p=2, seed=0, n_projections=projections)),
        "latent_sw2": float(sliced_wasserstein(
            latents.float(), reference_latents.float(), p=2, seed=0,
            n_projections=projections)),
        "colour_w1_uniform": uniform_w1(colour),
        "colour_mean": float(colour.mean()),
        "colour_second_moment": float(colour.square().mean()),
    }


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv_atomic(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_arrays_atomic(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def run_seed(args, data_seed, run_dir, device):
    """One logged sample and frozen AE, shared by the whole severity grid."""
    start = time.perf_counter()
    seed = 1000 * data_seed
    pools = load_digit_pools(args.mnist_root, digits=(1, 6), device=device)
    dgp = {"w_conf": 10.0, "alpha": 0.2, "tau": 0.08, "k_mask": 20.0}
    X_raw, A, images = sample_observed(
        pools, args.n_obs, generator=generator(data_seed, device), **dgp)
    data = to_model_range(images)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"seed {data_seed}: AE on all {args.n_obs} logged images", flush=True)
    ae, _ = train_conv_ae(
        data, latent_dim=args.latent_dim, n_iterations=args.ae_iterations,
        batch_size=128, learning_rate=1e-3, device=device,
        generator=generator(seed, device), verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    torch.save({"state_dict": {k: v.cpu() for k, v in ae.state_dict().items()},
                "latent_dim": args.latent_dim}, run_dir / "autoencoder.pt")
    Z = encode_in_batches(ae.encode, data)
    with torch.no_grad():
        reconstruction_mse = float((ae(data[:2000]) - data[:2000]).square().mean())
    torch.manual_seed(seed)
    fitted = LogisticPropensityModel().fit(X_raw[:, None].double().cpu(), A.double().cpu())
    e1 = fitted.predict(X_raw[:, None].double().cpu()).double().to(device)
    if not bool(torch.isfinite(e1).all() and (e1 > 0).all() and (e1 < 1).all()):
        raise FloatingPointError("invalid fitted propensity; no clipping applied")
    X = X_raw.double()
    kernel_x = RBF(use_length_scale_heuristic=True).to(device)
    anchors = torch.randperm(args.n_obs, generator=generator(0))[:min(4000, args.n_obs)]
    kernel_x.calculate_and_fix_length_scale(X[anchors.to(device), None])
    diagnostic_kernel = RBF(use_length_scale_heuristic=True).to(device)
    diagnostic_kernel.calculate_and_fix_length_scale(Z[:min(2000, len(Z))].double())
    save_arrays_atomic(run_dir / "logged_support.npz", X=X_raw.cpu().numpy(),
                       A=A.cpu().numpy(), Z=Z.cpu().numpy(), e1=e1.cpu().numpy())
    sync(device)
    representation_seconds = time.perf_counter() - start
    rows, total_flows = [], 0
    n_weights = sum(1 if f == 1 else len(args.mass_modes) for f in args.weight_factors)
    flows_per_arm = len(args.cme_cutoffs) + n_weights + len(args.cme_cutoffs) * n_weights
    expected_flows = flows_per_arm * len(args.arms)
    expected_rows = len(args.cme_cutoffs) * n_weights * 3 * len(args.arms)
    for arm in args.arms:
        arm_start = time.perf_counter()
        probability = e1 if arm == 1 else 1 - e1
        baseline_w = (A == arm).double() / probability
        print(f"seed {data_seed}, arm {arm}: build severity targets", flush=True)
        targets, fits, weight_settings = build_severity_targets(
            X, A, Z, baseline_w, kernel_x, arm=arm,
            cme_cutoffs=args.cme_cutoffs, weight_factors=args.weight_factors,
            mass_modes=args.mass_modes, weight_threshold=args.weight_threshold,
            n_landmarks=args.n_landmarks, gram_ridge=args.gram_ridge, seed=seed,
            cv_max_n=args.cv_max_n, block_size=args.rhs_block_size)
        fit_by_cutoff = {fit["cme_cutoff"]: fit for fit in fits}
        write_json_atomic(run_dir / f"nuisance_fits_arm{arm}.json", fits)
        unique_targets = {}
        for key, v in targets.items():
            unique_targets.setdefault(target_flow_id(key), v)
        save_arrays_atomic(run_dir / f"targets_arm{arm}.npz",
                           **{key: v.cpu().numpy() for key, v in unique_targets.items()},
                           **{f"weights_f{number_token(f)}_{m}": w.cpu().numpy()
                              for (f, m), w in weight_settings.items()})
        candidates = torch.where(A == arm)[0]
        if len(candidates) < args.n_particles:
            raise ValueError(f"arm {arm} has fewer initial observations than M")
        pick = torch.randperm(len(candidates), generator=generator(seed + 1))[:args.n_particles]
        initial_indices = candidates[pick.to(device)]
        initial = Z[initial_indices].clone()
        c_imq = imq_c_from_median(Z)
        if not math.isfinite(c_imq) or c_imq <= 0:
            raise FloatingPointError("degenerate IMQ bandwidth")
        flow_kernel = IMQKernel(c=c_imq, beta=-0.5)
        reference_kwargs = {k: v for k, v in dgp.items() if k != "w_conf"}
        _, reference = sample_interventional(
            pools, args.n_ref, a=arm, generator=generator(seed + 9, device), **reference_kwargs)
        _, reference2 = sample_interventional(
            pools, args.n_ref, a=arm, generator=generator(seed + 10, device), **reference_kwargs)
        Z_reference = encode_in_batches(ae.encode, to_model_range(reference))
        Z_reference2 = encode_in_batches(ae.encode, to_model_range(reference2))
        save_arrays_atomic(run_dir / f"evaluation_arm{arm}.npz",
                           initial_indices=initial_indices.cpu().numpy(),
                           initial_particles=initial.cpu().numpy(),
                           reference_images=reference.cpu().numpy(),
                           reference2_images=reference2.cpu().numpy())
        distances = target_distances(
            diagnostic_kernel, Z, unique_targets, Z_reference, args.metric_block_size)
        k = args.n_particles
        with torch.no_grad():
            initial_images = to_image_range(ae.decode(initial))
            roundtrip = to_image_range(ae.decode(Z_reference2[:k]))
        shared = {}
        for label, image_sample, latent_sample in (
                ("initial", initial_images, initial),
                ("floor", reference2[:k], Z_reference2[:k]),
                ("roundtrip_floor", roundtrip, Z_reference2[:k])):
            scores = sample_metrics(image_sample, latent_sample, reference[:k],
                                    Z_reference[:k], args.n_projections)
            shared.update({f"{name}_{label}": value for name, value in scores.items()})
        shared["target_rbf_oracle_floor"] = float(squared_mmd(
            diagnostic_kernel, Z_reference2.double(), Z_reference.double()))
        sync(device)
        shared["seconds_arm_setup"] = time.perf_counter() - arm_start
        for flow_id, v in unique_targets.items():
            print(f"seed {data_seed}, arm {arm}: {flow_id} "
                  f"({total_flows + 1}/{expected_flows})", flush=True)
            if args.targets_only:
                output_metrics = {}
            else:
                flow_start = time.perf_counter()
                final, history = mmd_gradient_flow(
                    flow_kernel, Z, initial, step_size=0.15 * c_imq ** 2,
                    n_iterations=args.flow_iterations, target_weights=v.to(Z),
                    log_every=args.flow_iterations, verbose=False)
                sync(device)
                seconds_flow = time.perf_counter() - flow_start
                if not bool(torch.isfinite(final).all()):
                    raise FloatingPointError(f"nonfinite particles: {flow_id}")
                with torch.no_grad():
                    decoded = to_image_range(ae.decode(final))
                scores = sample_metrics(decoded, final, reference[:k],
                                        Z_reference[:k], args.n_projections)
                output_metrics = {
                    **{f"{name}_flow": value for name, value in scores.items()},
                    "objective_after_first_update": history["squared_mmd"][0],
                    "objective_final": history["squared_mmd"][-1],
                    "seconds_unique_flow": seconds_flow,
                }
                save_arrays_atomic(run_dir / f"particles_arm{arm}_{flow_id}.npz",
                                   particles=final.cpu().numpy())
            total_flows += 1
            for key, target in targets.items():
                if target_flow_id(key) != flow_id:
                    continue
                cutoff, factor, mass_mode, estimator = key
                weights = weight_settings[factor, mass_mode]
                plugin = targets[cutoff, factor, mass_mode, "plugin"]
                raw_corrupted = corrupt_importance_weights(
                    baseline_w, X, threshold=args.weight_threshold, downweight=factor)
                fit = fit_by_cutoff[cutoff]
                rows.append({
                    "data_seed": data_seed, "pipeline_seed": seed, "repeat": 0,
                    "arm": arm, "cme_cutoff": cutoff, "weight_factor": factor,
                    "corruption_mass": mass_mode, "estimator": estimator,
                    "weight_threshold": args.weight_threshold,
                    "n_obs": args.n_obs, "n_ae": args.n_obs,
                    "n_particles": k, "n_ref": args.n_ref,
                    "dr_fit": "nosplit", "ae_split": "none", "solver": "projected_nystrom",
                    "n_landmarks": args.n_landmarks, "gram_ridge": fit["gram_ridge"],
                    "n_cme_train": fit["n_train"],
                    "n_cme_train_arm": fit[f"n_train_arm{arm}"],
                    "n_landmarks_arm": fit[f"n_landmarks_arm{arm}"],
                    "weight_mass": float(weights.mean()),
                    "baseline_weight_mass": float(baseline_w.mean()),
                    "weight_ess": float(weights.sum().square() / weights.square().sum()),
                    "weight_max": float(weights.max()),
                    "corruption_factor": float(weights.sum() / raw_corrupted.sum()),
                    "target_rbf_squared_distance": distances[flow_id],
                    "correction_l1": float((target - plugin).abs().sum()) if estimator == "dr" else None,
                    "c_imq": c_imq, "step_size": 0.15 * c_imq ** 2,
                    "context_lengthscale": float(kernel_x.length_scale),
                    "diagnostic_lengthscale": float(diagnostic_kernel.length_scale),
                    "reconstruction_mse": reconstruction_mse,
                    "seconds_representation_and_propensity": representation_seconds,
                    "shared_flow_id": f"seed{data_seed}_arm{arm}_{flow_id}",
                    **weight_diagnostics(target, X), **shared, **output_metrics,
                })
            write_csv_atomic(run_dir / "results.csv", rows)
            write_json_atomic(run_dir / "progress.json", {
                "completed_unique_targets": total_flows,
                "completed_unique_flows": 0 if args.targets_only else total_flows,
                "expected_unique_flows": 0 if args.targets_only else expected_flows,
                "completed_rows": len(rows), "expected_rows": expected_rows,
                "last_shared_flow_id": f"seed{data_seed}_arm{arm}_{flow_id}",
                "elapsed_seconds": time.perf_counter() - start,
            })
    return {"rows": len(rows), "unique_targets": total_flows,
            "unique_flows": 0 if args.targets_only else total_flows,
            "seconds_total": time.perf_counter() - start}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--n-obs", type=int, default=20000)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument("--n-ref", type=int, default=2000)
    parser.add_argument("--ae-iterations", type=int, default=3000)
    parser.add_argument("--flow-iterations", type=int, default=3000)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-landmarks", type=int, default=1024)
    parser.add_argument("--cv-max-n", type=int, default=2000)
    parser.add_argument("--gram-ridge", type=float)
    parser.add_argument("--cme-cutoffs", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1])
    parser.add_argument("--weight-factors", nargs="+", type=float, default=[1.0, 0.25, 0.05, 0.01])
    parser.add_argument("--weight-threshold", type=float, default=0.5)
    parser.add_argument("--mass-modes", nargs="+", choices=("raw", "preserve"), default=["raw", "preserve"])
    parser.add_argument("--rhs-block-size", type=int, default=2048)
    parser.add_argument("--metric-block-size", type=int, default=256)
    parser.add_argument("--n-projections", type=int, default=1000)
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mnist-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--tag", default="paper_severity")
    args = parser.parse_args(argv)
    for name in ("n_obs", "n_particles", "n_ref", "ae_iterations", "flow_iterations",
                 "latent_dim", "n_landmarks", "rhs_block_size", "metric_block_size", "n_projections"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_ref < args.n_particles or args.cv_max_n < 3:
        parser.error("n_ref must be >= n_particles and cv_max_n >= 3")
    if args.gram_ridge is not None and (not math.isfinite(args.gram_ridge) or args.gram_ridge <= 0):
        parser.error("gram_ridge must be finite and positive")
    for values, name in ((args.cme_cutoffs, "CME cutoffs"), (args.weight_factors, "weight factors")):
        if len(set(values)) != len(values) or any(not 0 < value <= 1 for value in values):
            parser.error(f"{name} must be unique and in (0, 1]")
    if not 0 < args.weight_threshold < 1:
        parser.error("weight_threshold must be in (0, 1)")
    if len(set(args.mass_modes)) != len(args.mass_modes):
        parser.error("mass modes must be unique")
    if len(set(args.data_seeds)) != len(args.data_seeds) or any(seed < 0 for seed in args.data_seeds):
        parser.error("data seeds must be unique and nonnegative")
    if len(set(args.arms)) != len(args.arms):
        parser.error("arms must be unique")
    if Path(args.tag).name != args.tag or args.tag in ("", ".", ".."):
        parser.error("tag must be a nonempty directory name")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this execution context")
    if not (args.mnist_root / "MNIST/raw/train-images-idx3-ubyte").exists():
        raise FileNotFoundError(f"local MNIST assets are missing under {args.mnist_root}")
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    source_paths = [Path(__file__), HERE / "colormnist_doublegen_misspecification.py"]
    source_paths += [REPO_ROOT / relative for relative in (
        "src/causality_utils/cpme.py", "src/data_utils/colormnist.py",
        "src/autoencoder_utils/colormnist_ae.py", "src/dens_ratio_utils/propensity.py",
        "src/kernel_utils/kernels.py", "src/kernel_utils/linear_algebra.py",
        "src/metric_utils/metrics.py", "src/mmd_flow_utils/mmd_flow.py")]
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
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "reference": "https://arxiv.org/html/2509.16842v1#A13.SS1.SSS1",
            "paper_configuration": "full-sample nuisances; AE on all logged rows; projected Nystrom; IMQ Euler",
            "sw2_dtype": "float32",
            "interpretation": "baseline fitted nuisances are not oracle-correct; FID not computed",
            "timing_note": "shared costs repeat on rows; deduplicate shared_flow_id for sampler times",
        }
        write_json_atomic(run_dir / "provenance.json", provenance)
        print(f"device={device}; output={run_dir}", flush=True)
        try:
            provenance.update(run_seed(args, data_seed, run_dir, device))
            provenance["status"] = "complete"
        except Exception as error:
            provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json_atomic(run_dir / "provenance.json", provenance)
            raise
        write_json_atomic(run_dir / "provenance.json", provenance)
        print(f"completed {provenance['rows']} rows in {provenance['seconds_total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
