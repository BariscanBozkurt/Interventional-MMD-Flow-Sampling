"""Compare exact ColorMNIST sampling with full/coarse cached RFF requests.

Inputs are explicit completed baseline directories from the public ColorMNIST
nuisance experiment. Only their uncorrupted DR baseline is used. This runner
does not require an earlier RFF run, a particular machine, or archived results.
Full runs reuse the baseline's exact 2,000-particle endpoint; structural smokes
recompute the exact comparator at their declared particle/iteration budgets.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from autoencoder_utils import ConvAE
from data_utils import to_image_range
from reference_field import (
    CUDAGraphReferenceSampler, apply_reference_flow,
    integrate_coefficient_blocks, record_reference_flow,
)
from rff_imq import cache_target_mean, exact_imq_euler_flow, sample_imq_frequencies


class CUDAGraphImageSampler(CUDAGraphReferenceSampler):
    """Fixed-shape field plus frozen decoder; each call returns owned images.

    One instance is not safe for concurrent host calls or different streams.
    Initializer RNG, input gathering and host transfers remain outside capture.
    """

    def __init__(self, initial, coefficients, omega, c, step_size, image_decoder):
        self.image_decoder = image_decoder
        self.image_buffer = None
        super().__init__(initial, coefficients, omega, c, step_size)

    def _execute(self):
        super()._execute()
        self.image_buffer = self.image_decoder(self.output_buffer)
        return self.image_buffer

    @torch.no_grad()
    def __call__(self, initial, clone_output=True):
        super().__call__(initial, clone_output=False)
        return self.image_buffer.clone() if clone_output else self.image_buffer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_arrays(path, **values):
    np.savez_compressed(path, **{
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for key, value in values.items()
    })


def read_arrays(source, filename, fingerprints):
    path = source / filename
    fingerprints[filename] = sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_source(source, args):
    """Accept a complete baseline-only run without requiring a severity grid."""
    metadata = json.loads((source / "provenance.json").read_text())
    config = metadata["config"]
    if metadata.get("status") != "complete" or int(metadata["data_seed"]) < 0:
        raise ValueError("Source must be a completed run with a nonnegative data seed")
    if config.get("latent_dim") != 32:
        raise ValueError("This ColorMNIST comparison requires 32-dimensional latents")
    if not args.smoke:
        expected = {"n_obs": 20000, "n_particles": 2000, "n_ref": 2000,
                    "ae_iterations": 3000, "flow_iterations": 3000,
                    "n_landmarks": 1024, "cv_max_n": 2000}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Full comparison requires the paper baseline budgets; use --smoke for a structural check")
    with (source / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    baselines = {}
    for arm in (0, 1):
        matches = [row for row in rows if int(row["arm"]) == arm and row["estimator"] == "dr"
                   and float(row["cme_cutoff"]) == 1 and float(row["weight_factor"]) == 1
                   and row["corruption_mass"] == "baseline"]
        if len(matches) != 1:
            raise ValueError(f"Expected one uncorrupted DR baseline row for arm {arm}")
        row = matches[0]
        if (row["dr_fit"] != "nosplit" or row["ae_split"] != "none"
                or row["solver"] != "projected_nystrom"
                or int(row["n_obs"]) != config["n_obs"]
                or int(row["n_cme_train"]) != config["n_obs"]):
            raise ValueError("Baseline must use full-sample nuisances and all observed CME training rows")
        c, eta = float(row["c_imq"]), float(row["step_size"])
        if not math.isfinite(c) or c <= 0 or not math.isclose(eta, .15 * c * c, rel_tol=1e-12):
            raise ValueError("Invalid IMQ bandwidth or Euler step size")
        baselines[arm] = row
    fingerprints = {name: sha256(source / name) for name in
                    ("provenance.json", "results.csv", "autoencoder.pt")}
    return metadata, baselines, fingerprints


def load_projections(args):
    """Share actual normalized directions across methods, arms and datasets."""
    if args.projection_file is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
        values = torch.randn(args.projections, 2352, generator=generator)
        values = values / values.norm(dim=1, keepdim=True)
        origin = {"kind": "generated_cpu_torch", "seed": 0}
    else:
        with np.load(args.projection_file, allow_pickle=False) as archive:
            values = torch.from_numpy(archive["pixel_directions"][:args.projections].copy())
        origin = {"kind": "supplied", "file_sha256": sha256(args.projection_file)}
    if (values.shape != (args.projections, 2352) or values.dtype != torch.float32
            or not torch.isfinite(values).all()
            or not torch.allclose(values.norm(dim=1), torch.ones(args.projections), atol=1e-5, rtol=1e-5)):
        raise ValueError("Expected enough normalized float32 pixel directions of dimension 2352")
    return values, origin


def make_fid_scorers(args, device):
    if args.skip_fid:
        return {}, {}
    from metric_utils.fid import CleanFIDScorer, LegacyFIDScorer, default_clean_fid_weights_dir
    paths = {"legacy": Path(torch.hub.get_dir()) / "checkpoints" / "inception_v3_google-0cc3c7bd.pth",
             "clean": Path(default_clean_fid_weights_dir()) / "inception-2015-12-05.pt"}
    if not args.allow_fid_download and any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("FID weights are not cached; pre-cache them or pass --allow-fid-download")
    scorers = {"legacy": LegacyFIDScorer(device, batch_size=64, distance_device=device),
               "clean": CleanFIDScorer(device, batch_size=64, num_workers=args.threads,
                                        allow_download=args.allow_fid_download)}
    fingerprints = {name: sha256(path) for name, path in paths.items()}
    return scorers, fingerprints


@torch.no_grad()
def score_images(images, reference, projections, scorers, reference_features):
    if images.shape != reference.shape or not torch.isfinite(images).all():
        raise ValueError("Generated/reference pools must have equal finite image shapes")
    generated = (images.flatten(1) @ projections.T).sort(dim=0).values
    oracle = (reference.flatten(1) @ projections.T).sort(dim=0).values
    metrics = {"pixel_sw2": float((generated - oracle).square().mean().sqrt()),
               "fid_legacy": None, "fid_clean": None}
    for name, scorer in scorers.items():
        features = scorer.features(images, image_range="zero_one")
        if features.shape != (len(images), 2048):
            raise ValueError("FID must use the complete pooled image sample")
        metrics[f"fid_{name}"] = scorer.distance_from_features(features, reference_features[name])
    if any(value is not None and not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite image metric")
    return metrics


@torch.no_grad()
def generate_requests(bank, candidates, expected_indices, expected_initial, coefficients,
                      omega, c, eta, decoder, query_seed, args):
    """Perform real sequential requests and retain owned outputs and latents."""
    device = bank.device
    setup_start = time.perf_counter()
    sampler = None
    if device.type == "cuda" and not args.no_graph:
        sampler = CUDAGraphImageSampler(bank[:args.request_size], coefficients, omega, c, eta, decoder)
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    generator = np.random.default_rng(query_seed)
    images, latents, indices, starts = [], [], [], []
    first_snapshot = None
    for _ in range(args.requests):
        chosen = generator.choice(candidates, size=args.request_size, replace=True)
        initial = bank[torch.as_tensor(np.searchsorted(candidates, chosen), device=device)]
        if sampler is None:
            endpoint = apply_reference_flow(initial, coefficients, omega, c, eta)
            returned = decoder(endpoint).clone()
        else:
            returned = sampler(initial, clone_output=True)
            endpoint = sampler.output_buffer.clone()
        images.append(returned)
        latents.append(endpoint)
        indices.append(chosen)
        starts.append(initial)
        if first_snapshot is None:
            first_snapshot = returned.clone()
    pooled_images, pooled_latents = torch.cat(images), torch.cat(latents)
    np.testing.assert_array_equal(np.concatenate(indices), expected_indices)
    if not torch.equal(torch.cat(starts), expected_initial):
        raise AssertionError("Request starts differ from the shared query pool")
    last_snapshot = images[-1].clone()
    if sampler is not None:
        sampler(expected_initial[-args.request_size:].flip(0).contiguous())
    first_survives = torch.equal(images[0], first_snapshot)
    last_survives = torch.equal(images[-1], last_snapshot)
    pointers = len({value.data_ptr() for value in images})
    if not (first_survives and last_survives and pointers == args.requests):
        raise AssertionError("Returned samples must own their storage across requests")
    if not torch.isfinite(pooled_latents).all() or not torch.isfinite(pooled_images).all():
        raise FloatingPointError("Nonfinite request output")
    check = {"request_count": args.requests, "request_size": args.request_size,
             "distinct_owned_outputs": pointers, "starts_identical": True,
             "first_output_survives": first_survives, "last_output_survives": last_survives,
             "additional_graph_ownership_replays": int(sampler is not None),
             "seconds_graph_setup": setup_seconds, "cuda_graph": sampler is not None}
    return pooled_images, pooled_latents, check


@torch.no_grad()
def run_arm(source, output, metadata, baseline, support, observed, ae, fingerprints,
            arm, args, projections, scorers):
    seed, device = int(metadata["data_seed"]), support.device
    evaluation = read_arrays(source, f"evaluation_arm{arm}.npz", fingerprints)
    targets = read_arrays(source, f"targets_arm{arm}.npz", fingerprints)
    raw_weights = targets["dr_c1_f1_baseline"]
    count = args.request_size * args.requests
    guide_indices = evaluation["initial_indices"][:args.guide_size]
    guide = torch.as_tensor(evaluation["initial_particles"][:args.guide_size], device=device)
    if (raw_weights.shape != (len(support),) or not np.isfinite(raw_weights).all()
            or guide.shape != (args.guide_size, 32)
            or len(np.unique(guide_indices)) != args.guide_size
            or not np.all(observed[guide_indices] == arm)
            or not torch.equal(support[torch.as_tensor(guide_indices, device=device)], guide)):
        raise ValueError("Invalid raw target or observed-arm guide initialization")
    reference = torch.as_tensor(evaluation["reference_images"][:count], device=device)
    if reference.shape != (count, 3, 28, 28) or not torch.isfinite(reference).all():
        raise ValueError("Source does not contain enough finite reference images")
    weights = torch.as_tensor(raw_weights, dtype=support.dtype, device=device)
    c, eta = float(baseline["c_imq"]), float(baseline["step_size"])
    candidates = np.flatnonzero(observed == arm)
    bank = support[torch.as_tensor(candidates, device=device)]
    query_seed = args.query_seed + 100 * seed + arm
    query_indices = np.random.default_rng(query_seed).choice(candidates, size=count, replace=True)
    query_initial = support[torch.as_tensor(query_indices, device=device)]
    save_arrays(output / f"queries_arm{arm}.npz", initial_indices=query_indices, initial_latents=query_initial,
                request_boundaries=np.arange(0, count + 1, args.request_size), guide_indices=guide_indices)

    def decode(z):
        return to_image_range(ae.decode(z)).clamp(0, 1)

    reference_features = {name: scorer.features(reference, image_range="zero_one")
                          for name, scorer in scorers.items()}
    rows, checks = [], []

    def evaluate(method, dimension, stride, images, latents, cache_bytes):
        metrics = score_images(images, reference, projections, scorers, reference_features)
        filename = f"samples_arm{arm}_{method}_D{dimension}.npz"
        save_arrays(output / filename, images=images, latents=latents)
        rows.append({"data_seed": seed, "arm": arm, "method": method, "n_features": dimension,
                     "feature_seed": args.feature_seed, "effective_updates": math.ceil(args.iterations / stride),
                     "stride": stride, "n_generated": count, "n_reference": len(reference),
                     "request_size": count if dimension == 0 else args.request_size,
                     "request_count": 1 if dimension == 0 else args.requests,
                     "field_cache_bytes": cache_bytes, "field_cache_MB": None if cache_bytes is None else cache_bytes / 1e6,
                     "smoke": args.smoke, "sample_file": filename, **metrics})
        print(f"seed{seed} arm{arm} {method} D{dimension}: SW2={metrics['pixel_sw2']:.6f}", flush=True)

    if args.smoke:
        initial = torch.as_tensor(evaluation["initial_particles"][:count], device=device)
        if initial.shape != (count, 32):
            raise ValueError("Smoke exact comparator needs enough source initial particles")
        exact = exact_imq_euler_flow(initial, support, weights, c, eta, args.iterations, block_size=2048)
        exact_method = "exact_recomputed_smoke"
    else:
        archived = read_arrays(source, f"particles_arm{arm}_dr_c1_f1_baseline.npz", fingerprints)
        exact = torch.as_tensor(archived["particles"], device=device)
        if exact.shape != (count, 32) or not torch.isfinite(exact).all():
            raise ValueError("Archived exact cloud must match the complete requested budget")
        exact_method = "exact_archived"
    evaluate(exact_method, 0, 1, decode(exact), exact, None)
    for dimension in args.feature_dimensions:
        omega = sample_imq_frequencies(32, dimension, c, args.feature_seed, device=device)
        target_mean = cache_target_mean(support, weights, omega, c)
        guide_final, coefficients = record_reference_flow(guide, target_mean, omega, c, eta, args.iterations)
        replay_count = min(16, len(guide))
        replay = apply_reference_flow(guide[:replay_count], coefficients, omega, c, eta)
        replay_error = float((replay - guide_final[:replay_count]).abs().max())
        if not torch.isfinite(guide_final).all() or replay_error > 2e-4:
            raise AssertionError("Recorded field does not replay its own reference trajectory")
        coarse = integrate_coefficient_blocks(coefficients, args.coarse_stride)
        save_arrays(output / f"field_arm{arm}_D{dimension}.npz", coefficients=coefficients,
                    coarse_coefficients=coarse, omega=omega, target_mean=target_mean,
                    guide_initial=guide, guide_final=guide_final, c=np.array(c), step_size=np.array(eta),
                    coarse_stride=np.array(args.coarse_stride), guide_indices=guide_indices)
        for method, stride, field in (("cached_full", 1, coefficients),
                                      ("cached_coarse", args.coarse_stride, coarse)):
            images, latents, check = generate_requests(bank, candidates, query_indices, query_initial,
                field, omega, c, eta, decode, query_seed, args)
            check.update(arm=arm, n_features=dimension, method=method,
                         guide_replay_count=replay_count, guide_replay_max_error=replay_error)
            checks.append(check)
            cache_bytes = (field.numel() + omega.numel()) * field.element_size()
            evaluate(method, dimension, stride, images, latents, cache_bytes)
            del images, latents
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows, checks


@torch.no_grad()
def run_source(source, output, args, projections, scorers):
    metadata, baselines, fingerprints = validate_source(source, args)
    output.mkdir(parents=True, exist_ok=False)
    seed, device = int(metadata["data_seed"]), torch.device(args.device)
    config_keys = ("n_obs", "n_particles", "n_ref", "ae_iterations", "flow_iterations",
                   "latent_dim", "n_landmarks", "cv_max_n")
    provenance = {"status": "running", "data_seed": seed, "smoke": args.smoke,
                  "source_config": {key: metadata["config"][key] for key in config_keys},
                  "source_files_sha256": fingerprints,
                  "initializer": "uniform with replacement from the observed-arm latent bank",
                  "target": "raw signed baseline DR coefficients; no normalization or clipping",
                  "exact_comparator": "recomputed at smoke budgets" if args.smoke else "complete archived endpoint",
                  "reference_guide": "saved observed-arm initialization without replacement"}
    write_json(output / "provenance.json", provenance)
    started = time.perf_counter()
    try:
        arrays = read_arrays(source, "logged_support.npz", fingerprints)
        support = torch.as_tensor(arrays["Z"], device=device)
        observed = arrays["A"]
        if (support.shape != (metadata["config"]["n_obs"], 32) or support.dtype != torch.float32
                or observed.shape != (len(support),) or set(np.unique(observed)) != {0, 1}
                or not torch.isfinite(support).all()):
            raise ValueError("Invalid logged support or intervention-arm labels")
        checkpoint = torch.load(source / "autoencoder.pt", map_location="cpu", weights_only=True)
        if checkpoint["latent_dim"] != 32:
            raise ValueError("Checkpoint latent dimension differs from the saved support")
        ae = ConvAE(latent_dim=32).to(device).eval()
        ae.load_state_dict(checkpoint["state_dict"])
        for parameter in ae.parameters():
            parameter.requires_grad_(False)
        rows, checks = [], []
        for arm in (0, 1):
            arm_rows, arm_checks = run_arm(source, output, metadata, baselines[arm], support, observed,
                                          ae, fingerprints, arm, args, projections, scorers)
            rows.extend(arm_rows)
            checks.extend(arm_checks)
            write_csv(output / "results.csv", rows)
            write_json(output / "checks.json", checks)
        provenance.update(status="complete", quality_rows=len(rows), checks=len(checks),
                          seconds_total=time.perf_counter() - started,
                          output_sha256={p.name: sha256(p) for p in output.iterdir() if p.name != "provenance.json"})
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "provenance.json", provenance)
    return rows


def summarize(rows):
    """Average arms within each dataset before computing dataset mean/sample SD."""
    groups = {}
    for row in rows:
        key = (row["method"], row["n_features"], row["effective_updates"], row["stride"])
        groups.setdefault(key, {}).setdefault(row["data_seed"], []).append(row)
    summary = []
    for key, seeds in groups.items():
        if any(len(values) != 2 or {row["arm"] for row in values} != {0, 1} for values in seeds.values()):
            raise ValueError("Each dataset must contribute both intervention arms")
        first = next(iter(seeds.values()))[0]
        result = dict(zip(("method", "n_features", "effective_updates", "stride"), key))
        result.update(n_data_seeds=len(seeds), field_cache_MB=first["field_cache_MB"])
        for metric in ("pixel_sw2", "fid_legacy", "fid_clean"):
            values = [np.mean([row[metric] for row in arms]) for arms in seeds.values()
                      if all(row[metric] is not None for row in arms)]
            result[f"{metric}_mean"] = float(np.mean(values)) if values else None
            result[f"{metric}_SD"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(result)
    order = {"exact_archived": 0, "exact_recomputed_smoke": 0, "cached_full": 1, "cached_coarse": 2}
    return sorted(summary, key=lambda row: (order[row["method"]], row["n_features"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", nargs="+", type=Path, required=True,
                        help="One or more completed baseline directories, with distinct data seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--feature-dimensions", nargs="+", type=int)
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--guide-size", type=int)
    parser.add_argument("--coarse-stride", type=int)
    parser.add_argument("--request-size", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--query-seed", type=int, default=20261119)
    parser.add_argument("--projections", type=int)
    parser.add_argument("--projection-file", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Structural check; recompute exact comparator at these budgets")
    parser.add_argument("--skip-fid", action="store_true", help="Skip both FIDs in a structural smoke only")
    parser.add_argument("--no-graph", action="store_true", help="Use eager replay, including on CUDA")
    parser.add_argument("--allow-fid-download", action="store_true")
    args = parser.parse_args()
    defaults = {"feature_dimensions": ([4096, 16384], [64, 128]), "iterations": (3000, 3),
                "guide_size": (2000, 32), "coarse_stride": (100, 2), "request_size": (8, 8),
                "requests": (250, 4), "projections": (1000, 8)}
    for name, budgets in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, budgets[int(args.smoke)])
    if (not args.feature_dimensions or len(set(args.feature_dimensions)) != len(args.feature_dimensions)
            or any(d < 2 or d % 2 for d in args.feature_dimensions)):
        parser.error("Feature dimensions must be distinct positive even integers")
    if min(args.iterations, args.guide_size, args.coarse_stride, args.request_size,
           args.requests, args.projections, args.threads) < 1 or min(args.feature_seed, args.query_seed) < 0:
        parser.error("Budgets must be positive and random seeds nonnegative")
    if not args.smoke and (args.iterations != 3000 or args.guide_size != 2000
                           or args.request_size != 8 or args.requests != 250 or args.projections != 1000):
        parser.error("Changed particle/iteration/request/projection budgets require --smoke")
    if args.skip_fid and not args.smoke:
        parser.error("--skip-fid is permitted only with --smoke")
    if args.request_size * args.requests < 2:
        parser.error("The evaluated pool needs at least two images")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; structural CPU checks use --device cpu --smoke")
    torch.set_num_threads(args.threads)
    sources = [(source, validate_source(source, args)[0]) for source in args.source_dir]
    seeds = [int(metadata["data_seed"]) for _, metadata in sources]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Input directories must have distinct data seeds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    projections, projection_origin = load_projections(args)
    save_arrays(args.output_dir / "evaluation_projections.npz", pixel_directions=projections)
    scorers, fid_hashes = make_fid_scorers(args, device)
    code = [Path(__file__), HERE / "rff_imq.py", HERE / "reference_field.py",
            REPO_ROOT / "src/autoencoder_utils/colormnist_ae.py",
            REPO_ROOT / "src/data_utils/colormnist.py", REPO_ROOT / "src/metric_utils/fid.py"]
    config = {key: value for key, value in vars(args).items()
              if key not in ("source_dir", "output_dir", "projection_file")}
    provenance = {"status": "running", "config": config, "data_seeds": seeds,
                  "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in code},
                  "projections": projection_origin, "fid_weights_sha256": fid_hashes,
                  "fid_scorers": {name: scorer.metadata() for name, scorer in scorers.items()},
                  "torch_version": torch.__version__, "numpy_version": np.__version__,
                  "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_tf32": torch.backends.cudnn.allow_tf32},
                  "storage": "Float32 field coefficients plus frequencies per arm; excludes bank, AE and graph workspaces",
                  "aggregation": "Equal arm average within dataset, then mean and sample SD across distinct data seeds",
                  "timing": "Setup and total durations are incidental; no isolated latency benchmark",
                  "limitations": "Finite empirical initializer; fixed-feature and reference-field approximations; no exact target recovery claim"}
    write_json(args.output_dir / "provenance.json", provenance)
    try:
        rows = []
        for source, metadata in sources:
            folder = args.output_dir / f"seed{int(metadata['data_seed'])}"
            rows.extend(run_source(source, folder, args, projections.to(device), scorers))
        write_csv(args.output_dir / "results.csv", rows)
        summary = summarize(rows)
        write_csv(args.output_dir / "summary.csv", summary)
        provenance.update(status="complete", quality_rows=len(rows),
                          output_sha256={str(p.relative_to(args.output_dir)): sha256(p)
                                         for p in args.output_dir.rglob("*") if p.is_file()
                                         and p != args.output_dir / "provenance.json"})
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output_dir / "provenance.json", provenance)
    print(f"Completed comparison and arm-then-dataset summary: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
