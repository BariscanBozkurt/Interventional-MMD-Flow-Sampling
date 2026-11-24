"""Run the headline ColorMNIST sampler with independent encoder observations.

Twenty thousand logged observations are partitioned into 10,000 encoder and
10,000 causal observations. The existing simulator implements the split; this
launcher freezes its sources, records the partition, and validates the outputs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPT = Path("Simulations/ColorMNIST/dr_mmd_flow/colormnist_dr_mmd_flow_simulation.py")
FLAGS = [
    "--n_obs", "20000", "--n_particles", "2000", "--n_ref", "2000",
    "--eval_particles", "2000", "--target_arms", "0", "1", "--n_repeats", "1",
    "--estimators", "dr", "--dr_fit", "nosplit", "--solver", "projected_nystrom",
    "--n_landmarks", "1024", "--lambda_rule", "cv", "--n_folds", "5",
    "--init", "observed", "--ae_split", "half", "--ae_fraction", "0.5",
    "--latent_dim", "32", "--ae_iterations", "3000", "--flow_iterations", "3000",
    "--w_conf", "10", "--alpha", "0.2", "--k_mask", "20", "--tau", "0.08",
    "--fid", "both", "--fid_workers", "8", "--device", "cuda",
    "--tag", "independent_ae10k_causal10k",
]


def sha256(path):
    """Hash files in bounded memory, including the existing MNIST assets."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a complete JSON record atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(run_root, mnist_root):
    """Freeze Python sources and copy the existing three-seed headline scores."""
    sources = [(p, p.relative_to(REPO)) for p in sorted((REPO / "src").rglob("*.py"))]
    sources.append((REPO / SCRIPT, SCRIPT))
    for name in ("run_independent_encoder.py", "summarize_results.py"):
        sources.append((HERE / name, Path(name)))
    raw = mnist_root / "MNIST" / "raw"
    data_paths = [raw / name for name in (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    )]
    baseline_paths = [REPO / SCRIPT.parent / "results" / (
        "colormnist_est_headline_nobs20000_M2000_w10_a0.2_k20_tau0.08_"
        f"arms01_ds{seed}_nosplit_drnosplit_projnys_estplugin-ipw-dr.csv"
    ) for seed in range(3)]
    for path in [p for p, _ in sources] + data_paths + baseline_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("results", "logs", "provenance", "analysis", "baseline"):
        (run_root / name).mkdir()
    snapshot = run_root / "source_snapshot"
    hashes = {}
    for original, relative in sources:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[str(relative)] = sha256(destination)
        if hashes[str(relative)] != sha256(original):
            raise RuntimeError(f"Source changed during snapshot: {relative}")
    baseline_hashes = {}
    for original in baseline_paths:
        destination = run_root / "baseline" / original.name
        shutil.copy2(original, destination)
        baseline_hashes[original.name] = sha256(destination)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_seeds": [0, 1, 2], "flags": FLAGS,
        "n_obs": 20000, "n_ae": 10000, "n_causal": 10000,
        "n_particles_per_arm": 2000, "source_sha256": hashes,
        "mnist_root": str(mnist_root),
        "mnist_sha256": {str(p.relative_to(mnist_root)): sha256(p) for p in data_paths},
        "baseline_sha256": baseline_hashes,
        "baseline_description": "Original headline seeds 0-2; all 20,000 rows shared by encoder and target.",
        "split_rule": "CUDA randperm(20000), generator seed 1000*data_seed+2; first/last 10000.",
        "storage_policy": "CSV, partition and initial-particle indices only; no models, images or trajectories.",
    }
    write_json(run_root / "manifest.json", manifest)
    print(json.dumps({"prepared": str(run_root), "seeds": manifest["data_seeds"],
                      "source_files": len(hashes)}))


def validate_rows(run_root, seed):
    """Reject missing arms, changed sample budgets or nonfinite final metrics."""
    files = list((run_root / "results").glob(f"*_ds{seed}_*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one CSV for seed {seed}; found {len(files)}")
    with files[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 or {int(row["target_a"]) for row in rows} != {0, 1}:
        raise RuntimeError("Expected two complete DR arms")
    for row in rows:
        for key, expected in {"estimator": "dr", "ae_split": "half", "dr_fit": "nosplit",
                              "solver": "projected_nystrom", "init": "observed"}.items():
            if row[key] != expected:
                raise RuntimeError(f"Unexpected {key}: {row[key]}")
        for key, expected in {
            "data_seed": seed, "pipeline_seed": 1000 * seed, "repeat": 0,
            "n_obs": 20000, "n_ae": 10000, "n_causal": 10000, "n_dr": 10000,
            "n_particles_requested": 2000, "n_particles": 2000, "n_flow_eval": 2000,
            "n_ref": 2000, "n_landmarks": 1024, "latent_dim": 32,
            "ae_iterations": 3000, "flow_iterations": 3000,
        }.items():
            if int(row[key]) != expected:
                raise RuntimeError(f"Unexpected {key}: {row[key]}")
        for metric in ("sw2_flow", "fid_legacy_flow", "fid_clean_flow"):
            if not math.isfinite(float(row[metric])):
                raise RuntimeError(f"Nonfinite {metric}")
    return files[0]


def audit_partition(run_root, seed, result_path):
    """Reconstruct the frozen runner's split and check actual starting indices.

    This audits independent observation indices, not disjoint MNIST templates:
    the simulator samples handwriting templates with replacement from a fixed bank.
    """
    import numpy as np
    import torch

    generator = torch.Generator(device="cuda").manual_seed(1000 * seed + 2)
    split = torch.randperm(20000, generator=generator, device="cuda").cpu().numpy()
    ae_idx, causal_idx = split[:10000], split[10000:]
    if np.intersect1d(ae_idx, causal_idx).size or np.unique(split).size != 20000:
        raise RuntimeError("Encoder and causal partitions are not disjoint and exhaustive")
    particles_path = result_path.with_name(result_path.stem + "_particles.npz")
    with np.load(particles_path, allow_pickle=False) as particles:
        expected_keys = {f"ds{seed}_r0_a{arm}" for arm in (0, 1)}
        if set(particles.files) != expected_keys:
            raise RuntimeError("Unexpected particle-index keys")
        for key in expected_keys:
            indices = particles[key]
            if len(indices) != 2000 or len(np.unique(indices)) != 2000:
                raise RuntimeError("Expected 2,000 distinct starting observations per arm")
            if not np.isin(indices, causal_idx).all():
                raise RuntimeError("Initial particles include observations outside the causal half")
    path = run_root / "provenance" / f"partition_seed{seed}.npz"
    np.savez_compressed(path, ae_idx=ae_idx, causal_idx=causal_idx)
    return {"n_ae": len(ae_idx), "n_causal": len(causal_idx), "overlap": 0,
            "all_initial_particles_in_causal_half": True,
            "partition_sha256": sha256(path), "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name()}


def run(run_root, seed, dry_run):
    """Run one frozen pipeline and preserve failure as well as completion evidence."""
    manifest = json.loads((run_root / "manifest.json").read_text())
    if seed not in manifest["data_seeds"]:
        raise ValueError("Expected data seed 0, 1 or 2")
    snapshot = run_root / "source_snapshot"
    mnist_root = Path(manifest["mnist_root"])
    for base, hashes in ((snapshot, manifest["source_sha256"]),
                         (mnist_root, manifest["mnist_sha256"]),
                         (run_root / "baseline", manifest["baseline_sha256"])):
        for relative, digest in hashes.items():
            if sha256(base / relative) != digest:
                raise RuntimeError(f"Frozen input changed: {base / relative}")
    command = [sys.executable, "-B", "-u", str(snapshot / SCRIPT), *manifest["flags"],
               "--data_seeds", str(seed), "--mnist_root", str(mnist_root),
               "--out_dir", str(run_root / "results")]
    if dry_run:
        print(json.dumps(command))
        return
    if list((run_root / "results").glob(f"*_ds{seed}_*")):
        raise FileExistsError(f"Results already exist for seed {seed}")
    record_path = run_root / "provenance" / f"seed{seed}.json"
    with record_path.open("x") as stream:
        json.dump({"seed": seed, "status": "starting"}, stream)
    record = {"seed": seed, "command": command, "status": "running",
              "manifest_sha256": sha256(run_root / "manifest.json"),
              "start_unix": time.time(), "python": sys.version}
    write_json(record_path, record)
    try:
        result = subprocess.run(command, cwd=run_root, check=False)
        record["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Simulation exited with code {result.returncode}")
        result_path = validate_rows(run_root, seed)
        record["partition_audit"] = audit_partition(run_root, seed, result_path)
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.time() - record["start_unix"]
        record["output_sha256"] = {
            p.name: sha256(p) for p in sorted((run_root / "results").glob(f"*_ds{seed}_*"))
        }
        write_json(record_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--seed", type=int)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, default=REPO / "data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.dry_run:
            parser.error("--dry-run applies to --seed")
        prepare(args.run_root.resolve(), args.mnist_root.resolve())
    else:
        run(args.run_root.resolve(), args.seed, args.dry_run)


if __name__ == "__main__":
    main()
