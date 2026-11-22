"""Freeze the original ColorMNIST implementation and run its DR-only 20-seed study.

Prepare once before submitting jobs. Each job reads the same hashed source snapshot,
uses one independent data seed, and writes only the original compact outputs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPT = Path("Simulations/ColorMNIST/dr_mmd_flow/colormnist_dr_mmd_flow_simulation.py")
DEFAULT_FLAGS = [
    "--n_obs", "20000", "--n_particles", "2000", "--n_ref", "2000",
    "--eval_particles", "2000", "--target_arms", "0", "1", "--n_repeats", "1",
    "--estimators", "dr", "--dr_fit", "nosplit", "--solver", "projected_nystrom",
    "--n_landmarks", "1024", "--lambda_rule", "cv", "--n_folds", "5",
    "--init", "observed", "--ae_split", "none", "--latent_dim", "32",
    "--ae_iterations", "3000", "--flow_iterations", "3000",
    "--w_conf", "10", "--alpha", "0.2", "--k_mask", "20", "--tau", "0.08",
    "--fid", "both", "--fid_workers", "8", "--device", "cuda",
    "--tag", "twenty_seeds",
]


def sha256(path):
    """Hash a file without loading large dataset files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Write a complete record atomically beside the results."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prepare(run_root, mnist_root):
    """Copy only Python sources and the existing three-seed reference CSVs."""
    run_root.mkdir(parents=True, exist_ok=False)
    snapshot = run_root / "source_snapshot"
    sources = [(p, p.relative_to(REPO)) for p in sorted((REPO / "src").rglob("*.py"))]
    sources.append((HERE / "simulation_scripts" / SCRIPT.name, SCRIPT))
    sources.append((Path(__file__).resolve(), Path("run_twenty_seeds.py")))
    hashes = {}
    for original, relative in sources:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[str(relative)] = sha256(destination)
        if hashes[str(relative)] != sha256(original):
            raise RuntimeError(f"Source changed during snapshot: {relative}")
    raw = mnist_root / "MNIST" / "raw"
    data_paths = [raw / name for name in (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    )]
    data_hashes = {str(p.relative_to(mnist_root)): sha256(p) for p in data_paths}
    baseline = run_root / "baseline"
    baseline.mkdir()
    baseline_hashes = {}
    for seed in range(3):
        name = ("colormnist_est_headline_nobs20000_M2000_w10_a0.2_k20_tau0.08_"
                f"arms01_ds{seed}_nosplit_drnosplit_projnys_estplugin-ipw-dr.csv")
        original = REPO / SCRIPT.parent / "results" / name
        shutil.copy2(original, baseline / name)
        baseline_hashes[name] = sha256(baseline / name)
    for name in ("results", "logs", "provenance", "analysis"):
        (run_root / name).mkdir()
    manifest = {
        "data_seeds": list(range(20)), "flags": DEFAULT_FLAGS,
        "source_sha256": hashes, "mnist_sha256": data_hashes,
        "baseline_sha256": baseline_hashes, "mnist_root": str(mnist_root),
        "original_runner_sha256": sha256(REPO / SCRIPT),
        "copied_runner_sha256": hashes[str(SCRIPT)],
        "storage_policy": "CSV and initial-particle indices only; no trajectories or models",
    }
    if manifest["original_runner_sha256"] != manifest["copied_runner_sha256"]:
        raise RuntimeError("Copied runner differs from the original; inspect before launching")
    write_json(run_root / "manifest.json", manifest)
    print(json.dumps({"prepared": str(run_root), "source_files": len(hashes),
                      "source_bytes": sum((snapshot / p).stat().st_size for p in hashes)}))


def run(run_root, seed, dry_run):
    """Validate the frozen inputs and execute one original, unmodified simulation."""
    manifest = json.loads((run_root / "manifest.json").read_text())
    if seed not in manifest["data_seeds"]:
        raise ValueError("Seed must be one of 0 through 19")
    snapshot = run_root / "source_snapshot"
    for relative, digest in manifest["source_sha256"].items():
        if sha256(snapshot / relative) != digest:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    mnist_root = Path(manifest["mnist_root"])
    for relative, digest in manifest["mnist_sha256"].items():
        if sha256(mnist_root / relative) != digest:
            raise RuntimeError(f"MNIST hash mismatch: {relative}")
    command = [sys.executable, "-B", "-u", str(snapshot / SCRIPT), *manifest["flags"],
               "--data_seeds", str(seed), "--mnist_root", str(mnist_root),
               "--out_dir", str(run_root / "results")]
    if dry_run:
        print(json.dumps(command))
        return
    record_path = run_root / "provenance" / f"seed{seed}.json"
    # Refuse to overwrite either a previous attempt or its original result files.
    with record_path.open("x") as stream:
        json.dump({"seed": seed, "status": "starting"}, stream)
    if list((run_root / "results").glob(f"*_ds{seed}_*.csv")):
        raise FileExistsError(f"Results already exist for seed {seed}")
    record = {"seed": seed, "command": command, "status": "running",
              "manifest_sha256": sha256(run_root / "manifest.json"),
              "start_unix": time.time(), "python": sys.version}
    write_json(record_path, record)
    result = subprocess.run(command, cwd=run_root, check=False)
    record.update(status="complete" if result.returncode == 0 else "failed",
                  returncode=result.returncode, elapsed_seconds=time.time()-record["start_unix"])
    record["output_sha256"] = {
        p.name: sha256(p) for p in sorted((run_root / "results").glob(f"*_ds{seed}_*"))
    }
    if result.returncode == 0:
        files = list((run_root / "results").glob(f"*_ds{seed}_*.csv"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one results CSV for seed {seed}; found {len(files)}")
        with files[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 2 or {int(row["target_a"]) for row in rows} != {0, 1}:
            raise RuntimeError(f"Expected exactly two arm rows for seed {seed}")
        for row in rows:
            if row["estimator"] != "dr" or int(row["data_seed"]) != seed:
                raise RuntimeError("Wrong estimator or seed in output")
            if any(int(row[key]) != expected for key, expected in (
                ("n_obs", 20000), ("n_ae", 20000), ("n_dr", 20000),
                ("n_particles", 2000), ("n_flow_eval", 2000), ("n_ref", 2000),
                ("repeat", 0), ("ae_iterations", 3000), ("flow_iterations", 3000),
            )):
                raise RuntimeError("Output does not match the fixed headline configuration")
            for metric in ("sw2_flow", "fid_legacy_flow", "fid_clean_flow"):
                if not math.isfinite(float(row[metric])):
                    raise RuntimeError(f"Nonfinite output metric: {metric}")
        record["validation"] = "two complete DR arms; fixed counts; finite SW2 and both FIDs"
    write_json(record_path, record)
    if result.returncode:
        raise SystemExit(result.returncode)


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
            parser.error("--dry-run applies to --seed, not --prepare")
        prepare(args.run_root.resolve(), args.mnist_root.resolve())
    else:
        run(args.run_root.resolve(), args.seed, args.dry_run)


if __name__ == "__main__":
    main()
