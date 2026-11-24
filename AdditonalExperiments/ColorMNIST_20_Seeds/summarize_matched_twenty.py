"""Validate the revised ColorMNIST controls before summarizing the 20 DR seeds."""

import argparse
import csv
from hashlib import sha256
from io import StringIO
import math
from pathlib import Path
import re
from statistics import mean, stdev
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.stats import t

from summarize_results import NUMERIC_CONFIG, TEXT_CONFIG


METRICS = {"sw2_flow": "Pixel SW2", "latent_sw2_flow": "Latent SW2",
           "fid_legacy_flow": "Paper-compatible FID", "fid_clean_flow": "Clean-FID"}
TOLERANCES = {"sw2_flow": 1e-7, "latent_sw2_flow": 1e-7, "fid_legacy_flow": 1e-3}
TENSOR_HASHES = ("initial_sha256", "support_sha256", "oracle_sha256", "indices_sha256")
SEEDS = range(20)
ARMS = (0, 1)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_seed(seed, rows):
    check(len(rows) == 2 and {int(r["target_a"]) for r in rows} == set(ARMS),
          f"Seed {seed}: expected exactly two distinct arms")
    for row in rows:
        label = f"Seed {seed}, arm {row['target_a']}"
        counts = dict(NUMERIC_CONFIG, data_seed=seed, pipeline_seed=seed * 1000)
        for field, expected in counts.items():
            check(float(row[field]) == expected, f"{label}: unexpected {field}")
        settings = dict(TEXT_CONFIG, estimator="dr", encoder_source="archived" if seed < 3 else "trained")
        for field, expected in settings.items():
            check(row[field] == expected, f"{label}: unexpected {field}")
        for field in (*TENSOR_HASHES, "particle_indices_sha256", "encoder_sha256"):
            check(re.fullmatch("[0-9a-f]{64}", row[field]), f"{label}: invalid {field}")
        for field in ("step_size", "c_imq", "ridge_gram"):
            value = float(row[field])
            check(math.isfinite(value) and value > 0, f"{label}: invalid {field}")
        for field in METRICS:
            value = float(row[field])
            check(math.isfinite(value) and value >= 0, f"{label}: invalid {field}")
    for field in ("encoder_sha256", "support_sha256", "gpu_name"):
        check(rows[0][field] == rows[1][field] and rows[0][field],
              f"Seed {seed}: expected both arms to share {field}")


def compare_reference(seed, rows, reference_dir):
    stem = f"quality_n20000_seed{seed}"
    controls = [r for r in read_rows(reference_dir / f"{stem}.csv")
                if r["estimator"] == "dr" and r["solver"] == "projected_nystrom"]
    by_arm = {int(r["target_a"]): r for r in controls}
    check(len(controls) == 2 and set(by_arm) == set(ARMS), f"Invalid archived arms for seed {seed}")
    checkpoint = reference_dir / f"ae_n20000_seed{1000 * seed}.pt"
    checkpoint_digest = sha256(checkpoint.read_bytes()).hexdigest()
    differences = {field: 0.0 for field in TOLERANCES}
    with np.load(reference_dir / f"{stem}_particles.npz", allow_pickle=False) as archive:
        for row in rows:
            arm = int(row["target_a"])
            ref = by_arm[arm]
            label = f"Seed {seed}, arm {arm}"
            for field in ("data_seed", "n_obs", "n_particles", "c_imq", "step_size", "ridge_gram"):
                check(float(row[field]) == float(ref[field]), f"{label}: changed {field}")
            particles = archive[f"ds{seed}_r0_a{arm}"]
            expected = {field: ref[field] for field in (*TENSOR_HASHES, "gpu_name")}
            expected.update(encoder_sha256=checkpoint_digest,
                            particle_indices_sha256=sha256(particles.tobytes()).hexdigest())
            for field, value in expected.items():
                check(row[field] == value, f"{label}: changed {field}")
            for field, tolerance in TOLERANCES.items():
                difference = abs(float(row[field]) - float(ref[field]))
                check(difference <= tolerance,
                      f"{label}: {field} differs from revised control by {difference:.12g}")
                differences[field] = max(difference, differences[field])
    return {field: mean(float(r[field]) for r in controls) for field in TOLERANCES}, differences


def summarize(results_dir, reference_dir):
    seed_rows = []
    controls = []
    largest_difference = {field: 0.0 for field in TOLERANCES}
    for seed in SEEDS:
        rows = read_rows(results_dir / f"seed_{seed}.csv")
        validate_seed(seed, rows)
        if seed < 3:
            control, differences = compare_reference(seed, rows, reference_dir)
            controls.append(control)
            for field, diff in differences.items():
                largest_difference[field] = max(largest_difference[field], diff)
        else:
            checkpoint = results_dir / f"ae_seed{seed}.pt"
            check(sha256(checkpoint.read_bytes()).hexdigest() == rows[0]["encoder_sha256"],
                  f"Seed {seed}: saved encoder checksum differs from result")
        seed_row = {"data_seed": seed, "gpu_name": rows[0]["gpu_name"],
                    "encoder_sha256": rows[0]["encoder_sha256"]}
        for field in METRICS:
            seed_row[field] = mean(float(r[field]) for r in rows)
        seed_rows.append(seed_row)
    check(len({r["encoder_sha256"] for r in seed_rows}) == len(SEEDS),
          "Expected a distinct seed-specific encoder for each data seed")
    # Revised Table 1 reports the legacy FID mean and sample SD to three decimals.
    for aggregate in (mean, stdev):
        replay = aggregate(r["fid_legacy_flow"] for r in seed_rows[:3])
        original = aggregate(r["fid_legacy_flow"] for r in controls)
        check(f"{replay:.3f}" == f"{original:.3f}",
              f"Revised Table 1 FID {aggregate.__name__} differs at displayed precision: {replay} vs {original}")

    csv_text = StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=list(seed_rows[0]))
    writer.writeheader()
    writer.writerows(seed_rows)
    lines = ["# ColorMNIST repetition with the revised Table 1 encoders", "",
             "Each data seed uses 20,000 observations, 2,000 particles per intervention arm "
             "and 2,000 oracle references per arm. Seeds 0–2 reuse their revised Table 1 "
             "autoencoder checkpoints. Each additional seed uses its own encoder.", "",
             "We average the two arms within each data seed. The table reports the mean, "
             "sample standard deviation and two-sided 95% Student’s t interval for the mean "
             "across seed averages. SW2 is unscaled. The 20-seed estimate includes seeds 0–2. "
             "The archived quality runs measured only the paper-compatible FID.", "",
             "| Seeds | Metric | Mean | Sample SD | 95% interval for mean |",
             "|---|---|---:|---:|---|"]
    for name, sample in (("Revised Table 1, 0–2", controls), ("Rerun, 0–2", seed_rows[:3]),
                         ("Rerun, 0–19", seed_rows)):
        critical = t.ppf(.975, df=len(sample) - 1)
        for field, title in METRICS.items():
            if field not in sample[0]:
                continue
            values = [r[field] for r in sample]
            avg, sd = mean(values), stdev(values)
            half = critical * sd / math.sqrt(len(values))
            lines.append(f"| {name} | {title} | {avg:.6f} | {sd:.6f} | "
                         f"[{avg-half:.6f}, {avg+half:.6f}] |")
    lines += ["", "The six rerun arm rows for seeds 0–2 matched the archived encoder files, "
              "starting particles, latent supports, oracle images, evaluation indices, GPU model, "
              "IMQ scales and step sizes. The largest absolute per-arm score differences were "
              + ", ".join(f"{METRICS[field]} {v:.4g}" for field, v in largest_difference.items()) + "."]
    return csv_text.getvalue(), "\n".join(lines) + "\n"


def publish(path, content):
    output = NamedTemporaryFile("w", dir=path.parent, newline="", encoding="utf-8", delete=False)
    temporary = Path(output.name)
    try:
        with output:
            output.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    per_seed, report = summarize(args.results_dir, args.reference_dir)
    publish(args.results_dir / "per_seed.csv", per_seed)
    publish(args.results_dir / "comparison.md", report)
    print(report)
