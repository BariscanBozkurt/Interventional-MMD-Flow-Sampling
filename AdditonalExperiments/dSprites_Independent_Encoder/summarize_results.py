"""Validate and summarize the three-seed dSprites encoder-split experiment."""

import argparse
import csv
import statistics
from pathlib import Path


METRICS = (
    ("sw2_flow_mean", "Pixel SW2 x 1,000", 1000.0),
    ("latent_sw2_flow_mean", "Latent SW2 x 1,000", 1000.0),
    ("fid_legacy_flow", "FID", 1.0),
    ("fid_clean_flow", "Clean-FID", 1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[2]
                 / "Simulations/dSprites/dr_mmd_flow/results_confounded"),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_rows(paths):
    rows = []
    for path in sorted(paths):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return [row for row in rows if row.get("estimator") == "dr"]


def validate(rows, split):
    expected_seeds = {0, 1, 2}
    observed = {int(row["data_seed"]) for row in rows}
    if observed != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, found {observed}")
    if len(rows) != 3:
        raise ValueError(f"Expected one DR row per seed, found {len(rows)} rows")

    n_ae = 10000 if split else 20000
    n_causal = 10000 if split else 20000
    ae_split = "half" if split else "none"
    for row in rows:
        checks = {
            "design": "confounded",
            "n_obs": "20000",
            "n_ae": str(n_ae),
            "n_causal": str(n_causal),
            "ae_split": ae_split,
            "dr_fit": "nosplit",
            "solver": "projected_nystrom",
            "n_landmarks": "1024",
            "n_particles": "2000",
            "n_resample": "8",
            "ae_iterations": "4000",
            "flow_iterations": "2000",
            "target_mass": "rescale",
        }
        wrong = {key: (row.get(key), value) for key, value in checks.items()
                 if row.get(key) != value}
        if wrong:
            raise ValueError(f"Invalid row for seed {row.get('data_seed')}: {wrong}")


def summary(rows):
    result = {}
    for key, _, scale in METRICS:
        values = [scale * float(row[key]) for row in rows]
        result[key] = (statistics.mean(values), statistics.stdev(values))
    return result


def format_value(value):
    return f"{value[0]:.3f} ± {value[1]:.3f}"


def main():
    args = parse_args()
    result_paths = list((args.run_root / "results").glob("*.csv"))
    split_rows = read_rows(result_paths)

    baseline_paths = list(args.baseline_dir.glob(
        "dsprites_confounded_confounded_nobs20000_*_M2000.csv"
    ))
    baseline_rows = read_rows(baseline_paths)

    complete = len(split_rows) == 3
    if args.require_complete and not complete:
        seeds = sorted({int(row["data_seed"]) for row in split_rows})
        raise SystemExit(f"Incomplete split experiment: found seeds {seeds}")

    validate(baseline_rows, split=False)
    if complete:
        validate(split_rows, split=True)

    analysis_dir = args.run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = summary(baseline_rows)
    split_summary = summary(split_rows) if complete else None

    with (analysis_dir / "per_seed.csv").open("w", newline="") as handle:
        fields = ["protocol", "data_seed"] + [key for key, _, _ in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for protocol, rows in (("shared", baseline_rows), ("disjoint", split_rows)):
            for row in rows:
                writer.writerow({"protocol": protocol, "data_seed": row["data_seed"],
                                 **{key: row[key] for key, _, _ in METRICS}})

    lines = [
        "# dSprites independent-encoder comparison",
        "",
        "Mean ± sample SD across data seeds 0--2. Lower is better.",
        "",
        "| Encoder / causal protocol | AE rows | Causal rows | "
        + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---:|---:|" + "---:|" * len(METRICS),
        "| Shared observations | 20,000 | 20,000 | "
        + " | ".join(format_value(baseline_summary[key]) for key, _, _ in METRICS)
        + " |",
    ]
    if split_summary is not None:
        lines.append(
            "| Disjoint 10k / 10k split | 10,000 | 10,000 | "
            + " | ".join(format_value(split_summary[key]) for key, _, _ in METRICS)
            + " |"
        )
    else:
        lines.append("| Disjoint 10k / 10k split | 10,000 | 10,000 | pending |")
    lines.extend([
        "",
        "The split condition uses 2,000 particles for the single stochastic-policy "
        "target. It halves both training samples, so this is a fixed-budget robustness "
        "check rather than a sample-size-controlled estimate of the cost of independence.",
        "",
    ])
    (analysis_dir / "comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

