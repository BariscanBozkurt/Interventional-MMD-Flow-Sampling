"""Summarize the prespecified ColorMNIST DR-only extension across data seeds.

Arms are paired and averaged within seed before estimating variability. The
first three seeds are a subset of the twenty, so their comparison is descriptive,
not a test between independent samples. Missing seeds are never imputed.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


METRICS = (
    "sw2_flow", "sw2_initial", "sw2_floor_matched",
    "fid_legacy_flow", "fid_clean_flow",
    "fid_legacy_initial", "fid_clean_initial",
)
NUMERIC_CONFIG = {
    "n_obs": 20000, "n_ae": 20000, "n_causal": 20000, "n_dr": 20000,
    "n_particles_requested": 2000, "n_particles": 2000,
    "n_flow_eval": 2000, "n_ref": 2000, "n_landmarks": 1024,
    "w_conf": 10, "alpha": 0.2, "tau": 0.08, "k_mask": 20,
    "ae_iterations": 3000, "flow_iterations": 3000, "latent_dim": 32,
    "repeat": 0,
}
TEXT_CONFIG = {
    "init": "observed", "ae_split": "none", "dr_fit": "nosplit",
    "solver": "projected_nystrom", "lambda_rule": "cv", "ipw_form": "ht",
}


def finite_number(value, label):
    """Parse a required finite number, with a useful failure location."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return number


def load_seeds(directory, expected, source):
    """Validate rows and average complete two-arm seeds only.

    A seed with one missing arm is reported as incomplete and excluded. Duplicate
    rows, malformed configurations, nonfinite required metrics, and unexpected
    seeds raise instead of being silently averaged or dropped.
    """
    rows = {}
    files = []
    if directory is not None:
        if not directory.is_dir():
            raise ValueError(f"Missing input directory: {directory}")
        for path in sorted(directory.glob("*.csv")):
            files.append(path.name)
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("estimator") != "dr":
                        continue
                    location = path.name
                    for key, wanted in TEXT_CONFIG.items():
                        if row.get(key) != wanted:
                            raise ValueError(f"{location}: {key} != {wanted!r}")
                    for key, wanted in NUMERIC_CONFIG.items():
                        actual = finite_number(row[key], f"{location}: {key}")
                        if not math.isclose(actual, wanted, rel_tol=0, abs_tol=1e-10):
                            raise ValueError(f"{location}: {key}={actual}, expected {wanted}")
                    seed = int(row["data_seed"])
                    arm = int(row["target_a"])
                    if seed not in expected or arm not in (0, 1):
                        raise ValueError(f"{location}: unexpected seed={seed}, arm={arm}")
                    if int(row["pipeline_seed"]) != 1000 * seed:
                        raise ValueError(f"{location}: unexpected pipeline seed")
                    key = (seed, arm)
                    if key in rows:
                        raise ValueError(f"Duplicate DR row for seed={seed}, arm={arm}")
                    values = {}
                    for metric in METRICS:
                        value = row.get(metric, "")
                        if value == "" and metric.startswith("fid_"):
                            values[metric] = None
                        else:
                            values[metric] = finite_number(value, f"{location}: {metric}")
                    rows[key] = (values, path.name)
    per_seed = []
    incomplete = {}
    for seed in sorted(expected):
        missing = [arm for arm in (0, 1) if (seed, arm) not in rows]
        if missing:
            incomplete[str(seed)] = missing
            continue
        pair = [rows[(seed, arm)] for arm in (0, 1)]
        record = {"source": source, "data_seed": seed, "n_arms": 2,
                  "source_files": ";".join(sorted({item[1] for item in pair}))}
        for metric in METRICS:
            values = [item[0][metric] for item in pair]
            record[metric] = mean(values) if all(v is not None for v in values) else None
        per_seed.append(record)
    coverage = {"expected_seeds": sorted(expected),
                "complete_seeds": [row["data_seed"] for row in per_seed],
                "missing_arms_by_seed": incomplete, "source_files": files,
                "provided": directory is not None}
    return per_seed, coverage


def summarize_group(name, records, expected, provided=True):
    """Report sample SD and a two-sided Student-t interval for the seed mean."""
    selected = [row for row in records if row["data_seed"] in expected]
    summaries = []
    for metric in METRICS:
        observed = [row for row in selected if row[metric] is not None]
        values = [row[metric] for row in observed]
        center = mean(values) if values else None
        sd = stdev(values) if len(values) > 1 else None
        half = float(t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values)) if sd is not None else None
        seeds = [row["data_seed"] for row in observed]
        missing = sorted(set(expected) - set(seeds))
        status = "not_provided" if not provided else ("complete" if not missing else "partial")
        summaries.append({
            "group": name, "metric": metric, "status": status,
            "n_expected": len(expected), "n_seeds": len(values),
            "seeds": ";".join(map(str, seeds)),
            "missing_seeds": ";".join(map(str, missing)),
            "mean": center, "sample_sd": sd,
            "mean_ci95_low": center - half if half is not None else None,
            "mean_ci95_high": center + half if half is not None else None,
        })
    return summaries


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    fresh, coverage = load_seeds(args.results_dir, set(range(20)), "new_run")
    baseline, old_coverage = load_seeds(args.baseline_dir, {0, 1, 2}, "original_baseline")
    summary = summarize_group("original_baseline_0_2", baseline, {0, 1, 2}, args.baseline_dir is not None)
    summary += summarize_group("new_run_0_2", fresh, {0, 1, 2})
    summary += summarize_group("new_seeds_3_19", fresh, set(range(3, 20)))
    summary += summarize_group("all_20_seeds", fresh, set(range(20)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["source", "data_seed", "n_arms", "source_files", *METRICS]
    write_csv(args.out_dir / "per_seed.csv", baseline + fresh, fields)
    write_csv(args.out_dir / "summary.csv", summary, list(summary[0]))
    notes = {
        "new_run": coverage, "original_baseline": old_coverage,
        "aggregation": "Average two arms within each data seed, then mean and sample SD across data seeds.",
        "interval": "Two-sided Student-t 95% confidence interval for the mean across independent data seeds; not an equivalence test.",
        "comparison": "Seeds 0-2 are included in all 20. These groups are not independent. Original versus rerun seeds 0-2 is a reproduction check.",
        "interpretation": "Partial summaries exclude missing or incomplete seeds and must not be presented as completed 20-seed results.",
    }
    (args.out_dir / "coverage.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps({"complete_seeds": len(fresh), "expected_seeds": 20,
                      "missing_seeds": sorted(set(range(20)) - {r["data_seed"] for r in fresh}),
                      "status": "complete" if len(fresh) == 20 else "partial"}))
    if args.require_complete and (len(fresh) != 20 or (args.baseline_dir is not None and len(baseline) != 3)):
        raise SystemExit("Incomplete seed coverage; partial summaries were written and labeled.")


if __name__ == "__main__":
    main()
