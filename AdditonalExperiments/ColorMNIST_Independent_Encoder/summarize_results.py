"""Validate and summarize the three-seed independent-encoder ColorMNIST study.

Average arms within each data seed before reporting the mean and sample SD.
Incomplete seeds are excluded explicitly; the original headline CSVs supply
the shared-data baseline, rather than the later twenty-seed rerun.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


SEEDS = (0, 1, 2)
REQUIRED_METRICS = (
    "sw2_flow", "fid_legacy_flow", "fid_clean_flow",
    "recon_mse_train", "recon_mse_heldout",
)
METRICS = (*REQUIRED_METRICS, "latent_sw2_flow")
LABELS = {
    "sw2_flow": "Pixel SW2",
    "fid_legacy_flow": "Legacy FID",
    "fid_clean_flow": "Clean-FID",
    "recon_mse_train": "AE training-subset reconstruction MSE",
    "recon_mse_heldout": "AE causal-subset reconstruction MSE",
    "latent_sw2_flow": "Latent SW2 (representation dependent)",
}
NUMERIC_CONFIG = {
    "n_obs": 20000, "n_particles_requested": 2000, "n_particles": 2000,
    "n_flow_eval": 2000, "n_ref": 2000, "n_landmarks": 1024,
    "n_folds": 5, "w_conf": 10, "alpha": 0.2, "tau": 0.08,
    "k_mask": 20, "ae_iterations": 3000, "flow_iterations": 3000,
    "latent_dim": 32, "repeat": 0,
}
TEXT_CONFIG = {
    "init": "observed", "dr_fit": "nosplit", "solver": "projected_nystrom",
    "lambda_rule": "cv", "ipw_form": "ht",
}
GROUPS = {
    "original_headline": {"folder": "baseline", "count": 20000, "ae_split": "none"},
    "independent_encoder": {"folder": "results", "count": 10000, "ae_split": "half"},
}


def finite_number(value, location):
    """Parse a finite scalar and retain the field location in any error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location}: expected a number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{location}: expected a finite number, got {value!r}")
    return result


def integer(value, location):
    result = finite_number(value, location)
    if not result.is_integer():
        raise ValueError(f"{location}: expected an integer, got {value!r}")
    return int(result)


def validate_row(row, group, location):
    """Check the fixed protocol and return the seed, arm and metric values."""
    specification = GROUPS[group]
    for field, expected in {**TEXT_CONFIG, "ae_split": specification["ae_split"]}.items():
        if row.get(field) != expected:
            raise ValueError(f"{location}: {field}={row.get(field)!r}, expected {expected!r}")
    counts = {field: specification["count"] for field in ("n_ae", "n_causal", "n_dr")}
    for field, expected in {**NUMERIC_CONFIG, **counts}.items():
        actual = finite_number(row.get(field), f"{location}: {field}")
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"{location}: {field}={actual}, expected {expected}")
    seed = integer(row.get("data_seed"), f"{location}: data_seed")
    arm = integer(row.get("target_a"), f"{location}: target_a")
    if seed not in SEEDS or arm not in (0, 1):
        raise ValueError(f"{location}: unexpected seed={seed}, arm={arm}")
    if integer(row.get("pipeline_seed"), f"{location}: pipeline_seed") != 1000 * seed:
        raise ValueError(f"{location}: pipeline_seed must equal 1000 * data_seed")
    values = {}
    for metric in METRICS:
        value = row.get(metric)
        if metric not in REQUIRED_METRICS and value in (None, ""):
            values[metric] = None
        else:
            values[metric] = finite_number(value, f"{location}: {metric}")
            if values[metric] < 0:
                raise ValueError(f"{location}: {metric} must be nonnegative")
    return seed, arm, values


def load_group(run_root, group):
    """Read DR rows; exclude missing-arm seeds without hiding invalid rows."""
    directory = run_root / GROUPS[group]["folder"]
    paths = sorted(directory.glob("*.csv")) if directory.is_dir() else []
    rows = {}
    for path in paths:
        with path.open(newline="") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                location = f"{path.name}:{line}"
                if group == "original_headline" and row.get("estimator") in ("plugin", "ipw"):
                    continue
                if row.get("estimator") != "dr":
                    raise ValueError(f"{location}: expected a DR row")
                seed, arm, values = validate_row(row, group, location)
                if (seed, arm) in rows:
                    raise ValueError(f"{location}: duplicate DR row for seed={seed}, arm={arm}")
                rows[(seed, arm)] = (values, path.name)
    records, missing = [], {}
    for seed in SEEDS:
        absent = [arm for arm in (0, 1) if (seed, arm) not in rows]
        if absent:
            missing[str(seed)] = absent
            continue
        pair = [rows[(seed, arm)] for arm in (0, 1)]
        for metric in ("recon_mse_train", "recon_mse_heldout"):
            if not math.isclose(pair[0][0][metric], pair[1][0][metric], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{group}, seed={seed}: shared {metric} differs between arms")
        record = {"group": group, "data_seed": seed, "n_arms": 2,
                  "n_ae": GROUPS[group]["count"], "n_causal": GROUPS[group]["count"],
                  "n_dr": GROUPS[group]["count"],
                  "source_files": ";".join(sorted({item[1] for item in pair}))}
        for metric in METRICS:
            values = [item[0][metric] for item in pair]
            record[metric] = mean(values) if all(v is not None for v in values) else None
        records.append(record)
    coverage = {
        "status": "complete" if not missing else "partial",
        "expected_seeds": list(SEEDS), "complete_seeds": [r["data_seed"] for r in records],
        "missing_arms_by_seed": missing, "source_files": [p.name for p in paths],
        "directory_present": directory.is_dir(), "n_valid_dr_rows": len(rows),
    }
    return records, coverage


def summarize(records):
    """Produce one row per metric/group, retaining coverage for every mean."""
    summaries = []
    for group in GROUPS:
        for metric in METRICS:
            selected = [r for r in records if r["group"] == group and r[metric] is not None]
            values = [r[metric] for r in selected]
            seeds = [r["data_seed"] for r in selected]
            summaries.append({
                "group": group, "metric": metric,
                "status": "complete" if len(values) == len(SEEDS) else ("partial" if values else "unavailable"),
                "n_expected": len(SEEDS), "n_seeds": len(values),
                "seeds": ";".join(map(str, seeds)),
                "missing_seeds": ";".join(str(s) for s in SEEDS if s not in seeds),
                "mean": mean(values) if values else None,
                "sample_sd": stdev(values) if len(values) > 1 else None,
            })
    return summaries


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formatted_summary(row):
    if row["mean"] is None:
        return "Pending / unavailable (0/3 seeds)"
    value = f"{row['mean']:.6g}"
    if row["sample_sd"] is not None:
        value += f" ± {row['sample_sd']:.6g}"
    return f"{value} ({row['n_seeds']}/3 seeds; {row['status']})"


def comparison_markdown(summary, coverage):
    lookup = {(r["group"], r["metric"]): r for r in summary}
    lines = [
        "# ColorMNIST independent-encoder comparison", "",
        f"Status: **{coverage['status']}**. Only complete two-arm seeds enter each estimate.", "",
        "The baseline uses the original three-seed headline CSVs: 20,000 images train the encoder "
        "and the same 20,000 observations fit the causal target. The new run partitions 20,000 "
        "observations into 10,000 encoder-training and 10,000 causal-estimation rows.", "",
        "| Metric | Original shared-data headline | Independent encoder |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| {LABELS[metric]} | {formatted_summary(lookup['original_headline', metric])} "
                     f"| {formatted_summary(lookup['independent_encoder', metric])} |")
    lines += [
        "", "Values are unscaled means ± sample standard deviations after averaging the two "
        "arms within each seed. A single seed has no estimated sample SD. Missing seeds are "
        "not imputed, and partial estimates are not final three-seed results.", "",
        "Reconstruction errors are recorded once per trained encoder and repeated on the two "
        "arm rows; averaging those copies does not create additional replicates. They use at "
        "most 2,000 images from each subset. In the shared-data baseline, the CSV's 'heldout' "
        "subset is the same as the training subset and is not held out.", "",
        "The comparison changes encoder-training and causal-estimation sample sizes as well "
        "as their overlap. It also changes the eligible initial-particle pool. The common seed "
        "labels do not imply identical initial particles or latent coordinates; latent SW2 "
        "depends on the fitted representation. No runtime or statistical equivalence claim "
        "is inferred from this comparison.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    out = args.run_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    records, groups = [], {}
    try:
        for group in GROUPS:
            selected, groups[group] = load_group(args.run_root, group)
            records.extend(selected)
    except ValueError as error:
        coverage = {"status": "invalid", "error": str(error), "groups": groups}
        (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        # Replace any stale summaries from a previous invocation with an error report.
        write_csv(out / "per_seed.csv", [], ["group", "data_seed", *METRICS])
        write_csv(out / "summary.csv", [], ["group", "metric", "status", "mean", "sample_sd"])
        (out / "comparison.md").write_text(f"# Invalid inputs\n\n{error}\n")
        raise SystemExit(f"Invalid input; see {out / 'coverage.json'}: {error}") from error
    coverage = {
        "status": "complete" if all(g["status"] == "complete" for g in groups.values()) else "partial",
        "groups": groups,
        "aggregation": "Average arms within each data seed, then mean and sample SD across seeds.",
        "baseline": "Original three-seed headline DR rows, not the twenty-seed rerun.",
        "interpretation": "Incomplete seeds are excluded; partial estimates are not final three-seed results.",
    }
    summary = summarize(records)
    write_csv(out / "per_seed.csv", records,
              ["group", "data_seed", "n_arms", "n_ae", "n_causal", "n_dr", "source_files", *METRICS])
    write_csv(out / "summary.csv", summary, list(summary[0]))
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (out / "comparison.md").write_text(comparison_markdown(summary, coverage))
    print(json.dumps({"status": coverage["status"],
                      "complete_seeds": {k: v["complete_seeds"] for k, v in groups.items()}}))
    if args.require_complete and coverage["status"] != "complete":
        raise SystemExit("Incomplete seed coverage; labeled reports and coverage.json were written.")


if __name__ == "__main__":
    main()
