"""Audit and aggregate the paired CelebA sampler-ablation CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]
DEFAULT_RESULTS = HERE / "results"
HEADLINE_RESULTS = DEPLOY / "Simulations" / "CelebA" / "dr_mmd_flow" / "results"
SAMPLERS = (
    "riesz_adamw",
    "imq_euler_deterministic",
    "imq_euler_injected_noise",
)
METRICS = (
    "latent_sw2_flow",
    "sw2_flow",
    "fid_legacy_flow",
    "fid_clean_flow",
    "mean_transport",
    "seconds_flow",
)
PAIR_FIELDS = (
    "sum_v",
    "sum_abs_v",
    "sum_vX",
    "v_ess",
    "n_particles",
    "n_flow_eval",
    "latent_sw2_initial",
    "sw2_observed",
    "fid_legacy_observed",
    "fid_clean_observed",
    "fold_lambdas",
)


def _read_rows(results: Path) -> list[dict]:
    files = sorted(results.glob("celeba_theory_sampler_ablation_*.csv"))
    rows = []
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = path.name
                rows.append(row)
    return rows


def _as_float(row: dict, field: str) -> float:
    return float(row[field])


def _assert_complete_and_paired(rows: list[dict]) -> None:
    expected = {(seed, arm, sampler) for seed in range(3) for arm in range(2) for sampler in SAMPLERS}
    actual = {
        (int(row["data_seed"]), int(row["target_a"]), row["sampler"])
        for row in rows
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicates = len(rows) != len(actual)
    if missing or extra or duplicates:
        raise RuntimeError(
            f"incomplete result grid: missing={missing}, extra={extra}, "
            f"rows={len(rows)}, unique={len(actual)}"
        )

    by_pair = defaultdict(list)
    for row in rows:
        by_pair[(row["data_seed"], row["target_a"])].append(row)
    for key, group in by_pair.items():
        for field in PAIR_FIELDS:
            values = [row[field] for row in group]
            if field == "fold_lambdas":
                good = len(set(values)) == 1
            else:
                floats = np.asarray([float(value) for value in values])
                good = np.allclose(floats, floats[0], rtol=0.0, atol=1e-12, equal_nan=True)
            if not good:
                raise AssertionError(f"unpaired field {field} at seed/arm {key}: {values}")


def _headline_rows() -> dict[tuple[int, int], dict]:
    rows = {}
    pattern = "celeba_est_headline_sz256_nobs20000_M2000_arms01_ds*_drnosplit_projnys_estplugin-ipw-dr.csv"
    for path in sorted(HEADLINE_RESULTS.glob(pattern)):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["estimator"] == "dr":
                    rows[(int(row["data_seed"]), int(row["target_a"]))] = row
    return rows


def _compare_riesz_to_headline(rows: list[dict]) -> list[str]:
    headline = _headline_rows()
    notes = []
    for row in rows:
        if row["sampler"] != "riesz_adamw":
            continue
        key = (int(row["data_seed"]), int(row["target_a"]))
        if key not in headline:
            notes.append(f"No archived headline row for seed={key[0]}, a={key[1]}")
            continue
        differences = {
            metric: _as_float(row, metric) - float(headline[key][metric])
            for metric in METRICS[:-1]
        }
        largest = max(abs(value) for value in differences.values())
        notes.append(
            f"seed={key[0]}, a={key[1]}: largest Riesz/headline metric difference "
            f"{largest:.3e}"
        )
    return notes


def _seed_arm_averages(rows: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    values = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        seed = int(row["data_seed"])
        arm = int(row["target_a"])
        sampler = row["sampler"]
        for metric in METRICS:
            values[sampler][metric][(seed, arm)] = _as_float(row, metric)

    result = defaultdict(dict)
    for sampler in SAMPLERS:
        for metric in METRICS:
            result[sampler][metric] = np.asarray(
                [
                    np.mean([values[sampler][metric][(seed, arm)] for arm in (0, 1)])
                    for seed in range(3)
                ]
            )
    return result


def _fmt(values: np.ndarray, digits: int = 4) -> str:
    return f"{values.mean():.{digits}f} +/- {values.std(ddof=1):.{digits}f}"


def _write_outputs(rows: list[dict], out_dir: Path) -> None:
    aggregated = _seed_arm_averages(rows)
    summary_path = out_dir / "sampler_ablation_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sampler", "metric", "mean", "std", "seed_arm_averages"],
        )
        writer.writeheader()
        for sampler in SAMPLERS:
            for metric in METRICS:
                values = aggregated[sampler][metric]
                writer.writerow(
                    {
                        "sampler": sampler,
                        "metric": metric,
                        "mean": values.mean(),
                        "std": values.std(ddof=1),
                        "seed_arm_averages": ";".join(f"{value:.12g}" for value in values),
                    }
                )

    labels = {
        "riesz_adamw": "Riesz--AdamW",
        "imq_euler_deterministic": "IMQ--Euler",
        "imq_euler_injected_noise": "IMQ--Euler + injected noise",
    }
    lines = [
        "# CelebA paired IMQ--Euler versus Riesz--AdamW ablation",
        "",
        "Each seed is first averaged over `a=0` and `a=1`; `+/-` is the sample standard deviation across the three seed averages.",
        "",
        "| Sampler | Latent SW2 | Pixel SW2 | Legacy FID | Clean FID | Flow seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sampler in SAMPLERS:
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[sampler],
                    _fmt(aggregated[sampler]["latent_sw2_flow"], 5),
                    _fmt(aggregated[sampler]["sw2_flow"], 5),
                    _fmt(aggregated[sampler]["fid_legacy_flow"], 2),
                    _fmt(aggregated[sampler]["fid_clean_flow"], 2),
                    _fmt(aggregated[sampler]["seconds_flow"], 1),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Paired differences from Riesz--AdamW", ""])
    for sampler in SAMPLERS[1:]:
        lines.append(f"### {labels[sampler]}")
        for metric in METRICS[:4]:
            delta = aggregated[sampler][metric] - aggregated["riesz_adamw"][metric]
            lines.append(f"- `{metric}`: {_fmt(delta, 5)}")
        lines.append("")

    lines.extend(["## Audit", ""])
    lines.extend(f"- {note}" for note in _compare_riesz_to_headline(rows))
    lines.extend(
        [
            "",
            "The IMQ rows instantiate the smooth positive-definite Euler algorithmic form. The empirical step size was frozen after a seed-0/arm-0 pilot; this table does not assert that it verifies every conservative sufficient condition of the finite-particle theorem.",
            "",
        ]
    )
    (out_dir / "sampler_ablation_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = _read_rows(args.results)
    if not rows:
        raise SystemExit(f"No ablation CSVs found under {args.results}")
    _assert_complete_and_paired(rows)
    _write_outputs(rows, args.results)


if __name__ == "__main__":
    main()
