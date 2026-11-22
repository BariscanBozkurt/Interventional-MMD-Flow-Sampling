"""Aggregate the streamlined kernel-herding comparisons.

The image scripts store several outputs in one row because they reuse the paper
simulation drivers.  This file maps only explicit columns to a small, long-form
table.  In particular, it never interprets a ``*_flow`` compatibility alias in
a herding CSV as DR--IMMD.

Distribution metrics are first averaged over the two intervention arms within
each data seed.  Times are summed over arms.  The reported uncertainty is the
sample standard deviation over data seeds 0, 1, and 2, matching the paper.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_ARMS = (0, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=HERE / "results")
    parser.add_argument("--tag", default="streamlined")
    parser.add_argument(
        "--decfm_immd_dir",
        type=Path,
        default=HERE.parents[1] / "Simulations" / "DecFM_2D" / "dr_mmd_flow" / "results",
        help="directory containing the matched paper headline 2D DR--IMMD CSVs",
    )
    parser.add_argument(
        "--decfm_tag",
        default="rebuttal",
        help="2D herding tag (the image jobs use --tag separately)",
    )
    parser.add_argument("--out_dir", type=Path, default=HERE / "results" / "analysis")
    return parser.parse_args()


def read_csvs(paths, label):
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no {label} CSVs found")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        source = pd.Series(
            str(path.resolve()), index=frame.index, name="source_file"
        )
        frame = pd.concat([frame, source], axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def validate_design(frame, label):
    require_columns(
        frame,
        ["data_seed", "target_a", "estimator", "n_obs", "n_particles"],
        label,
    )
    frame = frame.copy()
    frame = frame[frame["estimator"] == "dr"]
    if frame.empty:
        raise ValueError(f"{label} contains no estimator='dr' rows")
    if set(frame["data_seed"].astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError(f"{label} must contain exactly data seeds {EXPECTED_SEEDS}")
    if set(frame["target_a"].astype(int)) != set(EXPECTED_ARMS):
        raise ValueError(f"{label} must contain exactly arms {EXPECTED_ARMS}")
    if set(frame["n_obs"].astype(int)) != {20000}:
        raise ValueError(f"{label} must use n_obs=20000")
    if set(frame["n_particles"].astype(int)) != {2000}:
        raise ValueError(f"{label} must return n_particles=2000")
    counts = frame.groupby(["data_seed", "target_a"]).size()
    if len(counts) != 6 or not (counts == 1).all():
        raise ValueError(
            f"{label} must contain one DR row per seed and arm; got\n{counts}"
        )
    return frame.sort_values(["data_seed", "target_a"]).reset_index(drop=True)


def value(row, column):
    if column is None or column not in row.index:
        return np.nan
    return row[column]


def append_method(records, row, dataset, method, columns):
    record = {
        "dataset": dataset,
        "data_seed": int(row["data_seed"]),
        "target_a": int(row["target_a"]),
        "method": method,
        "sw1": value(row, columns.get("sw1")),
        "sw2": value(row, columns.get("sw2")),
        "mmd2": value(row, columns.get("mmd2")),
        "latent_sw2": value(row, columns.get("latent_sw2")),
        "fid_legacy": value(row, columns.get("fid_legacy")),
        "fid_clean": value(row, columns.get("fid_clean")),
        "sampler_seconds": value(row, columns.get("sampler_seconds")),
        "decode_seconds": value(row, columns.get("decode_seconds")),
        "n_unique": value(row, columns.get("n_unique")),
        "unique_fraction": value(row, columns.get("unique_fraction")),
        "max_multiplicity": value(row, columns.get("max_multiplicity")),
        "selected_target_arm_fraction": value(
            row, columns.get("selected_target_arm_fraction")
        ),
        "source_file": row["source_file"],
    }
    records.append(record)


def color_records(frame):
    require_columns(
        frame,
        [
            "sw2_herding",
            "sw2_herding_decoded",
            "sw2_immd",
            "latent_sw2_herding",
            "latent_sw2_immd",
            "fid_legacy_herding",
            "fid_legacy_herding_decoded",
            "fid_legacy_immd",
            "fid_clean_herding",
            "fid_clean_herding_decoded",
            "fid_clean_immd",
            "seconds_herding",
            "seconds_herding_decode",
            "seconds_immd_flow",
            "seconds_immd_decode",
        ],
        "ColorMNIST",
    )
    records = []
    for _, row in frame.iterrows():
        append_method(
            records,
            row,
            "ColorMNIST",
            "Initial particles",
            {
                "sw1": "sw1_initial",
                "sw2": "sw2_initial",
                "mmd2": "mmd2_initial",
                "latent_sw2": "latent_sw2_initial",
                "fid_legacy": "fid_legacy_initial",
                "fid_clean": "fid_clean_initial",
            },
        )
        append_method(
            records,
            row,
            "ColorMNIST",
            "DR-IMMD Flow",
            {
                "sw1": "sw1_immd",
                "sw2": "sw2_immd",
                "mmd2": "mmd2_immd",
                "latent_sw2": "latent_sw2_immd",
                "fid_legacy": "fid_legacy_immd",
                "fid_clean": "fid_clean_immd",
                "sampler_seconds": "seconds_immd_flow",
                "decode_seconds": "seconds_immd_decode",
            },
        )
        herding_common = {
            "latent_sw2": "latent_sw2_herding",
            "sampler_seconds": "seconds_herding",
            "n_unique": "n_unique",
            "unique_fraction": "unique_fraction",
            "max_multiplicity": "max_multiplicity",
            "selected_target_arm_fraction": "selected_target_arm_fraction",
        }
        append_method(
            records,
            row,
            "ColorMNIST",
            "Kernel herding (indexed)",
            {
                **herding_common,
                "sw1": "sw1_herding",
                "sw2": "sw2_herding",
                "mmd2": "mmd2_herding",
                "fid_legacy": "fid_legacy_herding",
                "fid_clean": "fid_clean_herding",
            },
        )
        append_method(
            records,
            row,
            "ColorMNIST",
            "Kernel herding (decoded)",
            {
                **herding_common,
                "sw1": "sw1_herding_decoded",
                "sw2": "sw2_herding_decoded",
                "mmd2": "mmd2_herding_decoded",
                "fid_legacy": "fid_legacy_herding_decoded",
                "fid_clean": "fid_clean_herding_decoded",
                "decode_seconds": "seconds_herding_decode",
            },
        )
    return records


def celeba_records(herding):
    require_columns(
        herding,
        [
            "sw2_herding",
            "sw2_herding_decoded",
            "latent_sw2_herding",
            "fid_legacy_herding",
            "fid_legacy_herding_decoded",
            "fid_clean_herding",
            "fid_clean_herding_decoded",
            "sw2_immd",
            "latent_sw2_immd",
            "fid_legacy_immd",
            "fid_clean_immd",
            "seconds_herding",
            "seconds_herding_decode",
            "seconds_immd_flow",
            "seconds_immd_decode",
        ],
        "CelebA herding",
    )
    records = []
    for _, row in herding.iterrows():
        append_method(
            records,
            row,
            "CelebA",
            "Initial particles",
            {
                "sw1": "sw1_observed",
                "sw2": "sw2_observed",
                "mmd2": "mmd2_observed",
                "latent_sw2": "latent_sw2_initial",
                "fid_legacy": "fid_legacy_observed",
                "fid_clean": "fid_clean_observed",
            },
        )
        append_method(
            records,
            row,
            "CelebA",
            "DR-IMMD Flow",
            {
                "sw1": "sw1_immd",
                "sw2": "sw2_immd",
                "mmd2": "mmd2_immd",
                "latent_sw2": "latent_sw2_immd",
                "fid_legacy": "fid_legacy_immd",
                "fid_clean": "fid_clean_immd",
                "sampler_seconds": "seconds_immd_flow",
                "decode_seconds": "seconds_immd_decode",
            },
        )
        herding_common = {
            "latent_sw2": "latent_sw2_herding",
            "sampler_seconds": "seconds_herding",
            "n_unique": "n_unique",
            "unique_fraction": "unique_fraction",
            "max_multiplicity": "max_multiplicity",
            "selected_target_arm_fraction": "selected_target_arm_fraction",
        }
        append_method(
            records,
            row,
            "CelebA",
            "Kernel herding (indexed)",
            {
                **herding_common,
                "sw1": "sw1_herding",
                "sw2": "sw2_herding",
                "mmd2": "mmd2_herding",
                "fid_legacy": "fid_legacy_herding",
                "fid_clean": "fid_clean_herding",
            },
        )
        append_method(
            records,
            row,
            "CelebA",
            "Kernel herding (decoded)",
            {
                **herding_common,
                "sw1": "sw1_herding_decoded",
                "sw2": "sw2_herding_decoded",
                "mmd2": "mmd2_herding_decoded",
                "fid_legacy": "fid_legacy_herding_decoded",
                "fid_clean": "fid_clean_herding_decoded",
                "decode_seconds": "seconds_herding_decode",
            },
        )
    return records


def validate_decfm(frame, label, samplers):
    require_columns(
        frame,
        ["data_seed", "target_a", "estimator", "sampler", "n_obs", "n_particles"],
        label,
    )
    frame = frame[frame["estimator"] == "dr"].copy()
    if set(frame["data_seed"].astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError(f"{label} must contain exactly data seeds {EXPECTED_SEEDS}")
    if set(frame["target_a"].astype(int)) != set(EXPECTED_ARMS):
        raise ValueError(f"{label} must contain exactly arms {EXPECTED_ARMS}")
    if set(frame["sampler"]) != set(samplers):
        raise ValueError(f"{label} must contain samplers {samplers}")
    if set(frame["n_obs"].astype(int)) != {20000}:
        raise ValueError(f"{label} must use n_obs=20000")
    if set(frame["n_particles"].astype(int)) != {2000}:
        raise ValueError(f"{label} must return n_particles=2000")
    counts = frame.groupby(["data_seed", "target_a", "sampler"]).size()
    if len(counts) != 12 or not (counts == 1).all():
        raise ValueError(
            f"{label} must contain one DR row per seed, arm, and sampler; got\n{counts}"
        )
    return frame.sort_values(["data_seed", "target_a", "sampler"]).reset_index(drop=True)


def load_decfm_immd(directory):
    paths = directory.glob(
        "decfm2d_est_headline_nobs20000_M2000_*_ds*_drnosplit_projnys*.csv"
    )
    frame = read_csvs(paths, "two-dimensional headline DR--IMMD")
    return validate_decfm(
        frame,
        "two-dimensional headline DR--IMMD",
        ("rbf_euler", "riesz_adamw"),
    )


def decfm_records(herding, immd):
    require_columns(
        herding,
        [
            "base_sampler",
            "sw2_flow",
            "seconds_herding",
            "n_unique",
            "unique_fraction",
            "max_multiplicity",
            "selected_target_arm_fraction",
            "target_mass_handling",
        ],
        "two-dimensional herding",
    )
    herding = herding.copy()
    expected_base = {
        "kernel_herding_rbf": "rbf_euler",
        "kernel_herding_riesz": "riesz_adamw",
    }
    observed = dict(zip(herding["sampler"], herding["base_sampler"]))
    if observed != expected_base:
        raise ValueError(
            "two-dimensional herding sampler/base_sampler mapping is not the audited "
            f"mapping: {observed}"
        )
    paired = herding.merge(
        immd,
        left_on=["data_seed", "target_a", "base_sampler"],
        right_on=["data_seed", "target_a", "sampler"],
        suffixes=("_herding_file", "_immd_file"),
        validate="one_to_one",
    )
    for column in ("sw1_initial", "sw2_initial", "mmd2_initial", "sw2_floor"):
        left = paired[f"{column}_herding_file"].to_numpy(float)
        right = paired[f"{column}_immd_file"].to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            maximum = float(np.nanmax(np.abs(left - right)))
            raise ValueError(
                f"2D herding and IMMD artifacts are not exactly paired in {column} "
                f"(maximum absolute difference {maximum:.3g})"
            )

    records = []
    for _, row in herding.iterrows():
        match = immd[
            (immd["data_seed"] == row["data_seed"])
            & (immd["target_a"] == row["target_a"])
            & (immd["sampler"] == row["base_sampler"])
        ]
        if len(match) != 1:
            raise ValueError("2D IMMD pairing is not one-to-one")
        immd_row = match.iloc[0]

        base_sampler = row["base_sampler"]
        if base_sampler == "rbf_euler":
            append_method(
                records,
                row,
                "Two-dimensional mixture",
                "Initial particles",
                {"sw1": "sw1_initial", "sw2": "sw2_initial", "mmd2": "mmd2_initial"},
            )
            flow_name = "DR-IMMD Flow (RBF--Euler)"
            herding_name = "Kernel herding (RBF)"
        else:
            flow_name = "DR-IMMD Flow (Riesz--AdamW)"
            herding_name = "Kernel herding (Riesz, unit-mass target)"

        append_method(
            records,
            immd_row,
            "Two-dimensional mixture",
            flow_name,
            {
                "sw1": "sw1_flow",
                "sw2": "sw2_flow",
                "mmd2": "mmd2_flow",
                "sampler_seconds": "seconds_flow",
            },
        )
        # Here *_flow is not a compatibility alias: the row is first validated as
        # a row-oriented kernel_herding_* method emitted by the 2D wrapper.
        append_method(
            records,
            row,
            "Two-dimensional mixture",
            herding_name,
            {
                "sw1": "sw1_flow",
                "sw2": "sw2_flow",
                "mmd2": "mmd2_flow",
                "sampler_seconds": "seconds_herding",
                "n_unique": "n_unique",
                "unique_fraction": "unique_fraction",
                "max_multiplicity": "max_multiplicity",
                "selected_target_arm_fraction": "selected_target_arm_fraction",
            },
        )
    return records


def reduce_arms(per_arm):
    identifiers = ["dataset", "data_seed", "method"]
    mean_columns = [
        "sw1",
        "sw2",
        "mmd2",
        "latent_sw2",
        "fid_legacy",
        "fid_clean",
        "n_unique",
        "unique_fraction",
        "max_multiplicity",
        "selected_target_arm_fraction",
    ]
    sum_columns = ["sampler_seconds", "decode_seconds"]
    grouped = per_arm.groupby(identifiers, sort=False)
    counts = grouped["target_a"].nunique()
    if not (counts == 2).all():
        raise ValueError("every dataset/seed/method must contain both intervention arms")
    means = grouped[mean_columns].mean().reset_index()
    sums = grouped[sum_columns].sum(min_count=1).reset_index()
    return means.merge(sums, on=identifiers, validate="one_to_one")


def aggregate_seeds(per_seed):
    metrics = [column for column in per_seed.columns if column not in {
        "dataset", "data_seed", "method"
    }]
    rows = []
    for (dataset, method), group in per_seed.groupby(["dataset", "method"], sort=False):
        if set(group["data_seed"].astype(int)) != set(EXPECTED_SEEDS):
            raise ValueError(f"{dataset}/{method} does not have all three data seeds")
        row = {"dataset": dataset, "method": method, "n_seeds": len(group)}
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def formatted(mean, sd, digits):
    if pd.isna(mean):
        return "--"
    if pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def markdown_report(aggregate):
    lines = [
        "# Streamlined kernel-herding comparison",
        "",
        "All distribution metrics average arms within each data seed and then report "
        "mean ± sample standard deviation over seeds 0, 1, and 2. Runtime is summed "
        "over arms within a seed before the same three-seed summary.",
        "",
        "## Interpretation guardrails",
        "",
        "- Indexed image herding returns logged pixels verbatim. Its FID therefore "
        "includes the advantage of bypassing autoencoder or SD-VAE reconstruction; "
        "the decoded-herding row is the representation-matched comparison.",
        "- Herding is a finite-candidate selector, whereas DR--IMMD moves particles "
        "off the logged support. SW2 and FID alone do not distinguish retrieval from "
        "synthesis, so the selection multiplicities are reported separately.",
        "- The two-dimensional Riesz row uses an explicitly unit-normalized DR target "
        "because the conditionally positive-definite energy discrepancy requires "
        "equal mass. The RBF row uses the estimated DR coefficients verbatim.",
        "",
    ]
    for dataset in aggregate["dataset"].drop_duplicates():
        table = aggregate[aggregate["dataset"] == dataset]
        lines.extend([
            f"## {dataset}",
            "",
            "| Method | SW2 | Latent SW2 | Legacy FID | CleanFID |",
            "|---|---:|---:|---:|---:|",
        ])
        for _, row in table.iterrows():
            lines.append(
                f"| {row['method']} | "
                f"{formatted(row.sw2_mean, row.sw2_sd, 6)} | "
                f"{formatted(row.latent_sw2_mean, row.latent_sw2_sd, 6)} | "
                f"{formatted(row.fid_legacy_mean, row.fid_legacy_sd, 3)} | "
                f"{formatted(row.fid_clean_mean, row.fid_clean_sd, 3)} |"
            )
        lines.extend([
            "",
            "| Method | Sampler seconds | Decode seconds | Unique / 2,000 | "
            "Unique fraction | Max multiplicity | Target-arm fraction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for _, row in table.iterrows():
            lines.append(
                f"| {row['method']} | "
                f"{formatted(row.sampler_seconds_mean, row.sampler_seconds_sd, 2)} | "
                f"{formatted(row.decode_seconds_mean, row.decode_seconds_sd, 2)} | "
                f"{formatted(row.n_unique_mean, row.n_unique_sd, 1)} | "
                f"{formatted(row.unique_fraction_mean, row.unique_fraction_sd, 4)} | "
                f"{formatted(row.max_multiplicity_mean, row.max_multiplicity_sd, 2)} | "
                f"{formatted(row.selected_target_arm_fraction_mean, row.selected_target_arm_fraction_sd, 4)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    decfm_tag = args.decfm_tag
    color = read_csvs(
        (args.results_root / "ColorMNIST").glob(
            f"colormnist_kernel_herding_{args.tag}_*.csv"
        ),
        "ColorMNIST herding",
    )
    color = validate_design(color, "ColorMNIST herding")

    celeba = read_csvs(
        (args.results_root / "CelebA").glob(
            f"celeba_kernel_herding_{args.tag}_*.csv"
        ),
        "CelebA herding",
    )
    celeba = validate_design(celeba, "CelebA herding")

    decfm = read_csvs(
        (args.results_root / "DecFM_2D").glob(
            f"decfm2d_kernel_herding_{decfm_tag}_*.csv"
        ),
        "two-dimensional herding",
    )
    decfm = validate_decfm(
        decfm,
        "two-dimensional herding",
        ("kernel_herding_rbf", "kernel_herding_riesz"),
    )
    decfm_immd = load_decfm_immd(args.decfm_immd_dir)

    records = color_records(color)
    records.extend(celeba_records(celeba))
    records.extend(decfm_records(decfm, decfm_immd))
    per_arm = pd.DataFrame(records)
    per_seed = reduce_arms(per_arm)
    aggregate = aggregate_seeds(per_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_arm.to_csv(args.out_dir / "kernel_herding_per_arm.csv", index=False)
    per_seed.to_csv(args.out_dir / "kernel_herding_per_seed.csv", index=False)
    aggregate.to_csv(args.out_dir / "kernel_herding_aggregate.csv", index=False)
    (args.out_dir / "kernel_herding_report.md").write_text(
        markdown_report(aggregate), encoding="utf-8"
    )
    print(markdown_report(aggregate))


if __name__ == "__main__":
    main()
