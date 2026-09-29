"""
Plain-text tables for the simulation analysis notebooks.

The notebooks exist to be read, so every table here is monospaced, aligned, and
formatted as `mean +/- sd` in the layout the collaborators already circulate. No
styled DataFrames, no multi-index, no colour.

Aggregation is the house order throughout: where a run scores two intervention
arms, the arms are averaged within the run first, then repeats within a data
seed, and the reported spread is the standard deviation ACROSS SEEDS.
"""

import glob
import os

import pandas as pd

RESULTS = "../{dataset}/dr_mmd_flow/results"

#: Reference numbers from H. Dance, reproduced verbatim for the head-to-head
#: tables. FID here is the legacy TorchVision Inception-v3 convention, which is
#: the one these tables use; clean-FID is a different scorer and the two must
#: never appear in the same column.
DECFM_CMNIST = {
    "DecFM-Plugin": {"sw2": (0.01137, 0.00078), "fid": (32.005, 1.075),
                     "latent_sw2": (0.08206, 0.00754), "colour_w2": (0.01991, 0.00285)},
    "DecFM-DR-I":   {"sw2": (0.01392, 0.00090), "fid": (33.650, 1.151),
                     "latent_sw2": (0.09217, 0.00657), "colour_w2": (0.03665, 0.01237)},
    "DecFM-DR-OT":  {"sw2": (0.01190, 0.00106), "fid": (33.824, 1.219),
                     "latent_sw2": (0.07794, 0.00384), "colour_w2": (0.02840, 0.00866)},
}
DECFM_CELEBA = {
    "DecFM-Plugin": {"sw2": (0.03272, 0.00183), "fid": (11.497, 0.156),
                     "latent_sw2": (0.06220, 0.00071)},
    "DecFM-DR-I":   {"sw2": (0.03356, 0.00318), "fid": (14.609, 0.525),
                     "latent_sw2": (0.06241, 0.00405)},
    "DecFM-DR-OT":  {"sw2": (0.03636, 0.00245), "fid": (15.521, 1.307),
                     "latent_sw2": (0.06746, 0.00257)},
}

#: Column label -> CSV column, per dataset. `None` means the dataset does not
#: report that metric (DecFM_2D has no images, dSprites is single-channel).
METRICS = {
    "ColorMNIST": [("SW2", "sw2_flow"), ("FID", "fid_legacy_flow"),
                   ("Latent SW2", "latent_sw2_flow"), ("Color-W2", "colour_w2_flow")],
    "CelebA":     [("SW2", "sw2_flow"), ("FID", "fid_legacy_flow"),
                   ("Latent SW2", "latent_sw2_flow")],
    "dSprites":   [("SW2", "sw2_flow_mean"), ("SW1", "sw1_flow_mean"),
                   ("Energy", "energy_flow_mean")],
    "DecFM_2D":   [("SW2", "sw2_flow"), ("SW1", "sw1_flow"), ("MMD2", "mmd2_flow")],
}


#: DGP parameters that must be constant across every row of a dataset's results.
#: `w_conf` is excluded for ColorMNIST because the confounding sweep varies it on
#: purpose; the recolouring parameters are never a sweep axis.
DGP_INVARIANTS = {"ColorMNIST": ["tau", "alpha", "k_mask"]}


def _check_dgp(frame, dataset):
    """Raise if a dataset's CSVs disagree on the design that generated them.

    Mixing two designs in one table is silent and fatal: the numbers average
    across incompatible ground truths. Results from a superseded design belong
    in a `results_archive_*` folder, not in `results/`.
    """
    columns = [c for c in DGP_INVARIANTS.get(dataset, []) if c in frame.columns]
    if not columns:
        return
    unique = frame[columns].drop_duplicates()
    if len(unique) > 1:
        raise ValueError(
            f"{dataset} results mix {len(unique)} DGP configurations:\n"
            f"{unique.to_string(index=False)}\n"
            "Archive the superseded runs before building tables.")


def load(dataset: str) -> pd.DataFrame:
    """Every projected-Nystrom result CSV for `dataset`, concatenated.

    Stages are separated downstream by the `init` / `dr_fit` / `n_obs` /
    `n_particles` COLUMNS, never by filename, so a renamed run cannot be
    silently misfiled.
    """
    pattern = os.path.join(RESULTS.format(dataset=dataset), "*.csv")
    files = [f for f in sorted(glob.glob(pattern)) if "_projnys" in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"no projected-Nystrom CSVs under {pattern}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    _check_dgp(frame, dataset)
    if "n_particles_requested" not in frame.columns:
        # Added by concatenation rather than by insertion or assign(): these frames
        # carry hundreds of columns, and adding one in place emits a pandas
        # fragmentation warning straight into the notebook output.
        frame = pd.concat(
            [frame, frame["n_particles"].rename("n_particles_requested")], axis=1)
    return frame


def default_rows(frame: pd.DataFrame, n_obs: int = 20000,
                 n_particles: int = 2000) -> pd.DataFrame:
    """The default configuration: no-split nuisance, no-split autoencoder,
    observed initialisation, projected Nystrom, at one sample and particle size.

    This is the configuration the headline table reports; everything else in the
    notebooks is an ablation away from it.
    """
    keep = ((frame.dr_fit == "nosplit") & (frame.init == "observed")
            & (frame.n_obs == n_obs) & (frame.n_particles_requested == n_particles))
    if "ae_split" in frame.columns:
        keep &= frame.ae_split == "none"
    if "tag" in frame.columns:                       # the w-sweep shares the grid
        keep &= ~frame.tag.astype(str).str.contains("wsweep")
    return frame[keep]


def summarise(frame: pd.DataFrame, column: str):
    """(mean, sd) of `column` in the house aggregation order."""
    per_run = frame.groupby(["data_seed", "repeat"])[column].mean()
    per_seed = per_run.groupby("data_seed").mean()
    return per_seed.mean(), per_seed.std()


def _cell(mean, sd, width: int, digits=None) -> str:
    """`mean +/- sd`, with the decimals chosen by magnitude so columns align.

    `digits` overrides that choice, which is what ratio columns want (a ratio
    straddling 1 would otherwise switch precision from row to row).
    """
    if mean is None:
        return "--".ljust(width)
    if digits is None:
        digits = 5 if abs(mean) < 1 else 3
    return f"{mean:.{digits}f} ± {sd:.{digits}f}".ljust(width)


def plain_table(rows: list, headers: list, title: str = "", note: str = "",
                label_width: int = 22, cell_width: int = 21,
                first_column: str = "Method", arrow: bool = True,
                digits=None) -> str:
    """
    A monospaced table in the layout the collaborators circulate.

    Parameters
    ----------
    rows : list of (label, [(mean, sd) or None, ...])
    headers : list of str
        Metric names; an arrow is appended since every metric here is
        lower-is-better.
    title, note : str
        Printed above and below the table.
    arrow : bool
        Append the lower-is-better arrow to each header. Switch it off for
        ratio columns, where the best value is 1 rather than 0.
    digits : int or list of int, optional
        Fixed number of decimals, either for every cell or one per column.
    """
    per_column = digits if isinstance(digits, (list, tuple)) else [digits] * len(headers)
    suffix = " ↓" if arrow else ""
    head = (first_column.ljust(label_width)
            + "".join((h + suffix).ljust(cell_width) for h in headers)).rstrip()
    lines = []
    if title:
        lines += [title, ""]
    lines += [head, "-" * len(head)]
    for label, values in rows:
        lines.append(str(label).ljust(label_width)
                     + "".join(_cell(*(v if v else (None, None)), cell_width, d)
                               for v, d in zip(values, per_column)).rstrip())
    if note:
        lines += ["", note]
    return "\n".join(lines)


def headline_table(frame: pd.DataFrame, dataset: str, reference: dict,
                   ours: str = "DR MMD Flow (ours)", n_obs: int = 20000) -> str:
    """The punch-line table: the reference method's rows, then ours."""
    metrics = METRICS[dataset]
    keys = ["sw2", "fid", "latent_sw2", "colour_w2"][:len(metrics)]
    rows = [(name, [stats.get(k) for k in keys]) for name, stats in reference.items()]
    default = default_rows(frame, n_obs=n_obs)
    rows.append((ours, [summarise(default, column) for _, column in metrics]))
    seeds = [int(s) for s in sorted(default.data_seed.unique())]
    return plain_table(
        rows, [label for label, _ in metrics],
        title=f"{dataset}   (n_obs = {n_obs}, M = 2000, seeds {seeds})",
        note="Ours: projected Nystrom, no-split nuisance, no-split autoencoder, "
             "P_Y initialisation.\nFID is the legacy TorchVision Inception-v3 scorer, "
             "matching the reference convention.")


def ablation_table(frame: pd.DataFrame, dataset: str, by: str, labels: dict,
                   title: str, note: str = "", label_width: int = 22,
                   first_column: str = "Setting", order=None, metrics=None,
                   arrow: bool = True, digits=None) -> str:
    """One row per value of `by`, same metric columns as the headline table.

    Parameters
    ----------
    order : list, optional
        The values of `by` in the order the rows should appear. The default is
        sorted order, which is what a numeric sweep wants; name the order
        explicitly when the default row should come first.
    metrics : list of (label, column), optional
        Replaces the dataset's default metric columns, e.g. with ratios to the
        count-matched floor.
    digits : int or list of int, optional
        Fixed number of decimals, either for every cell or one per column.
    """
    metrics = METRICS[dataset] if metrics is None else metrics
    values = sorted(frame[by].dropna().unique()) if order is None else order
    rows = []
    for value in values:
        subset = frame[frame[by] == value]
        if not len(subset):
            continue
        rows.append((labels.get(value, value),
                     [summarise(subset, column) for _, column in metrics]))
    return plain_table(rows, [label for label, _ in metrics], title=title, note=note,
                       label_width=label_width, first_column=first_column,
                       arrow=arrow, digits=digits)
