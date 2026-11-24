"""
Plain-text tables for the simulation analysis notebooks.

The notebooks exist to be read, so every table here is monospaced, aligned, and
formatted as `mean +/- sd` in the layout the collaborators already circulate. No
styled DataFrames, no multi-index, no colour.

Aggregation is the house order throughout: where a run scores two intervention
arms, the arms are averaged within the run first, then repeats within a data
seed, and the reported spread is the standard deviation ACROSS SEEDS.

The `estimator` column names which target-embedding estimator the flow was matched
to -- plug-in, IPW, or the doubly robust combination of the two. The three come out
of one nuisance solve on one logged sample and are handed identical initial
particles, so the axis is a label ablation like `init` and `dr_fit`, not a separate
design: the default table reports `dr` and `estimator_table` puts all three side by
side. Results written before the axis existed carry no such column and are read as
`dr`, which is what those runs sampled from.

That pin belongs at the SOURCE, not in each slice, and it is what `load` does by
DEFAULT: a notebook's main frame carries one estimator, so a three-estimator CSV
landing in `results/` cannot be averaged into an ablation table even if a slice forgets
the axis. The estimator section is the one place that wants every row, and it asks for
them explicitly with `load(dataset, estimator=None)`; `pin_estimator` is the pin on its
own, for a frame that did not come from `load`.

The last two sections of this module are the WALL CLOCK, which does not aggregate the
house way at all. A `seconds_*` column is paid once per DATA SEED, once per REPEAT, or by
one estimator row, and only the third may be summed down a frame's rows; `repeat_seconds`,
`seed_seconds`, `total_seconds` and `sampling_report` are where that distinction is
implemented, so that no notebook has to remember which column is which.

Two different costs are reported from those columns and neither replaces the other.
`sampling_table` is the per-estimator cost of DRAWING one sample, which charges each
estimator only the half of the nuisance solve it is spent on so that the three can be
compared. `training_table` is the compute-time table for the METHOD: the nuisance fit plus
the MMD flow, over all arms of a repeat, for the doubly robust embedding, averaged over the
data seeds -- with the representation, the per-data-seed setup, the decode and all scoring
printed beside it and excluded from it.
"""

import glob
import os
import textwrap
import warnings

import numpy as np
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

#: Column label -> CSV column, per dataset. A metric simply absent from a dataset's
#: list is one that dataset does not report: DecFM_2D has no images at all, and dSprites
#: has no FID because its outcome is single-channel, so Inception features are not
#: meaningful on it and would not be comparable with the other designs' FID. dSprites
#: DOES report latent SW2, in the same standardized-latent space CelebA uses.
#: Warfarin leads with its EXACT columns: the outcome there is scalar and the
#: interventional CDF is available in closed form, so `w1_exact_*` / `ks_exact_*`
#: measure distance to the law itself and carry no Monte Carlo error, while
#: `w1_flow` / `ks_flow` score against a finite oracle draw that a sampler can
#: beat by chance. Report the exact pair; keep the sample pair beside it.
METRICS = {
    "ColorMNIST": [("SW2", "sw2_flow"), ("FID", "fid_legacy_flow"),
                   ("Latent SW2", "latent_sw2_flow"), ("Color-W2", "colour_w2_flow")],
    "CelebA":     [("SW2", "sw2_flow"), ("FID", "fid_legacy_flow"),
                   ("Latent SW2", "latent_sw2_flow")],
    # dSprites leads with SW2 / FID / Latent SW2 in the SAME ORDER as ColorMNIST and
    # CelebA, so the three image designs line up column-for-column in a combined table;
    # SW1 and Energy follow as this design's extras. Its FID is on channel-replicated
    # greyscale (see the simulation's `to_rgb`) and is a WITHIN-DESIGN number only --
    # Inception never saw binary sprites, so it does not compare with the other two
    # designs' FID despite the shared column name. Read it against its own floor row.
    # dSprites' latent SW2 is recorded on the HEADLINE stage only (added 2026-10-05).
    # This list reaches exactly one dSprites table: `estimator_table` via
    # `estimator_report`, which is already pinned to the headline cell, so the column
    # prints a NUMBER there. The dSprites notebook's own ladders and headline table are
    # built from a LOCAL metrics list plus `plain_table` and do not read this one, so
    # adding an entry here does not put latent SW2 on them -- and it cannot simply be
    # appended to that local list either, because the latent rows use a different row
    # vocabulary (`floor` / `floor_matched` / `initial`, no `roundtrip` and no
    # noisy-versus-clean split). Those tables need their own cell.
    # The pixel columns carry the `_mean` suffix because this design averages each
    # metric over several fresh reference draws.
    "dSprites":   [("SW2", "sw2_flow_mean"), ("FID", "fid_legacy_flow"),
                   ("Latent SW2", "latent_sw2_flow_mean"),
                   ("SW1", "sw1_flow_mean"), ("Energy", "energy_flow_mean")],
    "DecFM_2D":   [("SW2", "sw2_flow"), ("SW1", "sw1_flow"), ("MMD2", "mmd2_flow")],
    "Warfarin":   [("W1 (exact)", "w1_exact_flow"), ("KS (exact)", "ks_exact_flow"),
                   ("W1", "w1_flow"), ("KS", "ks_flow")],
}


#: DGP parameters that must be constant across every row of a dataset's results.
#: A mismatch here is an accident -- two incompatible designs sharing one folder --
#: so `load` refuses to build a table at all. Parameters a launcher varies ON
#: PURPOSE do not belong here, however much they change the law: they live in
#: `SWEEP_AXES` and are pinned per table instead. That is why ColorMNIST's
#: `w_conf` and Warfarin's `theta` and `delta_scale` are absent.
DGP_INVARIANTS = {"ColorMNIST": ["tau", "alpha", "k_mask"],
                  "Warfarin": ["tolerance", "outcome_noise", "max_dose"]}

#: Design axes a launcher sweeps on purpose, with the value the DEFAULT table
#: reports, per dataset. Each distinct value is a DIFFERENT interventional law, so
#: pooling two of them averages incomparable ground truths -- for Warfarin, two
#: different closed-form reference CDFs. `default_rows` pins every one of these,
#: and `ablation_table` unpins the single axis it varies.
#:
#: These are NUMERIC design parameters: the pinning compares with `np.isclose`. The
#: `estimator` axis is a categorical label, so it is pinned by the explicit
#: `estimator=` argument of `default_rows` instead, beside `dr_fit` and `init` --
#: which is also the axis it resembles, since all three label a choice of method
#: made on one fixed design rather than a change to the design itself.
SWEEP_AXES = {"ColorMNIST": {"w_conf": 10.0},
              "Warfarin": {"theta": 0.5, "delta_scale": 1.0}}

#: Tag substrings that mark rows belonging to a dedicated sweep launcher rather
#: than the main grid. They share a results folder with the main sweep, so the
#: default table must exclude them even when their pinned axes happen to match.
#: A launcher that only adds estimators needs no entry here: its rows are separated
#: by the `estimator` column, and its `dr` rows are the same estimator on the same
#: design as the main grid's, so they pool into the default table as further repeats
#: rather than as a second law.
ABLATION_TAGS = ("wsweep", "osweep")

#: Target-embedding estimators, in the order the tables report them: the plug-in
#: conditional mean embedding, the importance-weighted outcomes, and the doubly
#: robust combination v_DR = v_PI + v_correction that is the default.
ESTIMATORS = ("plugin", "ipw", "dr")
ESTIMATOR_LABELS = {"plugin": "Plug-in (CME only)",
                    "ipw": "IPW (ratios only)",
                    "dr": "Doubly robust (default)"}

#: Columns that, where present, identify one pipeline run: the unit the estimators
#: are paired on. `sampler` is a DecFM_2D column, `target_a` an arm label, and
#: `target_policy` Warfarin's intervention axis -- each of its values is a different
#: closed-form interventional law, so a frame covering several of them must not be
#: paired or summarised across them. A design lacking a key pairs on the rest.
PAIRING_KEYS = ("data_seed", "repeat", "target_a", "sampler", "target_policy")

#: The two weight-vector diagnostics the estimator table reports, on ONE definition for
#: every dataset: the Kish effective sample size (sum v)^2 / sum v^2 of the SIGNED weight
#: vector, and the largest single atom max |v|. All five simulations write exactly these
#: two, under exactly these names, so the estimator table reads them as they stand.
#: Results predating the columns simply lack them and print as `--`.
V_ESS = "v_ess"
V_LARGEST = "v_max_abs"


def _paired_estimator_rows(frame: pd.DataFrame, estimators) -> pd.DataFrame:
    """The rows of `frame` whose run carries every one of `estimators`.

    A results folder holds whatever has been launched, and the launchers still run
    the doubly robust estimator alone, so the `dr` rows normally outnumber the
    others. Summarising each estimator over whatever rows it happens to have would
    then compare one estimator's seeds against another's -- the seed noise the axis
    was built to remove -- so the comparison is restricted to the runs that scored
    all of them.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with an `estimator` column already resolved.
    estimators : sequence of str
        The estimator labels the table will place side by side.

    Returns
    -------
    pd.DataFrame
        The rows of the fully covered runs, in `frame`'s order.
    """
    keys = [column for column in PAIRING_KEYS if column in frame.columns]
    if not keys or len(estimators) < 2:
        return frame
    # dropna=False so that a key left blank by some design still forms a group;
    # dropped groups would come back as a shorter mask and select columns instead.
    covered = frame.groupby(keys, dropna=False).estimator.transform(
        lambda names: set(estimators) <= set(names))
    return frame[covered.astype(bool)]


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


def _estimator_labels(frame: pd.DataFrame) -> pd.Series:
    """The `estimator` column of `frame`, with a missing column or value read as "dr".

    Every result written before the estimator axis existed sampled from the doubly
    robust embedding, because it was the only one the scripts could target. Those
    CSVs have no `estimator` column at all, and concatenating them with newer ones
    leaves NaN in their rows; both cases mean "dr".

    Parameters
    ----------
    frame : pd.DataFrame

    Returns
    -------
    pd.Series of str, indexed like `frame`.
    """
    if "estimator" not in frame.columns:
        return pd.Series("dr", index=frame.index, dtype=object)
    return frame.estimator.fillna("dr").astype(str)


def pin_estimator(frame: pd.DataFrame, estimator: str = "dr") -> pd.DataFrame:
    """The rows of `frame` that sampled from one target-embedding estimator.

    The pin `load` applies, exposed on its own for a notebook that keeps the
    unpinned frame for its estimator section and the pinned one for everything else.
    Every other table in the notebooks reports a single estimator, so it must be
    pinned BEFORE the ablation axes are sliced: `frame[frame.init == "observed"]` on
    an unpinned frame silently averages the three.

    Parameters
    ----------
    frame : pd.DataFrame
    estimator : str
        "dr", "plugin" or "ipw". Rows without an `estimator` column count as "dr".

    Returns
    -------
    pd.DataFrame
    """
    return frame[_estimator_labels(frame) == estimator]


#: Filename token each solver writes, which is how `load` selects files. The exact solve
#: is the reference the Nystrom approximations are measured against, so it is admissible
#: whenever a caller asks for it; the non-projected `nystrom` is NOT sound at a small ridge
#: and few landmarks (see `_check_solver`) and has to be named explicitly to be read.
SOLVER_TOKENS = {"projected_nystrom": "_projnys", "exact": "_exact", "nystrom": "_nys"}


def _check_solver(frame: pd.DataFrame, dataset: str,
                  allowed=("projected_nystrom",)) -> None:
    """Raise if a loaded row came from a solver the caller did not ask for.

    `load` selects files by the `_projnys` filename token and the other solvers write
    tokens of their own, so the name and the column agree unless a CSV was renamed or
    hand-copied -- and they must agree, because the non-projected `nystrom` solver is
    not sound at a small ridge and few landmarks. Its Woodbury operator
    S(r) = (r - Phi (Phi^T Phi + lam I)^{-1} Phi^T r) / (lam n) divides by lam n, and on
    a 4-d Gaussian-policy bandit at n = 800 with RBF Grams the total mass of the vector
    it returns -- its ORDINARY doubly robust return, with no component split anywhere in
    it -- came out at -2.71 at lambda = 1e-3, m / n = 0.125 and 10.81 at m / n = 0.062,
    where the exact solve gives 0.998 and projected Nystrom stays within 0.993 to 0.999
    throughout. A total mass of -2.71 is not the embedding of a probability measure. The
    hazard is that solver, for every estimator it is asked for, and not the plug-in /
    correction split: S is linear, so splitting the right-hand side reproduces the
    unsplit return to 1.5e-12. Reading such rows as the default solver's would be silent.
    """
    if "solver" not in frame.columns:
        return
    other = sorted(set(frame.solver.dropna().astype(str)) - set(allowed))
    if other:
        raise ValueError(
            f"{dataset} results/ holds rows from the solver(s) {', '.join(other)} where "
            f"only {', '.join(allowed)} was asked for -- the filename token and the "
            "`solver` column disagree, which means a CSV was renamed or hand-copied.\n"
            "Either move those runs out of results/, or name the solver explicitly: "
            "load(dataset, solvers=(\"projected_nystrom\", \"exact\")).")


def _warn_total_mass(frame: pd.DataFrame, dataset: str) -> None:
    """Warn about loaded rows whose target embedding carries non-positive total mass.

    chi_hat(pi) embeds a probability measure, so `sum_v` is 1 up to estimation error --
    0.982 to 1.021 across every result on disk. A non-positive value is far outside that,
    but it is not by itself a broken solve: v_PI is a SIGNED ridge solution and a
    ridge-dominated plug-in row can legitimately carry non-positive mass, while a `dr`
    row that does is the signature of the non-projected Nystrom solver (`_check_solver`).

    So this warns rather than refusing the folder, and it is called on the rows `load`
    returns rather than on every CSV in it. A run's three estimator rows share one CSV,
    so a refusal would make every table in the notebook unbuildable -- including a doubly
    robust table that never reads the plug-in row -- with no remedy available short of
    deleting the run. The message names the estimators involved, which is what says
    whether the value needs acting on.
    """
    if "sum_v" not in frame.columns:
        return
    flagged = frame[frame.sum_v.astype(float) <= 0]
    if not len(flagged):
        return
    names = ", ".join(sorted(set(_estimator_labels(flagged))))
    warnings.warn(
        f"{dataset}: {len(flagged)} of the {len(frame)} rows loaded carry non-positive "
        f"target-embedding mass (min sum_v = {flagged.sum_v.min():.4f}), on the "
        f"estimator(s) {names}. A signed plug-in solve can do this under a heavy ridge; "
        "a `dr` row that does is a broken nuisance solve. Do not quote such a row.",
        stacklevel=2)


def _check_ipw_form(frame: pd.DataFrame, dataset: str) -> None:
    """Raise if the loaded rows mix the two IPW normalisations.

    `ipw_form` records which v_IPW a run was matched to: the Horvitz-Thompson w / n,
    which is the term that appears inside v_DR and carries no unit mass, or the
    self-normalised Hajek w / sum(w), which sums to one by construction. Those are two
    DIFFERENT estimators, not two scalings of one, so pooling them puts a total mass
    neither produced in the row labelled "IPW (ratios only)" -- 0.958 from one run of
    each, against 0.916 and 1.000. There is no ablation section for this axis, so the
    rows cannot be told apart in any table here and the folder is refused instead.
    """
    if "ipw_form" not in frame.columns:
        return
    forms = sorted(set(frame.ipw_form.dropna().astype(str)))
    if len(forms) > 1:
        raise ValueError(
            f"{dataset} results/ mixes the IPW normalisations {', '.join(forms)}; "
            "Horvitz-Thompson (w / n) and Hajek (w / sum w) are different estimators "
            "and\nno table here separates them. Keep one form per results/ folder.")


def _normalise_seconds(frame: pd.DataFrame) -> None:
    """Turn a -0.0 in any `seconds_*` column of `frame` into 0.0, in place.

    Three of the five designs compute `seconds_other` as a residual of rounded parts
    without the trailing `+ 0.0` the other two carry, so a residual that vanishes reaches
    the CSV as the string "-0.0" -- arithmetically zero, but it prints as a negative cost
    in any table built from the column. Normalising on the way in fixes every table at
    once; the guard still belongs at the source, in dSprites, ColorMNIST and CelebA.
    """
    for column in [name for name in frame.columns if name.startswith("seconds_")]:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column] + 0.0


def load(dataset: str, estimator: str = "dr",
         solvers=("projected_nystrom",)) -> pd.DataFrame:
    """Every projected-Nystrom result CSV for `dataset`, concatenated, for ONE estimator.

    Stages are separated downstream by the `init` / `dr_fit` / `estimator` /
    `n_obs` / `n_particles` COLUMNS, never by filename, so a renamed run cannot be
    silently misfiled.

    Parameters
    ----------
    dataset : str
    estimator : str, optional
        Which target-embedding estimator's rows to keep. "dr" by default, because that
        is the embedding every table but the estimator section reports and the only one
        the runs written before the axis existed could have sampled from -- so the frame
        a notebook binds at the top is pinned unless it says otherwise. None keeps every
        row, which is what `estimator_table` and `estimator_report` need.

    Returns
    -------
    pd.DataFrame
    """
    wanted = (solvers,) if isinstance(solvers, str) else tuple(solvers)
    unknown = set(wanted) - set(SOLVER_TOKENS)
    if unknown:
        raise ValueError(f"unknown solver(s) {sorted(unknown)}; "
                         f"choose from {sorted(SOLVER_TOKENS)}")
    tokens = tuple(SOLVER_TOKENS[name] for name in wanted)
    pattern = os.path.join(RESULTS.format(dataset=dataset), "*.csv")
    files = [f for f in sorted(glob.glob(pattern))
             # "_projnys" also contains "_nys", so a token match has to be exact on the
             # solver, not a substring of a longer one.
             if any(os.path.basename(f).endswith(t + ".csv")
                    or t + "_" in os.path.basename(f) for t in tokens)
             and not (("_nys" in tokens) is False and
                      os.path.basename(f).endswith("_nys.csv"))]
    if not files:
        raise FileNotFoundError(
            f"no CSVs for solver(s) {list(wanted)} under {pattern}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    _normalise_seconds(frame)
    _check_dgp(frame, dataset)
    _check_solver(frame, dataset, allowed=wanted)
    _check_ipw_form(frame, dataset)
    if "n_particles_requested" not in frame.columns:
        # Added by concatenation rather than by insertion or assign(): these frames
        # carry hundreds of columns, and adding one in place emits a pandas
        # fragmentation warning straight into the notebook output.
        frame = pd.concat(
            [frame, frame["n_particles"].rename("n_particles_requested")], axis=1)
    if "estimator" not in frame.columns:
        frame = pd.concat(
            [frame, _estimator_labels(frame).rename("estimator")], axis=1)
    else:
        # Filled in place: the column already exists, so no block is added and no
        # fragmentation warning follows. A folder mixing pre-axis and post-axis CSVs
        # would otherwise leave the older rows at NaN, where `groupby` drops them.
        frame["estimator"] = _estimator_labels(frame)
    selected = frame if estimator is None else pin_estimator(frame, estimator)
    _warn_total_mass(selected, dataset)
    return selected


def _sole_target_policy(frame: pd.DataFrame):
    """The one target policy these rows cover, or a refusal to choose between them.

    Each target policy is a different intervention, so it has its own
    interventional law and its own closed-form reference CDF; pooling two of them
    averages incomparable ground truths. They are meant to share a results folder,
    unlike a DGP invariant, so the axis is pinned rather than forbidden.

    Returns
    -------
    str or None
        The single policy present, or None when no row is left to disambiguate.
    """
    available = sorted(frame.target_policy.dropna().unique())
    if len(available) > 1:
        raise ValueError(
            f"these rows cover {len(available)} target policies "
            f"({', '.join(map(str, available))}); name one with target_policy=.")
    return available[0] if available else None


def default_rows(frame: pd.DataFrame, n_obs: int = 20000,
                 n_particles: int = 2000,
                 target_policy: str = None,
                 dataset: str = None, estimator: str = "dr",
                 init: str = "observed", dr_fit: str = "nosplit",
                 **axes) -> pd.DataFrame:
    """The default configuration: no-split nuisance, no-split autoencoder,
    observed initialisation, projected Nystrom, doubly robust target embedding,
    at one sample and particle size.

    This is the configuration the headline table reports; everything else in the
    notebooks is an ablation away from it. Filters are applied only where the
    corresponding column exists, so a design without autoencoder splits or
    without a policy axis passes through untouched.

    Parameters
    ----------
    frame : pd.DataFrame
        As returned by `load`.
    n_obs, n_particles : int
        The logged sample size and the REQUESTED particle count to report. Either set
        to None is left unpinned, which is what the sample-size and particle-count
        ablations read.
    target_policy : str, optional
        Which intervention to report, for a design that sweeps several (Warfarin).
        The default resolves it when the selected rows carry exactly one and
        raises when they carry more, rather than averaging over interventions.
    dataset : str, optional
        Names the dataset whose `SWEEP_AXES` to pin. When omitted, every axis in
        `SWEEP_AXES` whose column is present in `frame` is pinned -- the axes are
        named distinctly enough across datasets that this is unambiguous.
    estimator : str, optional
        Which target-embedding estimator to report: "dr" (the default), "plugin" or
        "ipw". None leaves the axis unpinned, which is what `estimator_table` needs.
        Rows without an `estimator` column count as "dr".
    init, dr_fit : str, optional
        The initialisation and the nuisance fitting to report. Either set to None is
        left unpinned, which is what those two ablations read. Unpinning ONE axis and
        leaving the rest pinned is the whole point: nothing else may vary underneath
        the axis being read.
    **axes
        Overrides for individual swept axes, e.g. ``delta_scale=2.0``. An axis set
        to None is left unpinned, which is what `ablation_table` needs for the one
        axis it varies.
    """
    keep = pd.Series(True, index=frame.index)
    for column, value in [("dr_fit", dr_fit), ("init", init), ("n_obs", n_obs),
                          ("n_particles_requested", n_particles)]:
        if value is not None:
            keep &= frame[column] == value
    if "ae_split" in frame.columns:
        keep &= frame.ae_split == "none"
    if estimator is not None:
        keep &= _estimator_labels(frame) == estimator
    pinned = dict(SWEEP_AXES.get(dataset, {})) if dataset else {
        column: value
        for spec in SWEEP_AXES.values() for column, value in spec.items()
        if column in frame.columns}
    pinned.update(axes)
    # The dedicated sweep launchers write into the same results folder, so their
    # rows are excluded from the default table -- UNLESS the caller unpinned an
    # axis, which is how an ablation asks for exactly those rows.
    if "tag" in frame.columns and not any(v is None for v in axes.values()):
        tags = frame.tag.astype(str)
        for marker in ABLATION_TAGS:
            keep &= ~tags.str.contains(marker)
    for column, value in pinned.items():
        if value is None or column not in frame.columns:
            continue
        keep &= np.isclose(frame[column].astype(float), float(value))
    if "target_policy" in frame.columns:
        policy = (_sole_target_policy(frame[keep]) if target_policy is None
                  else target_policy)
        keep &= frame.target_policy == policy
    return frame[keep]


def summarise(frame: pd.DataFrame, column: str):
    """(mean, sd) of `column` in the house aggregation order."""
    per_run = frame.groupby(["data_seed", "repeat"])[column].mean()
    per_seed = per_run.groupby("data_seed").mean()
    return per_seed.mean(), per_seed.std()


def _with_blank_columns(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """`frame` with any of `columns` it lacks added as all-NaN.

    A metric added to a design partway through its run exists in the CSVs written
    after the change and in none of the ones written before it. `summarise` raises a
    `KeyError` on a column no CSV carries, which turns "this table has no number yet"
    into a dead notebook; an all-NaN column instead reaches `_cell` and prints `--`,
    which is what a missing number should look like. Columns that are present in only
    SOME rows need nothing: `pd.concat` in `load` already pads them with NaN.

    This does NOT make a partially filled column safe to quote. `summarise` skips NaN,
    so a half-finished rerun prints the mean of whichever seeds have landed with a `nan`
    spread beside it -- read `n_floor_matched` or the row count before trusting a cell
    whose sd is `nan`.

    Parameters
    ----------
    frame : pd.DataFrame
    columns : iterable of str

    Returns
    -------
    pd.DataFrame
        `frame` itself when nothing is missing, so the common path copies nothing.
    """
    blanks = [pd.Series(np.nan, index=frame.index, dtype=float, name=name)
              for name in columns if name not in frame.columns]
    # Added by concatenation rather than insertion: these frames carry well over a
    # hundred columns, and adding one in place emits a pandas fragmentation warning
    # straight into the notebook output.
    return pd.concat([frame, *blanks], axis=1) if blanks else frame


def _cell(mean, sd, width: int, digits=None) -> str:
    """`mean +/- sd`, with the decimals chosen by magnitude so columns align.

    `digits` overrides that choice, which is what ratio columns want (a ratio
    straddling 1 would otherwise switch precision from row to row). A missing value --
    None, or a diagnostic this dataset does not record -- prints as `--`.
    """
    if mean is None or pd.isna(mean):
        return "--".ljust(width)
    if digits is None:
        digits = 5 if abs(mean) < 1 else 3
    # `+ 0.0` is not a no-op: it turns a -0.0 -- which a residual rounded from a tiny
    # negative float carries all the way out of the CSV -- into 0.0, so a cost that
    # vanished prints as 0.0 and not as -0.0.
    return f"{mean + 0.0:.{digits}f} ± {sd:.{digits}f}".ljust(width)


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
                   ours: str = "IMMD-Flow (ours)", n_obs: int = 20000) -> str:
    """The punch-line table: the reference method's rows, then ours."""
    metrics = METRICS[dataset]
    # POSITIONAL alignment: slot i of METRICS[dataset] is paired with key i of this list.
    # Valid only for the datasets whose METRICS entries are in exactly this order, which
    # is ColorMNIST and CelebA -- the two that have a reference dict. dSprites now leads
    # with the same three (SW2 / FID / Latent SW2) but has five entries, so the slice
    # would run off the end of this list and silently drop its last two columns; it has
    # no reference dict and never reaches here. Giving any dataset one means looking
    # these up by LABEL rather than by position.
    keys = ["sw2", "fid", "latent_sw2", "colour_w2"][:len(metrics)]
    rows = [(name, [stats.get(k) for k in keys]) for name, stats in reference.items()]
    default = default_rows(frame, n_obs=n_obs)
    rows.append((ours, [summarise(default, column) for _, column in metrics]))
    seeds = [int(s) for s in sorted(default.data_seed.unique())]
    return plain_table(
        rows, [label for label, _ in metrics],
        title=f"{dataset}   (n_obs = {n_obs}, M = 2000, seeds {seeds})",
        note="Ours: doubly robust target embedding, projected Nystrom, no-split "
             "nuisance, no-split autoencoder, P_Y initialisation.\nFID is the legacy "
             "TorchVision Inception-v3 scorer, matching the reference convention.")


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
    # A metric this design only started recording partway through its run prints `--`
    # rather than killing the table; see `_with_blank_columns`. Only a column the dataset
    # DECLARES is blanked: a typo or a stale name in a caller-supplied `metrics=` list must
    # still raise out of `summarise`, because a wrong column that renders as a missing
    # number is a table someone can quote.
    declared = {column for _, column in METRICS[dataset]}
    frame = _with_blank_columns(frame, [column for _, column in metrics
                                        if column in declared])
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


def estimator_table(frame: pd.DataFrame, dataset: str, title: str = None,
                    note: str = None, label_width: int = 26,
                    first_column: str = "Target embedding", metrics=None,
                    arrow: bool = True, digits=None) -> str:
    """The three target-embedding estimators side by side, one row each.

    A special case of `ablation_table` over the `estimator` column, and the table the
    estimator axis exists for: the plug-in estimator uses the conditional mean
    embedding alone, IPW the importance ratios alone, and the doubly robust estimator
    is their sum v_DR = v_PI + v_correction. Within a run all three come from one
    nuisance solve on one logged sample and start from the same particles, so the
    differences here are the estimators' own and not seed noise.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows covering the estimators to compare, typically
        ``default_rows(runs, dataset=..., estimator=None)``. Rows with no
        `estimator` column are read as "dr", so a pre-axis frame gives one row.
        Runs that scored only some of the estimators present are left out, so every
        row of the table is built from the same set of runs; the note says how many
        rows that dropped.
    dataset : str
        Names the dataset whose `METRICS` become the columns.
    title, note : str, optional
        Defaults name the dataset and state the pairing.
    metrics : list of (label, column), optional
        Replaces the dataset's metric columns, e.g. with the weight-concentration
        diagnostics each run records per row -- total mass, effective sample size and
        largest atom, on the one definition every design writes. Those differ sharply
        between the three estimators, which is why they are per row and not shared.
    label_width, first_column, arrow, digits
        Passed to `plain_table`.

    Returns
    -------
    str
    """
    rows = frame.assign(estimator=_estimator_labels(frame))
    present = [name for name in ESTIMATORS if name in set(rows.estimator)]
    paired = _paired_estimator_rows(rows, present)
    if title is None:
        title = f"{dataset}   target embedding estimator"
    if note is None:
        note = ("One logged sample, one nuisance solve, one set of initial particles "
                "and one flow budget per run;\nonly the target weight vector differs.")
        if len(paired) < len(rows):
            note += (f"\n{len(rows) - len(paired)} of {len(rows)} rows come from runs "
                     "that scored only some of the estimators and are left out.")
    return ablation_table(paired, dataset, "estimator", ESTIMATOR_LABELS,
                          title=title, note=note, label_width=label_width,
                          first_column=first_column, order=list(ESTIMATORS),
                          metrics=metrics, arrow=arrow, digits=digits)


def weight_diagnostics(frame: pd.DataFrame):
    """`frame` with both weight columns present, and the metrics list that reads them.

    The three estimators put very different profiles on the same logged outcomes -- IPW
    lives on the intervened arm alone and inherits the importance-ratio tails undamped,
    the plug-in vector is a signed ridge solution spread over every row, and DR is their
    sum -- so these columns are recorded per row and are most of what the estimator table
    is for. A results folder that predates one of them gets an all-NaN column, which
    prints as `--`, instead of a `KeyError` out of `summarise`.

    Parameters
    ----------
    frame : pd.DataFrame

    Returns
    -------
    (pd.DataFrame, list of (str, str))
        The rows with `V_ESS` and `V_LARGEST` present, and the `metrics` argument
        `estimator_table` needs to print total mass, ESS and largest atom.
    """
    return (_with_blank_columns(frame, (V_ESS, V_LARGEST)),
            [("sum v", "sum_v"), ("ESS(v)", V_ESS), ("max |v|", V_LARGEST)])


def estimator_report(frame: pd.DataFrame, dataset: str, title: str = None,
                     note: str = None, digits=None, label_width: int = 26) -> str:
    """The estimator comparison for `dataset`, or one line saying the axis was not run.

    Two tables: the dataset's metric columns, then the weight vector each estimator
    handed the flow. Both are restricted by `estimator_table` to the runs that scored
    every estimator present, since the axis is only interpretable paired.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows at ONE configuration with the estimator axis left UNPINNED, i.e.
        ``default_rows(runs, dataset=..., estimator=None)``.
    dataset : str
    title, note : str, optional
        Passed to `estimator_table` for the metric table.
    digits : int or list of int, optional
        Decimals for the metric table.
    label_width : int

    Returns
    -------
    str
        The two tables, or one line naming the only estimator on disk. Runs written
        before the axis existed carry no `estimator` column and count as "dr", so a
        dataset whose estimator stage has not been launched says so rather than
        printing a one-row table that looks like a comparison.
    """
    present = [name for name in ESTIMATORS if name in set(_estimator_labels(frame))]
    if len(present) < 2:
        only = ESTIMATOR_LABELS[present[0]].lower() if present else "nothing at all"
        return (f"Estimator comparison NOT RUN for {dataset}: the only target embedding "
                f"on disk is\n{only}, so there is nothing to compare it against. "
                "Launching the estimator\nstage fills this section in with no change "
                "here -- the rows are separated by the\n`estimator` column, not by "
                "filename.")
    extended, metrics = weight_diagnostics(frame)
    return "\n\n\n".join([
        estimator_table(frame, dataset, title=title, note=note,
                        label_width=label_width, digits=digits),
        estimator_table(
            extended, dataset, metrics=metrics, arrow=False, digits=[5, 0, 6],
            label_width=label_width,
            title=f"{dataset}   the weight vector behind each row",
            note="sum v is the total mass of the target embedding, 1 for the embedding "
                 "of a probability measure;\nESS(v) the Kish ratio (sum v)^2 / sum v^2 "
                 "of the SIGNED vector, which cancellation between\nopposite signs "
                 "lowers; max |v| the largest single atom. None of the three is an "
                 "objective.")])


# ---------------------------------------------------------------------------------------
# Wall clock
#
# THREE tiers, not two. A repeat writes one row PER ESTIMATOR -- per arm and per sampler
# where a design has those axes -- and a phase is paid either once for a whole DATA SEED,
# once for a REPEAT, or by one row alone. The three add up differently:
#
#   per data seed  Paid OUTSIDE `seconds_repeat` and amortised over that seed's repeats:
#                  `SEED_SECONDS`. CelebA's frozen SD-VAE encode (75.5 s, the largest
#                  single phase of a run) and its Inception reference features are the
#                  measured instances. Summing such a column down the rows reports it once
#                  per row -- six times over on a two-arm three-estimator repeat, twice
#                  that again if the seed has two repeats -- while leaving it out of a
#                  total under-reports the run by all of it. Both mistakes were live.
#   per repeat     Inside `seconds_repeat`, identical on every row of the repeat:
#                  `SHARED_SECONDS`, which the simulations also sum into `seconds_shared`.
#   per estimator  This row's own work: `OWN_SECONDS`, the only tier free to be summed
#                  across the rows of a repeat.
#
# A BREAKDOWN is not a fourth tier: it is already inside one of the three, so adding it
# double counts. `seconds_ratio` and `seconds_cme` are breakdowns of `seconds_nuisance`.
# `seconds_fid` is a breakdown of `seconds_eval` in one design and a per-estimator column
# beside it in another, which is why `own="auto"` resolves it from the frame rather than
# asking every caller to remember which design it is holding.
#
# `seconds_data` is the per-data-seed data preparation -- the cohort load, the image cache,
# the logged and oracle draws -- which sat outside every column in all five designs until
# 2026-10-03 and which only Warfarin admitted to. Where a frame carries it, a total here is
# complete but for process start-up; where it does not, the total is a lower bound and
# `_uncharged_note` says which of the two the reader is holding.
#
# `repeat_seconds` implements the distinction, `seed_seconds` is the tier outside it,
# `total_seconds` is what a notebook calls for "how long did this cost", and
# `sampling_report` is the table. None of them raises on a frame written before the
# contract: the decomposition is skipped and reported missing, while a frame that CLAIMS
# the contract and then does not add up still raises.
# ---------------------------------------------------------------------------------------

#: Timing columns SHARED BY EVERY ROW OF A REPEAT, which the simulations also sum into the
#: convenience column `seconds_shared`. They record work done ONCE for every estimator of
#: the repeat -- the autoencoder training, the single CME / ratio solve, the reference and
#: floor scoring, the shared FID features -- so summing one of them down a repeat's rows
#: multiplies it by the number of rows. A design omits the columns whose phase it lacks.
SHARED_SECONDS = ("seconds_ae", "seconds_nuisance", "seconds_eval_shared",
                  "seconds_fid_shared")

#: Timing columns shared by every row of every repeat of ONE DATA SEED, and measured
#: OUTSIDE `seconds_repeat`: the data preparation, CelebA's frozen SD-VAE encode and its
#: Inception reference features are all pipeline-independent, so they are hoisted out of
#: the repeat loop and paid once per data seed. They are added to a total once per data
#: seed, not once per repeat, and they are the tier the two-tier contract had nowhere to
#: put -- which is how a measured 75.5 s encode came to be both omitted from a CelebA
#: budget and, by the other reading, counted once for every row of every repeat of it.
SEED_SECONDS = ("seconds_data", "seconds_encode", "seconds_fid_ref")

#: The per-data-seed TOTAL, standing to `SEED_SECONDS` exactly as `seconds_shared` stands
#: to `SHARED_SECONDS`: one column to add once per data seed, checked against its parts.
SEED_TOTAL = "seconds_setup"

#: Timing columns that are THIS ROW'S OWN work, and the only ones free to be summed across
#: the rows of a repeat.
OWN_SECONDS = ("seconds_flow", "seconds_decode", "seconds_eval")

#: The one column whose SCOPE is not the same in every CSV that carries it. Both designs
#: that write it now time their FID INSIDE `seconds_eval`, as a breakdown of it: ColorMNIST
#: always did, and CelebA adopted the same convention on 2026-10-03, so a current frame of
#: either reads as "breakdown". CelebA's ARCHIVED CSVs still carry it BESIDE `seconds_eval`,
#: as a fourth per-estimator column, and so does the synthetic CelebA frame in
#: `tests/test_timing_columns.py`. Both readings are therefore on disk, and `fid_scope`
#: decides per frame -- by which reading closes the repeat identity -- instead of leaving a
#: caller to pass the right `own=`.
FID_SECONDS = "seconds_fid"

#: The two halves of the one nuisance solve, as a BREAKDOWN of `seconds_nuisance`, and
#: which half each estimator is actually spent on: the plug-in vector needs the conditional
#: mean embedding alone, IPW the importance ratios alone, the doubly robust vector both.
#: One solve returns every requested weight vector, so the halves are shared per repeat
#: exactly like `seconds_nuisance`; charging an estimator its own halves is an ATTRIBUTION
#: -- what drawing that one sample would have cost on its own -- and never a decomposition
#: of `seconds_repeat`, which is why `sampling_seconds` output must not be summed into it.
NUISANCE_SECONDS = ("seconds_ratio", "seconds_cme")
ESTIMATOR_NUISANCE = {"plugin": ("seconds_cme",),
                      "ipw": ("seconds_ratio",),
                      "dr": ("seconds_ratio", "seconds_cme")}

#: The isolated sampling cost, which the simulations write PER ROW as that row's charged
#: nuisance halves plus its own flow and decode. It belongs to neither identity, so
#: `repeat_seconds` never adds it; `sampling_seconds` aggregates it to the repeat the only
#: way that does not double count -- halves once, flow and decode summed -- and checks the
#: recorded column against its own definition row by row.
SAMPLING_SECONDS = "seconds_sampling"

#: What a design pays to put the outcome in the space the flow runs in, with the column's
#: table label and the TIER it is paid on. dSprites and ColorMNIST train their own
#: autoencoder once per repeat; CelebA encodes with a frozen SD-VAE once per data seed.
#: The tier is printed in the column header, because a column whose scope is invisible is
#: how a 75.5 s phase came to be both omitted from a budget and counted once per row.
REPRESENTATION_SECONDS = {"seconds_ae": ("AE", "repeat"),
                          "seconds_encode": ("Encode", "data seed")}

#: The per-repeat cost of SCORING a row, which the sampling table prints beside the
#: sampling cost so that what was excluded from it is visible rather than implied.
#: `seconds_fid` joins these only where it is a column of its own (see `fid_scope`), and
#: `seconds_fid_ref` never does: it is a per-data-seed phase and is reported as one.
EVALUATION_SECONDS = ("seconds_eval", "seconds_eval_shared", "seconds_fid_shared")

#: The columns identifying one repeat: the unit `seconds_repeat` measures. The arm and
#: sampler axes are deliberately NOT here -- every design runs its arms and samplers INSIDE
#: one repeat clock and accumulates their shared costs into that repeat's shared columns, so
#: grouping by arm would count a repeat's shared work once per arm. The frame must already be
#: pinned to ONE configuration, as `default_rows` leaves it: two launcher tasks differing
#: only in `n_particles` reuse the same (data_seed, repeat) labels, and pooling them makes a
#: group's shared columns disagree -- which raises instead of being averaged away.
REPEAT_KEYS = ("data_seed", "repeat")

#: The one cost no design measures, named wherever a total is printed. It is per data seed
#: and it is large: a cold 16 GB CelebA cache is minutes to tens of minutes, and Warfarin's
#: cohort load plus its three draws sit outside every repeat clock too. A design that
#: records `seconds_data` has charged it; one that does not has not, and says so here.
SECONDS_UNCHARGED = (
    "Per-data-seed DATA PREPARATION -- the cohort load or image cache, and the logged and "
    "oracle draws -- is timed by no column in these runs, so every total here is a LOWER "
    "BOUND on the task's own wall clock.")

#: What is still outside the per-task identity once `seconds_data` is recorded, named
#: rather than left for a reader to find in a gap between two numbers.
SECONDS_UNCHARGED_WITH_DATA = (
    "Data preparation IS timed here, as seconds_data inside seconds_setup, once per data "
    "seed. What remains outside every column is process start-up -- the imports and, on "
    "CUDA, the context -- and the CSV rewrites between repeats.")


def _uncharged_note(frame: pd.DataFrame) -> str:
    """Which of the two uncharged-cost sentences applies to `frame`."""
    return (SECONDS_UNCHARGED_WITH_DATA if "seconds_data" in frame.columns
            else SECONDS_UNCHARGED)


def _wrapped(*paragraphs) -> str:
    """The paragraphs wrapped to the width the notebooks' other tables are written to.

    Notes here are assembled from measured numbers, so they cannot carry hand-placed line
    breaks the way a fixed caption can.
    """
    return "\n".join(textwrap.fill(paragraph, width=100)
                     for paragraph in paragraphs if paragraph)


def _repeat_groups(frame: pd.DataFrame, keys) -> "pd.core.groupby.DataFrameGroupBy":
    """`frame` grouped by the `keys` it carries, as a single group when it carries none."""
    present = [frame[key] for key in keys if key in frame.columns]
    if not present:
        present = [pd.Series(0, index=frame.index, name="run")]
    # dropna=False so a key left blank by some design still forms a group rather than
    # dropping its rows out of the total.
    return frame.groupby(present, dropna=False)


def _estimator_groups(frame: pd.DataFrame, keys) -> "pd.core.groupby.DataFrameGroupBy":
    """`frame` grouped by repeat AND estimator: the unit a sampling cost is paid for."""
    present = [frame[key] for key in keys if key in frame.columns]
    present.append(_estimator_labels(frame).rename("estimator"))
    return frame.groupby(present, dropna=False)


def _shared_value(groups, column: str, scope: str) -> pd.Series:
    """The one value `column` takes in each group, raising if a group carries several.

    The regression this exists to catch: before 2026-10-03 `seconds_repeat` was the
    elapsed time AT each estimator row, so it grew down the rows of a repeat and was
    neither a per-row cost nor a constant total. A column in this scope that is not
    constant inside its group cannot be de-duplicated at all, so it is refused rather
    than averaged or taken from an arbitrary row.

    Parameters
    ----------
    groups : pd.core.groupby.DataFrameGroupBy
    column : str
    scope : str
        How the column is described in the error message.

    Returns
    -------
    pd.Series
        The value per group, indexed by the group keys.
    """
    spread = groups[column].nunique(dropna=False)
    if (spread > 1).any():
        offenders = groups[column].agg(["min", "max", "count"])[spread > 1]
        raise ValueError(
            f"{column} is {scope}, so it must be identical on every row of a group, but "
            f"{int((spread > 1).sum())} group(s) carry several values:\n"
            f"{offenders.to_string()}\n"
            "A cost that varies down the rows of a repeat cannot be de-duplicated. Either "
            "the simulation wrote a per-row value into a shared column, or the frame pools "
            "two configurations that reuse the same (data_seed, repeat) labels -- pin it "
            "with `default_rows` first.")
    return groups[column].first()


def _present_sum(table: pd.DataFrame, columns) -> pd.Series:
    """The row-wise sum of whichever `columns` `table` carries, NaN where one is missing.

    A design omits a phase it does not have rather than writing 0.0, so a sum over a fixed
    list has to skip the absent columns -- but a sum over NONE of them, or over a blank
    value, is UNKNOWN rather than zero. `min_count` is what keeps an unmeasured cost
    printing as `--` instead of as free.
    """
    present = [column for column in columns if column in table.columns]
    if not present:
        return pd.Series(np.nan, index=table.index)
    return table[present].sum(axis=1, min_count=len(present))


def timing_reason(frame: pd.DataFrame):
    """Why `frame`'s phases cannot be added up, or None when they can.

    The three tiers are read from the columns themselves, so the one thing that cannot be
    inferred is the RESIDUAL: a run written before the wall-clock contract measured its
    phases but never measured what was left over, so its parts fall short of
    `seconds_repeat` by an amount nobody recorded. Such a frame is reported as
    undecomposable rather than refused -- every archived CSV in the repository is one --
    while a frame that carries `seconds_shared` and has lost `seconds_other` is a
    half-written contract and still raises in `repeat_seconds`.

    Parameters
    ----------
    frame : pd.DataFrame

    Returns
    -------
    str or None
        A sentence naming what is missing, or None when the identity is checkable.
    """
    present = sorted(column for column in frame.columns if column.startswith("seconds_"))
    if not present:
        return "these runs recorded no timings at all: no `seconds_*` column is present"
    if "seconds_other" not in present and "seconds_shared" not in present:
        return ("these runs predate the wall-clock contract -- they carry "
                f"{', '.join(present)} and neither `seconds_shared` nor `seconds_other`, "
                "so the residual that closes the identity was never measured and the "
                "phases cannot be added up")
    return None


def fid_scope(frame: pd.DataFrame, keys=REPEAT_KEYS, tol: float = 0.05):
    """Whether `seconds_fid` is a per-estimator column of its own or a breakdown.

    Decided by which reading closes the repeat identity, which is the only evidence that
    distinguishes them: CelebA times FID BESIDE `seconds_eval`, so the identity needs it
    added; ColorMNIST times it INSIDE `seconds_eval`, so adding it over-counts by exactly
    the FID time. The smaller reading is tried first, so a design whose FID is a breakdown
    is never charged for it twice.

    Parameters
    ----------
    frame : pd.DataFrame
    keys : sequence of str
    tol : float

    Returns
    -------
    str or None
        "breakdown", "own", or None when the frame has no `seconds_fid`, carries no
        checkable identity, or closes under neither reading.
    """
    if FID_SECONDS not in frame.columns or timing_reason(frame) is not None:
        return None
    for reading, own in (("breakdown", OWN_SECONDS),
                         ("own", OWN_SECONDS + (FID_SECONDS,))):
        try:
            _repeat_table(frame, keys, own, tol)
        except ValueError:
            continue
        return reading
    return None


def _shared_table(frame: pd.DataFrame, groups, tol: float) -> pd.DataFrame:
    """The per-repeat shared columns, de-duplicated, with `seconds_shared` checked.

    Shared by `repeat_seconds`, which goes on to the identity, and by `_check_scopes`,
    which stops here: a shared cost is one number for its repeat whichever of the repeat's
    rows are in hand, while the identity's SUM needs all of them.
    """
    table = pd.DataFrame(index=groups.size().index)
    shared = [column for column in SHARED_SECONDS if column in frame.columns]
    for column in shared:
        table[column] = _shared_value(groups, column, "shared per repeat")

    parts = table[shared].sum(axis=1)
    if "seconds_shared" in frame.columns:
        table["seconds_shared"] = _shared_value(groups, "seconds_shared",
                                                "shared per repeat")
        gap = (table.seconds_shared - parts).abs()
        if (gap > tol).any():
            raise ValueError(
                "seconds_shared is the sum of "
                f"{', '.join(shared) if shared else 'the shared columns'}, but it differs "
                f"from that sum by up to {gap.max():.2f} s on "
                f"{int((gap > tol).sum())} repeat(s). One shared phase is being written "
                "into the convenience column and not into a column of its own, or the "
                "other way round.")
    else:
        table["seconds_shared"] = parts
    _check_nuisance_split(frame, groups, tol)
    return table


def _check_nuisance_split(frame: pd.DataFrame, groups, tol: float) -> None:
    """Raise if the recorded nuisance halves do not add up to `seconds_nuisance`.

    `seconds_ratio` and `seconds_cme` are a BREAKDOWN of the one solve, so the two must sum
    to it on the rounded values the CSV carries. If they drift apart, one of the three is
    measuring something the other two do not, and the per-estimator attribution in
    `sampling_seconds` stops being a split of a cost that was really paid.
    """
    columns = NUISANCE_SECONDS + ("seconds_nuisance",)
    if not all(column in frame.columns for column in columns):
        return
    values = {column: _shared_value(groups, column, "shared per repeat")
              for column in columns}
    gap = (values["seconds_nuisance"]
           - values["seconds_ratio"] - values["seconds_cme"]).abs()
    if (gap > tol).any():
        raise ValueError(
            "seconds_nuisance is seconds_ratio + seconds_cme, but it differs from that sum "
            f"by up to {gap.max():.2f} s on {int((gap > tol).sum())} repeat(s). The two "
            "halves are a breakdown of the one solve, so a phase is being timed into one "
            "and not the other, or into neither.")


def _check_scopes(frame: pd.DataFrame, keys=REPEAT_KEYS, tol: float = 0.05) -> None:
    """Verify every scope rule that holds on a SLICE of a repeat, and no more.

    What a cost table can check before printing: that each shared column is one value per
    repeat, that `seconds_shared` is their sum, that each per-data-seed column is one value
    per seed and `SEED_TOTAL` their sum. What it cannot: the identity, whose per-estimator
    sum needs every row of the repeat and is therefore left to `repeat_seconds`.
    """
    groups = _repeat_groups(frame, keys)
    _shared_table(frame, groups, tol)
    for column, scope in (("seconds_other", "shared per repeat"),
                          ("seconds_repeat", "the total of the whole repeat")):
        if column in frame.columns:
            _shared_value(groups, column, scope)
    seed_seconds(frame, tol)


def _repeat_table(frame: pd.DataFrame, keys, own, tol: float) -> pd.DataFrame:
    """`repeat_seconds` for one explicit set of per-estimator columns."""
    reason = timing_reason(frame)
    groups = _repeat_groups(frame, keys)
    table = _shared_table(frame, groups, tol)

    own_columns = [column for column in own if column in frame.columns]
    table["seconds_own"] = (sum(groups[column].sum() for column in own_columns)
                            if own_columns else 0.0)
    if reason is not None:
        # Pre-contract: the phases above were measured, the residual never was, so the
        # total is left unknown rather than reported short. `seconds_repeat` still carries
        # what the run actually took.
        table["seconds_other"] = np.nan
        table["seconds_total"] = np.nan
    else:
        table["seconds_other"] = (
            _shared_value(groups, "seconds_other", "shared per repeat")
            if "seconds_other" in frame.columns else 0.0)
        table["seconds_total"] = (table.seconds_shared + table.seconds_own
                                  + table.seconds_other)

    if "seconds_repeat" in frame.columns:
        table["seconds_repeat"] = _shared_value(groups, "seconds_repeat",
                                                "the total of the whole repeat")
        gap = (table.seconds_repeat - table.seconds_total).abs()
        if (gap > tol).any():
            raise ValueError(
                "the timing columns do not add up to seconds_repeat on "
                f"{int((gap > tol).sum())} repeat(s), by up to {gap.max():.2f} s with "
                f"own={tuple(own)}. Either a phase is missing from `own` -- `seconds_fid` "
                "is the one whose scope differs between designs, and `own=\"auto\"` "
                "resolves it -- or time is being spent outside every named phase without "
                "landing in seconds_other.")
    return table


def repeat_seconds(frame: pd.DataFrame, keys=REPEAT_KEYS, own="auto",
                   tol: float = 0.05) -> pd.DataFrame:
    """One row per repeat: its shared cost once, its per-estimator cost summed, the total.

    The per-data-seed tier is deliberately absent: it is paid outside `seconds_repeat`, so
    `seed_seconds` reports it and `total_seconds` adds it once per seed.

    Parameters
    ----------
    frame : pd.DataFrame
        Result rows at ONE configuration, as `default_rows` leaves them.
    keys : sequence of str
        The columns identifying a repeat; `REPEAT_KEYS` by default. Keys the frame does
        not carry are skipped, and a frame carrying none is treated as one repeat.
    own : sequence of str or "auto"
        The per-estimator timing columns to sum. "auto", the default, is `OWN_SECONDS`
        plus `seconds_fid` where `fid_scope` finds it to be a column of its own.
    tol : float
        Allowed slack, in seconds, on both `seconds_shared` against its parts and
        `seconds_total` against `seconds_repeat`. The simulations round every part to
        0.1 s and define `seconds_other` as the residual OF THE ROUNDED PARTS, so both
        identities close exactly; the tolerance only absorbs float representation.

    Returns
    -------
    pd.DataFrame
        Indexed by `keys`, with every shared column present taken once, `seconds_shared`,
        `seconds_own` (the `own` columns summed over the repeat's rows), `seconds_other`,
        `seconds_total` = shared + other + own, and, where the frame carries it,
        `seconds_repeat` as the simulation measured it. On a frame `timing_reason` calls
        pre-contract, `seconds_other` and `seconds_total` are NaN: those repeats cost real
        time whose split was never recorded, and a number that looked right would be worse
        than a blank.

    Raises
    ------
    ValueError
        If a shared column, `seconds_other` or `seconds_repeat` varies inside a repeat, if
        `seconds_shared` is not the sum of the shared columns, or if the parts of a
        contract frame do not add up to `seconds_repeat`. Each of those means a reader
        adding the columns up would get the wrong number, so none is tolerated silently.
    """
    if isinstance(own, str):
        if own != "auto":
            raise ValueError(f"own must be a sequence of column names or \"auto\", "
                             f"not {own!r}")
        own = OWN_SECONDS + ((FID_SECONDS,) if fid_scope(frame, keys, tol) == "own" else ())
    return _repeat_table(frame, keys, own, tol)


def seed_seconds(frame: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """One row per data seed: the phases paid once for that seed, outside every repeat.

    The tier the two-tier contract had nowhere to put. A design with no such phase pays
    0.0 here, which is a measured zero and not a gap -- the gap is `SECONDS_UNCHARGED`.

    Parameters
    ----------
    frame : pd.DataFrame
    tol : float
        Allowed slack, in seconds, on `SEED_TOTAL` against its parts.

    Returns
    -------
    pd.DataFrame
        Indexed by `data_seed`, with each `SEED_SECONDS` column present and the per-seed
        total `seconds_seed` -- `SEED_TOTAL` as the simulation wrote it where the frame
        carries it, and the sum of the parts otherwise.

    Raises
    ------
    ValueError
        If a per-data-seed column varies between the repeats or rows of one seed, which
        would mean the phase was re-run rather than amortised, or if `SEED_TOTAL` is not
        the sum of the parts.
    """
    seeds = _repeat_groups(frame, ("data_seed",))
    table = pd.DataFrame(index=seeds.size().index)
    present = [column for column in SEED_SECONDS if column in frame.columns]
    for column in present:
        table[column] = _shared_value(seeds, column, "shared per data seed")
    parts = _present_sum(table, present) if present else pd.Series(
        0.0, index=table.index)
    if SEED_TOTAL in frame.columns:
        table["seconds_seed"] = _shared_value(seeds, SEED_TOTAL, "shared per data seed")
        gap = (table.seconds_seed - parts).abs()
        if (gap > tol).any():
            raise ValueError(
                f"{SEED_TOTAL} is the sum of "
                f"{', '.join(present) if present else 'the per-data-seed columns'}, but it "
                f"differs from that sum by up to {gap.max():.2f} s on "
                f"{int((gap > tol).sum())} data seed(s). One per-data-seed phase is being "
                "written into the total and not into a column of its own, or the other way "
                "round.")
    else:
        table["seconds_seed"] = parts
    return table


def total_seconds(frame: pd.DataFrame, keys=REPEAT_KEYS, own="auto",
                  tol: float = 0.05, check: bool = True) -> float:
    """The wall clock a set of runs actually cost, with shared work counted once.

    The number to quote for "how long did this take": every repeat's measured clock once,
    plus every per-data-seed phase once per data seed. Summing the `seconds_*` columns of a
    three-estimator frame instead would report the autoencoder, the nuisance solve and the
    shared scoring three times each, and summing `seconds_encode` down a CelebA frame's
    rows would report one 75.5 s encode twelve times.

    Parameters
    ----------
    frame : pd.DataFrame
        Result rows at ONE configuration.
    keys, own, tol
        As in `repeat_seconds`.
    check : bool
        Whether to verify the per-repeat decomposition on the way. True for a frame holding
        every row of its repeats. False for a SLICE of them -- one sampler of a design that
        runs two inside one repeat clock -- where the rows left out carry part of that
        clock, so the identity cannot close however correct the columns are. The clock
        itself is a property of the repeat and is read once either way.

    Returns
    -------
    float
        Seconds, rounded to 0.1 s like the columns it is built from, and a LOWER BOUND on
        the task's own wall clock (`SECONDS_UNCHARGED`). NaN -- deliberately, via
        skipna=False -- if any repeat's clock is missing, which is what a frame
        concatenating rows written before these columns existed looks like: those repeats
        cost real time that cannot be recovered, and skipping them would return a smaller
        number that looks right. A pre-contract frame still totals, since `seconds_repeat`
        is a measurement and only its SPLIT into phases was never recorded.
    """
    if check or "seconds_repeat" not in frame.columns:
        table = repeat_seconds(frame, keys=keys, own=own, tol=tol)
        clocks = (table.seconds_repeat if "seconds_repeat" in table.columns
                  else table.seconds_total)
    else:
        clocks = _shared_value(_repeat_groups(frame, keys), "seconds_repeat",
                               "the total of the whole repeat")
    total = (float(clocks.sum(skipna=False))
             + float(seed_seconds(frame, tol).seconds_seed.sum(skipna=False)))
    return round(total, 1)


def _sole_method(frame: pd.DataFrame) -> None:
    """Raise if these rows mix two samplers or two target policies.

    Neither is a row axis a repeat accumulates: DecFM_2D runs both samplers inside one
    repeat clock, and Warfarin's policies are separate runs with separate nuisance solves.
    A timing table pooling two samplers would add both flows into one "sampling cost" and
    label it with one sampler's name, which is the kind of silent scope change this module
    exists to prevent.
    """
    for column in ("sampler", "target_policy"):
        if column not in frame.columns:
            continue
        values = sorted(set(frame[column].dropna().astype(str)))
        if len(values) > 1:
            raise ValueError(
                f"these rows cover {len(values)} values of {column} "
                f"({', '.join(values)}); a sampling cost is per {column}, so slice to one "
                "before timing it.")


def _attributed_nuisance(table: pd.DataFrame, names) -> pd.Series:
    """The share of the one nuisance solve each row of `table` is charged.

    Where the design records the `seconds_ratio` / `seconds_cme` breakdown, each estimator
    is charged the halves it is actually spent on (`ESTIMATOR_NUISANCE`). Where it records
    only the total, every estimator is charged the whole solve: exact for `dr`, an upper
    bound for the two one-sided estimators, and said to be so in the table's note.

    Parameters
    ----------
    table : pd.DataFrame
        Either a results frame (one row per estimator row) or the aggregate indexed by
        repeat and estimator.
    names : sequence of str
        The estimator each row belongs to, aligned with `table`.

    Returns
    -------
    pd.Series
    """
    split = [column for column in NUISANCE_SECONDS if column in table.columns]
    if len(split) < len(NUISANCE_SECONDS):
        return _present_sum(table, ("seconds_nuisance",))
    charged = pd.Series(np.nan, index=table.index, dtype=float)
    for name in sorted(set(names)):
        rows = np.asarray(names) == name
        parts = [column for column in ESTIMATOR_NUISANCE.get(name, NUISANCE_SECONDS)
                 if column in split]
        charged.loc[rows] = table.loc[rows, parts].sum(axis=1, min_count=len(parts))
    return charged


def representation_scope(frame: pd.DataFrame):
    """Which column pays for the space the flow runs in, its table label and its tier.

    Returns
    -------
    (str, str, str) or (None, None, None)
        The column, its short label, and "repeat" or "data seed".
    """
    for column, (label, tier) in REPRESENTATION_SECONDS.items():
        if column in frame.columns:
            return column, label, tier
    return None, None, None


def sampling_seconds(frame: pd.DataFrame, keys=REPEAT_KEYS,
                     tol: float = 0.05) -> pd.DataFrame:
    """One row per (repeat, estimator): what drawing that sample cost on its own.

    `seconds_sampling` spans two tiers -- a share of the repeat's shared nuisance solve
    plus this row's own flow and decode -- so it is an ATTRIBUTION, answering "what would
    drawing this one sample have cost", and belongs to NEITHER identity: the estimators of
    a repeat share one solve, so their shares overlap and must never be added together or
    into `seconds_repeat`. The simulations write the column per ROW; the column here is its
    repeat-level aggregate, the only one that does not double count -- the shared halves
    taken ONCE and the per-row flow and decode summed. The two coincide on a design with
    one row per estimator per repeat and differ on a two-arm design, which is why a
    recorded column is checked row by row (`_check_recorded_sampling`) rather than summed.

    The arms of a repeat are SUMMED, not averaged: both arms' flows were really run. That
    is the one departure from the house aggregation order, and `sampling_table` says so.

    Parameters
    ----------
    frame : pd.DataFrame
        Result rows at ONE configuration and for ONE sampler and target policy.
    keys : sequence of str
        The columns identifying a repeat; `REPEAT_KEYS` by default.
    tol : float
        Slack, in seconds, on a recorded `seconds_sampling` against this definition.

    Returns
    -------
    pd.DataFrame
        Indexed by `keys` and `estimator`, carrying whichever of `seconds_ratio`,
        `seconds_cme`, `seconds_nuisance`, `seconds_flow`, `seconds_decode` the design
        records, their attributed sum `seconds_sampling`, and the two costs excluded from
        it: `seconds_representation` and `seconds_evaluation`.

    Raises
    ------
    ValueError
        If the rows mix two samplers or target policies, if a shared column varies inside
        a (repeat, estimator) group, or if a recorded `seconds_sampling` disagrees with
        the definition above.
    """
    _sole_method(frame)
    groups = _estimator_groups(frame, keys)
    table = pd.DataFrame(index=groups.size().index)
    # SAMPLING_SECONDS is deliberately NOT summed here: it is rebuilt below from the parts
    # with the shared halves taken once, which is what the per-row column cannot be.
    for column in OWN_SECONDS + (FID_SECONDS,):
        if column in frame.columns:
            table[column] = groups[column].sum(min_count=1)
    for column in SHARED_SECONDS + NUISANCE_SECONDS:
        if column in frame.columns:
            table[column] = _shared_value(groups, column, "shared per repeat")
    for column in SEED_SECONDS:
        if column in frame.columns:
            table[column] = _shared_value(groups, column, "shared per data seed")

    table[SAMPLING_SECONDS] = (
        _attributed_nuisance(table, table.index.get_level_values("estimator"))
        + _present_sum(table, ("seconds_flow", "seconds_decode")))
    representation, _, _ = representation_scope(frame)
    table["seconds_representation"] = (table[representation] if representation
                                       else np.nan)
    scored = EVALUATION_SECONDS + ((FID_SECONDS,) if fid_scope(frame, keys, tol) == "own"
                                   else ())
    table["seconds_evaluation"] = _present_sum(
        table, [column for column in scored if column in table.columns])
    _check_recorded_sampling(frame, tol)
    return table


def _check_recorded_sampling(frame: pd.DataFrame, tol: float) -> None:
    """Raise if the recorded `SAMPLING_SECONDS` column disagrees with its definition.

    The simulations write it PER ROW as the estimator's charged nuisance halves plus that
    row's own flow and decode. Checked row by row rather than against the aggregate: on a
    design with several arms per repeat the column must NOT be summed down the rows, since
    each row carries the one shared solve, and a check that summed it would licence
    exactly the double count this module exists to prevent.
    """
    if SAMPLING_SECONDS not in frame.columns:
        return
    names = _estimator_labels(frame)
    defined = (_attributed_nuisance(frame, names)
               + _present_sum(frame, ("seconds_flow", "seconds_decode")))
    gap = (frame[SAMPLING_SECONDS].astype(float) - defined).abs()
    if (gap > tol).any():
        raise ValueError(
            f"{SAMPLING_SECONDS} differs from its definition -- the nuisance halves this "
            "estimator needs (plug-in the CME, IPW the ratios, DR both) plus this row's "
            f"flow and decode -- by up to {gap.max():.2f} s on "
            f"{int((gap > tol).sum())} row(s). One definition has to win, or two tables "
            "built from one CSV quote two different sampling costs.")


def _seconds_stats(table: pd.DataFrame, column: str):
    """(mean, sd) of `column`: repeats averaged within a data seed, sd ACROSS seeds.

    The house order, on a table whose arms are already summed. A column the design does
    not record is all-NaN and comes back as (nan, nan), which `_cell` prints as `--`.
    """
    if column not in table.columns:
        return np.nan, np.nan
    if "data_seed" in (table.index.names or []):
        per_seed = table.groupby(level="data_seed")[column].mean()
    else:
        per_seed = table[column]
    return per_seed.mean(), per_seed.std()


def sampling_table(frame: pd.DataFrame, dataset: str, title: str = None,
                   note: str = None, label_width: int = 26,
                   cell_width: int = 15) -> str:
    """Per estimator: the isolated sampling cost, its parts, and what it excludes.

    The table a cost comparison against another method's published timings is read from,
    which is why the excluded costs are printed beside it rather than left implied: the
    representation and the scoring are not part of drawing a sample, and each column
    header carries the tier its number is paid on.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows at ONE configuration with the estimator axis UNPINNED, as
        ``default_rows(runs, dataset=..., estimator=None)`` leaves it.
    dataset : str
    title, note : str, optional
        Defaults name the dataset and state the attribution, the arm summing and
        `SECONDS_UNCHARGED`.
    label_width, cell_width : int

    Returns
    -------
    str
    """
    _check_scopes(frame)
    table = sampling_seconds(frame)
    _, label, tier = representation_scope(frame)
    columns = [("Ratio", "seconds_ratio"), ("CME", "seconds_cme"),
               ("Flow", "seconds_flow"), ("Decode", "seconds_decode"),
               ("Sampling", "seconds_sampling"),
               (f"{label}/{'seed' if tier == 'data seed' else 'repeat'}"
                if label else "Represent.", "seconds_representation"),
               ("Scoring", "seconds_evaluation")]
    names = table.index.get_level_values("estimator")
    rows = []
    for estimator in ESTIMATORS:
        if estimator not in set(names):
            continue
        own = table[names == estimator]
        rows.append((ESTIMATOR_LABELS[estimator],
                     [_seconds_stats(own, column) for _, column in columns]))
    if title is None:
        title = (f"{dataset}   seconds to draw one sample   ({len(frame)} rows, "
                 f"{len(_repeat_groups(frame, REPEAT_KEYS))} repeat(s), "
                 f"{len(seed_seconds(frame))} data seed(s))")
    return plain_table(rows, [header for header, _ in columns], title=title,
                       note=_sampling_note(frame, note),
                       label_width=label_width, cell_width=cell_width,
                       first_column="Target embedding", arrow=False, digits=1)


def _sampling_note(frame: pd.DataFrame, note) -> str:
    """The note under `sampling_table`: the scope of every column it just printed."""
    if note is not None:
        return note
    lines = [
        "Sampling = the estimator's share of the one nuisance solve + its flow + its "
        "decode, summed over the arms of a repeat. It is an ATTRIBUTION, not a part of "
        "seconds_repeat: the estimators of a repeat SHARE that solve, so their shares "
        "overlap and must not be added together."]
    if not all(column in frame.columns for column in NUISANCE_SECONDS):
        lines.append(
            "These runs do not record the seconds_ratio / seconds_cme split, so every "
            "estimator is charged the WHOLE solve: exact for the doubly robust vector, an "
            "upper bound for the two one-sided ones.")
    _, label, tier = representation_scope(frame)
    excluded = ("The last two columns are EXCLUDED from Sampling: neither putting the "
                "outcome in the flow's space nor scoring the result is part of drawing a "
                "sample.")
    if tier == "data seed":
        excluded += (f" {label} is paid ONCE PER DATA SEED and amortised over that seed's "
                     "repeats, which is why its header says so; every other column is per "
                     "repeat.")
    lines.append(excluded)
    if "seconds_fid_ref" in frame.columns:
        reference = seed_seconds(frame).seconds_fid_ref
        lines.append(f"The FID reference features are a further per-data-seed cost "
                     f"({reference.mean():.1f} s per seed), outside every column above.")
    lines += [_identity_line(frame), _total_line(frame),
              _uncharged_note(frame)]
    return _wrapped(*lines)


def _identity_line(frame: pd.DataFrame) -> str:
    """One line on whether the per-repeat decomposition closes on the rows as given."""
    try:
        table = repeat_seconds(frame)
    except ValueError as refusal:
        first = str(refusal).splitlines()[0]
        if "do not add up to seconds_repeat" not in first:
            return f"The per-repeat identity cannot be checked here: {first}"
        return ("The per-repeat identity does not close on these rows ALONE, which is what "
                "a slice of a repeat looks like -- one sampler of a repeat that ran two, "
                "say, whose other rows carry part of the clock. Check it on the whole frame "
                "with `repeat_seconds`.")
    gap = ((table.seconds_repeat - table.seconds_total).abs().max()
           if "seconds_repeat" in table.columns else 0.0)
    return (f"The per-repeat identity closes on these rows: shared + other + summed "
            f"per-estimator columns is within {gap:.2f} s of seconds_repeat.")


def _total_line(frame: pd.DataFrame) -> str:
    """One line: what this set of runs cost in total, or why that cannot be said."""
    try:
        total = total_seconds(frame, check=False)
    except ValueError as refusal:
        return f"Wall clock NOT SUMMABLE: {str(refusal).splitlines()[0]}"
    if pd.isna(total):
        return ("Wall clock NOT SUMMABLE: some repeat's clock is blank, so the total "
                "would be short by it.")
    return (f"Wall clock of the repeat(s) these rows come from: {total:.1f} s "
            f"({total / 60:.1f} min) -- every repeat counted once whatever it ran inside "
            "it, and every per-data-seed phase once per seed.")


def sampling_report(frame: pd.DataFrame, dataset: str, title: str = None,
                    note: str = None) -> str:
    """The sampling-cost table for `dataset`, or the lines saying it was not recorded.

    The degrading half is the point: every CSV archived in this repository predates the
    wall-clock contract, so this prints what IS on disk -- the measured repeat clocks and
    the total they sum to -- and names what is not, in place of a table built from parts
    that do not close. Re-executing once contract runs land fills the section in with no
    code change, since the tiers are read from the `seconds_*` columns and not from a
    filename.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows at ONE configuration, estimator axis unpinned.
    dataset : str
    title, note : str, optional
        Passed to `sampling_table`.

    Returns
    -------
    str
    """
    reason = timing_reason(frame)
    if reason is None:
        try:
            return sampling_table(frame, dataset, title=title, note=note)
        except ValueError as refusal:
            reason = str(refusal)
    return "\n\n".join([
        _wrapped(f"Sampling cost NOT RECORDED for {dataset}: {reason}."),
        _wrapped(f"What IS on disk: {len(frame)} rows over "
                 f"{len(_repeat_groups(frame, REPEAT_KEYS))} repeat(s) and "
                 f"{len(seed_seconds(frame))} data seed(s).", _total_line(frame)),
        _wrapped(_uncharged_note(frame)),
        _wrapped("Re-executing once contract runs land fills this section in with no code "
                 "change: the three tiers are read from the `seconds_*` columns "
                 "themselves, not from a filename.")])


# ---------------------------------------------------------------------------------------
# Training time
#
# The compute-time table the paper quotes for OUR method, on one definition:
#
#     training(data seed) = fitting the nuisance + running the MMD flow,
#                           over ALL ARMS of that seed
#
# reported as mean +/- sd ACROSS the data seeds. It is a DIFFERENT quantity from
# `sampling_seconds`, which charges each estimator the HALVES of the solve it is spent on
# so that the three estimators can be compared; here the method is the doubly robust
# embedding, it needs the whole solve, and the arms are summed because both arms' flows
# were really run. Both quantities live in this module and neither replaces the other.
#
# The arm convention, read off the three simulation scripts rather than assumed, and the
# reason the two phases are combined differently:
#
#   seconds_nuisance  SHARED PER REPEAT and ALREADY SUMMED OVER ARMS. All three scripts
#                     accumulate each arm's ratios and each arm's target-weight solve into
#                     the repeat's `seconds_ratio` / `seconds_cme` and write the rounded sum
#                     onto every row ("Summed over arms", ColorMNIST and CelebA; dSprites
#                     has a single continuous-action target and no arm axis at all). So it
#                     is taken ONCE per repeat. Summing it down the rows would multiply one
#                     solve by n_arms x n_estimators.
#   seconds_flow      PER ROW, one row per (arm, estimator). So it is SUMMED over the arms
#                     of the repeat, after the estimator axis is pinned. Averaging it would
#                     report half of what a two-arm run spent on flow.
#
# Both mistakes are silent, which is why `_check_arm_rows` refuses a frame whose rows are
# not one per (repeat, arm) and `_shared_value` refuses a nuisance column that varies
# inside a repeat. What is EXCLUDED from the training number -- the representation, the
# per-data-seed setup, the decode and all scoring -- is printed beside it rather than
# dropped, so an end-to-end figure can be formed without re-reading the CSVs.
# ---------------------------------------------------------------------------------------

#: The columns identifying one repeat OF ONE LAUNCHER TASK. `tag` is in here and not in
#: `REPEAT_KEYS` because two launcher tasks reuse the same (data_seed, repeat) labels: the
#: ColorMNIST solver comparison writes `est_solver_projected_nystrom` rows at the same
#: design as the headline grid's `est_headline` rows, so they pool into the default table
#: as further repeats -- correct for a metric, and two distinct clocks for a cost. Grouping
#: by `tag` as well keeps each task's nuisance solve and flow with its own repeat; a frame
#: whose tag still does not separate two configurations is refused by `_shared_value`
#: rather than averaged.
RUN_KEYS = REPEAT_KEYS + ("tag",)

#: The two phases the training time is the sum of, in the order the table prints them.
TRAINING_PHASES = ("seconds_nuisance", "seconds_flow")

#: Every column `training_seconds` returns and `training_table` prints, in order: the two
#: phases, the training time itself, and the four costs EXCLUDED from it. Named here so a
#: pre-contract frame can return the same shape filled with NaN.
TRAINING_COLUMNS = ("seconds_nuisance", "seconds_flow", "seconds_training",
                    "seconds_decode", "seconds_representation", "seconds_setup",
                    "seconds_evaluation")


def _check_arm_rows(rows: pd.DataFrame, keys) -> None:
    """Raise unless each arm of each repeat contributes exactly ONE row to the flow sum.

    `seconds_flow` is written per (arm, estimator) row, so a training figure sums it over
    the arms of a repeat. That sum is the arms' flows only if the rows it runs over ARE
    distinct arms: two launcher tasks reusing one (data_seed, repeat) label, or an
    estimator axis left unpinned, would add two tasks' or three estimators' flows together
    and label the result one arm's cost.

    Parameters
    ----------
    rows : pd.DataFrame
        Already pinned to one estimator.
    keys : sequence of str
        The columns identifying a repeat.
    """
    present = [key for key in tuple(keys) + ("target_a",) if key in rows.columns]
    if not present:
        return
    repeated = rows.duplicated(subset=present, keep=False)
    if repeated.any():
        offenders = rows.loc[repeated].groupby(present, dropna=False).size()
        raise ValueError(
            f"{int(repeated.sum())} row(s) share a ({', '.join(present)}) label, so "
            "summing seconds_flow over them would add several runs' flows together:\n"
            f"{offenders.to_string()}\n"
            "Either the estimator axis is unpinned, or two launcher tasks reuse the same "
            "(data_seed, repeat) labels -- pin the configuration with `default_rows` and "
            "keep `tag` in `keys`.")


def _check_seed_scope(rows: pd.DataFrame, keys) -> None:
    """Raise if a per-data-seed column varies between the repeats of one data seed.

    `seed_seconds` makes the same check keyed by `data_seed` alone, which is right for a
    frame holding one launcher task and wrong for one holding two: each task prepares its
    own seed's data, so the per-seed phases are per (data seed, task). The keys here are
    `keys` without `repeat`, which is that unit.
    """
    seed_keys = [key for key in keys if key != "repeat"]
    seeds = _repeat_groups(rows, seed_keys)
    for column in SEED_SECONDS + (SEED_TOTAL,):
        if column in rows.columns:
            _shared_value(seeds, column, "shared per data seed")


def training_seconds(frame: pd.DataFrame, keys=RUN_KEYS, estimator: str = "dr",
                     tol: float = 0.05) -> pd.DataFrame:
    """One row per repeat: what TRAINING that repeat cost -- the nuisance fit + the flow.

    The user's definition, implemented exactly: the nuisance fit plus the MMD flow, over
    ALL ARMS of the repeat, for the estimator that IS the method. The two phases are
    combined differently because the simulations measure them on different tiers -- the
    nuisance once per repeat with the arms already summed into it, the flow once per
    (arm, estimator) row -- so the nuisance is taken ONCE and the flow is SUMMED over the
    arms. See the section comment above for the evidence in the three scripts.

    The whole solve is charged, not the halves `sampling_seconds` attributes: for the
    doubly robust embedding those coincide, and for any other `estimator` the whole solve
    is still what was run before its flow could start.

    Parameters
    ----------
    frame : pd.DataFrame
        Result rows at ONE configuration, for ONE sampler and target policy, with the
        estimator axis UNPINNED -- it is pinned here, and `fid_scope` needs every row of
        the repeat to resolve itself.
    keys : sequence of str
        The columns identifying one repeat of one launcher task; `RUN_KEYS` by default.
        Keys the frame does not carry are skipped.
    estimator : str
        Whose flow to sum. "dr" by default, because the doubly robust embedding is the
        method; the plug-in and IPW flows belong to the estimator ablation.
    tol : float
        Allowed slack, in seconds, on `seconds_shared` against its parts and on the
        nuisance halves against their total.

    Returns
    -------
    pd.DataFrame
        Indexed by `keys`, carrying `TRAINING_COLUMNS` -- the nuisance (once), the flow
        (summed over arms), their sum `seconds_training`, and the excluded `seconds_decode`
        (summed over arms), `seconds_representation`, `seconds_setup` (per data seed) and
        `seconds_evaluation` -- plus the shared and own columns they are built from and
        `seconds_repeat` where the frame carries it. On a frame `timing_reason` calls
        pre-contract, every column is NaN: those repeats cost real time whose split was
        never recorded, and a number that looked right would be worse than a blank.

    Raises
    ------
    ValueError
        If the rows mix two samplers or target policies, if the rows are not one per
        (repeat, arm) so the flow sum would add two runs together, if a shared or
        per-data-seed column varies inside its scope, or if `seconds_shared` is not the sum
        of its parts. Every one of those makes the sum silently wrong.
    """
    rows = pin_estimator(frame, estimator)
    groups = _repeat_groups(rows, keys)
    reason = timing_reason(frame)
    if reason is not None:
        # Pre-contract: reported missing rather than refused, as the other timing helpers
        # do. No check is run first, so this path cannot raise on an archived frame.
        table = pd.DataFrame(index=groups.size().index)
        for column in TRAINING_COLUMNS:
            table[column] = np.nan
        return table
    if not len(rows):
        raise ValueError(
            f"no rows for estimator {estimator!r}; the frame carries "
            f"{sorted(set(_estimator_labels(frame)))}.")
    _sole_method(rows)
    _check_arm_rows(rows, keys)
    _check_seed_scope(rows, keys)

    table = _shared_table(rows, groups, tol)
    for column in SEED_SECONDS:
        if column in rows.columns:
            table[column] = _shared_value(groups, column, "shared per data seed")
    # The only tier free to be summed down a repeat's rows, which after the estimator pin
    # is one row per arm: this is the arm summation.
    for column in OWN_SECONDS + (FID_SECONDS,):
        if column in rows.columns:
            table[column] = groups[column].sum(min_count=1)
    if "seconds_repeat" in rows.columns:
        table["seconds_repeat"] = _shared_value(groups, "seconds_repeat",
                                                "the total of the whole repeat")

    if "seconds_nuisance" not in table.columns:
        table["seconds_nuisance"] = _present_sum(table, NUISANCE_SECONDS)
    for column in TRAINING_PHASES:
        # Present but blank, never absent: `_present_sum` skips a column it cannot see,
        # which would quietly report the flow alone as the training time.
        if column not in table.columns:
            table[column] = np.nan
    table["seconds_training"] = _present_sum(table, TRAINING_PHASES)

    representation, _, _ = representation_scope(rows)
    table["seconds_representation"] = (table[representation]
                                       if representation in table.columns else np.nan)
    if SEED_TOTAL in rows.columns:
        table["seconds_setup"] = _shared_value(groups, SEED_TOTAL,
                                               "shared per data seed")
    else:
        parts = [column for column in SEED_SECONDS if column in table.columns]
        table["seconds_setup"] = _present_sum(table, parts) if parts else np.nan
    # `fid_scope` is resolved on the WHOLE frame: it closes the per-repeat identity, which
    # needs every estimator's rows, not the one estimator summed here.
    scored = EVALUATION_SECONDS + ((FID_SECONDS,)
                                   if fid_scope(frame, keys, tol) == "own" else ())
    table["seconds_evaluation"] = _present_sum(
        table, [column for column in scored if column in table.columns])
    if "seconds_decode" not in table.columns:
        table["seconds_decode"] = np.nan
    return table


def training_table(frame: pd.DataFrame, dataset: str, by: str = None, labels: dict = None,
                   order=None, title: str = None, note: str = None, keys=RUN_KEYS,
                   estimator: str = "dr", tol: float = 0.05, ours: str = "IMMD-Flow",
                   label_width: int = 28, cell_width: int = 15) -> str:
    """The compute-time table for OUR method: nuisance + flow, and what it excludes.

    One row unless `by` is given, because there is no competing method in this table: the
    published baselines quote no wall clock on this hardware, so a comparison column would
    be a comparison of two machines.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows at ONE configuration with the estimator axis UNPINNED, as
        ``default_rows(runs, dataset=..., estimator=None)`` leaves it.
    dataset : str
    by : str, optional
        A column to break the table down by -- `n_obs` or `n_particles_requested` -- one
        row per value. Each value is SLICED first and timed on its own, so two sweep points
        are never pooled into one repeat.
    labels : dict, optional
        Row labels for the values of `by`; the value itself by default.
    order : list, optional
        The values of `by` in row order; sorted order by default.
    title, note : str, optional
        Defaults state the definition, the arm convention, the aggregation and what is
        excluded.
    keys, estimator, tol
        As in `training_seconds`.
    ours : str
        The single row's label.
    label_width, cell_width : int

    Returns
    -------
    str
    """
    _, label, tier = representation_scope(frame)
    columns = [("Nuisance", "seconds_nuisance"), ("Flow", "seconds_flow"),
               ("TRAINING", "seconds_training"), ("Decode", "seconds_decode"),
               (f"{label}/{'seed' if tier == 'data seed' else 'repeat'}"
                if label else "Represent.", "seconds_representation"),
               ("Setup/seed", "seconds_setup"), ("Scoring", "seconds_evaluation")]
    if by is None:
        slices = [(ours, frame)]
    else:
        values = sorted(frame[by].dropna().unique()) if order is None else order
        slices = [((labels or {}).get(value, value), frame[frame[by] == value])
                  for value in values]
    rows, tables = [], []
    for row_label, subset in slices:
        if not len(subset):
            continue
        table = training_seconds(subset, keys=keys, estimator=estimator, tol=tol)
        tables.append(table)
        rows.append((row_label, [_seconds_stats(table, column) for _, column in columns]))
    if title is None:
        title = (f"{dataset}   training seconds: nuisance fit + MMD flow over all arms   "
                 f"({len(frame)} rows, {len(_repeat_groups(frame, keys))} repeat(s), "
                 f"{len(_repeat_groups(frame, ('data_seed',)))} data seed(s))")
    return plain_table(
        rows, [header for header, _ in columns], title=title,
        note=_training_note(frame, note, keys, estimator,
                            pd.concat(tables) if tables else None),
        label_width=label_width, cell_width=cell_width, first_column="Method",
        arrow=False, digits=1)


def _training_note(frame: pd.DataFrame, note, keys, estimator: str, table) -> str:
    """The note under `training_table`: the definition, the arm convention, the exclusions.

    `table` is every row the table above was built from, concatenated, or None when no
    slice had any rows -- it carries the measured repeat clock the training figure is a
    share of.
    """
    if note is not None:
        return note
    arms = (int(frame.target_a.nunique(dropna=False)) if "target_a" in frame.columns
            else 1)
    repeats = len(_repeat_groups(frame, keys))
    seeds = len(_repeat_groups(frame, ("data_seed",)))
    over = (f"over BOTH of the {arms} intervention arms" if arms > 1 else
            "for the one target this design intervenes on -- it has no arm axis")
    lines = [
        f"TRAINING = fitting the nuisance + running the flow, {over}, for the "
        f"{ESTIMATOR_LABELS.get(estimator, estimator)} embedding -- the method. Plug-in "
        "and IPW share the same solve and belong to the estimator ablation, not here.",
        "The two phases are measured on different tiers, so they are combined differently. "
        "Nuisance is a REPEAT TOTAL with the arms ALREADY SUMMED into it by the simulation "
        f"({arms} arm(s) of importance ratios and {arms} target-weight solve(s) inside that "
        "one number), so it is counted ONCE per repeat. Flow is written PER (arm, estimator) "
        "ROW, so it is SUMMED over the arms. Averaging the flow, or charging the nuisance "
        "once per arm, would both be wrong and neither would look it.",
        f"Repeats are averaged within a data seed and the spread is the standard deviation "
        f"ACROSS the {seeds} data seed(s), as everywhere else in these notebooks -- except "
        "that the arms are SUMMED rather than averaged, which is the point of the column.",
        "The last four columns are EXCLUDED from TRAINING and printed so an end-to-end "
        "figure can be formed without re-reading the CSVs. Decode is excluded because the "
        "definition is the nuisance and the flow; it is the cost of turning the flowed "
        "latents back into images, and a reader who counts it as part of the method can add "
        "it. Representation, Setup and Scoring are the space the flow runs in, the "
        "per-data-seed preparation and all measurement -- none of them training."]
    if tuple(representation_scope(frame)[1:]) == (None, None):
        lines.append("This design has no representation column: the flow runs on the "
                     "outcome as it stands.")
    elif representation_scope(frame)[2] == "data seed":
        label = representation_scope(frame)[1]
        lines.append(f"{label} is paid ONCE PER DATA SEED and is one COMPONENT of "
                     "Setup/seed, not a further cost beside it -- do not add the two.")
    reason = timing_reason(frame)
    if reason is not None:
        lines.append(f"Every cell above is blank because {reason}.")
    elif table is not None and "seconds_repeat" in table.columns:
        clock, spread = _seconds_stats(table, "seconds_repeat")
        setup, _ = _seconds_stats(table, "seconds_setup")
        lines.append(
            f"The measured clock of one repeat is {clock:.1f} +/- {spread:.1f} s and covers "
            f"every row that repeat ran -- {len(frame)} rows over {repeats} repeat(s) here, "
            f"every estimator and every arm -- plus {setup:.1f} s of per-data-seed setup "
            f"outside it. TRAINING is the {ESTIMATOR_LABELS.get(estimator, estimator)} "
            "share of that clock, not the clock itself.")
    if "tag" in frame.columns and frame.tag.nunique(dropna=False) > 1:
        tags = ", ".join(sorted(set(frame.tag.astype(str))))
        lines.append(
            f"These rows pool {frame.tag.nunique(dropna=False)} launcher tasks ({tags}) at "
            "the same design. Each is timed as a repeat of its own and they are averaged "
            "within a data seed, so the training figure is sound -- but a task's own "
            "Setup/seed differs from another's (the one-off digit-pool and Inception loads "
            "are charged to the first data seed of each process), so that column is an "
            "average of two different per-seed setups. Slice to one tag to read it.")
    lines.append(_uncharged_note(frame))
    return _wrapped(*lines)
