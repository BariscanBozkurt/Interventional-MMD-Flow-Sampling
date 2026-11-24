"""
Warfarin semi-synthetic: sampling an interventional outcome law under a
STOCHASTIC continuous-action policy.

Every other simulation in this repository intervenes deterministically -- a
single arm, or a fixed shift of a continuous action. This one uses a genuinely
stochastic target policy on real patient covariates, and in its headline setting
a MEAN-PRESERVING one: the target action law is a 50/50 mixture at -delta and
+delta around the logging mean, so E[A | X] is unchanged while the law becomes
bimodal with conditional variance larger by delta^2.

That choice is the point. The intervention is invisible to any summary that
tracks the policy through its mean action, yet the mis-dosing outcome
max(|a - TherDose(X)| - tol * TherDose(X), 0) is a kinked function of the
action, so the interventional OUTCOME law moves substantially -- at the
calibrated default the probability of severe mis-dosing rises by about 70 per
cent while the mean action moves under 1 per cent of an action standard
deviation. Recovering that requires sampling the law, not estimating a scalar
functional of it.

Pipeline, matching the other simulations: cross-fitted (or no-split) one-step DR
weights over a projected Nystrom CME, then a particle flow against the resulting
target embedding. The outcome is scalar, so the flow runs directly in outcome
space with no autoencoder and the oracle law is available exactly by drawing
A ~ pi(. | X) and evaluating the dosing cost.

Estimator axis: --estimators selects which target embeddings to sample against.
The plug-in expansion uses the conditional mean embedding alone, IPW the density
ratio alone (v = w / n), and the one-step DR estimator is their sum,
v_DR = v_PI + v_correction. That identity is why the three are one experiment and
not three: a single ridge solve, asked for its two components, yields all of them,
so a run that requests several of them shares the logged sample, the nuisance fit,
the starting particles, the oracle draws and the flow budget across all of them,
and the only thing that differs is the weight vector handed to the flow. An
unpaired comparison would be swamped by seed noise. Each estimator gets its own
CSV row, identified by the `estimator` column. The default is the DR estimator
alone, which takes the solver's single-vector path and is therefore bitwise what
it was before the other two existed. The IPW row is the Horvitz-Thompson form
v = w / n by default -- the term that appears inside v_DR, and the reason the
three are algebraically comparable, but one with no unit mass; --ipw_form hajek
switches it to the self-normalised w / sum(w), a different estimator, and the
form used is recorded in the ipw_form CSV column.

This is the design where that comparison is sharpest: the closed-form
interventional CDF behind the `*_exact` columns carries no Monte Carlo error, so
the three estimators are ranked against the true law rather than against a finite
draw of it.

Counting convention: the flow row is scored with --n_particles samples against
the --n_ref oracle reference draw, while the `floor` row scores a second,
independent oracle draw of --n_ref samples against the same reference. Every
metric here shrinks with the count it is evaluated at, so those two rows are not
comparable unless n_particles equals n_ref, and a sweep over --n_particles read
against `floor` alone would be a pure counting artefact -- the floor standing
still while the flow's best achievable value rises. The `*_floor_matched`
columns repeat the floor with that second draw randomly cut to
n_floor_matched = min(n_particles, n_ref), the count the row's own particles
were scored at, for the sample-based AND the exact-law metrics. Compare a swept
--n_particles against those.

Cross-fitting folds are assigned by PATIENT, not by row: the cohort has a few
thousand patients and --n_obs above that resamples them WITH replacement, so a
row-level split would put the same patient in several folds and every
cross-fitted nuisance would be evaluated on a patient it was fitted on. The
--dr_fit ablation would then measure that leakage rather than cross-fitting.

Seed scheme: `data_seed` fixes the DATA. The logged sample (drawn with seed
1000 * data_seed), the oracle reference draw (100000 + 1000 * data_seed) and the
floor draw (200000 + 1000 * data_seed) are drawn ONCE per data seed in `main`,
so every repeat of a seed sees the same three samples and the spread over
repeats is pipeline variance alone. `pipeline_seed = 1000 * data_seed + repeat`
then fixes everything the pipeline draws: the nuisance subsamples, the
target-policy resamples, the Nystrom landmarks, the count-matched floor cut, the
starting particles (unless --particle_seed overrides them) and the flow.

The design runs on a GPU: the flow is a dense n_particles x n_obs kernel at
every step, which is impractical on CPU at the default sizes.

Wall-clock accounting
---------------------
Every `seconds_*` column is wall clock (time.time(), rounded to 0.1 s) and belongs to
exactly ONE of three scopes. Mixing them is what makes a timing table wrong in both
directions: a cost measured once and written onto many rows is multiplied by any sum
down those rows, and a cost measured outside the repeat clock is lost by any sum of
repeats.

PER DATA SEED -- identical on every row of every repeat of that data seed, and measured
OUTSIDE seconds_repeat, because the work happens once before the repeat loop:

    seconds_data        this data seed's data preparation: the logged sample and the two
                        oracle draws. The cohort table is read once per PROCESS and is
                        charged to the FIRST data seed's seconds_data, so the per-task
                        identity below closes without a column a reader would have to
                        know to count once across seeds. Production tasks run one data
                        seed each, where the distinction does not arise.
    seconds_setup       the per-data-seed TOTAL, and the column to add once per data
                        seed. Here it is seconds_data alone: the outcome is scalar, so
                        this design has neither a frozen-encoder pass (seconds_encode)
                        nor shared FID reference features (seconds_fid_ref), the two
                        other components seconds_setup is broken into where they exist.

SHARED PER REPEAT -- identical on EVERY row of the repeat, so read it ONCE and never
sum it across rows:

    seconds_ratio       the density-ratio half of the nuisance fit: the cross-fitting
                        fold assignment, the logging-density fit, the ratio pi / pi_0 it
                        yields and the weight diagnostics. This is the whole of what the
                        IPW estimator needs.
    seconds_cme         the conditional-mean-embedding half: the two kernel lengthscale
                        locks, the target-policy action draws the target Gram averages
                        over, the per-fold ridge selection and the single solve that
                        returns v_PI and the one-step correction. This is the whole of
                        what the plug-in estimator needs; DR needs both halves. The
                        solve hands back every requested vector at once, so the
                        sub-millisecond IPW rescaling v = w / n is clocked here too.
    seconds_nuisance    seconds_ratio + seconds_cme, under the name it has always had.
                        It is their SUM: never add the three together, and never add
                        the two halves to seconds_shared.
    seconds_eval_shared the scoring the estimators share and that is therefore done
                        once: the metric kernel, the logged / initial / floor /
                        floor_matched rows, and the exact-CDF grid and its
                        evaluation. NOTE: until 2026-10-03 this time was ADDED to
                        every row's seconds_eval, so an archived CSV's seconds_eval
                        is this quantity plus that row's own, and the two cannot be
                        separated after the fact. An archived single-estimator run
                        has only one row to charge it to, so its total is unaffected.
    seconds_shared      seconds_nuisance + seconds_eval_shared, as one convenience
                        column

PER ESTIMATOR ROW -- this row's own work, so sum freely across the rows of a repeat:

    seconds_flow        this estimator's flow iterations
    seconds_eval        this estimator's OWN scoring: the metrics of ITS flowed
                        sample against the oracle draw and against the exact law,
                        and nothing else
    seconds_sampling    DERIVED, and a term of NEITHER identity below: the isolated cost
                        of the proposed sampling method for this row, composed for its
                        estimator out of the columns above --
                            plugin -> seconds_cme + seconds_flow
                            ipw    -> seconds_ratio + seconds_flow
                            dr     -> seconds_ratio + seconds_cme + seconds_flow
                        (no decode term: the outcome is scalar and the flow runs in it).
                        It deliberately EXCLUDES all evaluation, which is measurement
                        rather than sampling, and the shared starting-particle draw,
                        which seconds_other carries. It reuses shared columns, so it
                        must never be summed down the rows of a repeat; an end-to-end
                        figure is seconds_setup plus seconds_repeat, not a sum of this.

TOTAL, and the named remainder:

    seconds_other       exactly the residual of the per-repeat identity below: the
                        starting particle draw, the numpy conversions, the row assembly,
                        the caller's per-row CSV rewrite between two yields, and
                        whatever the 0.1 s rounding of the parts leaves over. It can be
                        slightly negative when that rounding overshoots.
    seconds_repeat      wall clock of the WHOLE repeat, identical on every row of it.
                        NOT cumulative: it is measured once, when the repeat ends,
                        and written back into every row. Until 2026-10-03 it was the
                        elapsed time AT each row, which on a multi-estimator repeat
                        could be neither summed nor averaged.

The two identities a reader needs, both exact on the rounded numbers the CSV carries:

    seconds_repeat == seconds_shared
                      + sum over the estimator rows of the repeat of
                            (seconds_flow + seconds_eval)
                      + seconds_other                                     [per repeat]

    task total     == sum over data seeds of seconds_setup
                      + sum over repeats of seconds_repeat                [per task]

reading seconds_setup once per data seed and seconds_repeat once per repeat, never once
per row. What the task identity does not cover is named rather than hidden: process
start-up (the imports and, on CUDA, the context) and the per-row CSV rewrites between
repeats. The run prints both the accounted total and the process wall clock on its last
line, so the gap is visible rather than inferred.

Columns this design does NOT write: seconds_ae, seconds_encode, seconds_decode,
seconds_fid, seconds_fid_ref, seconds_fid_shared. The outcome is scalar, so there is no
autoencoder, no frozen encoder, no decoding step and no FID, and the convention is to
omit a phase a design lacks rather than write 0.0 for it.

Every clock is stopped through `phase_seconds`, which drains the device first. How much
that recovers was measured rather than assumed: the flow is the only phase with GPU
work, and it leaves just 0.09 to 0.14 ms in flight when it returns (RTX 4500 Ada, 2000
particles against 20000 target points, 200 and 2000 iterations), because
`adamw_riesz_flow` logs the loss on its LAST iteration and that `.item()` already blocks
on the stream. The drain is therefore a guard against that incidental property of the
flow's logging changing, not a correction of a measured error -- and it changes no number
the run computes, being a host-side wait. Everything else here is CPU/numpy, where it is
a no-op.

A repeat killed between two rows leaves its surviving rows with an EMPTY
seconds_repeat and seconds_other -- the total is not known until the last row is
scored -- while their per-estimator, shared and per-data-seed columns are final.

Hardware provenance: every row also carries gpu_name, hostname, slurm_job_id,
slurm_array_task_id, slurm_nodelist, torch_version and cuda_version, so a published
timing can be tied to the node that produced it. The SLURM fields are read from the
environment and are empty off the cluster.

Design: data_utils.warfarin. Estimator: causality_utils.cpme. Flow:
mmd_flow_utils. Data: data/warfarin (IWPC/PharmGKB; see its SOURCE.txt).
"""

import argparse
import csv
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

from causality_utils import (cpme_dr_weights, cpme_dr_weights_crossfit,
                            cpme_dr_weights_crossfit_nystrom,
                            cpme_dr_weights_crossfit_projected_nystrom,
                            cpme_dr_weights_nystrom,
                            cpme_dr_weights_projected_nystrom,
                            cpme_ipw_weights, lambda_n_half,
                            make_crossfit_fold_ids, select_regularization)
from data_utils import (interventional_cdf, load_warfarin_cohort,
                        sample_warfarin_data)
from dens_ratio_utils import (crossfit_logging_policy_log_prob,
                             fit_linear_gaussian_policy,
                             importance_weight_diagnostics)
from kernel_utils import RBF
from metric_utils import sliced_wasserstein
from mmd_flow_utils import adamw_riesz_flow, squared_mmd

#: Estimators of the target embedding, in the order their rows are written and
#: their filename token is spelled. See `target_weight_vectors`.
ESTIMATORS = ("plugin", "ipw", "dr")

COLUMNS = [
    "tag", "device", "target_policy", "estimator", "init", "solver", "dr_fit",
    "lambda_rule",
    "data_seed", "repeat", "pipeline_seed",
    "n_obs", "n_cohort", "n_particles_requested", "n_particles", "n_ref",
    "n_floor_matched", "n_landmarks", "n_folds", "n_resample", "n_anchor",
    "n_lambda_cv", "theta", "delta_scale", "delta", "tolerance",
    "outcome_noise", "max_dose", "flow_iterations", "learning_rate",
    "warmup_steps", "weight_clip", "ipw_form",
    "lambda_constant", "fold_lambdas", "lengthscale_x", "lengthscale_a",
    "oracle_w_ess", "est_w_ess", "est_w_max", "est_w_log_rmse",
    "sum_v", "sum_abs_v", "v_ess", "v_max_abs",
    "action_mean_logged", "action_mean_target", "action_sd_logged",
    "action_sd_target",
    "loss_initial", "loss_final", "mean_transport",
]
#: Distributional metrics are recorded for five samples, so the table carries its
#: own references: `logged` is the no-intervention baseline, `floor` an
#: independent oracle draw (the best any sampler could do at n_ref points),
#: `floor_matched` that same draw cut to the flow's own particle count (the best
#: any sampler could do at THIS row's count), and `initial` the particle cloud
#: before the flow.
METRIC_ROWS = ("logged", "initial", "flow", "floor", "floor_matched")
METRIC_NAMES = ("sw1", "sw2", "w1", "w2", "mmd2", "ks", "mean_err", "sd_err",
                "q05_err", "q10_err", "q50_err", "q90_err", "q95_err",
                "tail_err", "cvar95_err")
#: The same two distances taken against the EXACT interventional law rather than
#: a finite draw of it; see `exact_law_metrics`.
EXACT_METRIC_NAMES = ("ks_exact", "w1_exact", "w2_exact")
COLUMNS += [f"{name}_{row}" for row in METRIC_ROWS
            for name in METRIC_NAMES + EXACT_METRIC_NAMES]
#: Largest fraction of any scored row lying past the exact-CDF grid's upper edge.
#: The grid reaches where the law has 1e-6 of mass left, so this is 0 unless a
#: sampler left the law's support; a positive value means that row's own tail was
#: truncated and its `*_exact` columns understate the distance.
COLUMNS += ["exact_grid_overflow"]
#: Wall clock, in scope order: per DATA SEED first, then shared-per-repeat, then this
#: row's own, then the total and the residual that closes the per-repeat identity. See
#: the module docstring for which of them may be summed across the estimator rows of a
#: repeat and which may not. `seconds_ratio` / `seconds_cme` are a BREAKDOWN of
#: `seconds_nuisance` and `seconds_data` one of `seconds_setup`, so neither pair may be
#: added to the total alongside the column it decomposes; `seconds_sampling` is derived
#: from several of the others and belongs to no sum at all.
COLUMNS += ["seconds_data", "seconds_setup",
            "seconds_ratio", "seconds_cme", "seconds_nuisance",
            "seconds_eval_shared", "seconds_shared",
            "seconds_flow", "seconds_eval", "seconds_sampling",
            "seconds_other", "seconds_repeat"]
#: Hardware provenance, so a published timing table can be audited against the node that
#: produced it. See `hardware_provenance`.
COLUMNS += ["gpu_name", "hostname", "slurm_job_id", "slurm_array_task_id",
            "slurm_nodelist", "torch_version", "cuda_version"]

#: Which half of the nuisance fit each estimator's target embedding actually needs:
#: v_PI is the CME solve alone, v_IPW = w / n the density ratios alone, and
#: v_DR = v_PI + v_correction both. Used by `sampling_seconds`.
SAMPLING_NUISANCE = {"plugin": ("seconds_cme",),
                     "ipw": ("seconds_ratio",),
                     "dr": ("seconds_ratio", "seconds_cme")}

#: Random directions for the sliced Wasserstein columns. A scalar outcome makes every
#: projection +/-1, so the value cannot change the result; it is the repo default so
#: that the call is the same one the image experiments make.
SW_PROJECTIONS = 1000

#: Outcome value above which a patient counts as severely mis-dosed, in mg/week
#: of dosing error. Used for the tail-probability and CVaR columns.
SEVERE_MISDOSE = 40.0

#: Default cap on the estimated importance weights, and the value Appendix 14.1.4
#: uses. It is a numerical guard on the ratio pi / pi_0 -- the appendix clips the
#: WEIGHTS, not the dose -- and the default design has healthy overlap, so at 1e5
#: the clip is inactive; the `est_w_max` column shows by how far.
WEIGHT_CLIP = 1e5

#: Starting-particle row indices per (data_seed, repeat), exported alongside the
#: CSV so another implementation can reproduce the same initial cloud.
PARTICLE_INDICES: dict = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=20000,
                        help="logged observations; patients are resampled with "
                             "replacement above the cohort size")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="size of the oracle reference draw every row is scored "
                             "against, and of the second draw that is the floor")
    parser.add_argument("--n_particles", type=int, default=2000)
    parser.add_argument("--target_policy", type=str, default="mean_preserving",
                        choices=("mean_preserving", "shift", "shifted_mixture"),
                        help="'mean_preserving' is the headline setting: E[A|X] is "
                             "unchanged and only the shape of the action law moves")
    parser.add_argument("--delta_scale", type=float, default=1.0,
                        help="target shift delta as a multiple of the logging policy's "
                             "conditional scale sigma_T sqrt(1 - theta). 1.0 is "
                             "calibrated: the mean-preserving target then has sqrt(2) "
                             "times the conditional action spread at about 63%% "
                             "effective sample size (measured at n_obs=20000; the "
                             "shift target is nearer 37%%), and 2.0 collapses it to "
                             "about 3%%. "
                             "NOTE the target family is anchored on the LOGGING "
                             "policy's conditional mean, not on the appendix's linear "
                             "X^T w_base, because mean preservation has to hold "
                             "relative to pi_0 for the sampling framing to mean "
                             "anything; delta_scale = 1 / sqrt(1 - theta) (sqrt(2) at "
                             "theta = 0.5) reproduces the appendix's Delta = sigma*_T")
    parser.add_argument("--theta", type=float, default=0.5,
                        help="fraction of the logged dose's variance explained by BMI, "
                             "i.e. the strength of the confounding")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="half-width of the acceptable dosing band, as a fraction "
                             "of the therapeutic dose")
    parser.add_argument("--outcome_noise", type=float, default=0.1)
    parser.add_argument("--max_dose", type=float, default=None,
                        help="drop cohort patients whose recorded therapeutic dose "
                             "exceeds this, in mg/week. None -- the default, and the "
                             "paper -- keeps them: Appendix 14.1.4 filters on a stable "
                             "INR only and its 10^5 clip is on the importance weights, "
                             "not on the dose. Set it to inspect the implausible upper "
                             "tail (a handful of patients above 100 mg/week)")
    parser.add_argument("--estimators", type=str, nargs="+", default=["dr"],
                        choices=ESTIMATORS,
                        help="target embeddings to sample against, one CSV row each: "
                             "'plugin' the conditional mean embedding alone, 'ipw' the "
                             "density ratio alone, 'dr' their sum. All of them come "
                             "from one solve and are run off the same particles, so a "
                             "multi-estimator request is a PAIRED comparison. The "
                             "default is 'dr' alone, which leaves the run bitwise "
                             "what it was before the axis existed")
    parser.add_argument("--init", type=str, default="observed",
                        choices=("observed", "normal", "moment"),
                        help="initial particles: the logged outcomes, N(0, I), or the "
                             "empirical N(mu, Sigma) of the flow's target support")
    parser.add_argument("--solver", type=str, default="projected_nystrom",
                        choices=("auto", "exact", "nystrom", "projected_nystrom"),
                        help="how each fold's CME ridge system is solved. "
                             "'projected_nystrom' (the DEFAULT) tracks the exact "
                             "solve at every landmark budget measured: DR total mass "
                             "0.993 to 0.999 against an exact 0.998. 'exact' is that "
                             "reference, affordable at small n_obs. 'nystrom' is the "
                             "non-projected Woodbury solve, kept to reproduce archived "
                             "tables: its weights coarsen as the landmark budget and "
                             "the ridge shrink -- for every estimator alike, the dr row "
                             "included -- and asking for it prints a warning carrying "
                             "the measured masses. 'auto' solves exactly up to "
                             "--exact_max_n_obs and REFUSES above it rather than "
                             "exchange the nuisance estimator silently partway through "
                             "an n_obs sweep. "
                             "The DR support is EVERY logged row under all of them, "
                             "so what the choice changes is the nuisance estimate, not "
                             "the sample.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly"
                             "; above it 'auto' refuses instead of choosing a solver "
                             "for you")
    parser.add_argument("--n_landmarks", type=int, default=1024)
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"))
    parser.add_argument("--lambda_rule", type=str, default="cv", choices=("cv", "n_half"))
    parser.add_argument("--lambda_constant", type=float, default=1.0,
                        help="the constant c in --lambda_rule n_half")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_resample", type=int, default=8,
                        help="target-policy draws averaged into the target action kernel")
    parser.add_argument("--n_anchor", type=int, default=4000,
                        help="rows the context and action median heuristics are locked "
                             "on; the median over all n_obs^2 pairs is neither "
                             "affordable nor necessary")
    parser.add_argument("--n_lambda_cv", type=int, default=2000,
                        help="rows the ridge cross-validation runs on, per training "
                             "block; the full block is itself an n^3 solve")
    parser.add_argument("--weight_clip", type=float, default=WEIGHT_CLIP,
                        help="cap on the estimated importance weights before the DR "
                             "stage (<= 0 disables). The default is the appendix's 10^5 "
                             "and is inactive on a design with healthy overlap")
    parser.add_argument("--ipw_form", type=str, default="ht", choices=("ht", "hajek"),
                        help="normalisation of the IPW target embedding, recorded in "
                             "the ipw_form column. 'ht' (the default) is the "
                             "Horvitz-Thompson form v = w / n, the term that appears "
                             "inside v_DR and therefore the one that keeps the three "
                             "estimators algebraically comparable; it does NOT have "
                             "unit mass, so read the IPW row's sum_v before reading "
                             "its metrics -- part of that row's apparent "
                             "disadvantage is a normalisation artefact rather than "
                             "an estimation error. 'hajek' is the "
                             "self-normalised form v = w / sum(w), which sums to one "
                             "by construction but is a DIFFERENT estimator. Not a cap "
                             "on the ratios themselves: that is --weight_clip.")
    parser.add_argument("--flow_iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--particle_seed", type=int, default=None,
                        help="base seed for the STARTING PARTICLE draw alone, kept "
                             "independent of the rest of the pipeline so another "
                             "implementation can reproduce the same initial cloud. The "
                             "draw uses particle_seed + 1000 * data_seed + repeat on a "
                             "CPU generator; None keeps the derivation from the "
                             "pipeline seed. The chosen logged rows also go to "
                             "<csv stem>_particles.npz, which needs no RNG agreement at "
                             "all -- but those indices address THIS run's logged "
                             "sample, so they are portable only to a run that shares "
                             "the data draw")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument("--tag", type=str, default="sweep")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--warfarin_csv", type=str,
                        default=str(REPO_ROOT / "data" / "warfarin"
                                    / "iwpc_warfarin_subject_data.csv"))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_solver(args: argparse.Namespace) -> str:
    """Which ridge solver this configuration uses; see --solver.

    Above `--exact_max_n_obs` the exact solve is unaffordable and `auto` falls back to
    the PROJECTED Nystrom solver. It resolved to the non-projected one historically,
    which was a latent defect rather than a choice: that solver's weights are coarse at
    small m / n (DR total mass -2.71 at lambda = 1e-3, m / n = 0.125, where the exact
    solve gives 0.998), while the projected solver stays at 0.993 to 0.999 across the
    same grid. `auto` still changes estimator partway through an n_obs sweep, which is
    why the sweep launchers name the solver explicitly; the resolved choice is recorded
    in the `solver` CSV column either way.
    """
    if args.solver != "auto":
        return args.solver
    if args.n_obs <= args.exact_max_n_obs:
        return "exact"
    return "projected_nystrom"


def resolve_estimators(args: argparse.Namespace) -> tuple:
    """The requested target embeddings, deduplicated and in `ESTIMATORS` order.

    Canonical order rather than the order the flag was typed in, so neither the
    row order of a run nor the filename it writes depends on how the request was
    spelled.
    """
    asked = set(args.estimators)
    return tuple(name for name in ESTIMATORS if name in asked)


def warn_nonprojected_solver(solver: str, n_obs: int, n_landmarks: int) -> None:
    """
    Warn that the non-projected Nystrom solve is coarse at this landmark budget.

    The hazard is the SOLVER, and it is the same hazard for every estimator the run
    reports. `cpme_dr_weights_nystrom` and its cross-fitted twin apply an APPROXIMATE
    Gram inverse to an exact right-hand side: truncation sets the Gram eigenvalues
    outside the landmark span to zero, so the exact damping 1 / (mu + lambda) of those
    directions becomes 1 / lambda, and whatever part of the right-hand side lies there
    comes back divided by lambda n. That lands on the single DR vector the solver
    returns by default -- the vector every archived `--solver nystrom` table holds.

    Splitting that vector into (v_PI, v_correction) adds nothing to the problem. The
    Woodbury operator S(r) = (r - Phi (Phi^T Phi + lam I)^{-1} Phi^T r) / (lam n) is
    linear, so S(r_PI) + S(r_corr) = S(r_DR) identically; measured, the split agrees
    with the default single-vector return to 1.5e-12 across the configurations
    tabulated below.

    Total mass of the returned DR weights -- 1 for the embedding of a probability
    measure -- on an n = 800, d = 4 bandit with RBF Grams:

        lambda   m/n     exact   nystrom   projected
        1e-1     0.500   0.9973    0.9984      0.9977
        1e-1     0.125   0.9973    0.9654      0.9965
        1e-1     0.062   0.9973    1.0912      0.9930
        1e-2     0.500   0.9979    1.0006      0.9983
        1e-2     0.125   0.9979    0.6328      0.9968
        1e-2     0.062   0.9979    1.9764      0.9934
        1e-3     0.500   0.9982    1.0073      0.9988
        1e-3     0.125   0.9982   -2.7062      0.9968
        1e-3     0.062   0.9982   10.8053      0.9934

    A negative total mass is not the embedding of a probability measure. The damage
    grows as m / n and lambda shrink, and `projected_nystrom` carries none of it.

    Hence a warning keyed on the solver, and not a refusal keyed on the estimators. A
    dr-only run has to stay possible on every solver, being what reproduces the
    archived non-projected tables; and refusing the component split while admitting the
    DR row would block a computation no less accurate than the one it waves through.
    `--solver auto` never arrives here: `resolve_solver` refuses rather than switch
    silently, so the solver is always one the operator named, and the `solver` CSV
    column records which.

    Parameters
    ----------
    solver : str
        The resolved solver. Anything but 'nystrom' returns silently, as does a budget
        that spans every logged row, where the factorisation is exact.
    n_obs : int
        Logged rows. The CME support is a subset of them, so m / n_obs bounds the
        m / |support| the solve runs at from below.
    n_landmarks : int
        Landmarks per solve.
    """
    if solver != "nystrom" or n_landmarks >= n_obs:
        return
    print(f"WARNING: --solver nystrom applies an approximate Gram inverse to an exact "
          f"right-hand side, and its weights coarsen as the landmark budget and the "
          f"ridge shrink -- for EVERY estimator, the dr row included.\n"
          f"  Here m = {n_landmarks} landmarks against n_obs = {n_obs}, so the solve "
          f"runs at m / |support| >= {n_landmarks / n_obs:.3f}; the ridge is whatever "
          f"--lambda_rule returns, and the error grows as it shrinks.\n"
          f"  Measured DR total mass at m / n = 0.125: 0.63 at lambda 1e-2 and -2.71 "
          f"at 1e-3, against 0.998 exact and 0.997 projected.\n"
          f"  Use --solver projected_nystrom unless this run is reproducing an "
          f"archived non-projected table.", file=sys.stderr, flush=True)


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block.

    The base name carries the outcome model -- the policy family, n_obs, the
    particle count, delta_scale, theta, the tolerance and the outcome noise --
    and every other option contributes a token only when it is off its default.
    Both halves matter: `write_rows` rewrites the whole file, so two sweep cells
    sharing a name would silently overwrite one another, and a knob absent from
    the name is a knob that cannot be swept.
    """
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    estimators = resolve_estimators(args)
    if estimators != ("dr",):
        suffix += "_est" + "-".join(estimators)
    if args.dr_fit != "crossfit":
        suffix += f"_dr{args.dr_fit}"
    if args.lambda_rule != "cv":
        suffix += f"_lam{args.lambda_rule}"
    if args.lambda_constant != 1.0:
        suffix += f"_lc{args.lambda_constant:g}"
    # One token per explicitly requested solver, so an --solver sweep cannot have
    # two cells write the same filename. Keyed on args.solver, not the resolved
    # one, so a run that passed --solver auto keeps its own name.
    if args.solver == "projected_nystrom":
        suffix += "_projnys"
    elif args.solver == "exact":
        suffix += "_exact"
    elif args.solver == "nystrom":
        suffix += "_nys"
    if args.n_landmarks != 1024:
        suffix += f"_L{args.n_landmarks}"
    if args.n_folds != 5:
        suffix += f"_k{args.n_folds}"
    if args.n_resample != 8:
        suffix += f"_rs{args.n_resample}"
    if args.n_ref != 2000:
        suffix += f"_ref{args.n_ref}"
    if args.n_anchor != 4000:
        suffix += f"_anchor{args.n_anchor}"
    if args.n_lambda_cv != 2000:
        suffix += f"_cv{args.n_lambda_cv}"
    if args.flow_iterations != 2000:
        suffix += f"_it{args.flow_iterations}"
    if args.learning_rate != 0.05:
        suffix += f"_lr{args.learning_rate:g}"
    if args.warmup_steps != 100:
        suffix += f"_wu{args.warmup_steps}"
    if args.weight_clip != WEIGHT_CLIP:
        suffix += f"_clip{args.weight_clip:g}"
    if args.ipw_form != "ht":
        suffix += f"_ipw{args.ipw_form}"
    if args.max_dose is not None:
        suffix += f"_maxdose{args.max_dose:g}"
    if args.particle_seed is not None:
        suffix += f"_ps{args.particle_seed}"
    name = (f"warfarin_{args.tag}_{args.target_policy}_nobs{args.n_obs}"
            f"_M{args.n_particles}_d{args.delta_scale:g}_th{args.theta:g}"
            f"_tol{args.tolerance:g}_noise{args.outcome_noise:g}"
            f"_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def particle_generator(args: argparse.Namespace, seed: int) -> torch.Generator:
    """CPU generator for the starting-particle draw; see --particle_seed.

    Offset by 3 from the seed it is built on, which is the stream the pipeline
    has always used, so an unset --particle_seed reproduces earlier runs exactly.
    A CPU generator so the stream does not depend on the device.
    """
    base = seed if args.particle_seed is None else args.particle_seed + seed
    return torch.Generator().manual_seed(base + 3)


def gaussian_init(reference: torch.Tensor, n_particles: int, mode: str,
                  generator: torch.Generator) -> torch.Tensor:
    """Particles from N(0, I) or the reference sample's empirical N(mu, Sigma).

    The moment-matched draw uses the centred data matrix rather than a Cholesky
    factor, which is exact, needs no jitter, and stays defined when n < d.
    """
    n, dim = reference.shape
    if mode == "normal":
        return torch.randn(n_particles, dim, generator=generator,
                           dtype=reference.dtype)
    mean = reference.mean(dim=0, keepdim=True)
    noise = torch.randn(n_particles, n, generator=generator, dtype=reference.dtype)
    return mean + noise @ (reference - mean) / (n - 1) ** 0.5


def select_ridge(gram, outcomes: torch.Tensor, index: np.ndarray,
                 n_subsample: int) -> float:
    """
    Cross-validated ridge for one training block, on a bounded subsample of it.

    Parameters
    ----------
    gram : callable
        (rows, cols) -> the joint Gram block K_X * K_A of those logged rows.
    outcomes : torch.Tensor of shape (n, 1)
        Logged outcomes, the validation target of the selection.
    index : np.ndarray
        Rows of the training block the penalty may be selected on.
    n_subsample : int
        Cap on how many of them are used; the selection is an n^3 solve.

    Returns
    -------
    float
        The selected ridge. The subsample is seeded, so it is reproducible.
    """
    order = torch.randperm(len(index), generator=torch.Generator().manual_seed(0))
    pick = index[order[:n_subsample].numpy()]
    chosen, _ = select_regularization(gram(pick, pick), outcomes[pick, 0], seed=0)
    return chosen


def target_weight_vectors(solve, weights: torch.Tensor, estimators: tuple,
                          ipw_form: str) -> dict:
    """
    The requested target embeddings, all from ONE nuisance fit.

    Each estimator is a weight vector over the logged outcomes, and the three are
    algebraically one object: v_DR = v_PI + v_correction, while v_IPW = w / n is
    the leading term of that correction. Asking the solver for its two components
    therefore produces all three at the cost of a single solve -- which is what
    makes the comparison paired, and is also necessary rather than merely thrifty:
    the Nystrom landmark draw is stateful, so a second call would fit the CME on a
    different subspace and the plug-in row would no longer belong to the same
    nuisance as the DR row.

    Parameters
    ----------
    solve : callable
        `solve(components)` -> the configured solver's return: the DR weight vector
        when False, the pair (v_PI, v_correction) when True.
    weights : torch.Tensor of shape (n,)
        The (clipped) estimated importance weights the solver was given.
    estimators : tuple of str
        A subset of `ESTIMATORS`.
    ipw_form : {'ht', 'hajek'}
        Normalisation of the IPW vector; see --ipw_form. 'ht' is the w / n that
        appears inside v_DR, so it is the form under which the three rows are
        algebraically one object.

    Returns
    -------
    dict {str: torch.Tensor of shape (n,)}
        One weight vector per requested estimator, in the order given.

    Notes
    -----
    A request for the DR estimator alone takes the solver's single-vector path, so
    the default configuration is arithmetically untouched by the existence of the
    other two. Requested alongside a component estimator it becomes
    v_PI + v_correction, which agrees with that path to floating point rather than
    bitwise.
    """
    if estimators == ("dr",):
        return {"dr": solve(False)}
    v_plugin, v_correction = solve(True)
    available = {"plugin": v_plugin,
                 "ipw": cpme_ipw_weights(weights, normalize=ipw_form == "hajek"),
                 "dr": v_plugin + v_correction}
    return {name: available[name] for name in estimators}


def weight_diagnostics(v: torch.Tensor) -> dict:
    """
    Mass and concentration of one estimator's target weight vector.

    Recorded per estimator, not per repeat: the three put very different profiles on
    the same logged outcomes -- IPW carries the raw density-ratio tail undamped, the
    plug-in carries none of it, DR sits between -- and that difference is the finding,
    so it cannot be a shared column.

    These are the CANONICAL per-estimator diagnostics of this repository: the same
    functionals, under the same names, in the same units, in all five simulations, so an
    estimator table can be read across designs. Two details are load-bearing. `v_ess` is
    Kish's ratio of the vector ITSELF, (sum v)^2 / sum v^2 -- on a signed vector
    (sum |v|)^2 / sum v^2 is a different functional and not a rescaling of it. And
    `v_max_abs` is the raw largest atom, never multiplied by n.

    Parameters
    ----------
    v : torch.Tensor of shape (n,)
        The estimator's weight vector over the logged outcomes.

    Returns
    -------
    dict
        'sum_v' : sum_i v_i, the total mass. chi(pi) is the embedding of a probability
            measure, so 1 is where a well-behaved estimator lands; the
            Horvitz-Thompson IPW form w / n does not, which is what --ipw_form records.
        'sum_abs_v' : sum_i |v_i|, above 'sum_v' exactly when v is signed.
        'v_ess' : (sum v)^2 / sum v^2, scale invariant. An effective sample size on the
            non-negative IPW weights; on the SIGNED plug-in and DR vectors a
            concentration ratio rather than a count, which may exceed n.
        'v_max_abs' : max_i |v_i|, in the units of v itself, so 1 / n is the
            uniform-weight reference.
    """
    total = float(v.sum())
    return {"sum_v": total, "sum_abs_v": float(v.abs().sum()),
            "v_ess": total ** 2 / float((v ** 2).sum()),
            "v_max_abs": float(v.abs().max())}


def matched_subsample(sample: np.ndarray, n_rows: int, seed: int) -> np.ndarray:
    """A uniform random `n_rows`-element cut of a scalar sample, seeded.

    Returns `sample` unchanged when it is already at most that long, so the cut
    is a no-op whenever the flow runs with at least n_ref particles.
    """
    if len(sample) <= n_rows:
        return sample
    rng = np.random.default_rng(seed)
    return sample[rng.choice(len(sample), size=n_rows, replace=False)]


def distribution_metrics(sample: torch.Tensor, reference: torch.Tensor,
                         kernel) -> dict:
    """
    Distance between two scalar samples, reported several ways.

    A one-dimensional outcome makes the optimal-transport distances exact: the
    sorted-difference formula is W_p, not an approximation, so `w1`/`w2` need no
    projections. The quantile, tail and CVaR entries are reported because the
    mean-preserving intervention moves the body and the tail of the law by very
    different amounts, and a single aggregate would hide that.

    `ks` is the two-sample Kolmogorov-Smirnov statistic. Both empirical CDFs are
    evaluated on the pooled support, which is where the supremum of |F_a - F_b| is
    attained and which stays correct when the two samples share exact values --
    `outcome_noise = 0` gives the cost a point mass at zero, so ties are the norm
    rather than the exception on that design.
    """
    a = np.sort(np.asarray(sample, dtype=float).reshape(-1))
    b = np.sort(np.asarray(reference, dtype=float).reshape(-1))
    size = min(len(a), len(b))
    grid = (np.arange(size) + 0.5) / size
    qa, qb = np.quantile(a, grid), np.quantile(b, grid)

    pool = np.concatenate([a, b])
    ks = np.abs(np.searchsorted(a, pool, "right") / len(a)
                - np.searchsorted(b, pool, "right") / len(b)).max()
    tail_a, tail_b = np.mean(a > SEVERE_MISDOSE), np.mean(b > SEVERE_MISDOSE)
    cvar = lambda v: v[v >= np.quantile(v, 0.95)].mean()
    # The project reports SLICED Wasserstein throughout, so this design reports it too
    # and stays comparable with the image experiments. A scalar outcome makes the
    # projections +/-1, so SW_p here IS the exact one-dimensional W_p -- verified
    # identical to twelve decimals, and independent of both the projection count and
    # the seed. It is kept as its own column rather than aliased to `w1`/`w2` because
    # `sliced_wasserstein` truncates to the smaller sample by taking its FIRST rows
    # (the convention of Dance et al., matched bitwise elsewhere in this repo), where
    # the quantile form below interpolates over all of them: identical at matched
    # counts, slightly different when n_particles and n_ref disagree.
    sliced = lambda order: float(sliced_wasserstein(
        a[:, None], b[:, None], n_projections=SW_PROJECTIONS, p=order, seed=0))
    out = {"sw1": sliced(1),
           "sw2": sliced(2),
           "w1": float(np.abs(qa - qb).mean()),
           "w2": float(np.sqrt(((qa - qb) ** 2).mean())),
           "ks": float(ks),
           "mean_err": float(a.mean() - b.mean()),
           "sd_err": float(a.std() - b.std()),
           "tail_err": float(tail_a - tail_b),
           "cvar95_err": float(cvar(a) - cvar(b))}
    for q in (0.05, 0.10, 0.50, 0.90, 0.95):
        out[f"q{int(100 * q):02d}_err"] = float(np.quantile(a, q) - np.quantile(b, q))
    x = torch.as_tensor(a, dtype=torch.float64)[:, None]
    y = torch.as_tensor(b, dtype=torch.float64)[:, None]
    out["mmd2"] = float(squared_mmd(kernel, x, y))
    return out


def exact_law_upper_edge(cdf, tail_mass: float = 1e-6, start: float = 1.0,
                         growth: float = 1.25) -> float:
    """
    Where to stop integrating against the exact CDF: the first point of the
    geometric sequence `start * growth**k` past which the law keeps less than
    `tail_mass`.

    W1 against the exact law is an integral over the whole line, so an edge taken
    from the SAMPLE truncates it by whatever mass the law keeps beyond the
    largest draw. That amount varies from row to row, and it penalises a
    heavier-tailed sampler twice -- once inside the integral, once through the
    interval it removes. This edge is a property of the LAW alone, so it is the
    same for every row of a run and for every run of one design, and the
    truncation it leaves is below `tail_mass`.

    Parameters
    ----------
    cdf : callable
        (array of y) -> F(y) of the law being scored against.
    tail_mass : float
        Largest 1 - F(hi) the returned edge may leave.
    start, growth : float
        First point of the search and its multiplicative step.

    Returns
    -------
    float
    """
    hi = float(start)
    for _ in range(200):
        if 1.0 - float(cdf(np.array([hi]))[0]) < tail_mass:
            return hi
        hi *= growth
    raise RuntimeError("the exact CDF does not reach 1 - tail_mass below "
                       f"{hi:g}; check that the law matches the sample")


def exact_cdf_grid(values: np.ndarray, outcome_noise: float, upper: float,
                   n_body: int = 1200, n_atom: int = 241,
                   n_tail: int = 300) -> np.ndarray:
    """
    Points at which to evaluate the exact CDF, in three legs.

    The body is uniform over the range the samples occupy. The atom patch is
    dense around zero: every correctly dosed patient scores exactly zero, and the
    observation noise smears that atom over a few `outcome_noise` -- far narrower
    than the rest of the law, which spans tens of mg/week, so one uniform grid
    would either miss the atom or be needlessly large.

    The third leg continues coarsely out to `upper`, the law's own upper edge
    (see `exact_law_upper_edge`). It is needed because the design's therapeutic
    doses reach 315 mg/week while the action law is centred near the cohort mean,
    so a handful of patients carry the cost law out an order of magnitude past
    anything a finite sample reaches -- about 7e-4 of the mass here, which a
    sample-tied edge silently drops from W1. Over that stretch every empirical
    CDF is already 1 and only the law still moves, so few points resolve it, and
    keeping it a separate leg leaves the body's resolution untouched by how far
    the tail reaches.
    """
    lo = min(float(np.min(values)), 0.0) - 1.0
    body_hi = max(float(np.max(values)), lo + 1.0)

    pad = max(6.0 * outcome_noise, 1e-3)
    legs = [np.linspace(lo, body_hi, n_body), np.linspace(-pad, pad + 1.0, n_atom)]
    if float(upper) > body_hi:
        legs.append(np.linspace(body_hi, float(upper), n_tail))
    return np.unique(np.concatenate(legs))


def exact_law_metrics(sample: np.ndarray, grid: np.ndarray,
                      exact_cdf: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov and Wasserstein-1 distance of a scalar sample from the
    EXACT interventional CDF.

    Scored against a finite oracle draw, a sampler can come out ahead of the
    reference purely through the reference's own Monte Carlo error, which makes
    the comparison meaningless. The exact law has no such error. In one
    dimension W1 is the integral of the absolute CDF difference, so integrating
    it over `grid` is exact up to the quadrature; KS is evaluated at the sample
    points, where the empirical CDF attains its extremes, with the exact CDF
    interpolated onto them.
    """
    x = np.sort(np.asarray(sample, dtype=float).reshape(-1))
    size = len(x)
    at_sample = np.interp(x, grid, exact_cdf)
    steps = np.arange(1, size + 1) / size
    ks = max(np.abs(steps - at_sample).max(),
             np.abs(steps - 1.0 / size - at_sample).max())
    empirical = np.searchsorted(x, grid, "right") / size
    # W2 against the exact law, the one-sample analogue of the SW2 column. In one
    # dimension W2^2 is the integral of the squared quantile difference, so it is
    # taken on a uniform probability grid with the exact quantile function obtained by
    # inverting the tabulated CDF, rather than from the CDF difference that gives W1.
    levels = (np.arange(size) + 0.5) / size
    exact_quantiles = np.interp(levels, exact_cdf, grid)
    return {"ks_exact": float(ks),
            "w1_exact": float(np.trapezoid(np.abs(empirical - exact_cdf), grid)),
            "w2_exact": float(np.sqrt(((x - exact_quantiles) ** 2).mean()))}


def scored_metrics(samples: dict, reference: np.ndarray, kernel,
                   grid: np.ndarray, exact_cdf: np.ndarray) -> dict:
    """
    Both metric families for every named sample, as `<metric>_<name>` keys.

    Parameters
    ----------
    samples : dict {str: np.ndarray}
        Named scalar samples; the names become the `_<row>` half of the column, so
        they must be drawn from `METRIC_ROWS`.
    reference : np.ndarray
        The oracle draw the sample-based metrics are taken against.
    kernel : Kernel
        Fixed outcome kernel of the `mmd2` column.
    grid, exact_cdf : np.ndarray
        The exact interventional CDF, tabulated; see `exact_cdf_grid`.

    Returns
    -------
    dict
    """
    out = {}
    for name, sample in samples.items():
        for metric, value in distribution_metrics(sample, reference, kernel).items():
            out[f"{metric}_{name}"] = value
        for metric, value in exact_law_metrics(sample, grid, exact_cdf).items():
            out[f"{metric}_{name}"] = value
    return out


def grid_overflow(samples: dict, grid: np.ndarray) -> float:
    """Largest fraction of any of these samples lying past the grid's upper edge."""
    return max(float(np.mean(sample > grid[-1])) for sample in samples.values())


def check_columns(row: dict) -> None:
    """Fail before anything is written if the row and COLUMNS disagree.

    A column declared but never assigned is a silently empty CSV field, and a key
    assigned but never declared is a result that never reaches the file; both
    have to be loud.
    """
    missing = [name for name in COLUMNS if name not in row]
    extra = [name for name in row if name not in COLUMNS]
    if missing or extra:
        raise KeyError(f"row does not match COLUMNS; unassigned {missing}, "
                       f"undeclared {extra}")


def hardware_provenance(device: torch.device) -> dict:
    """
    The machine this run was measured on, as CSV columns.

    A wall-clock table is only auditable against someone else's numbers if the node is
    recorded with them: these columns say which GPU, which host and which SLURM
    allocation produced the row, and which torch/CUDA build ran it.

    Parameters
    ----------
    device : torch.device
        The device the run uses. `gpu_name` is empty off CUDA, where there is none.

    Returns
    -------
    dict
        The seven provenance columns. The SLURM entries are empty strings off the
        cluster, where the variables are simply absent.
    """
    return {
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else ""),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }


def sampling_seconds(estimator: str, shared: dict, seconds_flow: float) -> float:
    """
    Isolated wall clock of the proposed sampling method for one estimator row.

    The three estimators do not need the same nuisances -- the plug-in expansion needs
    the CME solve and no ratios, IPW the ratios and no solve, DR both -- so the honest
    cost of sampling against one of them is its own half of the nuisance fit plus its own
    flow. Representation and measurement are excluded by construction: there is no
    autoencoder here, and no evaluation column enters.

    Parameters
    ----------
    estimator : {'plugin', 'ipw', 'dr'}
    shared : dict
        The repeat's shared columns, already ROUNDED to 0.1 s, so that the result equals
        what a reader gets by adding the CSV's own numbers.
    seconds_flow : float
        This row's rounded `seconds_flow`.

    Returns
    -------
    float
        Seconds, rounded to 0.1 s.
    """
    parts = [shared[column] for column in SAMPLING_NUISANCE[estimator]]
    return round(sum(parts) + seconds_flow, 1)


def drain(device: torch.device) -> None:
    """Block until the device has finished the work queued on it.

    A no-op off CUDA. Waiting changes nothing the run computes -- it only delays the
    host -- so it is safe on a results path.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def phase_seconds(t0: float, device: torch.device) -> float:
    """
    Wall clock since `t0`, with queued device work drained first.

    A CUDA kernel launch returns to Python before the kernel runs, so a bare
    `time.time()` around a GPU phase stops the clock at the LAUNCH and silently
    charges the unfinished work to whichever later phase first reads a device tensor
    -- here, the `.cpu()` inside the evaluation. Draining first puts the time where it
    was spent.

    On this design the flow is the only phase with outstanding device work, and it
    turns out to have almost none: measured, the drain after it recovers 0.09 to
    0.14 ms, because the flow logs its loss on the last iteration and that `.item()`
    blocks on the stream already. So this is a guard, not a repair -- the flow's
    logging is free to change -- and it is applied to every phase, the CPU ones
    included, so that moving one of them to the GPU later cannot mis-time it
    unnoticed.
    """
    drain(device)
    return time.time() - t0


def run_repeat(cohort: dict, data: dict, reference: np.ndarray,
               floor_draw: np.ndarray, setup: dict, args: argparse.Namespace,
               data_seed: int, repeat: int, device: torch.device):
    """One pipeline pass on the data seed's logged sample; one row per estimator.

    The logged sample and the two oracle draws are arguments rather than draws of
    this function, so that repeats of one data seed differ only in the pipeline;
    see the module docstring's seed scheme. `setup` carries that seed's per-DATA-SEED
    timing columns, measured in `main` outside this repeat's clock and copied onto every
    row unchanged.

    Everything up to and including the ridge solve is shared by the requested
    estimators, as are the starting particles, the reference and floor draws, the
    exact-CDF grid and the flow budget; only the target weight vector differs. The
    rows are yielded as each flow finishes so that a timeout keeps the ones
    already scored.

    That incremental yield is why `seconds_repeat` and `seconds_other` are filled in
    only after the loop: the repeat's total is not a property of any one row, so it is
    measured once at the end and written back into every row of the repeat -- which
    this function can still do, holding references to the rows it has yielded. The
    caller therefore has to rewrite the CSV after the generator is exhausted; until it
    does, the rows it already wrote carry those two fields empty. See the module
    docstring's wall-clock section for the whole convention.

    Yields
    ------
    dict
        One row per entry of `resolve_estimators`, in that order.
    """
    seed = 1000 * data_seed + repeat
    estimators = resolve_estimators(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # The repeat clock. Read once, at the end of the generator body, so every row of
    # the repeat carries the same total; see the module docstring.
    started = time.time()
    produced = []

    X = data["X"].numpy()
    A = data["A_logged"].numpy()
    Y = data["Y_logged"].double()
    rows = data["row_index"].numpy()[:, None].astype(float)

    # -- the ratio half of the nuisance fit: the logging density is fitted, the target
    # policy is known. This is all the IPW estimator needs; the fold assignment is
    # clocked here because this half consumes it first, and the CME half reuses it.
    t0 = time.time()
    # Folds are assigned over the COHORT and read off by patient: above the cohort
    # size the logged sample repeats patients, and a row-level split would hand a
    # fold rows of a patient its nuisances were fitted on.
    cohort_folds = make_crossfit_fold_ids(len(cohort["ther_dose"]),
                                          n_folds=args.n_folds, seed=0)
    fold_ids = cohort_folds[data["row_index"].numpy()]
    if args.dr_fit == "nosplit":
        logging_hat = fit_linear_gaussian_policy(X, A)
        log_pi0 = logging_hat.log_prob(X, A)
    else:
        log_pi0 = crossfit_logging_policy_log_prob(X, A, fold_ids=fold_ids)
    log_ratio = data["target_policy"].log_prob(rows, A) - log_pi0
    w_hat = np.exp(log_ratio)
    if args.weight_clip is not None and args.weight_clip > 0:
        w_hat = np.minimum(w_hat, args.weight_clip)
    oracle_w = data["importance_weights"].numpy()
    diagnostics = importance_weight_diagnostics(w_hat, oracle_w)
    oracle_diagnostics = importance_weight_diagnostics(oracle_w)
    seconds_ratio = phase_seconds(t0, device)

    # -- the CME half: the target embedding's kernel ridge solve over the logged
    # outcomes. This is all the plug-in estimator needs, and DR needs both halves.
    t0 = time.time()
    solver = resolve_solver(args)
    kernel_x = RBF(use_length_scale_heuristic=True)
    kernel_a = RBF(use_length_scale_heuristic=True)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    A_t = torch.as_tensor(A, dtype=torch.float64)
    anchor = torch.randperm(
        args.n_obs, generator=torch.Generator().manual_seed(0))[:args.n_anchor]
    kernel_x.calculate_and_fix_length_scale(X_t[anchor])
    kernel_a.calculate_and_fix_length_scale(A_t[anchor])

    # The target policy is stochastic, so the target action kernel is the policy
    # average E_{a ~ pi(.|x_j)} k_A(a_i, a); approximated by the same fixed set of
    # draws for every block so the Gram is consistent across calls.
    rng = np.random.default_rng(seed + 11)
    draws = [data["target_policy"].sample(rows, rng) for _ in range(args.n_resample)]
    draws = [torch.as_tensor(d, dtype=torch.float64) for d in draws]

    gram_x = lambda r, c: kernel_x(X_t[r], X_t[c])
    gram = lambda r, c: gram_x(r, c) * kernel_a(A_t[r], A_t[c])
    def gram_pi(r, c):
        total = sum(kernel_a(A_t[r], draw[c]) for draw in draws) / len(draws)
        return gram_x(r, c) * total

    # The penalty of a cross-fitted fold is selected on that fold's TRAINING block
    # only: one global choice would be tuned partly on the very rows it is then
    # applied to out of fold. The no-split fit has a single training block.
    fold_lambdas: dict = {}
    nosplit_lambda = None
    if args.lambda_rule == "n_half":
        nosplit_lambda = lambda_n_half(args.n_obs, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    elif args.dr_fit == "nosplit":
        nosplit_lambda = select_ridge(gram, Y, np.arange(args.n_obs),
                                      args.n_lambda_cv)
    else:
        for fold in np.unique(fold_ids):
            fold_lambdas[int(fold)] = select_ridge(gram, Y,
                                                   np.where(fold_ids != fold)[0],
                                                   args.n_lambda_cv)

    # The solver is bound to its arguments first and called once, so every
    # requested estimator reads the same CME fit and the same landmark draw.
    weights = torch.as_tensor(w_hat, dtype=torch.float64)
    landmark_generator = torch.Generator().manual_seed(seed)
    if solver in ("nystrom", "projected_nystrom"):
        projected = solver == "projected_nystrom"
        if args.dr_fit == "nosplit":
            nystrom = (cpme_dr_weights_projected_nystrom if projected
                       else cpme_dr_weights_nystrom)
            solve = lambda components: nystrom(
                gram, gram_pi, weights, nosplit_lambda,
                n_landmarks=args.n_landmarks, generator=landmark_generator,
                components=components)
        else:
            nystrom = (cpme_dr_weights_crossfit_projected_nystrom if projected
                       else cpme_dr_weights_crossfit_nystrom)
            solve = lambda components: nystrom(
                gram, gram_pi, weights, fold_ids, fold_lambdas,
                n_landmarks=args.n_landmarks, generator=landmark_generator,
                components=components)
    else:
        index = np.arange(args.n_obs)
        K_x, K_a, K_pi = gram_x(index, index), kernel_a(A_t, A_t), gram_pi(index, index)
        K_a_target = K_pi / K_x.clamp_min(1e-300)
        if args.dr_fit == "nosplit":
            solve = lambda components: cpme_dr_weights(
                K_x, K_a, K_a_target, weights, nosplit_lambda,
                components=components)
        else:
            solve = lambda components: cpme_dr_weights_crossfit(
                K_x, K_a, K_a_target, weights, fold_ids, fold_lambdas,
                components=components)
    weight_vectors = target_weight_vectors(solve, weights, estimators, args.ipw_form)
    seconds_cme = phase_seconds(t0, device)

    # -- starting particles, shared by every estimator --------------------------------
    flow_target = Y.to(device)
    n_particles = min(args.n_particles, args.n_obs)
    generator = particle_generator(args, seed)
    if args.init == "observed":
        pick = torch.randperm(args.n_obs, generator=generator)[:n_particles]
        PARTICLE_INDICES[f"ds{data_seed}_r{repeat}"] = pick.cpu().numpy()
        initial = flow_target[pick.to(device)].clone()
    else:
        initial = gaussian_init(Y, n_particles, args.init, generator).to(device)

    # -- evaluation scaffolding, also shared ------------------------------------------
    # The particle draw above left a device copy in flight; drain it before the
    # evaluation clock starts, so its tail is charged to the unattributed remainder
    # rather than to the scoring.
    drain(device)
    t0 = time.time()
    metric_kernel = RBF(use_length_scale_heuristic=True)
    metric_kernel.calculate_and_fix_length_scale(
        torch.as_tensor(reference, dtype=torch.float64)[:, None])
    # Every row but `flow` is estimator-independent, so these are scored once and
    # copied into each estimator's row: the three are being compared against
    # identical baselines, not against three draws of them.
    fixed_samples = {
        "logged": Y.numpy().ravel()[:n_particles],
        "initial": initial.detach().cpu().numpy().ravel(),
        "floor": floor_draw,
        "floor_matched": matched_subsample(floor_draw, n_particles,
                                           300000 + 1000 * data_seed)}

    # The rows are also scored against the closed-form law, which carries no Monte
    # Carlo error of its own. One CDF evaluation is shared by every row.
    law_cdf = lambda y: interventional_cdf(cohort, y, target=args.target_policy,
                                           theta=args.theta,
                                           delta_scale=args.delta_scale,
                                           tolerance=args.tolerance,
                                           outcome_noise=args.outcome_noise)
    # The grid spans the DATA-FIXED rows only -- the logged sample, the oracle
    # reference and the floor draw. Including the flow and initial clouds would let
    # the sampler's own output move the grid, and with it the logged and floor
    # scores that are meant to be fixed baselines for the data seed. With several
    # estimators that would be worse still: each would score the shared baselines
    # on its own grid and the comparison would no longer be against one yardstick.
    fixed = np.concatenate([fixed_samples["logged"], fixed_samples["floor"],
                            reference])
    grid = exact_cdf_grid(fixed, args.outcome_noise, exact_law_upper_edge(law_cdf))
    exact_cdf = law_cdf(grid)
    fixed_metrics = scored_metrics(fixed_samples, reference, metric_kernel, grid,
                                   exact_cdf)
    # A row reaching past the grid would have its own tail silently truncated, so
    # the overshoot is recorded rather than assumed away.
    fixed_overflow = grid_overflow(fixed_samples, grid)
    seconds_eval_shared = phase_seconds(t0, device)

    # Round every shared part FIRST, then build the sums from the rounded values, so both
    # identities hold on the numbers the CSV carries and not merely on the raw floats
    # behind them.
    seconds_ratio = round(seconds_ratio, 1)
    seconds_cme = round(seconds_cme, 1)
    seconds_nuisance = round(seconds_ratio + seconds_cme, 1)
    seconds_eval_shared = round(seconds_eval_shared, 1)
    seconds_shared = round(seconds_nuisance + seconds_eval_shared, 1)

    action_logged = A.ravel()
    action_target = data["A_target"].numpy().ravel()
    constants = {
        "tag": args.tag, "device": str(device), "target_policy": args.target_policy,
        "init": args.init, "solver": solver, "dr_fit": args.dr_fit,
        "lambda_rule": args.lambda_rule, "data_seed": data_seed, "repeat": repeat,
        "pipeline_seed": seed, "n_obs": args.n_obs,
        "n_cohort": len(cohort["ther_dose"]),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "n_ref": args.n_ref,
        "n_floor_matched": int(len(fixed_samples["floor_matched"])),
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "n_folds": args.n_folds, "n_resample": args.n_resample,
        "n_anchor": args.n_anchor, "n_lambda_cv": args.n_lambda_cv,
        "theta": args.theta,
        "delta_scale": args.delta_scale, "delta": data["policy_info"]["delta"],
        "tolerance": args.tolerance, "outcome_noise": args.outcome_noise,
        "max_dose": args.max_dose if args.max_dose is not None else float("inf"),
        "flow_iterations": args.flow_iterations, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_clip": (args.weight_clip if args.weight_clip is not None
                        and args.weight_clip > 0 else float("inf")),
        "ipw_form": args.ipw_form,
        "lambda_constant": args.lambda_constant,
        "fold_lambdas": (f"all:{nosplit_lambda:g}" if args.dr_fit == "nosplit"
                         else ";".join(f"{fold}:{lam:g}" for fold, lam
                                       in sorted(fold_lambdas.items()))),
        "lengthscale_x": float(kernel_x.length_scale),
        "lengthscale_a": float(kernel_a.length_scale),
        "oracle_w_ess": oracle_diagnostics["ess"], "est_w_ess": diagnostics["ess"],
        "est_w_max": diagnostics["max"], "est_w_log_rmse": diagnostics.get("log_rmse"),
        "action_mean_logged": float(action_logged.mean()),
        "action_mean_target": float(action_target.mean()),
        "action_sd_logged": float(action_logged.std()),
        "action_sd_target": float(action_target.std()),
        # Shared by every estimator row of this repeat, so they are part of the constants
        # and must NOT be summed across rows. seconds_nuisance is the SUM of its two
        # halves, so a reader adding the breakdown gets the column back exactly and must
        # not add all three.
        "seconds_ratio": seconds_ratio, "seconds_cme": seconds_cme,
        "seconds_nuisance": seconds_nuisance,
        "seconds_eval_shared": seconds_eval_shared,
        "seconds_shared": seconds_shared,
        # Identical on every row of every repeat of this data seed, and measured outside
        # this repeat's clock: it is the second term of the per-task identity.
        **setup,
        **hardware_provenance(device),
    }

    # -- one flow per estimator, off the shared particles and budget -------------------
    for estimator, v in weight_vectors.items():
        t0 = time.time()
        torch.manual_seed(seed + 1)
        flowed, info = adamw_riesz_flow(
            initial.clone(), flow_target, target_weights=v.to(device), sliced=False,
            n_iterations=args.flow_iterations, learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps, lr_schedule="cosine", verbose=False)
        seconds_flow = phase_seconds(t0, device)

        t0 = time.time()
        flow_sample = {"flow": flowed.detach().cpu().numpy().ravel()}
        metrics = dict(fixed_metrics)
        metrics.update(scored_metrics(flow_sample, reference, metric_kernel, grid,
                                      exact_cdf))
        metrics["exact_grid_overflow"] = max(fixed_overflow,
                                             grid_overflow(flow_sample, grid))
        seconds_eval = phase_seconds(t0, device)

        row = dict(constants)
        row["estimator"] = estimator
        row.update(weight_diagnostics(v))
        row.update({
            # Each estimator's loss is the distance to its OWN target embedding, so
            # these two are a convergence check within a row and not a ranking
            # across rows; the distances to the oracle draw and to the exact law
            # are what compare the estimators.
            "loss_initial": float(info["loss"][0]),
            "loss_final": float(info["loss"][-1]),
            "mean_transport": float((flowed - initial).norm(dim=1).mean()),
            "seconds_flow": round(seconds_flow, 1),
            # This row's OWN scoring alone. The shared scaffolding is in
            # seconds_eval_shared, so that a three-estimator repeat charges it once
            # instead of three times and these two columns can be summed over rows.
            "seconds_eval": round(seconds_eval, 1),
            # Derived, and in neither identity: this estimator's own half of the nuisance
            # fit plus its own flow, with no representation and no measurement in it.
            "seconds_sampling": sampling_seconds(estimator, constants,
                                                 round(seconds_flow, 1)),
            # Properties of the REPEAT, not of the row, so they are filled in below
            # once the last flow is scored. They stay empty on the rows of a repeat
            # that never reached its end.
            "seconds_other": "",
            "seconds_repeat": "",
        })
        row.update(metrics)
        check_columns(row)
        produced.append(row)
        yield row

    # -- the repeat's total, written back into every row of it ------------------------
    # Measured once, here, so it is a constant of the repeat rather than a cumulative
    # count down the rows. `seconds_other` is taken from the ROUNDED parts, so the
    # identity in the module docstring closes exactly on the numbers in the file and
    # not merely on raw floats a reader never sees; it collects the starting-particle
    # draw, the row bookkeeping and the caller's per-row CSV rewrite between yields.
    seconds_repeat = round(phase_seconds(started, device), 1)
    seconds_own = round(sum(row["seconds_flow"] + row["seconds_eval"]
                            for row in produced), 1)
    # The trailing `+ 0.0` is not a no-op: it turns the rounding's -0.0 into 0.0, so a
    # residual that vanishes does not reach the CSV as a negative zero.
    seconds_other = round(seconds_repeat - seconds_shared - seconds_own, 1) + 0.0
    for row in produced:
        row["seconds_repeat"] = seconds_repeat
        row["seconds_other"] = seconds_other


def write_rows(path: Path, rows: list) -> None:
    """Atomic-ish rewrite after every row, so a timeout keeps partial results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_particle_indices(path: Path) -> None:
    """Export the logged rows used as starting particles, for reuse elsewhere."""
    if PARTICLE_INDICES:
        np.savez(path.with_name(path.stem + "_particles.npz"), **PARTICLE_INDICES)


def draw_target_outcomes(cohort: dict, args: argparse.Namespace,
                         seed: int) -> np.ndarray:
    """One oracle draw from the interventional outcome law, as a flat array."""
    return sample_warfarin_data(cohort, n_samples=args.n_ref, theta=args.theta,
                                delta_scale=args.delta_scale,
                                target=args.target_policy,
                                tolerance=args.tolerance,
                                outcome_noise=args.outcome_noise,
                                seed=seed)["Y_target"].numpy().ravel()


def main() -> None:
    args = parse_args()
    warn_nonprojected_solver(resolve_solver(args), args.n_obs, args.n_landmarks)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    csv_file = Path(args.warfarin_csv)
    if not csv_file.exists():
        sys.exit(f"warfarin table not found at {csv_file}; see data/warfarin/SOURCE.txt")
    # The per-DATA-SEED setup clock. It starts HERE so that the once-per-process cohort
    # read is charged to the first data seed's seconds_data rather than to nothing at all;
    # it is restarted at the end of each seed's repeats for the next seed. See the module
    # docstring's wall-clock section.
    t_process = time.time()
    t_setup = t_process
    cohort = load_warfarin_cohort(str(csv_file), max_dose=args.max_dose)
    out_path = csv_path(args)
    print(f"device: {device}   cohort: {len(cohort['ther_dose'])} patients, "
          f"{cohort['X'].shape[1]} covariates", flush=True)
    print(f"output: {out_path}", flush=True)

    rows = []
    accounted = 0.0
    for data_seed in args.data_seeds:
        # One logged sample and one pair of oracle draws per DATA seed, so the
        # repeats of a seed vary only in the pipeline; see the seed scheme. The
        # draw seed is the first repeat's pipeline seed, which is what every
        # single-repeat run has always used.
        draw_seed = 1000 * data_seed
        data = sample_warfarin_data(cohort, n_samples=args.n_obs, theta=args.theta,
                                    delta_scale=args.delta_scale,
                                    target=args.target_policy,
                                    tolerance=args.tolerance,
                                    outcome_noise=args.outcome_noise,
                                    seed=draw_seed)
        reference = draw_target_outcomes(cohort, args, 100000 + draw_seed)
        floor_draw = draw_target_outcomes(cohort, args, 200000 + draw_seed)
        # This seed's whole per-data-seed tier: data preparation and nothing else here, so
        # seconds_setup equals seconds_data. Both sit outside every seconds_repeat.
        seconds_data = round(phase_seconds(t_setup, device), 1)
        setup = {"seconds_data": seconds_data, "seconds_setup": seconds_data}
        accounted += seconds_data
        for repeat in range(args.n_repeats):
            for row in run_repeat(cohort, data, reference, floor_draw, setup, args,
                                  data_seed, repeat, device):
                rows.append(row)
                write_rows(out_path, rows)
                write_particle_indices(out_path)
                # This row's own cost, not the repeat's: the repeat total is only
                # known when its last flow is scored, and is printed below.
                print(f"[seed {data_seed} repeat {repeat} {row['estimator']}] "
                      f"W1 logged {row['w1_logged']:.3f} -> flow {row['w1_flow']:.3f} "
                      f"(floor {row['w1_floor']:.3f}, "
                      f"matched {row['w1_floor_matched']:.3f})   "
                      f"exact W1 {row['w1_exact_logged']:.3f} -> "
                      f"{row['w1_exact_flow']:.3f} "
                      f"(floor {row['w1_exact_floor']:.3f}, "
                      f"matched {row['w1_exact_floor_matched']:.3f})   "
                      f"tail err {row['tail_err_flow']:+.4f}   "
                      f"{row['seconds_flow'] + row['seconds_eval']:.0f}s own",
                      flush=True)
            # `run_repeat` fills seconds_repeat and seconds_other into every row of the
            # repeat only after its last flow, so the file needs one more write. The
            # per-row writes above are the crash-safety net and leave those two empty.
            write_rows(out_path, rows)
            repeat_rows = rows[-len(resolve_estimators(args)):]
            own = sum(row["seconds_flow"] + row["seconds_eval"]
                      for row in repeat_rows)
            last = repeat_rows[-1]
            accounted += last["seconds_repeat"]
            print(f"[seed {data_seed} repeat {repeat}] "
                  f"{last['seconds_repeat']:.1f}s = shared {last['seconds_shared']:.1f}"
                  f" (nuisance {last['seconds_nuisance']:.1f}"
                  f" = ratio {last['seconds_ratio']:.1f}"
                  f" + cme {last['seconds_cme']:.1f}"
                  f", eval {last['seconds_eval_shared']:.1f})"
                  f" + own {own:.1f} over {len(repeat_rows)} row(s)"
                  f" + other {last['seconds_other']:.1f}"
                  f"   [setup {last['seconds_setup']:.1f}s this seed]", flush=True)
        # The next data seed's setup clock starts once this seed's repeats are done.
        t_setup = time.time()
    # The per-task identity, printed so the gap it does not cover is visible rather than
    # inferred: process start-up and the per-row CSV rewrites between repeats.
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)
    print(f"accounted {round(accounted, 1)}s = sum(seconds_setup) over data seeds "
          f"+ sum(seconds_repeat) over repeats, of "
          f"{phase_seconds(t_process, device):.1f}s process wall clock", flush=True)


if __name__ == "__main__":
    main()
