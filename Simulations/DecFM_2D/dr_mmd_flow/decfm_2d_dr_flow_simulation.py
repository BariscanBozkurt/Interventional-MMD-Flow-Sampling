"""
DecFM 2D mixture simulation: cross-fitted DR-CPME target, RBF-Euler and Riesz-AdamW flows.

Script version of the twin notebooks
`NotebookExamples/Interventional_MMD_Flow/DecFM_2D_Mixture_RBF_MMD_Flow_Euler.ipynb` and
`DecFM_2D_Mixture_Riesz_MMD_Flow_AdamW.ipynb`, which share the design (Dance et al.,
2026, Appendix B.2.2: Y = X + mixture noise, so both interventional laws equal the
marginal P_Y), the cross-fitted logistic propensity, and the DR weight fitting, and
differ only in the sampler. Each repeat therefore fits the nuisances once and runs
BOTH samplers on BOTH arms from the SAME per-arm initial particles — a paired
comparison of forward-Euler RBF flow against AdamW Riesz flow at every configuration.

Everything runs on CPU in float64 — there is no GPU code in this experiment, and the
script has been validated with CUDA hidden.

One CSV row per (data_seed, repeat, arm, sampler, estimator): DGP constants, weight
diagnostics (the ratio's on-arm ESS, and the row estimator's own weight sum, absolute
sum, Kish ESS, largest atom and on-arm ESS), per-fold lambdas, kernel lengthscales,
the sampler's own step/learning-rate, mean transport, and SW1 / SW2 / unbiased RBF
MMD^2 against the
oracle marginal for the observed arm and the flowed particles, with the
oracle-vs-oracle floor alongside.

Wall-clock accounting
---------------------
Every `seconds_*` column is wall clock (time.time(), rounded to 0.1 s) and belongs to
exactly ONE of three scopes. Keeping them apart is what makes them addable now that a
repeat writes many rows instead of one: a cost measured once and written onto every row
is multiplied by any sum down those rows, and a cost measured outside the repeat clock is
lost by any sum of repeats.

PER DATA SEED -- identical on every row of every repeat of that data seed, and measured
OUTSIDE seconds_repeat, because the work happens once before the repeat loop:

    seconds_data        this data seed's data generation: the logged draw and the fixed
                        2 x --n_ref oracle draw its halves are scored against. This
                        design loads nothing from disk and holds no model, so there is no
                        once-per-process cost to charge to the first seed.
    seconds_setup       the per-data-seed TOTAL, and the column to add once per data
                        seed. Here it is seconds_data alone: there is no frozen-encoder
                        pass (seconds_encode) and no shared FID reference feature
                        (seconds_fid_ref), the two other components seconds_setup is
                        broken into where a design has them.

SHARED PER REPEAT -- identical on EVERY row of the repeat, so take it ONCE and never
sum it across rows:

    seconds_ratio       the density-ratio half of the nuisance fit: the DR support
                        selection, the cross-fitting folds, the logistic propensity fit
                        and each arm's importance ratios w = 1{a_i = a} / e(a | x_i).
                        This is the whole of what the IPW estimator needs. The support
                        and the folds are clocked here because this half consumes them
                        first; the CME half reuses both.
    seconds_cme         the conditional-mean-embedding half: the context-kernel median
                        lock, the Gram blocks, the per-fold ridge selection and each
                        arm's target action-kernel block with the CME ridge solve that
                        turns it into the weight vectors. This is the whole of what the
                        plug-in estimator needs; DR needs both halves. One solve serves
                        every estimator and sampler of the arm and hands back all of its
                        vectors at once, so the sub-millisecond IPW rescaling v = w / n
                        is clocked here too.
    seconds_nuisance    seconds_ratio + seconds_cme, under the name it has always had.
                        It is their SUM: never add the three together, and never add the
                        two halves to seconds_shared. NOTE: until 2026-10-03 this column
                        stopped at the lambda selection, so an archived CSV's
                        seconds_nuisance is a smaller quantity and the per-arm solve sat
                        in the unattributed remainder.
    seconds_eval_shared the scoring that is independent of the estimator and the sampler
                        and is therefore done once: the outcome kernel's median lock, the
                        oracle-vs-oracle floor, its count-matched twin, and each arm's
                        initial-cloud metrics (accumulated over the two arms)
    seconds_shared      seconds_nuisance + seconds_eval_shared, as one convenience
                        column

PER ROW -- this row's own work, so sum freely across rows:

    seconds_flow        this (arm, sampler, estimator) row's flow iterations
    seconds_eval        this row's OWN scoring: the flowed cloud's SW1 / SW2 / MMD^2
                        against the oracle reference, and nothing else
    seconds_sampling    DERIVED, and a term of NEITHER identity below: the isolated cost
                        of the proposed sampling method for this row, composed for its
                        estimator out of the columns above --
                            plugin -> seconds_cme + seconds_flow
                            ipw    -> seconds_ratio + seconds_flow
                            dr     -> seconds_ratio + seconds_cme + seconds_flow
                        (no decode term: the outcome is 2-d and the flow runs in it).
                        It deliberately EXCLUDES all evaluation, which is measurement
                        rather than sampling, and the shared starting-particle draw,
                        which seconds_other carries. Two caveats, both consequences of
                        the shared columns it is built from: the nuisance halves are
                        repeat totals over BOTH arms, so run one arm for one arm's
                        figure; and the outcome-kernel median that the RBF-Euler step
                        size reads is locked inside seconds_eval_shared, so it is not
                        counted here. It reuses shared columns, so it must never be
                        summed down the rows; an end-to-end figure is seconds_setup plus
                        seconds_repeat, not a sum of this.

TOTAL, and the named remainder:

    seconds_other       exactly the residual of the per-repeat identity below -- the
                        starting-particle draw, the weight diagnostics and the row
                        bookkeeping, plus whatever the 0.1 s rounding of the parts
                        leaves over. A per-repeat constant like the shared columns,
                        so take it ONCE and never sum it across rows. It can be
                        slightly negative when that rounding overshoots.
    seconds_repeat      wall clock of the WHOLE repeat, identical on every row of it.
                        Not cumulative: it is measured once, when the repeat ends.

The two identities a reader needs, both exact on the rounded numbers the CSV carries:

    seconds_repeat == seconds_shared
                      + sum over ALL rows of the repeat of
                            (seconds_flow + seconds_eval)
                      + seconds_other                                     [per repeat]

    task total     == sum over data seeds of seconds_setup
                      + sum over repeats of seconds_repeat                [per task]

reading seconds_setup once per data seed and seconds_repeat once per repeat, never once
per row. What the task identity does not cover is named rather than hidden: process
start-up (the imports) and the per-repeat CSV rewrite. The run prints both the accounted
total and the process wall clock on its last line, so the gap is visible rather than
inferred.

"ALL rows of the repeat" runs over the arm, sampler AND estimator axes at once: this
design writes 2 arms x |--samplers| x |--estimators| rows per repeat and each one
carries its own flow and its own scoring, so the sum is over all three axes rather
than over the estimators alone. There is deliberately NO per-arm or per-sampler shared
column: work shared inside an arm (that arm's CME solve, that arm's initial-cloud
metrics) is accumulated over the arms into the per-repeat shared columns, so a repeat
has exactly one shared total however many arms, samplers and estimators it ran.

Columns this design does NOT write: seconds_ae, seconds_encode, seconds_decode,
seconds_fid, seconds_fid_ref, seconds_fid_shared. There is no autoencoder, no frozen
encoder, no decoding step and no FID here, and the convention is to omit a phase a design
lacks rather than write 0.0 for it.

Everything here runs on CPU in float64, so no torch.cuda.synchronize() is needed before
any clock is stopped: there is no asynchronous kernel launch whose cost a naive
time.time() could miss.

Hardware provenance: every row also carries gpu_name (always empty -- this design selects
no device), hostname, slurm_job_id, slurm_array_task_id, slurm_nodelist, torch_version
and cuda_version, so a published timing can be tied to the node that produced it. The
SLURM fields are read from the environment and are empty off the cluster.

Estimator axis (--estimators): the target embedding chi_hat(pi) = sum_i v_i k_Y(y_i, .)
IS a weight vector, so the plug-in, IPW and DR estimators differ only in the vector
handed to the flow. All requested estimators share the logged sample, the single
nuisance solve, the initial particles, the reference and floor draws and the flow
budget, so the comparison between them carries no seed noise. The default ["dr"]
reproduces the pre-estimator-axis runs exactly. The IPW row is the Horvitz-Thompson form
v = w / n by default -- the term that appears inside v_DR, and the reason the three are
algebraically comparable, but one with no unit mass; --ipw_form hajek switches it to the
self-normalised w / sum(w), a different estimator, and the form used is recorded in the
ipw_form CSV column.

Evaluation protocol (2026-09-15, matched to Dance et al.): the DR support is EVERY
logged row (--n_dr caps it only on request) solved with the Nystrom solver at every n;
--n_particles 2000 particles are generated per arm and scored against a FIXED
--n_ref 2000 counterfactual reference draw (seed 100000 + data_seed, independent of
n_obs), with the floor a second independent 2000-draw against the first, so every
row is a 2000-vs-2000 number regardless of n_obs. The outcome kernel median stays
locked on the FULL logged sample, as in the notebooks. The *_floor_matched columns
repeat that floor with both oracle draws cut to n_floor_matched = min(m, n_ref), the
count the row's own particles were scored at, because these metrics shrink with the
sample size they are evaluated at; compare a swept --n_particles against those.

Seed scheme: `data_seed` fixes the draw; `pipeline_seed = 1000 * data_seed + repeat`
fixes the propensity fit, the DR subsample, the particle picks and the Riesz flow.

Launched by `submit_decfm_2d_dr_flow_parallel.sh`; single-run example:

    python decfm_2d_dr_flow_simulation.py --n_obs 4000 --data_seeds 0 --n_repeats 3
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

from kernel_utils import RBF
from data_utils import sample_decfm_2d_data
from causality_utils import (cpme_dr_weights, cpme_dr_weights_crossfit,
                             cpme_dr_weights_crossfit_nystrom,
                             cpme_dr_weights_crossfit_projected_nystrom,
                             cpme_dr_weights_projected_nystrom, cpme_dr_weights_nystrom,
                             cpme_ipw_weights, lambda_n_half, make_crossfit_fold_ids,
                             select_regularization)
from dens_ratio_utils import LogisticPropensityModel, crossfit_propensity
from mmd_flow_utils import adamw_riesz_flow, mmd_gradient_flow, squared_mmd
from metric_utils import sliced_wasserstein

#: Report order of the estimator axis. `--estimators` is canonicalised to it, so
#: neither the row order nor the filename token depends on how the flags were typed.
ESTIMATORS = ("plugin", "ipw", "dr")

COLUMNS = [
    "tag", "init", "solver", "n_landmarks", "dr_fit", "lambda_rule", "ipw_form",
    "data_seed", "repeat", "pipeline_seed", "sampler", "estimator", "target_a",
    "n_obs", "n_dr", "n_ref", "n_arm", "n_particles_requested", "n_particles",
    "n_floor", "n_floor_matched",
    "confounding_strength", "mixture_delta", "noise_std", "n_folds",
    "ess_on_arm", "n_on_arm",
    "sum_v", "sum_abs_v", "v_ess", "v_max_abs", "v_ess_on_arm",
    "fold_lambdas",
    "lengthscale_x", "lengthscale_y",
    "flow_iterations", "step_or_lr", "mean_transport",
    "sw1_initial", "sw2_initial", "mmd2_initial",
    "sw1_flow", "sw2_flow", "mmd2_flow",
    "sw1_floor", "sw2_floor", "mmd2_floor",
    "sw1_floor_matched", "sw2_floor_matched", "mmd2_floor_matched",
    # Wall clock, in scope order: per DATA SEED first, then shared-per-repeat, then this
    # row's own, then the total. See the module docstring for which may be summed across
    # rows and which may not. `seconds_ratio` / `seconds_cme` are a BREAKDOWN of
    # `seconds_nuisance` and `seconds_data` one of `seconds_setup`, so neither pair may be
    # added alongside the column it decomposes; `seconds_sampling` is derived from several
    # of the others and belongs to no sum at all.
    "seconds_data", "seconds_setup",
    "seconds_ratio", "seconds_cme", "seconds_nuisance",
    "seconds_eval_shared", "seconds_shared",
    "seconds_flow", "seconds_eval", "seconds_sampling",
    "seconds_other", "seconds_repeat",
    # Hardware provenance, so a published timing table can be audited against the node
    # that produced it. See `hardware_provenance`.
    "gpu_name", "hostname", "slurm_job_id", "slurm_array_task_id", "slurm_nodelist",
    "torch_version", "cuda_version",
]

#: Which half of the nuisance fit each estimator's target embedding actually needs:
#: v_PI is the CME solve alone, v_IPW = w / n the density ratios alone, and
#: v_DR = v_PI + v_correction both. Used by `sampling_seconds`.
SAMPLING_NUISANCE = {"plugin": ("seconds_cme",),
                     "ipw": ("seconds_ratio",),
                     "dr": ("seconds_ratio", "seconds_cme")}


def hardware_provenance() -> dict:
    """
    The machine this run was measured on, as CSV columns.

    A wall-clock table is only auditable against someone else's numbers if the node is
    recorded with them: these columns say which host and which SLURM allocation produced
    the row, and which torch/CUDA build ran it.

    Returns
    -------
    dict
        The seven provenance columns. `gpu_name` is always empty: this design runs on CPU
        in float64 and selects no device, so even on a GPU node none is used. The SLURM
        entries are empty strings off the cluster, where the variables are simply absent.
    """
    return {
        "gpu_name": "",
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
    }


def sampling_seconds(estimator: str, shared: dict, seconds_flow: float) -> float:
    """
    Isolated wall clock of the proposed sampling method for one row.

    The three estimators do not need the same nuisances -- the plug-in expansion needs the
    CME solve and no ratios, IPW the ratios and no solve, DR both -- so the honest cost of
    sampling against one of them is its own half of the nuisance fit plus its own flow.
    Measurement is excluded by construction: no evaluation column enters, and there is no
    autoencoder or decoder in this design to exclude.

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


def write_rows(out_path: Path, rows: list) -> None:
    """Atomically rewrite the task CSV (tmp + rename).

    A plain open(out_path, "w") truncates the previous good version BEFORE writing,
    so a failure mid-write (disk quota, kill) destroys already-completed rows — this
    happened in production with EDQUOT. Writing to a task-unique tmp and renaming
    makes every failure leave the last complete CSV untouched.
    """
    tmp = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def gaussian_init(reference, n_particles: int, mode: str,
                  generator) -> "torch.Tensor":
    """Particles from N(0, I) or the reference sample's empirical N(mu, Sigma).

    The moment-matched draw uses the CENTRED DATA MATRIX rather than a D x D
    Cholesky: for centred Z (n, D) and g ~ N(0, I_n), Z^T g / sqrt(n - 1) has
    exactly the sample covariance. That costs O(n D), needs no jitter, and stays
    well defined when n < D (the empirical Gaussian is then degenerate on the
    data span, which is the correct object). `reference` is the flow's own target
    support, so both draws live in the coordinates the flow operates in.
    """
    n, dim = reference.shape
    if mode == "normal":
        return torch.randn(n_particles, dim, generator=generator,
                           device=reference.device, dtype=reference.dtype)
    mean = reference.mean(dim=0, keepdim=True)
    weights = torch.randn(n_particles, n, generator=generator,
                          device=reference.device, dtype=reference.dtype)
    return mean + weights @ (reference - mean) / (n - 1) ** 0.5

def subsample_rows(sample, n_rows: int, generator) -> "torch.Tensor":
    """A uniform random `n_rows`-row subsample, or `sample` itself when it is smaller.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, d)
    n_rows : int
        Target row count.
    generator : torch.Generator
        Seeded source of the permutation, so the subsample is reproducible.

    Returns
    -------
    torch.Tensor of shape (min(n, n_rows), d)
    """
    if len(sample) <= n_rows:
        return sample
    return sample[torch.randperm(len(sample), generator=generator)[:n_rows]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=4000,
                        help="logged sample size n")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="size of the fixed counterfactual reference draw used for "
                             "every metric (and of the second draw used for the floor)")
    parser.add_argument("--n_particles", type=int, default=2000,
                        help="flow particles m per arm (capped at the arm size)")
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
                             "the sample, and a results table mixing two solvers is "
                             "mixing two estimators.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly; above it a fold "
                             "would be a (4/5 n_obs)^2 float64 system and the Gram alone "
                             "3.2 GB at n_obs = 20000"
                             ", and where 'auto' refuses instead of choosing a "
                             "solver for you")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold")
    parser.add_argument("--n_dr", type=int, default=None,
                        help="optional cap on the DR support, kept for reproducing the "
                             "pre-2026-09 runs (--solver exact --n_dr 4000). Unset, the "
                             "support is every logged row.")
    parser.add_argument("--samplers", type=str, nargs="+",
                        default=["rbf_euler", "riesz_adamw"],
                        choices=("rbf_euler", "riesz_adamw"))
    parser.add_argument("--estimators", type=str, nargs="+", default=["dr"],
                        choices=ESTIMATORS,
                        help="which target-embedding estimators to flow to, one CSV row "
                             "each. They are three coefficient vectors over the SAME "
                             "logged support: 'plugin' uses the conditional mean "
                             "embedding alone, 'ipw' the density ratios alone, 'dr' their "
                             "one-step combination v_PI + v_correction. Every requested "
                             "estimator shares the nuisance solve, the initial particles, "
                             "the reference draws and the flow budget, so the rows differ "
                             "only in the target. Default ['dr'] keeps the historical "
                             "behaviour and the historical arithmetic.")
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
                             "by construction but is a DIFFERENT estimator. It is a "
                             "normalisation, not a cap on the ratios.")
    parser.add_argument("--confounding_strength", type=float, default=3.0,
                        help="gamma in the logistic propensity")
    parser.add_argument("--mixture_delta", type=float, default=2.0,
                        help="separation of the two outcome-noise components")
    parser.add_argument("--noise_std", type=float, default=0.2,
                        help="within-component outcome noise")
    parser.add_argument("--particle_seed", type=int, default=None,
                        help="base seed for the STARTING PARTICLE draw, kept independent of the "
                             "rest of the pipeline so another implementation can reproduce the "
                             "same initial cloud. The draw uses seed particle_seed + 1000 * "
                             "data_seed + repeat on a CPU generator (CPU so the stream does not "
                             "depend on the device). Default None keeps the historical derivation "
                             "from the pipeline seed. The chosen rows are also written verbatim "
                             "to <csv stem>_particles.npz, which is the exact way to share them: "
                             "it needs no RNG agreement at all. CAVEAT: those indices address "
                             "THIS run's logged sample, so they are portable only to a run that "
                             "shares the data draw. ColorMNIST draws its data with a CUDA "
                             "generator and CelebA from a cached file, so a different device or "
                             "cache yields a different sample; to hand starting particles to an "
                             "implementation that generates its own data, export the particle "
                             "IMAGES rather than these indices.")
    parser.add_argument("--init", type=str, default="observed",
                        choices=("observed", "normal", "moment"),
                        help="initial particles: 'observed' = the logged arm (default), "
                             "'normal' = N(0, I), 'moment' = the empirical N(mu, Sigma) "
                             "of the flow's target support. Both Gaussians are drawn in "
                             "the coordinates the flow operates in. NOTE: the *_initial "
                             "metric columns describe the run's actual starting cloud, so "
                             "under a Gaussian init they are not the observed-arm baseline "
                             "-- take that from the matching --init observed run.")
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"),
                        help="how the DR nuisances are fitted. DEFAULT CHANGED 2026-09-28 to 'nosplit', which fits the CME and the propensity on ALL rows; 'crossfit' (the previous default) fits both out of fold. The same-sample notes license reusing observations for the NUISANCES. NOTE the filename token is deliberately still keyed on 'crossfit': a 'nosplit' run appends _drnosplit and a 'crossfit' run appends nothing, exactly as before the default flipped, so every historical filename is still reproduced byte for byte and no new run can overwrite an archived one. The dr_fit CSV column is the authoritative record. "
                             "historical behaviour) fits the CME and the propensity out of "
                             "fold. 'nosplit' fits both on ALL rows and evaluates each row's "
                             "residual with a fit that saw it. The same-sample notes license "
                             "the latter for a FIXED kernel and feature map under bounded "
                             "nuisance norms and prediction consistency; it is NOT licensed "
                             "when the kernel or encoder is itself selected on these rows.")
    parser.add_argument("--lambda_rule", type=str, default="cv",
                        choices=("cv", "n_half"),
                        help="ridge selection. 'cv' (default, historical) cross-validates "
                             "over a grid per training block. 'n_half' uses the rate the "
                             "same-sample theory gives as sufficient, lambda = c * n^-1/2, "
                             "which also keeps the penalty MATCHED between --dr_fit arms: "
                             "cross-validated lambdas are chosen per block and so differ "
                             "systematically between a full-sample and a k-fold fit, which "
                             "would confound that comparison.")
    parser.add_argument("--lambda_constant", type=float, default=1.0,
                        help="the constant c in --lambda_rule n_half (the rate is fixed by "
                             "theory, the constant is not)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--step_factor", type=float, default=0.05,
                        help="Euler step = step_factor * (outcome lengthscale)^2")
    parser.add_argument("--flow_iterations", type=int, default=None,
                        help="sets BOTH --euler_iterations and --riesz_iterations, so a "
                             "sweep over sampling steps can drive one flag name here as it "
                             "does in the image simulations. Default None keeps the two "
                             "separate values below.")
    parser.add_argument("--euler_iterations", type=int, default=3000)
    parser.add_argument("--riesz_iterations", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="AdamW peak learning rate (Riesz sampler)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="pipeline repeats per data seed")
    parser.add_argument("--tag", type=str, default="twins")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()
    if args.flow_iterations is not None:
        args.euler_iterations = args.riesz_iterations = args.flow_iterations
    args.estimators = [name for name in ESTIMATORS if name in args.estimators]
    return args



PARTICLE_INDICES: dict = {}


def particle_generator(args, seed: int, offset: int = 0) -> "torch.Generator":
    """CPU generator for the starting-particle draw; see --particle_seed."""
    base = seed if args.particle_seed is None else args.particle_seed + seed
    return torch.Generator().manual_seed(base + offset)


def write_particle_indices(out_path: Path) -> None:
    """Save the logged-sample rows used as starting particles, for cross-implementation reuse."""
    if PARTICLE_INDICES:
        np.savez(out_path.with_name(out_path.stem + "_particles.npz"), **PARTICLE_INDICES)

def resolve_solver(args) -> str:
    """Which ridge solver this configuration uses; see --solver.

    Raises
    ------
    SystemExit
        When `--solver auto` would resolve above --exact_max_n_obs. Resolving there
        exchanges the exact solve for the non-projected Nystrom one, which is both a
        change of CME estimator partway through a sweep and a change to a solver whose
        weights are coarse at small m / n -- `warn_nonprojected_solver` has the
        numbers. Naming the solver is left to the caller.
    """
    if args.solver != "auto":
        return args.solver
    if args.n_obs <= args.exact_max_n_obs:
        return "exact"
    return "projected_nystrom"


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


def weight_diagnostics(v: torch.Tensor, on_arm_mask: torch.Tensor) -> dict:
    """
    Mass and concentration of one estimator's target weight vector.

    Recorded per estimator, not per repeat: the three put very different profiles on
    the same logged outcomes -- IPW carries the raw density-ratio tail undamped, the
    plug-in carries none of it, DR sits between -- and that difference is the finding,
    so it cannot be a shared column.

    The first four are the CANONICAL per-estimator diagnostics of this repository: the
    same functionals, under the same names, in the same units, in all five simulations,
    so an estimator table can be read across designs. Two details are load-bearing.
    `v_ess` is Kish's ratio of the vector ITSELF, (sum v)^2 / sum v^2 -- on a signed
    vector (sum |v|)^2 / sum v^2 is a different functional and not a rescaling of it.
    And `v_max_abs` is the raw largest atom, never multiplied by n.

    `v_ess_on_arm` is specific to this design and named so: the same Kish ratio
    restricted to the rows logged under the intervened action. It equals `v_ess` for
    IPW, which is zero off the arm, and differs for the estimators carrying a CME term,
    so it separates the concentration of the on-arm mass from the signed off-arm spread
    only those two have. It is not the `ess_on_arm` column, which is that ratio of the
    IMPORTANCE WEIGHTS on the arm and so takes one value per repeat and arm, whatever
    estimator the row carries: the two agree on the IPW row, where v is a positive
    rescaling of w, and part ways on the rows with a CME term.

    Parameters
    ----------
    v : torch.Tensor of shape (n,)
        The estimator's weight vector over the logged outcomes.
    on_arm_mask : torch.Tensor of shape (n,)
        Boolean mask of the rows logged under the intervened action.

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
        'v_ess_on_arm' : 'v_ess' of the on-arm restriction of v.
    """
    kish = lambda t: float(t.sum() ** 2 / (t ** 2).sum())
    return {"sum_v": float(v.sum()), "sum_abs_v": float(v.abs().sum()),
            "v_ess": kish(v), "v_max_abs": float(v.abs().max()),
            "v_ess_on_arm": kish(v[on_arm_mask])}


def estimator_weights(solve, importance_weights, estimators: list,
                      ipw_form: str) -> dict:
    """Target weight vectors for the requested estimators, from ONE nuisance solve.

    The three estimators are the same expansion over the same support with different
    coefficients, so fitting the nuisances once and reading all three off that fit is
    what makes their difference the estimators' own rather than the fit's noise.

    Parameters
    ----------
    solve : callable
        ``solve(components)`` returning v_DR when False and the pair
        (v_PI, v_correction) when True: the configured DR solver with its Gram blocks,
        lambdas and landmark generator already bound.
    importance_weights : torch.Tensor of shape (n,)
        w_i on the DR support. The IPW estimator is w / n and needs nothing else.
    estimators : list of str
        Subset of `ESTIMATORS`, in report order.
    ipw_form : {'ht', 'hajek'}
        Normalisation of the IPW vector; see --ipw_form. 'ht' is the w / n that appears
        inside v_DR, so it is the form under which the three rows are one object.

    Returns
    -------
    dict
        Estimator name -> weight vector of shape (n,).

    Notes
    -----
    A ["dr"] request takes the solver's own DR right-hand side, so that path is
    arithmetically the one used before this axis existed; an ["ipw"] request skips the
    CME solve altogether. Otherwise ONE `components` call serves every requested
    estimator and v_DR is re-formed as v_PI + v_correction, which agrees with the
    direct solve to floating point rather than bitwise. The split has to stay inside a
    single call: the landmark draw is stateful, so two calls could fit two different
    Nystrom subspaces and the estimators would no longer share a nuisance.
    """
    if estimators == ["dr"]:
        return {"dr": solve(components=False)}
    if estimators == ["ipw"]:
        return {"ipw": cpme_ipw_weights(importance_weights,
                                        normalize=ipw_form == "hajek")}
    v_plugin, v_correction = solve(components=True)
    available = {"plugin": v_plugin,
                 "ipw": cpme_ipw_weights(importance_weights,
                                         normalize=ipw_form == "hajek"),
                 "dr": v_plugin + v_correction}
    return {name: available[name] for name in estimators}


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    # Tokens are appended only when the option is off its default, so every filename a
    # previous run produced is reproduced byte for byte. Without the iteration token a
    # sweep over step counts silently overwrites itself: write_rows rewrites the whole
    # file atomically, so the last task to finish would win.
    if args.dr_fit != "crossfit":
        suffix += f"_dr{args.dr_fit}"
    if args.lambda_rule != "cv":
        suffix += f"_lam{args.lambda_rule}"
    # One token per explicitly requested solver, so an --solver sweep cannot have two
    # cells write the same filename. Keyed on args.solver (not the RESOLVED solver), so
    # historical runs that passed --solver auto keep their exact filenames.
    if args.solver == "projected_nystrom":
        suffix += "_projnys"
    elif args.solver == "exact":
        suffix += "_exact"
    elif args.solver == "nystrom":
        suffix += "_nys"
    if (args.euler_iterations, args.riesz_iterations) != (3000, 4000):
        suffix += f"_it{args.euler_iterations}-{args.riesz_iterations}"
    # The estimator axis multiplies the rows rather than replacing them, so a
    # multi-estimator run must not land on the dr-only run's filename.
    if args.estimators != ["dr"]:
        suffix += "_est" + "-".join(args.estimators)
    if args.ipw_form != "ht":
        suffix += f"_ipw{args.ipw_form}"
    name = (f"decfm2d_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
            f"_gamma{args.confounding_strength:g}_delta{args.mixture_delta:g}"
            f"_sig{args.noise_std:g}_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def run_repeat(data: dict, setup: dict, args: argparse.Namespace, data_seed: int,
               repeat: int) -> list:
    """One pipeline pass on a fixed logged sample; one row per (arm, estimator, sampler).

    `setup` carries this data seed's per-DATA-SEED timing columns, measured in `main`
    outside this repeat's clock and copied onto every row unchanged.
    """
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, A_logged, Y_logged = data["X"], data["A_logged"], data["Y_logged"]
    Y_oracle, Y_oracle_2 = data["Y_ref"], data["Y_ref_2"]
    A_flat = A_logged.ravel()
    n = len(A_flat)

    # -- the ratio half of the shared nuisance fit: the DR support, the folds and the
    # propensity. This is all the IPW estimator needs; the support and the folds are
    # clocked here because this half consumes them first, and the CME half reuses both.
    t0 = time.time()
    solver = resolve_solver(args)
    # The support is every logged row; --n_dr caps it only when asked for explicitly.
    if args.n_dr is not None and args.n_dr < n:
        pick_dr = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 5))[:args.n_dr]
    else:
        pick_dr = torch.arange(n)
    X_s, A_s = X[pick_dr], A_logged[pick_dr]
    Y_s = Y_logged[pick_dr]
    A_s_flat = A_s.ravel()
    n_dr = len(pick_dr)

    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    if args.dr_fit == "nosplit":
        # Fit once on every row and predict on those same rows. NOT crossfit_propensity
        # with a single fold: that has no guard and silently returns a constant 0.5,
        # which would leave the two --dr_fit arms differing only in the CME nuisance.
        model = LogisticPropensityModel()
        model.fit(X_s, A_s_flat)
        propensity = model.predict(X_s).reshape(-1).double()
    else:
        propensity = crossfit_propensity(X_s, A_s_flat, fold_ids,
                                         model_factory=LogisticPropensityModel)
    seconds_ratio = time.time() - t0

    # -- the CME half: the context kernel, the Gram blocks and the ridge selection. This
    # is all the plug-in estimator needs, and DR needs both halves.
    t0 = time.time()
    kernel_x = RBF(use_length_scale_heuristic=True)
    kernel_a = RBF(length_scale=0.5)
    kernel_y = RBF(use_length_scale_heuristic=True)
    # Lock the context lengthscale on a bounded subsample: the median over all
    # n_dr^2 pairs is neither affordable nor necessary at n_dr = 20000.
    anchor = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:4000]
    _ = kernel_x(X_s[anchor])
    seconds_cme = time.time() - t0

    # The OUTCOME kernel is the metric kernel of the MMD^2 columns, so its median lock is
    # shared evaluation, not nuisance estimation -- as in the Warfarin and ColorMNIST
    # designs, which lock their metric kernels inside the same clock. The RBF-Euler step
    # size reads the same locked lengthscale, which is why seconds_sampling says it does
    # not cover that median.
    t0 = time.time()
    _ = kernel_y(Y_logged)   # median locked on the FULL logged sample, as in the notebooks
    seconds_eval_shared = time.time() - t0

    t0 = time.time()

    def gram_x(rows, cols):
        "K_X block alone; K_pi needs it without the action factor."
        return kernel_x(X_s[rows], X_s[cols])

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_dr^2 Gram is never formed."
        return gram_x(rows, cols) * kernel_a(A_s[rows], A_s[cols])

    K_x = kernel_x(X_s) if solver == "exact" else None
    K_a = kernel_a(A_s) if solver == "exact" else None

    if args.lambda_rule == "n_half":
        # one value, matched across arms by construction
        nosplit_lambda = lambda_n_half(n_dr, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    else:
        fold_lambdas = {}
        for fold in np.unique(fold_ids):
            train = np.where(fold_ids != fold)[0]
            # lambda does not need the full training block, which is itself an n^3 solve
            pick = train[torch.randperm(len(train),
                                        generator=torch.Generator().manual_seed(0))[:2000].numpy()]
            fold_lambdas[int(fold)], _ = select_regularization(gram(pick, pick),
                                                               Y_s[pick, 0], seed=0)
        # the no-split fit has one training block (all rows), so it needs one scalar;
        # only selected when that arm is actually requested
        nosplit_lambda = None
        if args.dr_fit == "nosplit":
            pick = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:2000].numpy()
            nosplit_lambda, _ = select_regularization(gram(pick, pick), Y_s[pick, 0], seed=0)
    # Each arm's target action-kernel block and its CME ridge solve are timed into this
    # same accumulator inside the arm loop below: they are the rest of the one CME fit
    # every row of the repeat shares.
    seconds_cme += time.time() - t0

    base = {
        "tag": args.tag, "init": args.init, "solver": solver,
        "dr_fit": args.dr_fit, "lambda_rule": args.lambda_rule,
        "ipw_form": args.ipw_form,
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "n_obs": args.n_obs, "n_dr": int(n_dr), "n_ref": int(len(Y_oracle)),
        "n_particles_requested": args.n_particles,
        "confounding_strength": args.confounding_strength,
        "mixture_delta": args.mixture_delta, "noise_std": args.noise_std,
        "n_folds": args.n_folds,
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        "lengthscale_x": float(kernel_x.length_scale),
        "lengthscale_y": float(kernel_y.length_scale),
        # Identical on every row of every repeat of this data seed, and measured outside
        # this repeat's clock: the second term of the per-task identity.
        **setup,
        **hardware_provenance(),
    }

    def evaluate(cloud, reference=None):
        reference = Y_oracle if reference is None else reference
        return {
            "sw1": sliced_wasserstein(cloud, reference, p=1, seed=0).item(),
            "sw2": sliced_wasserstein(cloud, reference, p=2, seed=0).item(),
            "mmd2": squared_mmd(kernel_y, cloud, reference, unbiased=True).item(),
        }

    n_floor = len(Y_oracle_2)
    t0 = time.time()
    floor = evaluate(Y_oracle_2)      # second independent reference draw vs the first
    seconds_eval_shared += time.time() - t0

    matched_floors: dict = {}

    def matched_floor(n_matched: int) -> dict:
        """The oracle-vs-oracle floor recomputed with BOTH draws cut to `n_matched` rows.

        WHY this exists: every cheap distributional metric here is sample-size biased --
        two independent draws of the SAME law score SW2 ~ 0.0266 at 250 points but
        ~ 0.0070 at 4000. `sliced_wasserstein` truncates to the smaller sample, so a row
        with m particles is scored at n_matched = min(m, n_ref) points while `floor` is
        scored at the full n_ref. Sweeping --n_particles would then move the flow rows
        and leave the floor put, a purely statistical "more particles is better". This
        twin puts the floor at the row's own effective count. The subsample is seeded
        off the pipeline seed, so it is reproducible, and cached because both samplers
        on an arm share it.

        Parameters
        ----------
        n_matched : int
            Effective count at which this row's particles were scored.

        Returns
        -------
        dict
            The same metric keys as `evaluate`.
        """
        if n_matched not in matched_floors:
            ref = subsample_rows(Y_oracle, n_matched,
                                 torch.Generator().manual_seed(seed + 7))
            ref_2 = subsample_rows(Y_oracle_2, n_matched,
                                   torch.Generator().manual_seed(seed + 8))
            matched_floors[n_matched] = evaluate(ref_2, ref)
        return matched_floors[n_matched]

    rows = []
    for target_a in (0, 1):
        # This arm's importance ratios: the ratio half's share of the arm. One set serves
        # every estimator and sampler on the arm, so it is shared work rather than any
        # row's own, and it accumulates into seconds_ratio over the two arms.
        t0 = time.time()
        e_a = propensity if target_a == 1 else 1.0 - propensity
        w_hat = (A_s_flat == target_a).double() / e_a
        on_arm = w_hat[A_s_flat == target_a]
        seconds_ratio += time.time() - t0

        # This arm's target action-kernel block and the CME ridge solve behind every one
        # of its weight vectors: the CME half's share of the arm.
        t0 = time.time()
        if solver in ("nystrom", "projected_nystrom"):
            # K_A_pi[i, j] = k_A(a_i, a) is the same for every column j, so K_pi's
            # blocks are K_X blocks scaled row-wise.
            target_column = kernel_a(A_s, torch.full((1, 1), float(target_a))).reshape(-1)
            gram_pi = lambda rows, cols: (gram_x(rows, cols)
                                          * target_column[rows][:, None])
            if args.dr_fit == "nosplit":
                nystrom_solve = (cpme_dr_weights_projected_nystrom
                                 if solver == "projected_nystrom" else cpme_dr_weights_nystrom)
                solve = lambda components: nystrom_solve(
                    gram, gram_pi, w_hat, nosplit_lambda,
                    n_landmarks=args.n_landmarks,
                    generator=torch.Generator().manual_seed(seed),
                    components=components)
            else:
                nystrom_solve = (cpme_dr_weights_crossfit_projected_nystrom
                                 if solver == "projected_nystrom" else cpme_dr_weights_crossfit_nystrom)
                solve = lambda components: nystrom_solve(
                    gram, gram_pi, w_hat, fold_ids, fold_lambdas,
                    n_landmarks=args.n_landmarks,
                    generator=torch.Generator().manual_seed(seed),
                    components=components)
        else:
            K_a_target = kernel_a(A_s, torch.full((n_dr, 1), float(target_a)))
            if args.dr_fit == "nosplit":
                solve = lambda components: cpme_dr_weights(
                    K_x, K_a, K_a_target, w_hat, nosplit_lambda, components=components)
            else:
                solve = lambda components: cpme_dr_weights_crossfit(
                    K_x, K_a, K_a_target, w_hat, fold_ids, fold_lambdas,
                    components=components)
        weight_vectors = estimator_weights(solve, w_hat, args.estimators,
                                           args.ipw_form)
        seconds_cme += time.time() - t0

        arm_idx = torch.nonzero(A_flat == target_a).ravel()
        n_particles = min(args.n_particles, len(arm_idx))
        generator = particle_generator(args, seed)
        if args.init == "observed":
            pick = arm_idx[torch.randperm(len(arm_idx), generator=generator)[:n_particles]]
            PARTICLE_INDICES[f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"] = pick.cpu().numpy()
            init = Y_logged[pick].clone()
        else:
            init = gaussian_init(Y_s, n_particles, args.init,
                                 torch.Generator().manual_seed(seed + 3))
        # Score the cloud the flow ACTUALLY starts from, which is what --init selects and
        # what the flag's help promises. Until 2026-09-28 this scored the whole logged arm
        # regardless of --init, so every initialisation reported an identical *_initial row
        # and an --init sweep recorded nothing about where the Gaussian arms began. Under
        # --init observed this is now the sampled arm subset rather than the full arm, so it
        # is also counted consistently with the flow row it is compared against.
        t0 = time.time()
        initial_metrics = evaluate(init)
        # the count the flowed cloud is actually scored at (sliced_wasserstein truncates)
        n_matched = min(n_particles, len(Y_oracle))
        floor_matched = matched_floor(n_matched)
        # Both are copied into every row of this arm, so their cost is shared scoring;
        # `matched_floor` is cached, so the second arm usually adds nothing.
        seconds_eval_shared += time.time() - t0

        # Both samplers and every estimator start from `init`, and the budget below is
        # the same for all of them, so a row pair differs only in its target weights.
        for estimator, v_target in weight_vectors.items():
            for sampler in args.samplers:
                t0 = time.time()
                if sampler == "rbf_euler":
                    step = args.step_factor * kernel_y.length_scale.item() ** 2
                    flowed, _ = mmd_gradient_flow(kernel_y, Y_s, init.clone(),
                                                  step_size=step,
                                                  n_iterations=args.euler_iterations,
                                                  target_weights=v_target, verbose=False)
                    iterations, step_or_lr = args.euler_iterations, step
                else:
                    torch.manual_seed(seed + 1)
                    flowed, _ = adamw_riesz_flow(init.clone(), Y_s, target_weights=v_target,
                                                 n_iterations=args.riesz_iterations,
                                                 learning_rate=args.learning_rate,
                                                 warmup_steps=args.warmup_steps,
                                                 lr_schedule="cosine", betas=(0.9, 0.999),
                                                 verbose=False)
                    iterations, step_or_lr = args.riesz_iterations, args.learning_rate
                seconds_flow = time.time() - t0

                # This row's OWN scoring: the flowed cloud and nothing else. The
                # initial-cloud and floor metrics below were scored once and shared.
                t0 = time.time()
                flow_metrics = evaluate(flowed)
                seconds_eval = time.time() - t0

                row = dict(base)
                row.update({
                    "sampler": sampler, "estimator": estimator, "target_a": target_a,
                    "n_arm": int(len(arm_idx)), "n_particles": int(n_particles),
                    "n_floor": int(n_floor), "n_floor_matched": int(n_matched),
                    "ess_on_arm": float(on_arm.sum() ** 2 / (on_arm ** 2).sum()),
                    "n_on_arm": int(len(on_arm)),
                    **weight_diagnostics(v_target, A_s_flat == target_a),
                    "flow_iterations": iterations, "step_or_lr": float(step_or_lr),
                    "mean_transport": float((flowed - init).norm(dim=1).mean()),
                    "seconds_flow": round(seconds_flow, 1),
                    "seconds_eval": round(seconds_eval, 1),
                })
                row.update({f"{key}_initial": value
                            for key, value in initial_metrics.items()})
                row.update({f"{key}_flow": value for key, value in flow_metrics.items()})
                row.update({f"{key}_floor": value for key, value in floor.items()})
                row.update({f"{key}_floor_matched": value
                            for key, value in floor_matched.items()})
                rows.append(row)

    # Round FIRST and take the residual from the ROUNDED parts, so the identity in the
    # module docstring closes exactly on the numbers the CSV carries rather than only on
    # the raw floats a reader never sees.
    seconds_repeat = round(time.time() - t_start, 1)
    seconds_ratio = round(seconds_ratio, 1)
    seconds_cme = round(seconds_cme, 1)
    seconds_nuisance = round(seconds_ratio + seconds_cme, 1)
    seconds_eval_shared = round(seconds_eval_shared, 1)
    seconds_shared = round(seconds_nuisance + seconds_eval_shared, 1)
    seconds_own = round(sum(row["seconds_flow"] + row["seconds_eval"]
                            for row in rows), 1)
    # The trailing + 0.0 turns the -0.0 that float subtraction yields when the parts
    # cancel exactly into a plain 0.0, so the CSV reads "0.0" and not "-0.0".
    seconds_other = round(seconds_repeat - seconds_shared - seconds_own, 1) + 0.0
    shared = {"seconds_ratio": seconds_ratio, "seconds_cme": seconds_cme,
              "seconds_nuisance": seconds_nuisance,
              "seconds_eval_shared": seconds_eval_shared,
              "seconds_shared": seconds_shared}
    for row in rows:
        row.update(shared)
        # Derived, and in neither identity: this estimator's own half of the nuisance fit
        # plus this row's own flow, with no representation and no measurement in it.
        row["seconds_sampling"] = sampling_seconds(row["estimator"], shared,
                                                   row["seconds_flow"])
        row.update({"seconds_other": seconds_other,
                    "seconds_repeat": seconds_repeat})
    return rows


def main() -> None:
    args = parse_args()
    warn_nonprojected_solver(resolve_solver(args), args.n_obs, args.n_landmarks)
    torch.set_default_dtype(torch.float64)
    out_path = csv_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_path}", flush=True)

    rows = []
    accounted = 0.0
    # The per-DATA-SEED setup clock, restarted after each seed's repeats. Nothing is
    # loaded once per process here, so no seed carries another's cost.
    t_process = time.time()
    t_setup = t_process
    for data_seed in args.data_seeds:
        data = sample_decfm_2d_data(n_samples=args.n_obs,
                                    confounding_strength=args.confounding_strength,
                                    mixture_delta=args.mixture_delta,
                                    noise_std=args.noise_std, seed=data_seed)
        # Fixed reference draw, independent of n_obs: first half scores every row, the
        # second half is the floor's sample. Seeds are disjoint from every data seed.
        reference = sample_decfm_2d_data(n_samples=2 * args.n_ref,
                                         confounding_strength=args.confounding_strength,
                                         mixture_delta=args.mixture_delta,
                                         noise_std=args.noise_std,
                                         seed=100000 + data_seed)["Y_oracle"]
        data["Y_ref"], data["Y_ref_2"] = reference[:args.n_ref], reference[args.n_ref:]
        # This seed's whole per-data-seed tier: data generation and nothing else here, so
        # seconds_setup equals seconds_data. Both sit outside every seconds_repeat.
        seconds_data = round(time.time() - t_setup, 1)
        setup = {"seconds_data": seconds_data, "seconds_setup": seconds_data}
        accounted += seconds_data
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(data, setup, args, data_seed, repeat)
            accounted += new_rows[-1]["seconds_repeat"]
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            write_particle_indices(out_path)
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']} "
                      f"{row['estimator']:6s} {row['sampler']:11s}] "
                      f"SW2 {row['sw2_initial']:.3f} -> "
                      f"{row['sw2_flow']:.3f} (floor {row['sw2_floor']:.3f})   "
                      f"sum_v {row['sum_v']:.3f}   {row['seconds_flow']:.0f}s", flush=True)
        # The next data seed's setup clock starts once this seed's repeats are done.
        t_setup = time.time()
    # The per-task identity, printed so the gap it does not cover is visible rather than
    # inferred: process start-up and the per-repeat CSV rewrite.
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)
    print(f"accounted {round(accounted, 1)}s = sum(seconds_setup) over data seeds "
          f"+ sum(seconds_repeat) over repeats, of "
          f"{time.time() - t_process:.1f}s process wall clock", flush=True)


if __name__ == "__main__":
    main()
