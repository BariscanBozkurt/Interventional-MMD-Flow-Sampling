"""
ColorMNIST simulation: cross-fitted DR-CPME target + IMQ-kernel MMD flow (forward Euler).

Script version of
`NotebookExamples/Interventional_MMD_Flow/ColorMNIST_DR_IMQ_MMD_Flow_Euler_Arm1_RefParams_SplitAE.ipynb`,
step for step: the logged sample is split once into an autoencoder half and a causal
half, the frozen encoder defines the latent space, the cross-fitted DR weights define
the target embedding, and an IMQ MMD flow transports observed arm-a latents onto it.

Each repeat runs BOTH intervention arms by default (the evaluation protocol averages
over a in {0, 1}), sharing every arm-independent stage — the logged sample, the
split, the autoencoder, the DR subsample with its encoded support, the cross-fitted
propensity and the per-fold ridge lambdas — so the two arms are paired on identical
nuisances and only the importance weights, DR weight vector, flow and evaluation are
arm-specific.

`--estimators` chooses which target-embedding estimators are sampled from. All three
come out of the SAME nuisance solve -- v_DR = v_PI + v_correction and v_IPW = w / n --
so asking for several costs one extra flow each and not one extra nuisance, and they
share the logged sample, the autoencoder, the solve, the starting particles and the
oracle draws. Only the weight vector handed to the flow differs, which is what makes
the comparison paired rather than a comparison of seeds. The default is DR alone. The
IPW row is the Horvitz-Thompson form v = w / n by default -- the term that appears
inside v_DR, and the reason the three are algebraically comparable, but one with no
unit mass; --ipw_form hajek switches it to the self-normalised w / sum(w), a
different estimator, and the form used is recorded in the ipw_form CSV column.

One CSV row per (data_seed, repeat, arm, estimator), holding the DGP constants, split
sizes, nuisance diagnostics (on-arm ESS, the shape of that row's own weight vector,
per-fold lambdas), flow diagnostics (bandwidth, step, objective at start and end) and
the pixel-space evaluation against fresh oracle draws (colour-ratio law, latent SW2,
pixel SW1, SW2, RBF MMD^2) for the initial particles, the flowed particles, and the
oracle-vs-oracle floor. Every cheap distributional floor also gets a `*_floor_matched`
twin computed on both oracle draws subsampled to the count the row's own sample was
scored at, because those metrics are sample-size biased and a particle-count sweep
would otherwise move the rows while the floor stood still.

Evaluation protocol (2026-09-15, matched to Dance et al.): no autoencoder split
(--ae_split none: the AE trains on the whole logged sample and the DR support is
every row), Nystrom solver at every n, --n_particles 2000 per arm scored on
--eval_particles 2000 against a FIXED --n_ref 2000 oracle draw (seed offset +9),
floor = a second 2000-draw (+10) against the first, plus the decoder round-trip of
that second draw; FID with both scorers of `metric_utils.fid`; 3 data seeds, no
repeats. Every row is therefore a 2000-vs-2000 number at every n_obs.

Seed scheme, mirroring the notebook's SEED-offset convention:
`data_seed` fixes the logged sample; `pipeline_seed = 1000 * data_seed + repeat`
fixes everything downstream (the split, the autoencoder, the DR subsample, the
particle pick and the oracle evaluation draws). The arm-1 rows at the default
arguments follow the notebook's seed path exactly.

Wall-clock accounting (every `seconds_*` column; wall clock, time.time(), 0.1 s)
-------------------------------------------------------------------------------
A repeat writes n_arms x n_estimators rows, and the stages those rows SHARE are written
identically onto each of them. Every column therefore belongs to exactly ONE of three
scopes, and keeping them apart is the whole point: a cost measured once and written onto
every row is multiplied by any sum down those rows, and a cost measured outside the repeat
clock is lost by any sum of repeats.

PER DATA SEED -- identical on every row of every repeat of that data seed, and measured
OUTSIDE seconds_repeat, because the work happens once before the repeat loop:
    seconds_data     this data seed's data preparation: the confounded logged draw and its
                     conversion to model range. The MNIST digit pools and the FID scorers'
                     Inception weights are loaded once per PROCESS and are charged to the
                     FIRST data seed's seconds_data, so the per-task identity below closes
                     without a column a reader would have to know to count once across
                     seeds. Production tasks run one data seed each, where the distinction
                     does not arise.
    seconds_setup    the per-data-seed TOTAL, and the column to add once per data seed.
                     Here it is seconds_data alone: the autoencoder is trained inside each
                     repeat, so there is no frozen-encoder pass to hoist out of the repeats
                     (seconds_encode) and no reference feature that outlives one
                     (seconds_fid_ref) -- the two other components seconds_setup is broken
                     into where a design has them, both of which CelebA carries.

SHARED PER REPEAT -- repeat TOTALS, identical on every row; never sum across rows:
    seconds_ae          the whole autoencoder STAGE: training, its two reconstruction
                        probes, and the encoder passes that build the latents the rest of
                        the pipeline reads -- the DR support and each arm's observed
                        latents. Summed over arms.
    seconds_ratio       the density-ratio half of the nuisance fit: the DR support
                        selection, the cross-fitting folds, the logistic propensity fit and
                        each arm's importance ratios w = 1{a_i = a} / e(a | x_i). This is
                        the whole of what the IPW estimator needs. The support and the
                        folds are clocked here because this half consumes them first; the
                        CME half reuses both. Summed over arms.
    seconds_cme         the conditional-mean-embedding half: the context-kernel median
                        lock, the Gram preparation, the per-fold ridge selection and each
                        arm's single target-weight solve. This is the whole of what the
                        plug-in estimator needs; DR needs both halves. One solve serves
                        every estimator of the arm and hands back all of its vectors at
                        once, so the sub-millisecond IPW rescaling v = w / n is clocked
                        here too. Summed over arms.
    seconds_nuisance    seconds_ratio + seconds_cme, under the name it has always had. It
                        is their SUM: never add the three together, and never add the two
                        halves to seconds_shared.
    seconds_eval_shared the arm-level work every estimator of that arm reuses: the
                        starting-particle draw and the IMQ bandwidth / step heuristic,
                        both oracle draws with their encodings, the shared decodes (the
                        initial cloud and the round-trip) and every shared metric
                        (*_initial, *_floor, *_floor_matched, *_roundtrip, the latent
                        SW2 floors). FID excluded. Summed over arms.
    seconds_fid_shared  the shared FID: the reference features and the initial / floor /
                        round-trip distances. Summed over arms.
    seconds_shared      = seconds_ae + seconds_nuisance + seconds_eval_shared +
                        seconds_fid_shared, as one convenience column.

PER-ESTIMATOR -- this row's own work; sum freely across rows:
    seconds_flow     this estimator's flow iterations.
    seconds_decode   decoding this estimator's flowed particles.
    seconds_eval     this row's OWN scoring: its weight diagnostics, the *_flow pixel
                     metrics, its latent SW2 and its own FID. INCLUDES seconds_fid.
    seconds_fid      this row's own FID alone -- a BREAKDOWN of seconds_eval, already
                     inside it, so adding the two together double counts.
    seconds_sampling DERIVED, and a term of NEITHER identity below: the isolated cost of
                     the proposed sampling method for this row, composed for its estimator
                     out of the columns above --
                         plugin -> seconds_cme + seconds_flow + seconds_decode
                         ipw    -> seconds_ratio + seconds_flow + seconds_decode
                         dr     -> seconds_ratio + seconds_cme + seconds_flow
                                   + seconds_decode
                     It EXCLUDES the autoencoder (seconds_ae: training, probes and the
                     latent encodes) and all evaluation, because those are representation
                     and measurement rather than sampling, so an END-TO-END figure must add
                     seconds_ae and seconds_setup back. One caveat, a consequence of the
                     shared columns it is built from: the nuisance halves are repeat totals
                     over BOTH arms, so run one arm (--target_arms a) for one arm's figure.
                     It reuses shared columns, so never sum it down the rows of a repeat.

TOTAL:
    seconds_repeat   wall clock of the WHOLE repeat, identical on every row of that
                     repeat. Not cumulative down the rows, and not a per-row cost.
    seconds_other    the unattributed residual, carried as a column so the per-repeat
                     identity closes EXACTLY on the numbers the CSV holds rather than
                     approximately. What lands in it: the logged-sample split, the RNG
                     reseeding, the timing bookkeeping itself, and the 0.1 s rounding of
                     every part above -- which is also why it can come out slightly
                     negative on a short run. It is a residual by construction and never a
                     measured stage, so a reader never has to guess where time went.

The two identities a reader needs, both exact on the numbers the CSV holds:

    seconds_repeat == seconds_shared + seconds_other
                      + sum over the repeat's rows of
                        (seconds_flow + seconds_decode + seconds_eval)    [per repeat]

    task total     == sum over data seeds of seconds_setup
                      + sum over repeats of seconds_repeat                [per task]

reading seconds_setup once per data seed and seconds_repeat once per repeat, never once per
row. What the task identity does not cover is named rather than hidden: process start-up
(the imports and the CUDA context) and the per-repeat CSV rewrite. The run prints both the
accounted total and the process wall clock on its last line, so the gap is visible rather
than inferred.

Two asymmetries, both deliberate. The own FID sits INSIDE seconds_eval while the shared
FID sits BESIDE seconds_eval_shared: the own-side identity is stated over the same three
columns (flow, decode, eval) in every design that has them, while seconds_shared is the
plain sum of the shared columns, and a column cannot be in both. CelebA follows the same
convention, so `seconds_fid` means one thing across the two FID designs. And the ARM axis
is folded into the shared columns instead of getting a column of its own -- the two arms
share the logged sample, the autoencoder and the nuisance stage but not the oracle draws,
so each shared column is the repeat total over BOTH arms and never one arm's cost. Per-arm
costs come from running --target_arms with a single arm.

Columns carried over from before 2026-10-03: seconds_ae, seconds_nuisance, seconds_flow,
seconds_fid, seconds_repeat. Three of them changed meaning. seconds_fid no longer folds in
the shared FID (that is seconds_fid_shared now); seconds_nuisance now also carries the
per-arm target-weight solves, which previously were timed by nothing at all, and no longer
carries the encoder pass over the DR support, which moved to seconds_ae where the rest of
the autoencoder's work is; and seconds_ae now spans that pass and each arm's observed
latents as well as training and the probes. Archived CSVs predate all of it and hold one
row per (arm, estimator) with the old meanings. No column is dropped when a stage is
absent: under --fid none the two FID columns are written as 0.0, because they already
existed and every design here runs the same phases -- only the FID can be switched off.

Hardware provenance: every row also carries gpu_name, hostname, slurm_job_id,
slurm_array_task_id, slurm_nodelist, torch_version and cuda_version, so a published timing
can be tied to the node that produced it. The SLURM fields are read from the environment
and are empty off the cluster.

Launched by `submit_colormnist_dr_mmd_flow_parallel.sh`; single-run example:

    python colormnist_dr_mmd_flow_simulation.py --n_obs 20000 --data_seeds 0 --n_repeats 3
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

from data_utils import (color_ratio, load_digit_pools, sample_interventional,
                        sample_observed, to_image_range, to_model_range)
from autoencoder_utils import encode_in_batches, train_conv_ae
from kernel_utils import IMQKernel, RBF, imq_c_from_median
from causality_utils import (cpme_dr_weights, cpme_dr_weights_crossfit,
                             cpme_dr_weights_crossfit_nystrom, cpme_dr_weights_nystrom,
                             cpme_dr_weights_crossfit_projected_nystrom,
                             cpme_dr_weights_projected_nystrom,
                             cpme_ipw_weights, lambda_n_half,
                             make_crossfit_fold_ids, select_regularization)
from dens_ratio_utils import LogisticPropensityModel, crossfit_propensity
from mmd_flow_utils import mmd_gradient_flow, squared_mmd
from metric_utils import CleanFIDScorer, LegacyFIDScorer, sliced_wasserstein

COLUMNS = [
    "tag", "device", "init", "estimator", "ae_split", "solver", "dr_fit",
    "lambda_rule", "ipw_form", "n_landmarks",
    "data_seed", "repeat", "pipeline_seed", "target_a",
    "n_obs", "w_conf", "tau", "alpha", "k_mask",
    "n_ae", "n_causal", "n_arm_causal", "n_dr", "n_folds",
    "n_particles_requested", "n_particles", "n_flow_eval", "n_ref", "latent_dim",
    "ae_iterations", "flow_iterations",
    "recon_mse_train", "recon_mse_heldout",
    "ess_on_arm", "n_on_arm", "max_weight",
    "sum_v", "sum_abs_v", "v_ess", "v_max_abs", "fold_lambdas",
    "c_imq", "step_size", "objective_initial", "objective_final",
    "latent_sw2_initial", "latent_sw2_flow", "latent_sw2_floor",
    "colour_mean_initial", "colour_std_initial", "colour_w2_initial",
    "sw1_initial", "sw2_initial", "mmd2_initial",
    "colour_mean_flow", "colour_std_flow", "colour_w2_flow", "sw1_flow", "sw2_flow", "mmd2_flow",
    "colour_mean_floor", "colour_std_floor", "colour_w2_floor",
    "sw1_floor", "sw2_floor", "mmd2_floor",
    "colour_mean_floor_matched", "colour_std_floor_matched", "colour_w2_floor_matched",
    "sw1_floor_matched", "sw2_floor_matched", "mmd2_floor_matched",
    "latent_sw2_floor_matched", "n_floor_matched", "n_floor_matched_latent",
    "colour_mean_roundtrip", "colour_std_roundtrip", "colour_w2_roundtrip",
    "sw1_roundtrip", "sw2_roundtrip", "mmd2_roundtrip",
    "colour_mean_oracle", "colour_std_oracle",
    "fid_legacy_initial", "fid_legacy_flow", "fid_legacy_floor", "fid_legacy_roundtrip",
    "fid_clean_initial", "fid_clean_flow", "fid_clean_floor", "fid_clean_roundtrip",
    # Wall clock; see "Wall-clock accounting" in the module docstring. The first group is
    # per DATA SEED (outside every seconds_repeat); the second SHARED per repeat (repeat
    # totals, identical on every row of the repeat, NOT to be summed across rows); the
    # third per-estimator and summing freely. seconds_ratio / seconds_cme are a BREAKDOWN
    # of seconds_nuisance, seconds_data one of seconds_setup and seconds_fid one of
    # seconds_eval, so none of the three may be added alongside the column it decomposes;
    # seconds_sampling is derived from several of the others and belongs to no sum at all.
    # seconds_other closes the per-repeat identity exactly.
    "seconds_data", "seconds_setup",
    "seconds_ae", "seconds_ratio", "seconds_cme", "seconds_nuisance",
    "seconds_eval_shared", "seconds_fid_shared", "seconds_shared",
    "seconds_flow", "seconds_decode", "seconds_eval", "seconds_fid",
    "seconds_sampling",
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


def hardware_provenance(device: torch.device) -> dict:
    """
    The machine this run was measured on, as CSV columns.

    A wall-clock table is only auditable against someone else's numbers if the node is
    recorded with them: these columns say which GPU, which host and which SLURM allocation
    produced the row, and which torch/CUDA build ran it.

    Parameters
    ----------
    device : torch.device
        The device the run uses. `gpu_name` is empty off CUDA, where there is none.

    Returns
    -------
    dict
        The seven provenance columns. The SLURM entries are empty strings off the cluster,
        where the variables are simply absent.
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


def sampling_seconds(estimator: str, shared: dict, seconds_flow: float,
                     seconds_decode: float) -> float:
    """
    Isolated wall clock of the proposed sampling method for one estimator row.

    The three estimators do not need the same nuisances -- the plug-in expansion needs the
    CME solve and no ratios, IPW the ratios and no solve, DR both -- so the honest cost of
    sampling against one of them is its own half of the nuisance fit plus its own flow and
    the decode that turns the particles back into images. Representation and measurement
    are excluded by construction: the autoencoder stage (`seconds_ae`) and every evaluation
    column stay out, so an end-to-end figure must add `seconds_ae` and `seconds_setup`
    back.

    Parameters
    ----------
    estimator : {'plugin', 'ipw', 'dr'}
    shared : dict
        The repeat's shared columns, already ROUNDED to 0.1 s, so that the result equals
        what a reader gets by adding the CSV's own numbers.
    seconds_flow, seconds_decode : float
        This row's rounded `seconds_flow` and `seconds_decode`.

    Returns
    -------
    float
        Seconds, rounded to 0.1 s.
    """
    parts = [shared[column] for column in SAMPLING_NUISANCE[estimator]]
    return round(sum(parts) + seconds_flow + seconds_decode, 1)


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



def sync_device(device) -> None:
    """Wait for queued CUDA work, so the wall clock that stops next stops on real work.

    CUDA kernel launches are asynchronous: `ae.decode(...)` or `encode_in_batches(...)`
    returns as soon as the work is QUEUED, so a bare time.time() around it measures the
    launch and charges the computation to whichever later phase happens to block. The
    phases here that already end in a `.item()` or `float(...)` -- the flow (its last
    iteration always logs the objective), the metrics, both FID distances, the
    reconstruction probes -- are therefore timed correctly as they stand, and this call is
    a no-op for them; the decode and the encoder passes are not, and this is what fixes
    them. Synchronising only waits, so it cannot change any numerical result.

    Parameters
    ----------
    device : torch.device or str or torch.Tensor.device
        Non-CUDA devices return immediately.
    """
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


PARTICLE_INDICES: dict = {}


def particle_generator(args, seed: int, offset: int = 0) -> "torch.Generator":
    """CPU generator for the starting-particle draw; see --particle_seed."""
    base = seed if args.particle_seed is None else args.particle_seed + seed
    return torch.Generator().manual_seed(base + offset)


def write_particle_indices(out_path: Path) -> None:
    """Save the logged-sample rows used as starting particles, for cross-implementation reuse."""
    if PARTICLE_INDICES:
        np.savez(out_path.with_name(out_path.stem + "_particles.npz"), **PARTICLE_INDICES)


#: Canonical estimator order, used for the row order and the filename token so that
#: neither depends on how the flag was spelled on the command line.
ESTIMATOR_ORDER = ("plugin", "ipw", "dr")


def resolve_estimators(args) -> list:
    """The estimators this run samples from, deduplicated and in `ESTIMATOR_ORDER`."""
    requested = set(args.estimators)
    return [name for name in ESTIMATOR_ORDER if name in requested]


def resolve_solver(args) -> str:
    """Which ridge solver this configuration uses.

    'auto' keys off n_obs rather than support size: a split run and a no-split run at
    the same n_obs then agree on the solver, so comparing them never mixes a
    modelling change with a numerical one.

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=20000,
                        help="logged sample size n")
    parser.add_argument("--n_particles", type=int, default=2000,
                        help="flow particles M (capped at the on-arm causal-split size)")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="size of the fixed oracle reference draw every row is scored "
                             "against, and of the second draw that is the floor; "
                             "independent of M and of n_obs")
    parser.add_argument("--eval_particles", type=int, default=2000,
                        help="score the decoded initial/flow rows on a fixed random subset "
                             "of this many particles (the whole set when M <= this); keeps "
                             "the evaluated count at the reference count when M is larger")
    parser.add_argument("--n_dr", type=int, default=None,
                        help="optional cap on the DR support, kept for reproducing the "
                             "pre-2026-09 runs (--solver exact --n_dr 4000). Unset, the "
                             "support is every eligible row.")
    parser.add_argument("--ae_fraction", type=float, default=0.5,
                        help="fraction of the logged sample used to train the autoencoder")
    parser.add_argument("--target_arms", type=int, nargs="+", default=[0, 1],
                        choices=(0, 1), help="intervention arms run in each repeat")
    parser.add_argument("--w_conf", type=float, default=10.0,
                        help="confounding strength w in e(1|x) = sigmoid(w (x - 1/2))")
    parser.add_argument("--tau", type=float, default=0.08, help="foreground mask threshold")
    parser.add_argument("--alpha", type=float, default=0.2, help="fixed green channel")
    parser.add_argument("--k_mask", type=float, default=20.0, help="foreground mask sharpness")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="pipeline repeats per data seed")
    parser.add_argument("--fid", type=str, default="both",
                        choices=("none", "legacy", "clean", "both"),
                        help="FID of the decoded initial / flowed particles and of a second "
                             "oracle draw (floor) against the oracle draw, with the scorers "
                             "of metric_utils.fid: 'legacy' reproduces the FID column of the "
                             "Dance et al. (2026) tables (Frechet distance on the GPU, their "
                             "ColorMNIST convention), 'clean' is standard Clean-FID. "
                             "'clean' needs the weights cached in ~/.cache/torch/clean-fid.")
    parser.add_argument("--fid_workers", type=int, default=8,
                        help="threads for Clean-FID's per-image resize (results identical)")
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
    parser.add_argument("--ae_split", type=str, default="none", choices=("half", "none"),
                        help="'half' trains the autoencoder on a disjoint half of the "
                             "logged sample (the reference design); 'none' trains it on "
                             "every row and lets the causal stage reuse them, trading the "
                             "feature map's independence for twice the eligible data.")
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
                             "The DR support is EVERY eligible row under all of "
                             "them, so what the choice changes is the nuisance estimate, "
                             "not the sample, and a results table mixing two solvers is "
                             "mixing two estimators. The threshold is keyed on n_obs "
                             "rather than support size, so split and no-split runs at the "
                             "same n_obs always share a solver and their comparison is "
                             "never confounded by it.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly; above it a "
                             "fold would be a (4/5 n_obs)^2 float64 system"
                             ", and where 'auto' refuses instead of choosing a "
                             "solver for you")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold")
    parser.add_argument("--estimators", type=str, nargs="+", default=["dr"],
                        choices=("plugin", "ipw", "dr"),
                        help="which estimators of the target embedding to sample from, "
                             "one CSV row each: 'plugin' is the conditional mean "
                             "embedding alone, v_PI; 'ipw' the importance ratios alone, "
                             "v_IPW = w / n; 'dr' (the default) the one-step estimator "
                             "v_DR = v_PI + v_correction. They come out of ONE nuisance "
                             "solve and are run on the SAME logged sample, autoencoder, "
                             "starting particles, flow budget and oracle draws, so the "
                             "comparison is paired. Off the default the output filename "
                             "gains an _est token; the authoritative record is the "
                             "estimator column.")
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
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"),
                        help="how the DR nuisances are fitted. DEFAULT CHANGED 2026-09-28 to 'nosplit', which fits the CME and the propensity on ALL rows; 'crossfit' (the previous default) fits both out of fold. The same-sample notes license reusing observations for the NUISANCES. NOTE the filename token is deliberately still keyed on 'crossfit': a 'nosplit' run appends _drnosplit and a 'crossfit' run appends nothing, exactly as before the default flipped, so every historical filename is still reproduced byte for byte and no new run can overwrite an archived one. The dr_fit CSV column is the authoritative record. "
                             "historical behaviour) fits the CME and the propensity out of "
                             "fold; 'nosplit' fits both on ALL rows. NOTE this is a DIFFERENT "
                             "axis from --ae_split, which controls whether the autoencoder "
                             "sees the causal rows. The same-sample notes license reusing "
                             "observations for NUISANCES but explicitly not for an encoder "
                             "selected on those rows, so --ae_split half --dr_fit nosplit is "
                             "the combination their theory covers.")
    parser.add_argument("--lambda_rule", type=str, default="cv",
                        choices=("cv", "n_half"),
                        help="ridge selection. 'cv' (default, historical) cross-validates per "
                             "training block; 'n_half' uses lambda = c * n^-1/2, the rate the "
                             "same-sample theory gives as sufficient, and keeps the penalty "
                             "matched between --dr_fit arms.")
    parser.add_argument("--lambda_constant", type=float, default=1.0,
                        help="the constant c in --lambda_rule n_half")
    parser.add_argument("--snapshot_every", type=int, default=None,
                        help="store the particle cloud every this many iterations so one run "
                             "yields a whole iteration curve. This sampler uses a FIXED Euler "
                             "step, so a snapshot at step s is exactly a run of s steps -- "
                             "unlike the cosine-scheduled Riesz simulations.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--ae_iterations", type=int, default=3000)
    parser.add_argument("--flow_iterations", type=int, default=3000)
    parser.add_argument("--tag", type=str, default="refparams")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--mnist_root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available")
    return parser.parse_args()


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    suffix += "_nosplit" if args.ae_split == "none" else ""
    # Tokens appended only off-default so historical filenames are reproduced exactly.
    # `_dr*` is deliberately distinct from the `_nosplit` token just above: --ae_split
    # and --dr_fit are different ablations and their filenames must not collide.
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
    # Only off the default, so a dr-only run keeps its historical name and a
    # multi-estimator run can never overwrite it.
    estimators = resolve_estimators(args)
    if estimators != ["dr"]:
        suffix += "_est" + "-".join(estimators)
    if args.ipw_form != "ht":
        suffix += f"_ipw{args.ipw_form}"
    if args.flow_iterations != 3000:
        suffix += f"_it{args.flow_iterations}"
    name = (f"colormnist_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
            f"_w{args.w_conf:g}_a{args.alpha:g}_k{args.k_mask:g}_tau{args.tau:g}"
            f"_arms{arms}_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def evaluate_images(images_01: torch.Tensor, reference_01: torch.Tensor,
                    kernel_pixel: RBF) -> dict:
    """Colour law plus flat-pixel SW1 / SW2 / unbiased RBF MMD^2 against the reference.

    `colour_w2` is the one-dimensional Wasserstein-2 distance between the colour-ratio
    laws (exact for equal counts through the sorted-projection estimator), the
    "Color W2" column of the Dance et al. (2026) ColorMNIST table."""
    flatten = lambda images: images.reshape(images.shape[0], -1)
    ratio = color_ratio(images_01)
    ratio_reference = color_ratio(reference_01)
    return {
        "colour_mean": float(ratio.mean()), "colour_std": float(ratio.std()),
        "colour_w2": sliced_wasserstein(ratio[:, None], ratio_reference[:, None], p=2,
                                        n_projections=1, seed=0).item(),
        "sw1": sliced_wasserstein(flatten(images_01), flatten(reference_01), p=1, seed=0).item(),
        "sw2": sliced_wasserstein(flatten(images_01), flatten(reference_01), p=2, seed=0).item(),
        "mmd2": squared_mmd(kernel_pixel, flatten(images_01), flatten(reference_01),
                            unbiased=True).item(),
    }


def subsample_rows(sample: torch.Tensor, k: int,
                   generator: "torch.Generator") -> torch.Tensor:
    """A random k-row subset of `sample`, or the whole sample when it has at most k rows.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, ...)
    k : int
        Target row count.
    generator : torch.Generator
        CPU generator; the permutation is drawn on the CPU so the subset does not
        depend on the device.

    Returns
    -------
    torch.Tensor of shape (min(n, k), ...)
    """
    if sample.shape[0] <= k:
        return sample
    pick = torch.randperm(sample.shape[0], generator=generator)[:k]
    return sample[pick.to(sample.device)]


def matched_floor(images_01: torch.Tensor, reference_01: torch.Tensor,
                  kernel_pixel: RBF, k: int, seed: int) -> dict:
    """The oracle-vs-oracle floor recomputed at the count this row's flow was scored at.

    Sliced Wasserstein, the colour-ratio W2 and the unbiased MMD^2 are all SAMPLE-SIZE
    BIASED: two independent draws of the SAME law score further apart at small counts
    than at large ones. `sliced_wasserstein` truncates to the smaller of its two inputs,
    so a flow row is effectively scored at k = min(n_scored, n_ref) while the plain
    floor is scored at the full n_ref. Sweeping the particle count then moves the flow
    rows while the floor stays put, which reads as a spurious "more particles is
    better". Subsampling BOTH floor inputs to the same k removes that gap; the subsets
    come from a fixed seed derived from the pipeline seed, so the columns are
    reproducible.

    Parameters
    ----------
    images_01, reference_01 : torch.Tensor of shape (n, 3, 28, 28)
        The two independent oracle draws, in [0, 1].
    kernel_pixel : RBF
        Pixel kernel with its bandwidth already locked on the reference draw.
    k : int
        Effective count at which this row's flow sample was scored.
    seed : int
        Seed of the CPU generator drawing both subsets.

    Returns
    -------
    dict
        The keys of `evaluate_images`, to be suffixed by the caller.
    """
    generator = torch.Generator().manual_seed(seed)
    return evaluate_images(subsample_rows(images_01, k, generator),
                           subsample_rows(reference_01, k, generator), kernel_pixel)


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


def target_weight_vectors(target_a: int, A_s: torch.Tensor, w_hat: torch.Tensor,
                          K_x, K_a, gram, gram_x, solver: str, fold_ids: np.ndarray,
                          fold_lambdas: dict, args: argparse.Namespace, seed: int,
                          device: torch.device, nosplit_lambda: float,
                          estimators: list) -> dict:
    """The requested estimators' target weight vectors, out of ONE nuisance solve.

    Every estimator is a linear functional of the same fit -- v_DR = v_PI +
    v_correction by definition, and v_IPW = w / n uses no fit at all -- so one solve
    with `components=True` yields all three and no second nuisance is ever estimated.
    That is also what keeps the comparison paired: a second solve would redraw the
    Nystrom landmarks and the estimators would no longer share a CME.

    When DR alone is requested the solver is called WITHOUT `components`, because the
    component split reproduces the plain return only to floating-point precision and
    the DR-only path is the one every existing result CSV was produced by. A DR row
    from a multi-estimator run therefore need not agree to the last digit with a
    DR-only run of the same configuration: the weight vectors agree to float32, but
    `objective_final` is an MMD^2 near zero and the flow amplifies the last bits into
    it. The distributional columns are unaffected at reporting precision.

    Parameters
    ----------
    target_a : int
        Intervention arm. The target policy is deterministic, so
        K_A_pi[i, j] = 1{a_i == target_a} and K_pi = K_X * 1{a_i == target_a}.
    A_s : torch.Tensor of shape (n_dr,)
        Logged actions on the DR support.
    w_hat : torch.Tensor of shape (n_dr,)
        Importance ratios for this arm, 1{a_i == target_a} / e_hat(target_a | x_i).
    K_x, K_a : torch.Tensor of shape (n_dr, n_dr) or None
        Dense context and action Grams; the exact solver alone uses them.
    gram, gram_x : callable
        Block accessors for K = K_X * K_A and for K_X on its own.
    solver : {'exact', 'nystrom', 'projected_nystrom'}
        As resolved by `resolve_solver`.
    fold_ids : np.ndarray of shape (n_dr,)
        Cross-fitting folds, used when --dr_fit crossfit.
    fold_lambdas : dict
        Per-fold ridge values; `nosplit_lambda` is the single value of the no-split fit.
    args : argparse.Namespace
    seed : int
        Pipeline seed, which fixes the landmark draw.
    device : torch.device
        Where the flow runs; the vectors are moved there in float32.
    nosplit_lambda : float
    estimators : list of {'plugin', 'ipw', 'dr'}

    Returns
    -------
    dict
        `estimators` -> weight vector of shape (n_dr,), in that order.
    """
    n_dr = len(w_hat)
    components = set(estimators) != {"dr"}
    if solver in ("nystrom", "projected_nystrom"):
        # K_pi[i, j] = K_X[i, j] * 1{a_i == a}: the target action is deterministic
        on_arm_flag = (A_s == target_a).double().to(device)
        gram_pi = lambda rows, cols: gram_x(rows, cols) * on_arm_flag[rows][:, None]
        # `projected_nystrom` restricts the CME ridge regression to the landmark-spanned
        # subspace and builds the DR right-hand side only at the landmarks. That is the
        # estimator the imported consistency rate is stated for, and it touches an
        # m x |H| cross-kernel instead of a |T| x |H| one. `nystrom` applies an
        # approximate Gram inverse to the exact full right-hand side -- a different
        # estimator, kept as the baseline that produced every earlier table.
        projected = solver == "projected_nystrom"
        if args.dr_fit == "nosplit":
            solve = (cpme_dr_weights_projected_nystrom if projected
                     else cpme_dr_weights_nystrom)
            solved = solve(gram, gram_pi, w_hat, nosplit_lambda,
                           n_landmarks=args.n_landmarks,
                           generator=torch.Generator().manual_seed(seed),
                           components=components)
        else:
            solve = (cpme_dr_weights_crossfit_projected_nystrom if projected
                     else cpme_dr_weights_crossfit_nystrom)
            solved = solve(gram, gram_pi, w_hat, fold_ids, fold_lambdas,
                           n_landmarks=args.n_landmarks,
                           generator=torch.Generator().manual_seed(seed),
                           components=components)
    else:
        K_a_target = ((A_s == target_a).double().to(device)[:, None]
                      .expand(n_dr, n_dr).contiguous())
        if args.dr_fit == "nosplit":
            solved = cpme_dr_weights(K_x, K_a, K_a_target, w_hat, nosplit_lambda,
                                     components=components)
        else:
            solved = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_hat, fold_ids,
                                              fold_lambdas, components=components)
    if not components:
        return {"dr": solved.float().to(device)}
    v_plugin, v_correction = solved
    available = {"plugin": v_plugin,
                 "ipw": cpme_ipw_weights(w_hat, normalize=args.ipw_form == "hajek"),
                 "dr": v_plugin + v_correction}
    return {name: available[name].float().to(device) for name in estimators}


def flow_and_score(name: str, v_target: torch.Tensor, initial_particles: torch.Tensor,
                   Z_s: torch.Tensor, kernel, step_size: float,
                   eval_idx: torch.Tensor, ae, Y_oracle: torch.Tensor,
                   Z_oracle: torch.Tensor, kernel_pixel: RBF, fid_scorers: dict,
                   oracle_features: dict, args: argparse.Namespace) -> dict:
    """Flow one estimator's target embedding and score the cloud it produces.

    Returns exactly the columns that depend on the target weight vector: the shape of
    the vector itself, the flow objective, the `*_flow` metrics, and this row's four
    PER-ESTIMATOR timings. Everything else in the row is shared by the estimators of this
    arm, and no shared work is inside any timing window here -- that is what makes
    `seconds_flow`, `seconds_decode` and `seconds_eval` summable across the rows of a
    repeat. The flow budget is taken from `args` and is therefore identical across
    estimators, so nothing but the target distinguishes the clouds.

    Parameters
    ----------
    name : {'plugin', 'ipw', 'dr'}
        Estimator that supplied `v_target`; goes to the `estimator` column.
    v_target : torch.Tensor of shape (n_dr,)
        Target embedding weights over the support `Z_s`.
    initial_particles : torch.Tensor of shape (M, latent_dim)
        The shared starting cloud.
    Z_s : torch.Tensor of shape (n_dr, latent_dim)
        Encoded DR support, the target embedding's expansion points.
    kernel, step_size
        The shared IMQ kernel and forward-Euler step.
    eval_idx : torch.Tensor
        Particle rows the pixel metrics are computed on.
    ae : torch.nn.Module
        Frozen autoencoder; only its decoder is used here.
    Y_oracle : torch.Tensor of shape (n_ref, 3, 28, 28)
        The oracle reference draw, in [0, 1].
    Z_oracle : torch.Tensor of shape (n_ref, latent_dim)
        That draw encoded, for the latent SW2.
    kernel_pixel : RBF
        Pixel kernel with its bandwidth already locked on `Y_oracle`.
    fid_scorers : dict
        Scorer per implementation, or empty / None for no FID.
    oracle_features : dict
        `Y_oracle`'s features under each scorer, computed once per arm.
    args : argparse.Namespace

    Returns
    -------
    dict
        This row's weight diagnostics, flow objective and `*_flow` metrics, plus
        `seconds_flow`, `seconds_decode`, `seconds_eval` and `seconds_fid`. The last is a
        BREAKDOWN of `seconds_eval` (the FID window sits inside the evaluation window),
        not a further term to add to it.
    """
    device = initial_particles.device
    t0 = time.time()
    particles, history = mmd_gradient_flow(kernel, target_samples=Z_s,
                                           initial_particles=initial_particles,
                                           step_size=step_size,
                                           n_iterations=args.flow_iterations,
                                           target_weights=v_target,
                                           snapshot_every=args.snapshot_every,
                                           log_every=500, verbose=False)
    sync_device(device)
    seconds_flow = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        decoded_flowed = to_image_range(ae.decode(particles[eval_idx]))
    sync_device(device)
    seconds_decode = time.time() - t0

    # Everything from here to `seconds_eval` is this row's OWN scoring. No shared work is
    # inside the window, which is what makes the column summable across estimator rows.
    t0 = time.time()
    row = {"estimator": name}
    row.update(weight_diagnostics(v_target))
    row.update({
        "objective_initial": float(history["squared_mmd"][0]),
        "objective_final": float(history["squared_mmd"][-1]),
        "latent_sw2_flow": sliced_wasserstein(particles, Z_oracle, p=2, seed=0).item(),
    })
    row.update({f"{key}_flow": value for key, value
                in evaluate_images(decoded_flowed, Y_oracle, kernel_pixel).items()})
    t_fid = time.time()
    for impl, scorer in (fid_scorers or {}).items():
        row[f"fid_{impl}_flow"] = scorer.distance_from_features(
            scorer.features(decoded_flowed, image_range="zero_one"),
            oracle_features[impl])
    seconds_fid = time.time() - t_fid
    sync_device(device)
    seconds_eval = time.time() - t0

    row.update({"seconds_flow": round(seconds_flow, 1),
                "seconds_decode": round(seconds_decode, 1),
                "seconds_eval": round(seconds_eval, 1),
                "seconds_fid": round(seconds_fid, 1)})
    return row


def run_arm(target_a: int, ae, pools: list, data: torch.Tensor,
            causal_idx: torch.Tensor, A_obs: torch.Tensor, A_s: torch.Tensor,
            Z_s: torch.Tensor, e1: torch.Tensor, K_x, K_a, gram, gram_x, solver: str,
            fold_ids: np.ndarray, fold_lambdas: dict, args: argparse.Namespace,
            seed: int, device: torch.device, fid_scorers: dict = None,
            nosplit_lambda: float = None) -> tuple:
    """Arm-specific half of one repeat: one row per requested estimator.

    Everything but the flow and the metrics of its output is shared by the estimators:
    the importance ratios, the single nuisance solve behind all three weight vectors,
    the starting particles, the IMQ bandwidth and step, and both oracle draws with the
    floors, the decoder round-trip and the FID reference features taken from them. The
    estimators therefore differ ONLY in the target weight vector handed to the flow,
    which is what makes their comparison paired instead of a comparison of seeds.

    Returns
    -------
    rows : list of dict
        One row per requested estimator.
    arm_seconds : dict
        This ARM's shared wall clock, which belongs to no single row and so is returned
        for the caller to accumulate over arms instead of being written into the rows:
        'ae' (this arm's observed-latent encode, part of the repeat's `seconds_ae`),
        'ratio' and 'cme' (its importance ratios and its target-weight solve, the two
        halves of the repeat's `seconds_nuisance`), 'eval_shared' (the particle / kernel
        setup, the oracle draws and every shared metric, FID excluded) and 'fid_shared'
        (the shared FID). See "Wall-clock accounting" in the module docstring.
    """
    estimators = resolve_estimators(args)
    # This arm's importance ratios: the ratio half's share of the arm, nuisance estimation
    # rather than evaluation, so the caller folds it into the repeat's `seconds_ratio`.
    t0 = time.time()
    e_target = e1 if target_a == 1 else 1.0 - e1
    w_hat = ((A_s == target_a).double() / e_target).to(device)
    on_arm = w_hat[(A_s == target_a).to(device)]
    ess = float(on_arm.sum() ** 2 / (on_arm ** 2).sum())
    sync_device(device)
    seconds_ratio = time.time() - t0

    # This arm's target-weight solve: the CME half's share of the arm. One solve serves
    # every estimator, and returns all of their vectors.
    t0 = time.time()
    vectors = target_weight_vectors(target_a, A_s, w_hat, K_x, K_a, gram, gram_x,
                                    solver, fold_ids, fold_lambdas, args, seed, device,
                                    nosplit_lambda, estimators)
    sync_device(device)
    seconds_cme = time.time() - t0

    # This arm's observed latents: an encoder pass, so part of the autoencoder stage rather
    # than of the nuisance fit or the scoring. The starting particles are drawn from them.
    t0 = time.time()
    arm_causal = causal_idx[A_obs[causal_idx] == target_a]
    Z_arm = encode_in_batches(ae.encode, data[arm_causal])
    sync_device(device)
    seconds_ae = time.time() - t0

    # Shared across the estimators of this arm: particle setup, the oracle draws, the
    # floors and the round-trip. Timed as one window into `seconds_eval_shared`.
    t0 = time.time()
    n_particles = min(args.n_particles, Z_arm.shape[0])
    # CPU generator: the CUDA stream is device-dependent, so a GPU draw cannot be
    # reproduced by another implementation (or on another card).
    pick = torch.randperm(Z_arm.shape[0],
                          generator=particle_generator(args, seed, 1))[:n_particles].to(device)
    PARTICLE_INDICES[f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"] = (
        arm_causal[pick].cpu().numpy())
    if args.init == "observed":
        initial_particles = Z_arm[pick].clone()
    else:
        initial_particles = gaussian_init(
            Z_s, n_particles, args.init,
            torch.Generator(device=device).manual_seed(seed + 3))

    c_imq = imq_c_from_median(Z_s)
    kernel = IMQKernel(c=c_imq, beta=-0.5)
    step_size = 0.15 * c_imq ** 2

    eval_idx = torch.arange(n_particles, device=initial_particles.device)
    if args.eval_particles is not None and n_particles > args.eval_particles:
        eval_idx = torch.randperm(n_particles, generator=torch.Generator(device=device).manual_seed(seed + 11),
                                  device=device)[:args.eval_particles]
    with torch.no_grad():
        decoded_initial = to_image_range(ae.decode(initial_particles[eval_idx]))
    _, Y_oracle = sample_interventional(pools, args.n_ref, a=target_a, tau=args.tau,
                                        alpha=args.alpha, k_mask=args.k_mask,
                                        generator=torch.Generator(device=device).manual_seed(seed + 9))
    _, Y_oracle_2 = sample_interventional(pools, args.n_ref, a=target_a, tau=args.tau,
                                          alpha=args.alpha, k_mask=args.k_mask,
                                          generator=torch.Generator(device=device).manual_seed(seed + 10))
    # Decoder ceiling: the second oracle draw passed through the autoencoder. Its
    # distance to the first draw is what a PERFECT flow could reach through this
    # decoder; the floor row (raw second draw) is what a perfect sampler could reach.
    with torch.no_grad():
        decoded_oracle = to_image_range(ae.decode(encode_in_batches(
            ae.encode, to_model_range(Y_oracle_2))))
    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(Y_oracle.reshape(len(Y_oracle), -1))  # lock the bandwidth on the reference

    # Latent SW2: particles against the ENCODED oracle draw, in the raw AE latent the
    # flow operates in (ColorMNIST does not standardise its latents). The floor encodes
    # the second oracle draw, so it is the same quantity computed on two independent
    # samples of the true interventional law.
    with torch.no_grad():
        Z_oracle = encode_in_batches(ae.encode, to_model_range(Y_oracle))
        Z_oracle_2 = encode_in_batches(ae.encode, to_model_range(Y_oracle_2))
    # Count-matched floors (see `matched_floor`). The latent metrics score EVERY
    # particle while the pixel metrics score the --eval_particles subset, so the two
    # effective counts differ whenever M exceeds --eval_particles and each floor takes
    # its own k.
    k_pixel = min(len(eval_idx), len(Y_oracle))
    k_latent = min(n_particles, len(Z_oracle))
    latent_generator = torch.Generator().manual_seed(seed + 13)
    latent_floor_matched = sliced_wasserstein(
        subsample_rows(Z_oracle_2, k_latent, latent_generator),
        subsample_rows(Z_oracle, k_latent, latent_generator), p=2, seed=0).item()

    ratio_oracle = color_ratio(Y_oracle)
    shared = {
        "target_a": target_a,
        "n_arm_causal": int(len(arm_causal)),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "n_flow_eval": int(len(eval_idx)), "n_ref": int(len(Y_oracle)),
        "ess_on_arm": ess, "n_on_arm": int(len(on_arm)), "max_weight": float(w_hat.max()),
        "c_imq": float(c_imq), "step_size": float(step_size),
        "latent_sw2_initial": sliced_wasserstein(initial_particles, Z_oracle, p=2,
                                                 seed=0).item(),
        "latent_sw2_floor": sliced_wasserstein(Z_oracle_2, Z_oracle, p=2, seed=0).item(),
        "latent_sw2_floor_matched": latent_floor_matched,
        "n_floor_matched": int(k_pixel), "n_floor_matched_latent": int(k_latent),
        "colour_mean_oracle": float(ratio_oracle.mean()),
        "colour_std_oracle": float(ratio_oracle.std()),
    }
    for name, images in (("initial", decoded_initial), ("floor", Y_oracle_2),
                         ("roundtrip", decoded_oracle)):
        metrics = evaluate_images(images, Y_oracle, kernel_pixel)
        shared.update({f"{key}_{name}": value for key, value in metrics.items()})
    matched = matched_floor(Y_oracle_2, Y_oracle, kernel_pixel, k_pixel, seed + 12)
    shared.update({f"{key}_floor_matched": value for key, value in matched.items()})
    sync_device(device)
    seconds_eval_shared = time.time() - t0

    t0 = time.time()
    oracle_features = {}
    for impl, scorer in (fid_scorers or {}).items():
        oracle_features[impl] = scorer.features(Y_oracle, image_range="zero_one")
        for name, images in (("initial", decoded_initial), ("floor", Y_oracle_2),
                             ("roundtrip", decoded_oracle)):
            shared[f"fid_{impl}_{name}"] = scorer.distance_from_features(
                scorer.features(images, image_range="zero_one"), oracle_features[impl])
    sync_device(device)
    seconds_fid_shared = time.time() - t0

    rows = []
    for name in estimators:
        row = dict(shared)
        # `flow_and_score` returns this row's OWN timings only. The shared FID above used
        # to be folded into each row's `seconds_fid`, which made the column unsummable:
        # three estimators reported three copies of one cost. It is returned separately
        # now and surfaced as the `seconds_fid_shared` column.
        row.update(flow_and_score(name, vectors[name], initial_particles, Z_s, kernel,
                                  step_size, eval_idx, ae, Y_oracle, Z_oracle,
                                  kernel_pixel, fid_scorers, oracle_features, args))
        rows.append(row)
    return rows, {"ae": seconds_ae, "ratio": seconds_ratio, "cme": seconds_cme,
                  "eval_shared": seconds_eval_shared, "fid_shared": seconds_fid_shared}


def run_repeat(pools: list, data: torch.Tensor, X_obs: torch.Tensor, A_obs: torch.Tensor,
               setup: dict, args: argparse.Namespace, data_seed: int, repeat: int,
               device: torch.device, fid_scorers: dict = None) -> list:
    """One pipeline pass on a fixed logged sample; one CSV row per (arm, estimator).

    The timing columns are assembled here rather than in `run_arm`, because the shared
    stages (the autoencoder, the nuisance stage, each arm's shared scoring) belong to the
    repeat and not to any one row: they are totalled over the arms and written identically
    onto every row, and `seconds_other` is then the residual that closes the per-repeat
    identity in the module docstring exactly. `setup` carries this data seed's
    per-DATA-SEED columns, measured in `main` outside this repeat's clock.
    """
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- split: autoencoder half / causal half (disjoint) ------------------------------
    if args.ae_split == "none":
        ae_idx = causal_idx = torch.arange(args.n_obs, device=device)
        n_ae = args.n_obs
    else:
        n_ae = int(round(args.ae_fraction * args.n_obs))
        split = torch.randperm(args.n_obs,
                               generator=torch.Generator(device=device).manual_seed(seed + 2),
                               device=device)
        ae_idx, causal_idx = split[:n_ae], split[n_ae:]

    # -- autoencoder feature map on the AE half only -----------------------------------
    t0 = time.time()
    ae, _ = train_conv_ae(data[ae_idx], latent_dim=args.latent_dim,
                          n_iterations=args.ae_iterations, batch_size=128,
                          learning_rate=1e-3, device=device,
                          generator=torch.Generator(device=device).manual_seed(seed),
                          log_every=max(1, args.ae_iterations // 2), verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        mse_train = float((ae(data[ae_idx][:2000]) - data[ae_idx][:2000]).pow(2).mean())
        mse_heldout = float((ae(data[causal_idx][:2000]) - data[causal_idx][:2000]).pow(2).mean())
    sync_device(device)
    seconds_ae = time.time() - t0

    # -- the ratio half of the nuisance fit: the DR support, the folds and the propensity.
    # This is all the IPW estimator needs; the support and the folds are clocked here
    # because this half consumes them first, and the CME half reuses both.
    t0 = time.time()
    solver = resolve_solver(args)
    # The support is every eligible row; --n_dr caps it only when asked for explicitly.
    if args.n_dr is not None and args.n_dr < len(causal_idx):
        pick_dr = torch.randperm(len(causal_idx),
                                 generator=torch.Generator(device=device).manual_seed(seed + 5),
                                 device=device)[:args.n_dr]
        subsample = causal_idx[pick_dr]
    else:
        subsample = causal_idx
    n_dr = len(subsample)
    X_s, A_s = X_obs[subsample], A_obs[subsample]
    sync_device(device)
    seconds_ratio = time.time() - t0

    # The DR support's latents are the autoencoder's product, not a nuisance: the CME, the
    # flow and the metrics all read them, so the pass belongs to the autoencoder stage.
    # CelebA, whose encoder is frozen and pretrained, can hoist the same work out of the
    # repeat entirely (its `seconds_encode`); here the encoder is this repeat's own.
    t0 = time.time()
    Z_s = encode_in_batches(ae.encode, data[subsample])
    sync_device(device)
    seconds_ae += time.time() - t0

    t0 = time.time()
    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    torch.manual_seed(seed)
    if args.dr_fit == "nosplit":
        # Fit once on all rows and predict on them. NOT crossfit_propensity with one
        # fold: that silently returns a constant 0.5, which would leave the two arms
        # differing only in the CME nuisance.
        model = LogisticPropensityModel()
        model.fit(X_s.unsqueeze(1).double().cpu(), A_s.double().cpu())
        e1 = model.predict(X_s.unsqueeze(1).double().cpu()).reshape(-1).double().to(device)
    else:
        e1 = crossfit_propensity(X_s.unsqueeze(1).double().cpu(),
                                 A_s.double().cpu(), fold_ids).to(device)
    sync_device(device)
    seconds_ratio += time.time() - t0

    # -- the CME half: the context kernel, the Gram preparation and the ridge selection.
    # This is all the plug-in estimator needs, and DR needs both halves. Each arm's single
    # target-weight solve is added to this accumulator from the arm loop below.
    t0 = time.time()
    # The RBF lengthscale is locked once on a bounded subsample; the median over the
    # full n_dr^2 pairs is both unaffordable and unnecessary at n_dr = 20000.
    kernel_x = RBF(use_length_scale_heuristic=True)
    # The nuisance runs on the SAME device as the autoencoder, the flow and the evaluation,
    # so a reported wall clock is one device's. The projected solver only ever requests the
    # (all, L), (L, L) and (L, all) blocks, so the largest tensor it forms is n x
    # n_landmarks -- 164 MB in float64 at n_dr = 20000 against 3.2 GB for a dense n x n
    # block, which is why keeping these on the device is affordable.
    X_dr = X_s.unsqueeze(1).double().to(device)
    A_dr = A_s.to(device)
    anchor = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:4000]
    _ = kernel_x(X_dr[anchor])

    def gram_x(rows, cols):
        "K_X block alone; K_pi = K_X * 1{a_i == a} needs it without the action factor."
        return kernel_x(X_dr[rows], X_dr[cols])

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_dr^2 Gram is never formed."
        return gram_x(rows, cols) * (A_dr[rows][:, None] == A_dr[cols][None, :]).double()

    # lambda does not need the full training block, which would itself be an n^3 solve
    if args.lambda_rule == "n_half":
        nosplit_lambda = lambda_n_half(n_dr, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    else:
        fold_lambdas = {}
        for fold in np.unique(fold_ids):
            train = np.where(fold_ids != fold)[0]
            pick = train[torch.randperm(len(train),
                                        generator=torch.Generator().manual_seed(0))[:2000].numpy()]
            fold_lambdas[int(fold)], _ = select_regularization(gram(pick, pick),
                                                               Z_s[pick, 0].double().to(device), seed=0)
        nosplit_lambda = None
        if args.dr_fit == "nosplit":
            pick = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:2000].numpy()
            nosplit_lambda, _ = select_regularization(gram(pick, pick),
                                                      Z_s[pick, 0].double().to(device), seed=0)
    K_x = kernel_x(X_dr) if solver == "exact" else None
    K_a = (A_dr[:, None] == A_dr[None, :]).double() if solver == "exact" else None
    sync_device(device)
    seconds_cme = time.time() - t0

    # -- per-arm: DR weights, flow, evaluation -----------------------------------------
    shared = {
        "tag": args.tag, "device": str(device), "init": args.init,
        "ae_split": args.ae_split,
        "solver": solver, "dr_fit": args.dr_fit, "lambda_rule": args.lambda_rule,
        "ipw_form": args.ipw_form,
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "n_obs": args.n_obs, "w_conf": args.w_conf,
        "tau": args.tau, "alpha": args.alpha, "k_mask": args.k_mask,
        "n_ae": int(n_ae), "n_causal": int(len(causal_idx)),
        "n_dr": int(n_dr), "n_folds": args.n_folds, "latent_dim": args.latent_dim,
        "ae_iterations": args.ae_iterations, "flow_iterations": args.flow_iterations,
        "recon_mse_train": mse_train, "recon_mse_heldout": mse_heldout,
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        # Identical on every row of every repeat of this data seed, and measured outside
        # this repeat's clock: the second term of the per-task identity.
        **setup,
        **hardware_provenance(device),
    }
    rows = []
    # Arm-level shared costs, accumulated over arms: the repeat's shared columns are
    # totals over BOTH arms, never one arm's own cost (see the module docstring).
    arm_totals = {"ae": 0.0, "ratio": 0.0, "cme": 0.0, "eval_shared": 0.0,
                  "fid_shared": 0.0}
    for target_a in args.target_arms:
        arm_rows, arm_seconds = run_arm(target_a, ae, pools, data, causal_idx, A_obs, A_s,
                                        Z_s, e1, K_x, K_a, gram, gram_x, solver, fold_ids,
                                        fold_lambdas, args, seed, device, fid_scorers,
                                        nosplit_lambda)
        for key in arm_totals:
            arm_totals[key] += arm_seconds[key]
        for arm_row in arm_rows:
            row = dict(shared)
            row.update(arm_row)
            rows.append(row)
    seconds_repeat = round(time.time() - t_start, 1)

    # -- wall clock: shared columns, then the residual that closes the identity ---------
    # Each shared column is a repeat TOTAL written identically onto every row, so it must
    # never be summed across the rows of a repeat; the per-estimator columns already on
    # the rows (flow / decode / eval) may be. `seconds_shared` is built from the ROUNDED
    # parts, and `seconds_other` from the rounded columns, so that
    #     seconds_repeat == seconds_shared + seconds_other
    #                       + sum_rows (seconds_flow + seconds_decode + seconds_eval)
    # holds exactly on the numbers the CSV carries rather than up to rounding.
    shared_seconds = {
        "seconds_ae": round(seconds_ae + arm_totals["ae"], 1),
        "seconds_ratio": round(seconds_ratio + arm_totals["ratio"], 1),
        "seconds_cme": round(seconds_cme + arm_totals["cme"], 1),
        "seconds_eval_shared": round(arm_totals["eval_shared"], 1),
        "seconds_fid_shared": round(arm_totals["fid_shared"], 1),
    }
    # The two nuisance halves are a BREAKDOWN, so seconds_nuisance is their rounded sum and
    # only it enters seconds_shared -- adding all three would count the fit twice.
    shared_seconds["seconds_nuisance"] = round(shared_seconds["seconds_ratio"]
                                               + shared_seconds["seconds_cme"], 1)
    # Named explicitly rather than summing the dict, so adding a shared column later is a
    # deliberate act: it has to be listed here or the identity will not close.
    shared_seconds["seconds_shared"] = round(
        shared_seconds["seconds_ae"] + shared_seconds["seconds_nuisance"]
        + shared_seconds["seconds_eval_shared"] + shared_seconds["seconds_fid_shared"], 1)
    own = sum(row["seconds_flow"] + row["seconds_decode"] + row["seconds_eval"]
              for row in rows)
    # The trailing + 0.0 turns the -0.0 that float subtraction yields when the parts cancel
    # exactly into a plain 0.0, so the CSV reads "0.0" and not "-0.0".
    seconds_other = round(seconds_repeat - shared_seconds["seconds_shared"] - own, 1) + 0.0
    for row in rows:
        row.update(shared_seconds)
        # Derived, and in neither identity: this estimator's own half of the nuisance fit
        # plus this row's own flow and decode, with no autoencoder and no measurement.
        row["seconds_sampling"] = sampling_seconds(row["estimator"], shared_seconds,
                                                   row["seconds_flow"],
                                                   row["seconds_decode"])
        row["seconds_other"] = seconds_other
        row["seconds_repeat"] = seconds_repeat
    return rows


def main() -> None:
    args = parse_args()
    warn_nonprojected_solver(resolve_solver(args), args.n_obs, args.n_landmarks)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = csv_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   output: {out_path}", flush=True)

    # Read MNIST straight from the repo's data/ folder; torchvision only downloads
    # when the files are absent, and compute nodes may have no internet, so fail
    # loudly here rather than let a missing folder turn into a download attempt.
    if not (Path(args.mnist_root) / "MNIST" / "raw" / "train-images-idx3-ubyte").exists():
        sys.exit(f"MNIST not found under {args.mnist_root}/MNIST/raw — "
                 "copy the repo's data/ folder there (no download on compute nodes).")
    # The per-DATA-SEED setup clock. It starts HERE so the once-per-process loads -- the
    # digit pools and the FID scorers' Inception weights -- are charged to the first data
    # seed's seconds_data rather than to nothing at all; it is restarted after each seed's
    # repeats. See "Wall-clock accounting" in the module docstring.
    t_process = time.time()
    t_setup = t_process
    pools = load_digit_pools(root=args.mnist_root, device=device)
    fid_scorers = {}
    if args.fid in ("legacy", "both"):
        fid_scorers["legacy"] = LegacyFIDScorer(device=device, batch_size=64,
                                                distance_device=device)
    if args.fid in ("clean", "both"):
        fid_scorers["clean"] = CleanFIDScorer(device=device, batch_size=64,
                                              num_workers=args.fid_workers, allow_download=False)

    rows = []
    accounted = 0.0
    for data_seed in args.data_seeds:
        generator = torch.Generator(device=device).manual_seed(data_seed)
        X_obs, A_obs, Y_obs = sample_observed(pools, args.n_obs, w_conf=args.w_conf,
                                              tau=args.tau, alpha=args.alpha,
                                              k_mask=args.k_mask, generator=generator)
        data = to_model_range(Y_obs)
        # This seed's whole per-data-seed tier: data preparation and nothing else here, so
        # seconds_setup equals seconds_data. Both sit outside every seconds_repeat.
        sync_device(device)
        seconds_data = round(time.time() - t_setup, 1)
        setup = {"seconds_data": seconds_data, "seconds_setup": seconds_data}
        accounted += seconds_data
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(pools, data, X_obs, A_obs, setup, args, data_seed, repeat,
                                  device, fid_scorers)
            accounted += new_rows[-1]["seconds_repeat"]
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            write_particle_indices(out_path)
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']} "
                      f"{row['estimator']}] "
                      f"colour flow {row['colour_mean_flow']:.3f} "
                      f"(initial {row['colour_mean_initial']:.3f}, oracle {row['colour_mean_oracle']:.3f})   "
                      f"SW2 flow {row['sw2_flow']:.4f} (floor {row['sw2_floor']:.4f})   "
                      + (f"FID legacy flow {row['fid_legacy_flow']:.1f} "
                         f"(initial {row['fid_legacy_initial']:.1f}, floor {row['fid_legacy_floor']:.1f})   "
                         if "fid_legacy_flow" in row else "")
                      + f"ESS {row['ess_on_arm']:.0f}/{row['n_on_arm']}   "
                      # own flow cost, then the whole repeat's wall clock (the same
                      # number on every row of the repeat, not this row's share of it)
                      f"{row['seconds_flow']:.0f}s flow / {row['seconds_repeat']:.0f}s repeat"
                      f" / {row['seconds_setup']:.0f}s setup this seed",
                      flush=True)
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
