"""
dSprites simulation: continuous actions, cross-fitted DR-CPME target, Riesz MMD flow (AdamW).

Script version of
`NotebookExamples/Interventional_MMD_Flow/dSprites_Continuous_Action_CrossFit_DR_Riesz_MMD_Flow_AdamW.ipynb`
with one deliberate change: the denoising autoencoder is trained on LOGGED IMAGES from
a split disjoint from the causal estimate, not on the sprite bank. The bank is the
entire noise-free outcome manifold, so training on it hands the feature map oracle
knowledge; here the logged sample is split once into an autoencoder half and a causal
half, the AE sees only the first (its denoising targets are the noisy logged images
themselves), and the nuisances, DR weights, flow particles and evaluation all live on
the second. The bank remains on the EVALUATION side only: denoise-MSE and probe
diagnostics, off-manifold distances, the snapped position law, and the clean-oracle
floor.

Everything else follows the notebook: estimated continuous-action importance weights
(known target policy over a cross-fitted logging-density fit), RBF product kernel on
(X, A) with a resampled target-action kernel mean, cross-fitted one-step DR weights,
and a dense Riesz flow under AdamW with linear warm-up on standardized latents.

Two designs
-----------
`--design` chooses the DGP, and the `design` CSV column records it. `cpme` (the DEFAULT,
and unchanged in every respect) renders a single heart sprite from the ACTION alone, so
the outcome is independent of the context given the action and no adjustment is strictly
needed. `confounded` lets the context set shape and scale as well as the action, so the
backdoor path is open and recovering nu(pi) genuinely requires adjusting for it; the
target is a shifted (or, under --mean_preserving, a mean-preserving) mixture that moves
only the position, leaving the shape and scale MARGINALS untouched, which is why a mean
summary of the outcome cannot detect the intervention and a distributional one can.

The two designs do not share their DGP arguments. `--policy_shift` belongs to `cpme`;
`--policy_gain`, `--policy_angle`, `--target_shift` and `--mean_preserving` to
`confounded`. Passing one to the other design is REFUSED rather than ignored, and
`--sigma_policy`, `--pos_gain`, `--pixel_noise` and `--weight_clip` default per design
(`DESIGN_DEFAULTS`) so that an unset flag always resolves to the value its design was
verified at.

What the confounded design does NOT have is a single-attribute (1024, 4096) position
bank: its outcome varies in shape and scale as well as position, so there is nothing to
snap a decoded image to and no clean position grid to probe. Every column that needs one
is written as NaN -- `denoise_mse_logged`, `probe_r2_*`, both `offmani_*` pairs, the
`floor_clean` pixel row and its count-matched twin, and the whole `pos_{energy,sw1,sw2}_*`
block (the flow row of it is unobtainable, and a floor without a flow row is not a
comparison). The DECODER is still measured, by the `roundtrip` pixel row and
`fid_*_roundtrip`, which need no bank. Everything the headline rests on -- pixel SW2,
`latent_sw2_*` and FID -- is bank-free and identical in construction across the two
designs. NOTE `pos_mean_logged_*` / `pos_mean_oracle_*` / `reweighted_pos_*` are kept and
are CONTINUOUS position means here against integer-grid means under `cpme`: read the
`design` column before comparing them.

`--estimators` chooses which target embedding estimators the flow is matched to:
the plug-in CME estimator, the IPW estimator, the DR estimator, or several at once.
All three are weight vectors over the logged outcomes and the plug-in weights and the
one-step correction come out of a SINGLE CME solve, so asking for all three costs one
extra flow apiece and no extra nuisance fitting. Estimators requested together are
paired: they share the logged sample, the autoencoder, the nuisance fit, the initial
particles and the reference draws, and differ only in the weight vector. The default
is DR alone. The IPW row is the Horvitz-Thompson form v = w / n by default -- the term
that appears inside v_DR, and the reason the three are algebraically comparable, but one
with no unit mass; --ipw_form hajek switches it to the self-normalised w / sum(w), a
different estimator, and the form used is recorded in the ipw_form CSV column.

Target mass
-----------
Whatever estimator produces it, the target embedding is projected onto TOTAL MASS 1 before
the flow (`--target_mass`, default `rescale`; see `project_to_unit_mass`). chi(pi) embeds a
probability measure, so its mass is 1 by construction, while the finite-sample estimate's is
not -- the ridge leaks mass and the IPW mean of estimated ratios misses 1. The Riesz kernel
is only conditionally positive definite, hence a proper discrepancy between mass-MATCHED
measures, so without the projection the particle cloud drifts instead of converging: at the
headline configuration latent SW2 sits at 17.41x its own count-matched floor against 1.14x
with the projection, and pixel SW2 is slightly better with it too. `sum_v` records the mass
the flow was handed and `sum_v_raw` the mass the estimator returned, so the deficit the
projection removed stays visible. CSVs written before 2026-10-05 have neither column and no
projection; `--target_mass none` reproduces them.

Two consequences, both deliberate: under `rescale` the IPW row is exactly the Hajek
estimator whatever `--ipw_form` requested (use `shift` to keep a Horvitz-Thompson row
distinct), and the exact decomposition v_DR = v_PI + v_correction does not survive the
projection, because the three vectors carry different masses. That identity belongs to the
raw estimators, and `sum_v_raw` records the masses it held for.

That a deficit of ~0.1% suffices to break the flow is established EMPIRICALLY and is not
explained here: the raw loss is confining under dilation and the deficit's own descent
direction is far too shallow to account for it. The live hypothesis is that AdamW, which
normalises the step by the gradient's running RMS, turns a persistent small imbalance near
the optimum into a full learning-rate step every iteration. Do not state the energy argument
as the cause.

One CSV row per (data_seed, repeat, estimator). Metrics (energy distance, SW1, SW2 in
flat pixel space, mean +/- std over --n_ref fresh interventional reference draws of
matched size) for: the logged images, their AE round-trip, the decoded flow output, the
noisy-oracle floor, and the clean-sprite floor (the appropriate one for a denoising
decoder).
The snapped position law (energy/SW1/SW2 on grid positions) is reported alongside,
with position means and off-manifold diagnostics.

`latent_sw2_*` reports the same comparison in the FLOW'S OWN SPACE -- the standardized
latents the particles live in -- for the initial particle cloud, the logged latents, the
flow output and an encoded independent draw of the interventional law, plus that floor's
count-matched twin. The pixel rows score the composition (sampler THEN decoder) and are
the stricter test; the latent row isolates the sampler. It follows CelebA's convention, but
the figure that travels between designs is the ratio of `latent_sw2_flow` to this design's
OWN `latent_sw2_floor_matched`, not the raw column: see `latent_scores` for why the raw
numbers are not interchangeable across designs.

`fid_{legacy,clean}_{initial,flow,floor,roundtrip}` reports FID for the decoded initial
cloud, the decoded flow output, the autoencoder round-trip and a second oracle draw, scored
against one fresh oracle draw. The outcome is SINGLE-CHANNEL, so the channel is replicated
into RGB for Inception (`to_rgb`): the image is unchanged, merely presented as three
channels. A binary sprite on black is far outside Inception's training distribution, so
these are a WITHIN-DESIGN comparison -- flow against initial against floor -- and are NOT
comparable with the ColorMNIST or CelebA FID columns despite sharing the column names. Read
the floor row as the scale. Unlike the SW and energy rows, FID uses a single reference draw
rather than an average over --n_ref, and is not count-matched, both to match those designs.

Wall-clock accounting
---------------------
Every `seconds_*` column is wall clock (time.time(), rounded to 0.1 s) and belongs to
exactly ONE of three scopes. Mixing them is what makes a timing table wrong in both
directions: a cost measured once and written onto every row is multiplied by any sum down
those rows, and a cost measured outside the repeat clock is lost by any sum of repeats.

PER DATA SEED -- identical on every row of every repeat of that data seed, and measured
OUTSIDE `seconds_repeat`, because the work happens once before the repeat loop:
`seconds_data` is this seed's data generation (the logged sample with its oracle
counterpart), and `seconds_setup` is the per-data-seed TOTAL, the column to add once per
data seed. Here the two are equal: the sprite bank is loaded once per PROCESS and is
charged to the FIRST data seed's `seconds_data`, and there is neither a frozen-encoder
pass (`seconds_encode`) nor a shared FID reference feature (`seconds_fid_ref`), the two
other components `seconds_setup` is broken into where a design has them.

SHARED PER REPEAT -- identical on every estimator row of a repeat, because the work was
done once and serves all of them, so they must NEVER be summed across those rows:
`seconds_ae` is the whole autoencoder STAGE -- the denoising training, its reconstruction
probes (the logged round-trip behind `denoise_mse_logged`, the sprite-bank encode and the
position probe) and the encode of the logged latents the CME and the flow both run on.
`seconds_ratio` is the density-ratio half of the nuisance fit: the cross-fitting folds,
the logging-density fit, the ratio pi / pi_0 it yields and the weight diagnostics, which
is the whole of what the IPW estimator needs. `seconds_cme` is the conditional-mean-
embedding half: the two lengthscale locks, the target-policy action resamples, the Gram
blocks, the per-fold ridge selection and the single solve that returns v_PI and the
one-step correction -- the whole of what the plug-in estimator needs, with DR needing
both halves. One solve hands back every requested vector, so the sub-millisecond IPW
rescaling v = w / n is clocked with the CME half too. `seconds_nuisance` is their SUM,
under the name it has always had, so never add the three together. `seconds_eval_shared`
is the fresh reference draws, the encode of those draws and of the noisy-oracle floor into
the latent space (a measurement pass, so it is billed here and not to `seconds_ae`), and
the logged / round-trip / floor / count-matched-floor scoring in both pixel and latent
space, plus the latent `initial` row, which scores the particle cloud before the flow and
has no pixel counterpart, and -- when --fid is on -- the Inception features of the reference
draw with the initial / floor / round-trip FID rows, which dominate this clock when enabled;
all computed once and copied into each row. ColorMNIST bills its
reference encode the same way. CelebA does NOT -- its encoder is a frozen pretrained VAE,
so it hoists that pass into the per-data-seed `seconds_encode`, which `analysis_tables`
maps to Representation rather than Scoring; a cross-design Representation-against-Scoring
split therefore places the same kind of work differently for CelebA, independently of this
design. `seconds_shared` is the sum of
`seconds_ae`, `seconds_nuisance` and `seconds_eval_shared` as a convenience column.

PER ESTIMATOR -- this row's own work, free to sum across rows: `seconds_flow` (this
estimator's flow iterations alone), `seconds_decode` (its decode of the final particles)
and `seconds_eval` (its OWN scoring -- the snap, the decoded-flow pixel metrics, the flow
position metrics, the flow's latent SW2 and its FID -- EXCLUDING the shared scoring above). `seconds_sampling` is
DERIVED and a term of NEITHER identity: the isolated cost of the proposed sampling method
for that row, composed for its estimator as

    plugin -> seconds_cme + seconds_flow + seconds_decode
    ipw    -> seconds_ratio + seconds_flow + seconds_decode
    dr     -> seconds_ratio + seconds_cme + seconds_flow + seconds_decode

It deliberately EXCLUDES the autoencoder (`seconds_ae`: training, probes and the latent
encode) and all evaluation, because those are representation and measurement rather than
sampling, and it excludes the shared latent standardisation and particle draw, which
`seconds_other` carries. An END-TO-END figure has to add `seconds_ae` and `seconds_setup`
back. It reuses shared columns, so it must never be summed down the rows of a repeat.

TOTAL -- `seconds_repeat`, the wall clock of the WHOLE repeat, IDENTICAL on every row of
that repeat; it is no longer cumulative down the estimator rows. `seconds_other` carries
the residual by construction -- the split, the latent standardisation and the particle
init, which are glue rather than named phases -- so that the two identities a reader
needs,

    seconds_repeat == seconds_shared + seconds_other
                      + sum over the repeat's rows of (flow + decode + eval)

    task total     == sum over data seeds of seconds_setup
                      + sum over repeats of seconds_repeat

close EXACTLY on the rounded numbers written to the CSV, not merely up to rounding,
reading `seconds_setup` once per data seed and `seconds_repeat` once per repeat and never
once per row. `seconds_other` is a signed residual and may be -0.1 when the rounded parts
happen to round up; expect a tenth or two, the CUDA context and the convolution kernels
being first touched inside the autoencoder stage, which is a named phase and carries that
warm-up itself. What the task identity does not cover is named rather than hidden: process
start-up (the imports and the CUDA context) and the per-repeat CSV rewrite. The run prints
both the accounted total and the process wall clock on its last line, so the gap is
visible rather than inferred.
Every GPU phase is synchronised before its clock is read (`sync` inside `run_repeat`):
CUDA launches are asynchronous, so without it a phase's tail is billed to whichever
later phase first reads a result back. Measured on an RTX 4500 Ada before the syncs were
added, every tail was already under 1% of its phase -- `train_dsprite_ae` and
`adamw_riesz_flow` both end their loop on a `.item()`, which drains the stream, and the
decode is launch-bound rather than kernel-bound at these particle counts -- so the syncs
make the attribution exact by construction rather than by luck, and they change nothing
numerically (a wait touches no tensor and draws no random number).
`seconds_encode` is deliberately ABSENT here. Unlike CelebA, which pulls a FROZEN
pretrained VAE over a fixed draw and can therefore hoist its encode into the
per-data-seed tier, this design trains its own autoencoder inside the repeat, so every
encoder pass depends on the repeat and lives in that repeat's `seconds_ae`. Columns absent
for want of a phase are omitted rather than written as 0.0.
Archived CSVs written before 2026-10-03 carry `seconds_eval` holding shared + own and a
`seconds_repeat` that is CUMULATIVE down the estimator rows of a repeat, and have no
`seconds_eval_shared` / `seconds_shared` / `seconds_other`: do not sum those files'
columns across estimator rows. In those files, and in the ones written earlier on
2026-10-03, `seconds_ae` is TRAINING ALONE -- the probes and the bank encode sat in
`seconds_other` and the latent encode in `seconds_nuisance` -- so that column is not
comparable with this one.
CSVs written before 2026-10-05 have NO `latent_sw2_*` columns at all, which is every
dSprites stage but the rerun headline: concatenating them leaves those rows NaN, so a
latent table has to be read off the headline stage alone. Their `seconds_eval_shared`
excludes the reference encode and the latent scoring and so runs short of this one, which
also moves `seconds_shared` and `seconds_repeat` with it, and `seconds_eval` by the flow
row's own latent call; `seconds_ae`, `seconds_nuisance`, `seconds_flow` and
`seconds_decode` are untouched, so the TRAINING total is comparable across the two.

What the latent rows do NOT do is change any number that was already there: they draw no
random value (every generator here is explicitly and locally seeded) and mutate no model
state (the autoencoder is GroupNorm-only, so it has no running statistics, and
`train_dsprite_ae` returns it in eval mode). Verified by running both versions at identical
seeds ON CPU: 134 of 134 shared non-timing columns bit-identical, ten columns added, none
removed.
That is NOT a promise that a rerun reproduces an archived GPU run. The autoencoder is
trained with cuDNN convolutions and nothing in this repository sets
`use_deterministic_algorithms` or `cudnn.deterministic`, so GPU training is not
run-to-run reproducible. Two archived headline runs of the SAME 148-column code at the
same seeds on the same L40S host differ in `ae_train_mse` (0.00252613 against 0.00251481),
`denoise_mse_logged` (2.375e-05 against 1.697e-05) and `sw2_flow_mean` (0.012362 against
0.012325), while every AE-INDEPENDENT column -- the `*_logged_*` metrics, both lengthscales,
`oracle_w_ess`, `est_w_ess`, `sum_v`, `v_ess`, both unmatched pixel floors -- is
bit-identical. So expect a rerun to move the third significant figure of anything downstream
of the autoencoder, and use those columns, not the pixel metrics, as the check that nothing
else changed. `fold_lambdas` is NOT in that set despite looking like a nuisance quantity: the
per-fold ridge is chosen by cross-validation over the OUTCOME LATENTS, so an autoencoder
perturbation can flip one fold across a grid boundary (observed: seed 0 fold 1, 0.001 to 0.01,
with every other fold and seed holding).

Hardware provenance: every row also carries gpu_name, hostname, slurm_job_id,
slurm_array_task_id, slurm_nodelist, torch_version and cuda_version, so a published
timing can be tied to the node that produced it. The SLURM fields are read from the
environment and are empty off the cluster.

Seed scheme: `data_seed` fixes the logged sample; `pipeline_seed = 1000 * data_seed +
repeat` fixes the split, the autoencoder and the flow; reference draws use seeds
100000 + 1000 * pipeline_seed + j, disjoint from every data seed.

Launched by `submit_dsprites_dr_riesz_flow_parallel.sh`; single-run example:

    python dsprites_dr_riesz_flow_simulation.py --n_obs 2000 --data_seeds 0 --n_repeats 3
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
                             cpme_dr_weights_nystrom, cpme_dr_weights_projected_nystrom,
                             cpme_ipw_weights, lambda_n_half, make_crossfit_fold_ids,
                             select_regularization)
from data_utils import (load_dsprite_bank, load_multifactor_bank,
                        sample_confounded_dsprite_data, sample_cpme_dsprite_data)
from dens_ratio_utils import (crossfit_logging_policy_log_prob,
                              fit_linear_gaussian_policy,
                              importance_weight_diagnostics,
                              importance_weights_from_logging_log_prob)
from autoencoder_utils import (encode_in_batches, position_probe, snap_to_bank,
                               train_dsprite_ae)
from kernel_utils import RBF, RieszKernel
from mmd_flow_utils import adamw_riesz_flow, squared_mmd
from metric_utils import sliced_wasserstein
from metric_utils.fid import CleanFIDScorer, LegacyFIDScorer

PIXEL_ROWS = ("logged", "roundtrip", "flow", "floor_noisy", "floor_clean")
POSITION_ROWS = ("logged", "flow", "floor")
METRICS = ("energy", "sw1", "sw2")
# Floor rows that also get a COUNT-MATCHED twin; see `score_matched`.
PIXEL_FLOOR_ROWS = ("floor_noisy", "floor_clean")
# Latent metric rows; see `latent_scores`. `initial` is the particle cloud before the
# flow and `logged` the logged latents -- the two baselines the flow has to beat --
# `flow` the flow output, and `floor` an independent draw of the interventional law
# encoded by the same autoencoder, which is the best any sampler can do at this count.
# The floor is the NOISY oracle only: a denoising encoder maps a clean sprite and its
# noisy copy to nearly the same latent, so a clean twin would not be a distinct floor
# here the way `floor_clean` is in pixel space.
LATENT_ROWS = ("initial", "logged", "flow", "floor")
# FID rows, matching ColorMNIST and CelebA so the three designs' columns line up.
FID_ROWS = ("initial", "flow", "floor", "roundtrip")
# Target embedding estimators, in the order their rows are emitted. Each is a weight
# vector over the logged outcomes: v_PI = (1/n) (K + lambda I)^-1 (K_X * K_A_pi) 1 uses
# the conditional mean embedding alone, v_IPW = w / n the importance ratios alone, and
# v_DR = v_PI + v_correction uses both, so one DR solve yields all three.
ESTIMATORS = ("plugin", "ipw", "dr")

COLUMNS = (
    ["tag", "design", "device", "init", "estimator", "data_seed", "repeat",
     "pipeline_seed",
     "n_obs", "sigma_policy", "policy_shift", "pos_gain", "pixel_noise",
     # What FID was actually scored on. `n_fid_ref` is the reference count and
     # `n_fid_row` the per-row count; they differ only when a row is smaller
     # than --fid_n_ref, which happens for the flow under a small M.
     "n_fid_ref", "n_fid_row",
     # The confounded design's own DGP parameters. Exactly one of `policy_shift` and
     # these four is a number on any row; the other is NaN, so no row can be read as
     # describing a DGP it did not use. CSVs written before 2026-10-06 have none of the
     # five columns and are all `design == cpme`.
     "policy_gain", "policy_angle", "target_shift", "mean_preserving",
     "n_ae", "n_causal", "n_folds", "n_resample", "n_ref",
     "latent_dim", "ae_iterations", "flow_iterations", "learning_rate", "warmup_steps",
     "ae_train_mse", "denoise_mse_logged", "probe_r2_x", "probe_r2_y",
     "weight_clip", "solver", "ae_split", "dr_fit", "lambda_rule", "ipw_form",
     "target_mass",
     "n_particles",
     "n_landmarks",
     "oracle_w_ess", "est_w_ess", "est_w_max", "est_w_log_rmse",
     "lengthscale_x", "lengthscale_a", "fold_lambdas", "sum_v", "sum_abs_v",
     "v_ess", "v_max_abs", "sum_v_raw",
     "reweighted_pos_x", "reweighted_pos_y",
     "loss_initial", "loss_final", "mean_transport",
     "offmani_roundtrip_median", "offmani_roundtrip_frac3",
     "offmani_flow_median", "offmani_flow_frac3"]
    + [f"{metric}_{row}_{stat}" for row in PIXEL_ROWS for metric in METRICS
       for stat in ("mean", "std")]
    + [f"pos_{metric}_{row}_{stat}" for row in POSITION_ROWS for metric in METRICS
       for stat in ("mean", "std")]
    + ["pos_mean_logged_x", "pos_mean_logged_y", "pos_mean_flow_x", "pos_mean_flow_y",
       "pos_mean_oracle_x", "pos_mean_oracle_y"]
    + [f"{metric}_{row}_matched_{stat}" for row in PIXEL_FLOOR_ROWS for metric in METRICS
       for stat in ("mean", "std")]
    + [f"pos_{metric}_floor_matched_{stat}" for metric in METRICS
       for stat in ("mean", "std")]
    # Latent SW2, in the flow's own space; see `latent_scores`. The count-matched floor
    # needs no count column of its own: its k is `n_particles`, which is exactly the
    # `n_floor_matched` the pixel floors already report, because the latent references
    # carry n_causal rows and n_particles never exceeds that.
    + [f"latent_sw2_{row}_{stat}" for row in LATENT_ROWS
       for stat in ("mean", "std")]
    + [f"latent_sw2_floor_matched_{stat}" for stat in ("mean", "std")]
    # FID on channel-replicated greyscale; see `to_rgb`. Single-reference, unlike the
    # SW / energy rows which average over --n_ref draws, and NOT count-matched -- the
    # flow row carries M atoms against the full reference, the ColorMNIST convention.
    + [f"fid_{impl}_{row}" for impl in ("legacy", "clean") for row in FID_ROWS]
    # Wall-clock columns: see the module docstring. Per DATA SEED first (outside every
    # seconds_repeat), then shared-per-repeat (never sum across the estimator rows of a
    # repeat), then per-estimator, then the total. `seconds_ratio` / `seconds_cme` are a
    # BREAKDOWN of `seconds_nuisance` and `seconds_data` one of `seconds_setup`, so
    # neither pair may be added alongside the column it decomposes; `seconds_sampling` is
    # derived from several of the others and belongs to no sum at all.
    + ["n_floor_matched",
       "seconds_data", "seconds_setup",
       "seconds_ae", "seconds_ratio", "seconds_cme", "seconds_nuisance",
       "seconds_eval_shared", "seconds_shared",
       "seconds_flow", "seconds_decode", "seconds_eval", "seconds_sampling",
       "seconds_other", "seconds_repeat"]
    # Hardware provenance, so a published timing table can be audited against the node
    # that produced it. See `hardware_provenance`.
    + ["gpu_name", "hostname", "slurm_job_id", "slurm_array_task_id", "slurm_nodelist",
       "torch_version", "cuda_version"]
)

#: Which half of the nuisance fit each estimator's target embedding actually needs:
#: v_PI is the CME solve alone, v_IPW = w / n the density ratios alone, and
#: v_DR = v_PI + v_correction both. Used by `sampling_seconds`.
SAMPLING_NUISANCE = {"plugin": ("seconds_cme",),
                     "ipw": ("seconds_ratio",),
                     "dr": ("seconds_ratio", "seconds_cme")}

#: DGP arguments whose default DEPENDS ON THE DESIGN. Those flags default to None, so an
#: unset flag resolves to the value its design was verified at while an explicit value
#: always wins. The `cpme` column is that design's historical default argument for
#: argument, so a cpme run's rows and its filename are unchanged.
#:
#: `pixel_noise` 0 is not a convenience. The confounded design renders by BILINEAR
#: interpolation, so the outcome is already continuous in the action and needs no noise to
#: avoid atoms -- and observation noise is what made FID on the cpme design measure the
#: missing noise rather than the sampler (raw oracle against oracle 145.9 there, 1.28
#: here). `weight_clip` is DISABLED because the shifted-mixture target contains the
#: logging policy as a component, so the true ratio is bounded below by 1/2 and overlap is
#: benign: measured at n = 4000, the estimated ratios give ESS 2596/4000, max 14.3 and
#: log-RMSE 0.045, no tail at all -- while the cpme cap of 5.0, calibrated on THAT
#: design's bad-overlap draws, would truncate a legitimate 14.3 and bias the position law
#: down. `est_w_ess` stays the monitor; revisit if any seed falls below ~50.
DESIGN_DEFAULTS = {
    "cpme": {"sigma_policy": 0.5, "pos_gain": 3.0, "pixel_noise": 0.05,
             "weight_clip": 5.0},
    "confounded": {"sigma_policy": 0.6, "pos_gain": 1.5, "pixel_noise": 0.0,
                   "weight_clip": 0.0},
}


def resolve_design_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the unset design-dependent DGP arguments from `DESIGN_DEFAULTS`.

    Called once at the end of `parse_args`, so every later reader -- `csv_path`, the DGP
    call and the CSV row -- sees ONE resolved value and the filename can never disagree
    with the DGP that produced the rows under it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, whose design-dependent entries are None when unset.

    Returns
    -------
    argparse.Namespace
        `args` itself, mutated in place.
    """
    for name, value in DESIGN_DEFAULTS[args.design].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def has_bank_design(args: argparse.Namespace) -> bool:
    """Whether this design has the single-attribute (1024, 4096) position bank.

    `cpme` renders one heart sprite on a 32 x 32 position grid, so every outcome is a
    point of a 1024-image bank and a decoded image can be snapped to it. `confounded`
    varies shape and scale too, so there is no such bank and every column derived from
    one -- the snap, the position law, the clean-sprite floor, the position probe -- is
    written as NaN rather than computed against the wrong bank.
    """
    return args.design == "cpme"


def sample_design(args: argparse.Namespace, sprite_bank: np.ndarray, n_samples: int,
                  seed: int) -> dict:
    """One draw of the design `--design` names: logged arm, oracle arm and ratios.

    Both the logged sample in `main` and the fresh reference draws in `run_repeat` go
    through this one function, so the sample being estimated from and the references it is
    scored against can never be generated from different policies. The `cpme` branch is
    the historical call argument for argument, so its draws are bitwise what they were.

    Parameters
    ----------
    args : argparse.Namespace
        Already passed through `resolve_design_defaults`.
    sprite_bank : np.ndarray
        The (32, 32, 64, 64) position bank under `cpme`, the
        (n_shapes, n_scales, 1, 32, 32, 64, 64) attribute bank under `confounded`.
    n_samples : int
    seed : int

    Returns
    -------
    dict
        The DGP's own dictionary. The confounded design adds "shape" and "scale" (oracle
        knowledge), which nothing downstream reads.
    """
    if args.design == "confounded":
        return sample_confounded_dsprite_data(sprite_bank, n_samples=n_samples,
                                              policy_gain=args.policy_gain,
                                              policy_angle=args.policy_angle,
                                              sigma_policy=args.sigma_policy,
                                              target_shift=args.target_shift,
                                              mean_preserving=args.mean_preserving,
                                              pos_gain=args.pos_gain,
                                              pixel_noise=args.pixel_noise,
                                              seed=seed)
    return sample_cpme_dsprite_data(sprite_bank, n_samples=n_samples,
                                    sigma_policy=args.sigma_policy,
                                    policy_shift=args.policy_shift,
                                    pos_gain=args.pos_gain,
                                    pixel_noise=args.pixel_noise,
                                    seed=seed)


def hardware_provenance(device: "torch.device") -> dict:
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
    are excluded by construction: the autoencoder stage (`seconds_ae`) and every
    evaluation column stay out, so an end-to-end figure must add `seconds_ae` and
    `seconds_setup` back.

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


def to_rgb(flat_images: torch.Tensor) -> torch.Tensor:
    """Flat single-channel 64x64 outcomes as the (N, 3, 64, 64) RGB batch FID needs.

    This design's outcome is GREY: one channel, a white sprite on black. The Inception
    network behind both scorers takes three channels, so the one channel is REPLICATED
    into all three -- the image is unchanged, it is simply presented as RGB.

    That convention has a consequence worth stating once. Inception's features were learned
    on natural colour photographs, and a binary sprite on black is far outside that
    distribution, so these numbers are a within-design comparison -- flow against initial
    against floor, on this benchmark -- and are NOT comparable with the ColorMNIST or CelebA
    FID columns even though they share the column names. Read the floor row as the scale.

    Parameters
    ----------
    flat_images : torch.Tensor of shape (N, 4096)
        Values in [0, 1]; the decoder ends in a sigmoid and the sprite bank is binary.

    Returns
    -------
    torch.Tensor of shape (N, 3, 64, 64)
    """
    return flat_images.reshape(-1, 1, 64, 64).expand(-1, 3, -1, -1)


def fid_rows(scorers: dict, reference_features: dict, named_images, n_fid: int,
             prefix: str = "") -> dict:
    """FID of each named image batch against the already-featurised reference.

    Parameters
    ----------
    scorers : dict
        {"legacy": LegacyFIDScorer, "clean": CleanFIDScorer}; empty disables FID.
    reference_features : dict
        Per implementation, the reference features from `reference_fid_features`.
    named_images : iterable of (str, torch.Tensor)
        Row name and its FLAT images.
    n_fid : int
        Images to score from each row. EVERY row and the reference are cut to this one
        number, so the column is comparable across a sweep; see `fid_subset`.
    prefix : str
        Unused hook kept for symmetry with `named_scores`.

    Returns
    -------
    dict
        e.g. {"fid_legacy_flow": ..., "fid_clean_flow": ...}; empty when no scorer.
    """
    out = {}
    for impl, scorer in (scorers or {}).items():
        for name, images in named_images:
            out[f"{prefix}fid_{impl}_{name}"] = scorer.distance_from_features(
                scorer.features(to_rgb(fid_subset(images, n_fid)), image_range="zero_one"),
                reference_features[impl])
    return out


def fid_subset(images: torch.Tensor, n_fid: int) -> torch.Tensor:
    """The first `n_fid` images of a row, or all of them when the row is smaller.

    FID is strongly biased upward at small sample sizes, so a column whose rows are
    scored at DIFFERENT counts cannot be compared. Until 2026-10-06 this simulation
    featurised each row whole: the reference and the floor were `n_causal` images while
    the flow was `n_particles`, so the reference count was the n_obs sweep axis ITSELF
    and the FID column moved with n for a purely statistical reason -- the split-half
    floor between two draws of the SAME law fell 0.53 -> 0.14 across n = 5000..20000
    while the truth it estimates did not change at all.

    Cutting every row and the reference to one fixed count removes that. The convention
    matches the CelebA simulation, which scores 2000 generated against 2000 held-out
    references at every n (the DecFM protocol), and through it the published tables.

    Rows are cut by taking the first `n_fid` entries rather than a random subset: every
    row here is already in a random order (fresh oracle draws, and particles drawn by a
    seeded permutation), so a slice is a uniform sample and stays reproducible without
    consuming more RNG.

    A row SMALLER than `n_fid` -- the flow under a small --n_particles -- is scored whole
    and is then NOT count-matched to the others. `n_fid_row` records what was actually
    used, so that case is visible rather than silent.
    """
    return images if len(images) <= n_fid else images[:n_fid]


def reference_fid_features(scorers: dict, reference: torch.Tensor, n_fid: int) -> dict:
    """Inception features of the reference draw, computed once and reused by every row."""
    return {impl: scorer.features(to_rgb(fid_subset(reference, n_fid)),
                                  image_range="zero_one")
            for impl, scorer in (scorers or {}).items()}


def project_to_unit_mass(v: torch.Tensor, mode: str) -> torch.Tensor:
    """The target weight vector with its total mass set to 1.

    Every estimator here returns the target embedding as a finite expansion
    chi_hat(pi) = sum_i v_i k(y_i, .), which is the embedding of the signed measure
    sum_i v_i delta_{y_i}. The thing being estimated is a PROBABILITY measure, so the
    truth satisfies sum_i v_i = 1 exactly; the finite-sample estimate does not. The ridge
    in the CME solve leaks mass, and the IPW mean of estimated ratios is not exactly 1.

    That matters here because the flow minimises ||mu_Q - chi_hat|| with the RIESZ
    (energy) kernel, which is only CONDITIONALLY positive definite -- a proper discrepancy
    between mass-MATCHED measures. mu_Q has mass 1 by construction, so a target of mass
    m != 1 leaves the particle cloud drifting instead of converging. Measured on this
    design at the headline configuration, latent SW2 as a ratio to its own count-matched
    floor: 17.41x without this projection against 1.14x with it, and pixel SW2 slightly
    better too (1.44x -> 1.43x). The damage tracks the deficit rank-perfectly across
    estimators: IPW's deficit is 65x the DR one and its latent SW2 is 16x worse.
    A BOUNDED kernel does not need this (ColorMNIST carries a deficit of the same size
    under IMQ and converges), and CelebA's deficit is exactly zero, so neither is changed.

    Imposing a constraint the truth is known to satisfy belongs to ESTIMATION, not to the
    sampler: it is a property of chi(pi) itself, independent of how one then samples from
    it. That is why it is applied here, to the weight vector, rather than inside the flow.

    Parameters
    ----------
    v : torch.Tensor of shape (n,)
        Target weights, possibly signed.
    mode : {"none", "rescale", "shift"}
        "none" leaves `v` untouched -- the pre-2026-10-05 behaviour, kept so the
        un-projected arm stays runnable. "rescale" divides by the total mass, which is
        the Hajek / self-normalised form and preserves the RELATIVE weights. "shift" adds
        the deficit evenly, which is the ORTHOGONAL projection onto {v : 1'v = 1} and so
        is the variant with the guarantee that projecting onto a convex set containing
        the truth cannot increase the distance to it.

    Returns
    -------
    torch.Tensor
        `v` itself when `mode` is "none", otherwise a new vector of total mass 1.

    Notes
    -----
    Two consequences worth knowing, both deliberate.

    Under "rescale", the IPW row becomes exactly the Hajek estimator w / sum(w) whatever
    `--ipw_form` says, because v_IPW = w / n rescaled by its own mass IS w / sum(w). The
    `ipw_form` column still records what was requested, so the pair is recoverable; use
    "shift", or "none", to keep a Horvitz-Thompson IPW row distinct.

    The exact decomposition v_DR = v_PI + v_correction does NOT survive either projection,
    since the three vectors carry different masses and are corrected by different amounts.
    That identity is a property of the raw estimators; `sum_v_raw` records the masses it
    held for.
    """
    if mode == "none":
        return v
    if mode == "rescale":
        return v / v.sum()
    if mode == "shift":
        return v + (1.0 - v.sum()) / v.numel()
    raise ValueError(f"unknown target_mass mode {mode!r}")


def resolve_estimators(requested) -> tuple:
    """The requested estimators, de-duplicated and in canonical `ESTIMATORS` order.

    Ordering the request makes the row order, and the filename token, independent of
    how the flags happened to be typed.

    Parameters
    ----------
    requested : sequence of str

    Returns
    -------
    tuple of str
    """
    wanted = set(requested)
    return tuple(name for name in ESTIMATORS if name in wanted)


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

    Notes
    -----
    On the IPW row `v_ess` reduces exactly to the `est_w_ess` column computed from the
    importance weights themselves, whatever --ipw_form is: Kish's ratio is scale
    invariant. That is a free cross-check between the two paths. The corresponding
    check on the largest atom holds only under --ipw_form ht, where v = w / n_causal
    and `v_max_abs` * n_causal is `est_w_max`; under --ipw_form hajek v = w / sum(w),
    so `v_max_abs` * n_causal is `est_w_max` / mean(w) and the identity fails. Either
    way `v_max_abs` is not itself `est_w_max`: that one is max w.
    """
    v = v.reshape(-1).double()
    total = float(v.sum())
    return {"sum_v": total, "sum_abs_v": float(v.abs().sum()),
            "v_ess": total ** 2 / float((v ** 2).sum()),
            "v_max_abs": float(v.abs().max())}


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

def particle_generator(args, seed: int, offset: int = 0) -> "torch.Generator":
    """CPU generator for the starting-particle draw, matching the sibling simulations.

    A CPU generator keeps the draw device-independent. `--particle_seed` is not a
    flag of this script; `getattr` keeps the signature identical to the siblings so
    the option can be added later without touching the call sites.
    """
    particle_seed = getattr(args, "particle_seed", None)
    base = seed if particle_seed is None else particle_seed + seed
    return torch.Generator().manual_seed(base + offset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=2000,
                        help="logged sample size (split between the AE and causal halves)")
    parser.add_argument("--design", type=str, default="cpme",
                        choices=("cpme", "confounded"),
                        help="which dSprites DGP to run, recorded in the `design` CSV "
                             "column. 'cpme' (the DEFAULT, unchanged) renders one heart "
                             "sprite from the ACTION alone, so the outcome is independent "
                             "of the context given the action. 'confounded' lets the "
                             "context set shape and scale as well as the action, so the "
                             "backdoor path is open and adjustment is genuinely required; "
                             "its target shifts only the POSITION, leaving the shape and "
                             "scale marginals fixed, so a mean summary of the outcome "
                             "cannot see the intervention. The two designs take different "
                             "DGP flags and write disjoint filenames; the confounded one "
                             "has no single-attribute position bank, so the snapped "
                             "position law, the probe, the off-manifold scores, the "
                             "clean-sprite floor and the denoise MSE are NaN there -- see "
                             "the module docstring.")
    parser.add_argument("--ae_split", type=str, default="half", choices=("half", "none"),
                        help="'half' (default, the reference design) trains the autoencoder "
                             "on a disjoint --ae_fraction of the logged sample, so the "
                             "feature map is independent of the causal stage. 'none' trains "
                             "it on EVERY row and lets the causal stage reuse them, matching "
                             "the ColorMNIST flag of the same name. NOTE the same-sample "
                             "notes license reusing observations for NUISANCES but explicitly "
                             "not for an encoder selected on those rows, so 'none' is an "
                             "empirical setting, not one their theory covers -- the licensed "
                             "combination is --ae_split half --dr_fit nosplit.")
    parser.add_argument("--ae_fraction", type=float, default=0.5,
                        help="fraction of the logged sample used to train the autoencoder; "
                             "ignored when --ae_split none")
    # The three DGP floats both designs take, defaulting PER DESIGN (DESIGN_DEFAULTS).
    parser.add_argument("--sigma_policy", type=float, default=None,
                        help="policy noise scale (logging and target); default 0.5 under "
                             "--design cpme, 0.6 under confounded")
    parser.add_argument("--pos_gain", type=float, default=None,
                        help="action-to-position gain of the outcome model; default 3.0 "
                             "under --design cpme, 1.5 under confounded, where the "
                             "bilinear renderer makes a softer gain the calibrated one")
    parser.add_argument("--pixel_noise", type=float, default=None,
                        help="observation noise on the sprite images; default 0.05 under "
                             "--design cpme, 0.0 under confounded. The confounded "
                             "renderer is bilinear, so the outcome is already continuous "
                             "in the action and needs no noise to avoid atoms, and the "
                             "noise is what made FID measure the missing noise rather "
                             "than the sampler -- raising it there is an ablation, not a "
                             "tuning knob.")
    # --design cpme only.
    parser.add_argument("--policy_shift", type=float, default=0.2,
                        help="--design cpme only: mean shift of the target policy against "
                             "the logging policy. The confounded design's shift is "
                             "--target_shift, and passing this one with it is refused.")
    # --design confounded only.
    parser.add_argument("--policy_gain", type=float, default=1.8,
                        help="--design confounded only: gain of the rotated logging mean "
                             "gain * R_angle (x - 1/2). Relative to --sigma_policy it "
                             "sets how strongly the action reveals the context, i.e. the "
                             "CONFOUNDING STRENGTH.")
    parser.add_argument("--policy_angle", type=float, default=np.pi / 6,
                        help="--design confounded only: rotation angle of the logging "
                             "mean, which mixes both context coordinates into both action "
                             "coordinates so an action is informative about shape and "
                             "scale alike")
    parser.add_argument("--target_shift", type=float, default=0.45,
                        help="--design confounded only: half-gap delta = (s, -s) of the "
                             "target mixture. The default target contains the logging "
                             "policy as a component, so the density ratio is bounded "
                             "below by 1/2 and overlap is benign.")
    parser.add_argument("--mean_preserving", action="store_true",
                        help="--design confounded only: put the mixture components at "
                             "-delta and +delta instead of 0 and +delta. The conditional "
                             "action mean is then EXACTLY the logging mean, so any method "
                             "reporting only E[A | X] -- or only the mean outcome -- is "
                             "blind to the intervention while the outcome law is visibly "
                             "bimodal. Adds a _meanpres token to the filename.")
    parser.add_argument("--weight_clip", type=float, default=None,
                        help="cap on the estimated importance weights before the DR stage. "
                             "DEFAULTS PER DESIGN: 5.0 under --design cpme (below), and "
                             "0.0 (DISABLED) under confounded, whose shifted-mixture "
                             "target contains the logging policy as a component -- "
                             "overlap is benign and the estimated ratios carry no tail "
                             "(ESS 2596/4000, max 14.3, log-RMSE 0.045 at n = 4000), so "
                             "the cpme cap would truncate a legitimate 14.3. "
                             "The cpme calibration: "
                             "(<= 0 disables). The estimated logging-density ratio grows "
                             "spurious tails on bad-overlap draws (observed: est max weight "
                             "192 where the oracle weights have ESS 126), and those runs "
                             "dominate the seed envelope. Default 5.0 from a 2026-09 search "
                             "over {inf, 20, 10, 5, 2.5} x failing+healthy seeds: 5 repairs "
                             "every failure, leaves healthy seeds unchanged, and keeps the "
                             "position mean unbiased; 2.5 begins to undershoot it.")
    parser.add_argument("--init", type=str, default="observed",
                        choices=("observed", "normal", "moment"),
                        help="initial particles: 'observed' = the logged arm (default), "
                             "'normal' = N(0, I), 'moment' = the empirical N(mu, Sigma) "
                             "of the flow's target support. Both Gaussians are drawn in "
                             "the coordinates the flow operates in. NOTE: the *_initial "
                             "metric columns describe the run's actual starting cloud, so "
                             "under a Gaussian init they are not the observed-arm baseline "
                             "-- take that from the matching --init observed run.")
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
                             "The DR support is the whole causal split under all "
                             "of them, so what the choice changes is the nuisance "
                             "estimate, not the sample; the threshold is keyed on n_obs, "
                             "so runs compared at the same n_obs always share a solver. "
                             "NOTE the causal split is n_obs/2 here, so the current grid "
                             "never leaves the exact regime at the default ae_fraction. "
                             "'projected_nystrom' is a DIFFERENT estimator of the CME "
                             "nuisance from 'nystrom', not a cheaper route to the same "
                             "number -- see the solver branch in `run_repeat` -- and "
                             "'auto' selects neither.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly"
                             "; above it 'auto' refuses instead of choosing a solver "
                             "for you")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold")
    parser.add_argument("--n_particles", type=int, default=None,
                        help="flow particles M, capped at the causal-split size. Default "
                             "None uses EVERY causal row, which is the historical behaviour "
                             "(the particle cloud was welded to the target support). A "
                             "measure-transport sampler with M particles standing in for N "
                             "target atoms pulls each particle toward a centroid of about "
                             "N/M of them, so M is a per-sample fidelity knob, not merely a "
                             "sample count.")
    parser.add_argument("--estimators", type=str, nargs="+", default=["dr"],
                        choices=ESTIMATORS,
                        help="which target embedding estimators to sample from; one CSV "
                             "row each, named by the `estimator` column. Default 'dr' "
                             "alone, which reproduces the historical rows. The three "
                             "come out of ONE nuisance fit and are run paired -- same "
                             "logged sample, same initial particles, same flow budget, "
                             "same reference draws -- because an unpaired comparison "
                             "would be swamped by seed noise. NOTE --weight_clip applies "
                             "before the DR stage, so 'ipw' inherits the CLIPPED ratios: "
                             "consistent with 'dr', but the clip matters most to the one "
                             "estimator that has nothing but the ratios to rely on.")
    parser.add_argument("--fid", type=str, default="both",
                        choices=("none", "legacy", "clean", "both"),
                        help="FID of the decoded initial / flowed particles, the "
                             "autoencoder round-trip and a second oracle draw (the floor) "
                             "against the oracle draw. The outcome is single-channel, so "
                             "the channel is replicated into RGB (see `to_rgb`) and these "
                             "numbers do NOT compare with the ColorMNIST or CelebA FID "
                             "columns -- read the floor row as the scale. 'legacy' is the "
                             "Dance et al. convention, 'clean' is Clean-FID and needs its "
                             "weights cached in ~/.cache/torch/clean-fid. 'none' skips it.")
    parser.add_argument("--fid_workers", type=int, default=8,
                        help="Dataloader workers for the Clean-FID scorer.")
    parser.add_argument("--fid_n_ref", type=int, default=2000,
                        help="images scored from EVERY FID row and from the reference. FID is "
                             "strongly sample-size biased, so a column whose rows are scored at "
                             "different counts cannot be compared -- before 2026-10-06 the "
                             "reference was the whole n_causal draw, i.e. the n_obs sweep axis "
                             "itself. 2000 matches the CelebA simulation and the DecFM protocol. "
                             "A row with fewer images than this is scored whole and `n_fid_row` "
                             "records it.")
    parser.add_argument("--target_mass", type=str, default="rescale",
                        choices=("none", "rescale", "shift"),
                        help="Project the target embedding onto total mass 1 before the "
                             "flow; see `project_to_unit_mass`. The truth has mass 1 and "
                             "the Riesz kernel needs mass-matched measures, so the "
                             "default is 'rescale'. 'none' reproduces the pre-2026-10-05 "
                             "behaviour. NOTE 'rescale' makes the IPW row the Hajek "
                             "estimator whatever --ipw_form says; use 'shift' to keep a "
                             "Horvitz-Thompson IPW row distinct.")
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
    parser.add_argument("--dr_fit", type=str, default="nosplit",
                        choices=("crossfit", "nosplit"),
                        help="how the DR nuisances are fitted. DEFAULT CHANGED 2026-09-28 to 'nosplit', which fits the CME and the propensity on ALL rows; 'crossfit' (the previous default) fits both out of fold. The same-sample notes license reusing observations for the NUISANCES. NOTE the filename token is deliberately still keyed on 'crossfit': a 'nosplit' run appends _drnosplit and a 'crossfit' run appends nothing, exactly as before the default flipped, so every historical filename is still reproduced byte for byte and no new run can overwrite an archived one. The dr_fit CSV column is the authoritative record. "
                             "historical behaviour) fits the CME and the logging density out "
                             "of fold; 'nosplit' fits both on ALL rows. NOTE the same-sample "
                             "notes require RKHS-norm control on the ratio itself, which "
                             "--weight_clip does NOT provide: range clipping bounds the "
                             "values but establishes neither norm control nor prediction "
                             "consistency for the true ratio.")
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
                             "yields a whole iteration curve. This sampler uses a CONSTANT "
                             "learning rate, so a snapshot at step s is exactly a run of s "
                             "steps.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_resample", type=int, default=8,
                        help="target-policy draws averaged into the target action kernel")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--ae_noise_std", type=float, default=0.05,
                        help="denoising corruption added during AE training")
    parser.add_argument("--ae_iterations", type=int, default=4000)
    parser.add_argument("--flow_iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--n_ref", type=int, default=10,
                        help="fresh interventional reference draws per metric")
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="DGP seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="pipeline repeats per data seed")
    parser.add_argument("--tag", type=str, default="splitae")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--dsprites_npz", type=str,
                        default=str(REPO_ROOT / "data" / "dsprites"
                                    / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available")
    args = parser.parse_args()
    # A flag belonging to the OTHER design is refused rather than ignored: ignored, it
    # would still be written into the CSV and (for --policy_shift) into the filename,
    # describing a DGP the run did not use.
    other = ({"policy_shift"} if args.design == "confounded"
             else {"policy_gain", "policy_angle", "target_shift", "mean_preserving"})
    wrong = [name for name in sorted(other)
             if getattr(args, name) != parser.get_default(name)]
    if wrong:
        parser.error(f"--design {args.design} does not use "
                     + ", ".join("--" + name for name in wrong))
    return resolve_design_defaults(args)


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    # Tokens appended only off-default so historical filenames are reproduced exactly.
    # `_nosplit` is the AUTOENCODER axis (same token as ColorMNIST); `_dr*` is the
    # separate nuisance axis. They are different ablations and must stay distinguishable.
    suffix += "_nosplit" if args.ae_split == "none" else ""
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
    if args.n_particles is not None:
        suffix += f"_M{args.n_particles}"
    if args.flow_iterations != 2000:
        suffix += f"_it{args.flow_iterations}"
    estimators = resolve_estimators(args.estimators)
    if estimators != ("dr",):
        suffix += "_est" + "-".join(estimators)
    if args.ipw_form != "ht":
        suffix += f"_ipw{args.ipw_form}"
    if args.target_mass != "rescale":
        suffix += f"_mass{args.target_mass}"
    if args.design == "confounded":
        # A design token FIRST, and this design's OWN parameters in place of the cpme
        # ones. The two templates diverge immediately after `_nobs<N>` -- `_tshift` here
        # against `_shift` below -- so no confounded name can equal a cpme name whatever
        # --tag is passed, and no filename can describe a DGP the run did not use.
        name = (f"dsprites_confounded_{args.tag}_nobs{args.n_obs}"
                f"_tshift{args.target_shift:g}_pgain{args.policy_gain:g}"
                f"_ang{args.policy_angle:g}_sig{args.sigma_policy:g}"
                f"_gain{args.pos_gain:g}_noise{args.pixel_noise:g}"
                f"{'_meanpres' if args.mean_preserving else ''}"
                f"_ds{seeds}{suffix}.csv")
    else:
        name = (f"dsprites_{args.tag}_nobs{args.n_obs}_shift{args.policy_shift:g}"
                f"_sig{args.sigma_policy:g}_gain{args.pos_gain:g}_noise{args.pixel_noise:g}"
                f"_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def score(sample: torch.Tensor, references: list, device: torch.device) -> dict:
    """Energy distance / SW1 / SW2 against each reference; mean and std over references.

    Scored on `device`: the O(n^2) pairwise distances behind the energy distance are
    prohibitive on CPU at the larger sweep sizes.
    """
    energy_kernel = RieszKernel()
    sample = sample.double().to(device)
    out = {}
    for metric, fn in (("energy", lambda a, b: float(squared_mmd(energy_kernel, a, b))),
                       ("sw1", lambda a, b: float(sliced_wasserstein(a, b, p=1, seed=0))),
                       ("sw2", lambda a, b: float(sliced_wasserstein(a, b, p=2, seed=0)))):
        values = [fn(sample, reference.double().to(device)) for reference in references]
        out[f"{metric}_mean"], out[f"{metric}_std"] = float(np.mean(values)), float(np.std(values))
    return out


def named_scores(metrics: dict, row: str, prefix: str = "") -> dict:
    """A `score` result keyed by the CSV columns of one metric row.

    Parameters
    ----------
    metrics : dict
        As returned by `score` or `score_matched`.
    row : str
        The metric row, e.g. "flow" or "floor_clean_matched".
    prefix : str
        "pos_" for the snapped-position metrics, "" for the pixel metrics.

    Returns
    -------
    dict
        e.g. {"pos_energy_flow_mean": ..., "pos_energy_flow_std": ..., ...}.
    """
    return {f"{prefix}{metric}_{row}_{stat}": metrics[f"{metric}_{stat}"]
            for metric in METRICS for stat in ("mean", "std")}


def subsample(sample: torch.Tensor, k: int, generator: "torch.Generator") -> torch.Tensor:
    """A uniform random k-row subsample; the sample itself when it has at most k rows."""
    if sample.shape[0] <= k:
        return sample
    pick = torch.randperm(sample.shape[0], generator=generator)[:k]
    return sample[pick.to(sample.device)]


def score_matched(sample: torch.Tensor, references: list, device: torch.device,
                  k: int, seed: int) -> dict:
    """`score` with BOTH sides randomly subsampled to k rows.

    The energy distance and the sliced Wasserstein distances are strongly
    SAMPLE-SIZE BIASED: on this design the SW2 between two independent draws of the
    same law falls from 0.027 at n = 250 to 0.007 at n = 4000. The references here
    are built at n_causal while the flow row carries only `n_particles` atoms (and
    `sliced_wasserstein` truncates to the smaller side), so a plain floor -- scored
    at the full reference size -- sits at a different point of that bias curve than
    the row it is supposed to bound. Re-scoring the floor at k = the flow's own
    effective count makes the two comparable, which is what a --n_particles sweep
    needs: otherwise the flow rows move with M for a purely statistical reason while
    the floor stays put.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, d)
    references : list of torch.Tensor
    device : torch.device
        Where the metrics are evaluated; see `score`.
    k : int
        Effective count of the row this floor is matched to.
    seed : int
        Fixed seed of the subsampling draw, derived from the pipeline seed.

    Returns
    -------
    dict
        The keys of `score`.
    """
    generator = torch.Generator().manual_seed(seed)
    return score(subsample(sample, k, generator),
                 [subsample(reference, k, generator) for reference in references],
                 device)


def latent_scores(sample: torch.Tensor, references: list, row: str) -> dict:
    """Latent SW2 of `sample` against each reference draw, as that row's two columns.

    Measured in the FLOW'S OWN SPACE -- standardized latents -- because that is where
    the particles live and where the MMD objective is written, so the number scores the
    thing the sampler actually optimises, with no decoder between. The pixel rows score
    the composition (sampler THEN decoder) and are the stricter test; this one isolates
    the sampler.

    This matches CelebA's CONVENTION exactly: that design also standardizes, and it
    standardizes its reference latents by the pooled LOGGED moments, which is what
    `to_flow` does here. It does NOT make the two numbers interchangeable, and neither is
    interchangeable with ColorMNIST's, which is measured in RAW latent units because that
    design does not standardize at all. Even between the two standardized designs the SCALE
    differs with the latent dimension: sliced Wasserstein averages over coordinates, so the
    same per-coordinate displacement costs about 63 times more at this design's d = 32 than
    at CelebA's d = 4096 (measured at n = 4000: 0.0499 of excess over the two-sample floor
    at d = 32, 0.0106 at d = 256, 0.0008 at d = 4096). There is also no DecFM reference
    number for dSprites anywhere in the repository.
    So the quantity that travels between designs is each one's ratio of `latent_sw2_flow`
    to its OWN `latent_sw2_floor_matched`, not the raw column. Quoting the raw columns of
    two designs side by side compares two different autoencoders over two different laws in
    two different dimensions.

    Only SW2 is reported: the pixel and position rows already carry the energy distance
    and SW1, and SW2 is the metric the latent comparison is written in elsewhere.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, latent_dim)
        Standardized latents.
    references : list of torch.Tensor
        Standardized latents of the fresh interventional reference draws.
    row : str
        The metric row, e.g. "flow" or "floor_matched".

    Returns
    -------
    dict
        {"latent_sw2_<row>_mean": ..., "latent_sw2_<row>_std": ...}
    """
    values = [float(sliced_wasserstein(sample, reference, p=2, seed=0))
              for reference in references]
    return {f"latent_sw2_{row}_mean": float(np.mean(values)),
            f"latent_sw2_{row}_std": float(np.std(values))}


def latent_scores_matched(sample: torch.Tensor, references: list, row: str,
                          k: int, seed: int) -> dict:
    """`latent_scores` with BOTH sides randomly subsampled to k rows.

    The same sample-size bias `score_matched` documents applies in the latent space, and
    for the same reason: the references carry n_causal rows while the flow row carries
    only `n_particles` atoms. The subsampling follows the same PROCEDURE as
    `score_matched` -- one generator advanced over the sample and then over each
    reference -- but not the same draw: this floor gets its own seed offset, so it selects
    a DIFFERENT subset of rows than either pixel floor or the position floor. The four
    matched floors are independent estimates of the same k-sample bias, not four views of
    one subsample, so a disagreement between the pixel and latent matched floors on one
    repeat is partly subsampling noise and not evidence that the two spaces disagree about
    the same images.

    Parameters
    ----------
    sample : torch.Tensor of shape (n, latent_dim)
    references : list of torch.Tensor
    row : str
        The metric row, e.g. "floor_matched".
    k : int
        Effective count of the row this floor is matched to.
    seed : int
        Fixed seed of the subsampling draw, derived from the pipeline seed.

    Returns
    -------
    dict
        The keys of `latent_scores`.
    """
    generator = torch.Generator().manual_seed(seed)
    return latent_scores(subsample(sample, k, generator),
                         [subsample(reference, k, generator) for reference in references],
                         row)


def run_repeat(sprite_bank: np.ndarray, bank_flat: torch.Tensor,
               bank_positions: torch.Tensor, data: dict, setup: dict,
               args: argparse.Namespace, data_seed: int, repeat: int,
               device: torch.device, fid_scorers: dict = None) -> list:
    """One full pipeline pass on a fixed logged sample; one CSV row per estimator.

    The wall-clock columns and the two identities they satisfy are specified in the module
    docstring; the per-phase clocks below are the implementation of it. `setup` carries
    this data seed's per-DATA-SEED timing columns, measured in `main` outside this repeat's
    clock and copied onto every row unchanged.
    """
    estimators = resolve_estimators(args.estimators)
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def sync() -> None:
        """Block until queued CUDA work is done, so a wall-clock stop means what it says.

        Kernel launches are asynchronous: `time.time()` read straight after a GPU phase
        stops the clock while that phase is still running, and its tail is then billed to
        whichever later phase first copies a result to the host. Waiting only reorders the
        host's idle time -- no tensor, no RNG draw and no result changes -- so it is safe
        to put at every phase boundary. A no-op on CPU.
        """
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # The single-attribute (1024, 4096) position bank exists only under --design cpme; see
    # `main`. Everything it supports -- the snap, the position probe, the clean-sprite
    # floor, the denoise MSE and the position law -- is skipped when it is absent and
    # written as NaN. The guard is on the BANK, not on a try/except: the confounded
    # design's `pos_logged` / `pos_target` are CONTINUOUS coordinates in [0, 31], so the
    # `.long()` indexing below would not raise, it would silently index the heart bank
    # with truncated coordinates and write plausible wrong numbers.
    has_bank = bank_flat is not None
    nan = float("nan")

    # -- split: autoencoder half / causal half (disjoint), or shared ---------------------
    if args.ae_split == "none":
        # every row trains the autoencoder AND carries the causal estimate
        ae_idx = causal_idx = torch.arange(args.n_obs)
        n_ae = args.n_obs
    else:
        n_ae = int(round(args.ae_fraction * args.n_obs))
        split = torch.randperm(args.n_obs, generator=torch.Generator().manual_seed(seed + 2))
        ae_idx, causal_idx = split[:n_ae], split[n_ae:]
    n_causal = len(causal_idx)

    Y_ae_img = data["Y_logged"][ae_idx].float().reshape(-1, 1, 64, 64).to(device)
    Y_log_img = data["Y_logged"][causal_idx].float().reshape(-1, 1, 64, 64).to(device)
    Y_target = data["Y_target"][causal_idx].float()
    pos_logged = data["pos_logged"][causal_idx].double()
    pos_target = data["pos_target"][causal_idx].double()
    X_np = data["X"][causal_idx].numpy()
    A_np = data["A_logged"][causal_idx].numpy()

    # -- denoising autoencoder on the AE half's logged images only ---------------------
    # The denoising targets are the noisy logged images themselves (no clean sprites,
    # no bank): honest for a logged-bandit application, at the price of the decoder
    # partially reproducing the observation noise instead of stripping it.
    t0 = time.time()
    ae, history = train_dsprite_ae(Y_ae_img, latent_dim=args.latent_dim,
                                   noise_std=args.ae_noise_std,
                                   n_iterations=args.ae_iterations, device=device,
                                   verbose=False)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    sync()
    seconds_ae = time.time() - t0

    # The autoencoder's reconstruction probes are part of its stage, not an unattributed
    # phase: the logged round-trip (which the `*_roundtrip` metrics reuse below, so its
    # decode is charged here rather than to the shared scoring), the sprite-bank encode
    # and the position probe all measure the feature map this repeat just trained.
    t0 = time.time()
    with torch.no_grad():
        recon = torch.cat([ae(Y_log_img[i:i + 256]) for i in range(0, n_causal, 256)])
    if has_bank:
        denoise_mse = float(((recon.reshape(n_causal, -1)
                              - bank_flat[pos_logged[:, 0].long() * 32
                                          + pos_logged[:, 1].long()]) ** 2).mean())
        bank_t = bank_flat.reshape(1024, 1, 64, 64)
        Z_bank = encode_in_batches(ae.encode,
                                   bank_t + args.pixel_noise * torch.randn(bank_t.shape,
                                                                           device=device))
        probe = position_probe(Z_bank, bank_positions)
    else:
        # The confounded design has no clean single-sprite target to difference against
        # and no position grid to probe: its content varies in shape and scale too. The
        # decoder is measured instead by the `roundtrip` pixel row and `fid_*_roundtrip`,
        # which need no bank, and by `ae_train_mse`.
        denoise_mse = nan
        probe = {"posX_r2": nan, "posY_r2": nan}
    sync()
    seconds_ae += time.time() - t0

    # -- the ratio half of the nuisance fit: cross-fitted continuous-action importance
    # weights. This is all the IPW estimator needs; the folds are clocked here because
    # this half consumes them first, and the CME half reuses them.
    t0 = time.time()
    fold_ids = make_crossfit_fold_ids(n_causal, n_folds=args.n_folds, seed=0)
    if args.dr_fit == "nosplit":
        # Fit the logging density once on all rows and score those same rows. The
        # cross-fitted helper has no single-fold path, and a degenerate fold split
        # would not be the full-sample fit the same-sample argument refers to.
        logging_hat = fit_linear_gaussian_policy(X_np, A_np)
        log_pi0_hat = logging_hat.log_prob(X_np, A_np)
    else:
        log_pi0_hat = crossfit_logging_policy_log_prob(X_np, A_np, fold_ids=fold_ids)
    w_hat = importance_weights_from_logging_log_prob(data["target_policy"], X_np, A_np,
                                                     log_pi0_hat)
    if args.weight_clip is not None and args.weight_clip > 0:
        w_hat = np.minimum(w_hat, args.weight_clip)
    oracle_diag = importance_weight_diagnostics(
        data["importance_weights"][causal_idx].numpy())
    diag = importance_weight_diagnostics(w_hat,
                                         data["importance_weights"][causal_idx].numpy())
    sync()
    seconds_ratio = time.time() - t0

    # The logged latents are the autoencoder's product, not a nuisance: the CME, the flow
    # and the particle cloud all read them, so the pass belongs to the autoencoder stage.
    # CelebA, whose encoder is frozen and pretrained, can hoist the same work out of the
    # repeat entirely (its `seconds_encode`); here the encoder is this repeat's own.
    t0 = time.time()
    Z_log = encode_in_batches(ae.encode, Y_log_img).cpu()
    sync()
    seconds_ae += time.time() - t0

    # -- the CME half: the lengthscale locks, the Gram blocks, the ridge selection and the
    # single solve behind every weight vector. All the plug-in estimator needs; DR needs
    # both halves.
    t0 = time.time()
    solver = resolve_solver(args)
    kernel_x = RBF(use_length_scale_heuristic=True)
    kernel_a = RBF(use_length_scale_heuristic=True)
    # The nuisance joins the autoencoder, the flow and the evaluation on the run
    # device, so a reported wall clock belongs to one device. The projected solver
    # only forms n x n_landmarks blocks, so this costs megabytes.
    X_t = data["X"][causal_idx].double().to(device)
    A_t = data["A_logged"][causal_idx].double().to(device)

    # Both lengthscales are locked ONCE on a bounded, SEEDED anchor subsample, as the
    # ColorMNIST simulation does. Two reasons. The median over all n_causal^2 pairs is
    # unaffordable at the sweep sizes; and `calculate_and_fix_length_scale` subsamples
    # with the GLOBAL torch RNG above its own 5000-row limit, so the lengthscale would
    # otherwise depend on how much randomness the pipeline consumed earlier -- a seeded
    # anchor is reproducible whatever else changes upstream. Unlike ColorMNIST/CelebA,
    # where the action factor is an equality indicator, the action here is CONTINUOUS
    # and 2-d, so kernel_a is a genuine RBF and needs a lock of its own.
    anchor = torch.randperm(n_causal, generator=torch.Generator().manual_seed(0))[:4000]
    kernel_x.calculate_and_fix_length_scale(X_t[anchor])
    kernel_a.calculate_and_fix_length_scale(A_t[anchor])

    # The target-policy action draws are stored once (n_resample x n_causal x action
    # dim, trivial) so the target action kernel can be evaluated blockwise instead of
    # accumulated into a full n_causal x n_causal matrix. `resample_rng` is seeded and
    # consumed in the same order as before, so the draws themselves are unchanged.
    resample_rng = np.random.default_rng(0)
    A_resamples = [torch.as_tensor(data["target_policy"].sample(X_np, resample_rng),
                                   dtype=torch.float64).to(device)
                   for _ in range(args.n_resample)]

    def gram_x(rows, cols):
        "K_X block alone; the target-action factor needs it without the logged one."
        return kernel_x(X_t[rows], X_t[cols])

    def gram_a(rows, cols):
        "K_A block over the LOGGED actions; a genuine RBF here, the action is continuous."
        return kernel_a(A_t[rows], A_t[cols])

    def gram_a_target(rows, cols):
        """mean_b k_A(a_i, a~_j^b) block.

        Note the ORIENTATION: `rows` index the LOGGED actions and `cols` the target
        policy DRAWS, matching K_a_target[i, j] = mean_b k_A(A_logged[i], A_draw_b[j]).
        Transposing it would silently estimate a different object.
        """
        block = kernel_a(A_t[rows], A_resamples[0][cols])
        for A_resample in A_resamples[1:]:
            block = block + kernel_a(A_t[rows], A_resample[cols])
        return block / len(A_resamples)

    def gram(rows, cols):
        "K = K_X * K_A on demand, so the full n_causal^2 Gram is never formed."
        return gram_x(rows, cols) * gram_a(rows, cols)

    def gram_pi(rows, cols):
        "K_pi = K_X * K_A_target on demand."
        return gram_x(rows, cols) * gram_a_target(rows, cols)

    # The exact solvers take whole matrices, so they -- and only they -- pay the O(n^2).
    if solver == "exact":
        K_x = kernel_x(X_t)
        K_a = kernel_a(A_t)
        K_a_target = torch.zeros(n_causal, n_causal, dtype=torch.float64)
        for A_resample in A_resamples:
            K_a_target += kernel_a(A_t, A_resample)
        K_a_target /= args.n_resample

    if args.lambda_rule == "n_half":
        nosplit_lambda = lambda_n_half(n_causal, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    else:
        fold_lambdas = {}
        for fold in np.unique(fold_ids):
            train = np.where(fold_ids != fold)[0]
            # lambda does not need the full training block, which is itself an n^3 solve
            pick = train[torch.randperm(len(train),
                                        generator=torch.Generator().manual_seed(0))[:2000].numpy()]
            fold_lambdas[int(fold)], _ = select_regularization(gram(pick, pick),
                                                               Z_log[pick, 0].double().to(device), seed=0)
        nosplit_lambda = None
        if args.dr_fit == "nosplit":
            pick = torch.randperm(n_causal,
                                  generator=torch.Generator().manual_seed(0))[:2000].numpy()
            nosplit_lambda, _ = select_regularization(gram(pick, pick),
                                                      Z_log[pick, 0].double().to(device), seed=0)
    w_t = torch.as_tensor(w_hat, dtype=torch.float64).to(device)
    # One solve serves every requested estimator: `components=True` returns the pair
    # (v_PI, v_correction) whose sum is the DR weight vector. The split has to happen
    # inside the solver, because the landmark draw is stateful and two calls would fit
    # two different Nystrom subspaces. It is requested only when a non-DR estimator is
    # wanted, so a DR-only run keeps the historical single-right-hand-side path.
    components = estimators != ("dr",)
    if solver in ("nystrom", "projected_nystrom"):
        # `projected_nystrom` restricts the CME ridge regression to the landmark-spanned
        # subspace and builds the DR right-hand side only at the landmarks -- the
        # estimator the imported consistency rate is stated for, touching an m x |H|
        # cross-kernel instead of a |T| x |H| one. `nystrom` applies an approximate Gram
        # inverse to the exact full right-hand side: a different estimator, kept as the
        # baseline that produced every earlier table.
        projected = solver == "projected_nystrom"
        if args.dr_fit == "nosplit":
            solve = (cpme_dr_weights_projected_nystrom if projected
                     else cpme_dr_weights_nystrom)
            solved = solve(gram, gram_pi, w_t, nosplit_lambda,
                           n_landmarks=args.n_landmarks,
                           generator=torch.Generator().manual_seed(seed),
                           components=components)
        else:
            solve = (cpme_dr_weights_crossfit_projected_nystrom if projected
                     else cpme_dr_weights_crossfit_nystrom)
            solved = solve(gram, gram_pi, w_t, fold_ids, fold_lambdas,
                           n_landmarks=args.n_landmarks,
                           generator=torch.Generator().manual_seed(seed),
                           components=components)
    else:
        if args.dr_fit == "nosplit":
            solved = cpme_dr_weights(K_x, K_a, K_a_target, w_t, nosplit_lambda,
                                     components=components)
        else:
            solved = cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_t, fold_ids,
                                              fold_lambdas, components=components)

    # The estimator axis lives entirely in this dictionary: below it, only the weight
    # vector handed to the flow changes.
    target_weights = {"ipw": cpme_ipw_weights(w_t,
                                          normalize=args.ipw_form == "hajek")}
    if components:
        v_plugin, v_correction = solved
        target_weights["plugin"] = v_plugin
        target_weights["dr"] = v_plugin + v_correction
    else:
        target_weights["dr"] = solved
    # The mass each estimator actually produced, kept before the projection below so the
    # deficit stays auditable: `sum_v` reports the mass the flow was given (1 under any
    # projection), `sum_v_raw` the mass the estimator returned.
    raw_mass = {name: float(v.sum()) for name, v in target_weights.items()}
    target_weights = {name: project_to_unit_mass(v, args.target_mass)
                      for name, v in target_weights.items()}
    sync()
    seconds_cme = time.time() - t0

    # Round every shared part FIRST and build the sums from the rounded values, so both
    # identities in the module docstring hold on the numbers the CSV carries and not merely
    # on the raw floats behind them.
    seconds_ae = round(seconds_ae, 1)
    seconds_ratio = round(seconds_ratio, 1)
    seconds_cme = round(seconds_cme, 1)
    seconds_nuisance = round(seconds_ratio + seconds_cme, 1)

    # -- shared flow geometry: standardized latents and the initial particle cloud -----
    latent_mean = Z_log.mean(dim=0, keepdim=True)
    latent_std = Z_log.std(dim=0, keepdim=True).clamp_min(1e-3)
    to_flow = lambda z: ((z - latent_mean) / latent_std).float()
    from_flow = lambda z: z.cpu().float() * latent_std + latent_mean

    flow_target = to_flow(Z_log).to(device)      # the logged latents: always the target
    # The particle cloud is drawn from the target support but is NOT the same object:
    # decoupling them is what makes a particle sweep possible without also changing the
    # DR support (which would confound M with the estimator).
    n_particles = (flow_target.shape[0] if args.n_particles is None
                   else min(args.n_particles, flow_target.shape[0]))
    if args.init == "observed":
        if n_particles == flow_target.shape[0]:
            initial_particles = flow_target.clone()   # historical path, bitwise unchanged
        else:
            pick = torch.randperm(flow_target.shape[0],
                                  generator=particle_generator(args, seed, 1))[:n_particles]
            initial_particles = flow_target[pick.to(device)].clone()
    else:
        initial_particles = gaussian_init(
            flow_target, n_particles, args.init,
            torch.Generator(device=device).manual_seed(seed + 3))

    def decode(particles):
        with torch.no_grad():
            return torch.cat([ae.decode(from_flow(particles[i:i + 256]).to(device))
                              for i in range(0, particles.shape[0], 256)]
                             ).reshape(particles.shape[0], -1)

    # -- evaluation against fresh matched-size interventional references ---------------
    # Drawn once per repeat: every estimator is scored against the SAME references, and
    # the rows that do not involve the flow -- the logged images, their round-trip and
    # both floors -- are computed once and copied into each estimator's row.
    t0 = time.time()
    # Only the oracle arm of each draw is scored, so the logged arm -- the same 4096-d
    # block again -- is dropped as each draw is made rather than held for the repeat.
    fresh_img, fresh_pos = [], []
    for j in range(args.n_ref):
        draw = sample_design(args, sprite_bank, n_causal, 100000 + 1000 * seed + j)
        fresh_img.append(draw["Y_target"].float())
        if has_bank:
            fresh_pos.append(draw["pos_target"].double())

    # Latent references and latent floor: the same fresh interventional draws, and this
    # repeat's own noisy-oracle counterpart, pushed through this repeat's encoder and
    # standardized exactly as `flow_target` is, so they land in the particles' own
    # coordinates. These passes are MEASUREMENT, not representation, so they are billed
    # to the shared evaluation clock below and NOT to `seconds_ae` -- unlike the `Z_log`
    # encode, which the CME and the flow themselves read.
    def to_latent(images: torch.Tensor) -> torch.Tensor:
        """Flat images -> standardized latents on the scoring device."""
        latents = encode_in_batches(ae.encode, images.reshape(-1, 1, 64, 64).to(device))
        return to_flow(latents.cpu()).to(device)

    fresh_latent = [to_latent(img) for img in fresh_img]
    Z_floor = to_latent(Y_target)

    obs_decode = recon.reshape(n_causal, -1).cpu()
    shared_pixel = {"logged": data["Y_logged"][causal_idx].float(),
                    "roundtrip": obs_decode,
                    "floor_noisy": Y_target}
    if has_bank:
        shared_pixel["floor_clean"] = bank_flat.cpu()[(pos_target.long()[:, 0] * 32
                                                       + pos_target.long()[:, 1])].float()
        _, obs_offmani = snap_to_bank(obs_decode.to(device), bank_flat)
        offmani_roundtrip = {
            "offmani_roundtrip_median": float(obs_offmani.median()),
            "offmani_roundtrip_frac3": float((obs_offmani > 3).float().mean())}
    else:
        # At --pixel_noise 0 the `floor_noisy` row IS the clean oracle draw, so no floor
        # is lost -- only the bank-snapped version of it.
        offmani_roundtrip = {"offmani_roundtrip_median": nan,
                             "offmani_roundtrip_frac3": nan}

    base_row = {
        "tag": args.tag, "design": args.design, "device": str(device), "init": args.init,
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "n_obs": args.n_obs, "sigma_policy": args.sigma_policy,
        "pos_gain": args.pos_gain, "pixel_noise": args.pixel_noise,
        # Exactly one of these groups is a number on any row; the other is NaN, so a row
        # can never be read as describing a DGP it did not use.
        "policy_shift": args.policy_shift if args.design == "cpme" else nan,
        "policy_gain": args.policy_gain if args.design == "confounded" else nan,
        "policy_angle": args.policy_angle if args.design == "confounded" else nan,
        "target_shift": args.target_shift if args.design == "confounded" else nan,
        "mean_preserving": (int(args.mean_preserving) if args.design == "confounded"
                            else nan),
        "n_ae": int(n_ae), "n_causal": int(n_causal), "ae_split": args.ae_split,
        "n_folds": args.n_folds,
        "n_resample": args.n_resample, "n_ref": args.n_ref,
        "latent_dim": args.latent_dim, "ae_iterations": args.ae_iterations,
        "flow_iterations": args.flow_iterations, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "ae_train_mse": float(history[-1][1]), "denoise_mse_logged": denoise_mse,
        "probe_r2_x": float(probe["posX_r2"]), "probe_r2_y": float(probe["posY_r2"]),
        "weight_clip": (args.weight_clip if args.weight_clip is not None
                        and args.weight_clip > 0 else float("inf")),
        "solver": solver, "dr_fit": args.dr_fit, "lambda_rule": args.lambda_rule,
        "ipw_form": args.ipw_form, "target_mass": args.target_mass, "n_particles": int(n_particles),
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "oracle_w_ess": float(oracle_diag["ess"]), "est_w_ess": float(diag["ess"]),
        "est_w_max": float(diag["max"]), "est_w_log_rmse": float(diag["log_rmse"]),
        "lengthscale_x": float(kernel_x.length_scale),
        "lengthscale_a": float(kernel_a.length_scale),
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        **offmani_roundtrip,
        "pos_mean_logged_x": float(pos_logged[:, 0].mean()),
        "pos_mean_logged_y": float(pos_logged[:, 1].mean()),
        "pos_mean_oracle_x": float(pos_target[:, 0].mean()),
        "pos_mean_oracle_y": float(pos_target[:, 1].mean()),
        "seconds_ae": seconds_ae, "seconds_ratio": seconds_ratio,
        "seconds_cme": seconds_cme, "seconds_nuisance": seconds_nuisance,
        # Identical on every row of every repeat of this data seed, and measured outside
        # this repeat's clock: the second term of the per-task identity.
        **setup,
        **hardware_provenance(device),
    }
    for name, sample in shared_pixel.items():
        base_row.update(named_scores(score(sample, fresh_img, device), name))
    if has_bank:
        for name, sample in (("logged", pos_logged), ("floor", pos_target)):
            base_row.update(named_scores(score(sample, fresh_pos, device), name, "pos_"))
    else:
        # Every column that needs the position bank, written as NaN EXPLICITLY rather than
        # left out: an omitted key would also come out as an empty field, so writing "nan"
        # is what distinguishes "not available for this design" from a computation that
        # silently failed to run. The whole position-law block goes, not only its flow row:
        # the flow row needs the snap, and a floor with no flow row to bound is not a
        # comparison. `pos_mean_*` and `reweighted_pos_*` are scalars off the DGP's own
        # continuous coordinates and are kept.
        base_row.update({f"{metric}_floor_clean{matched}_{stat}": nan
                         for metric in METRICS for matched in ("", "_matched")
                         for stat in ("mean", "std")})
        base_row.update({f"pos_{metric}_{row}_{stat}": nan
                         for row in POSITION_ROWS + ("floor_matched",)
                         for metric in METRICS for stat in ("mean", "std")})
    # The latent rows that do not involve the flow. `initial` and `logged` are the two
    # baselines the flow has to beat: the particle cloud it starts from, and the logged
    # latents themselves, which is the confounded answer. Like the pixel rows, `initial`
    # is scored at the particle count while `logged` and `floor` are scored at n_causal
    # (`sliced_wasserstein` truncates to the smaller side), which is what the matched
    # floor below exists to correct.
    for name, sample in (("initial", initial_particles), ("logged", flow_target),
                         ("floor", Z_floor)):
        base_row.update(latent_scores(sample, fresh_latent, name))

    # Count-matched floors: the same quantities at the flow row's effective count.
    n_matched = int(n_particles)
    for offset, name in enumerate(PIXEL_FLOOR_ROWS):
        if name not in shared_pixel:
            continue        # no clean-sprite floor without the bank; `offset` is unmoved
        base_row.update(named_scores(
            score_matched(shared_pixel[name], fresh_img, device, n_matched,
                          500000 + 10 * seed + offset), f"{name}_matched"))
    if has_bank:
        base_row.update(named_scores(
            score_matched(pos_target, fresh_pos, device, n_matched,
                          500000 + 10 * seed + len(PIXEL_FLOOR_ROWS)),
            "floor_matched", "pos_"))
    base_row.update(latent_scores_matched(
        Z_floor, fresh_latent, "floor_matched", n_matched,
        500000 + 10 * seed + len(PIXEL_FLOOR_ROWS) + 1))
    base_row["n_floor_matched"] = n_matched

    # FID on channel-replicated greyscale. The reference is a SINGLE fresh draw, not the
    # average over all --n_ref of them that the SW and energy rows use: Inception features
    # over n_causal images are the expensive part, and ColorMNIST and CelebA both score
    # against one draw, so matching them keeps the column comparable in construction even
    # though the numbers are not (see `to_rgb`). The floor is a SECOND fresh draw, which
    # is why --n_ref must be at least 2 for FID.
    if fid_scorers:
        if args.n_ref < 2:
            sys.exit("--fid needs --n_ref >= 2: the FID floor is a second oracle draw.")
        fid_reference = reference_fid_features(fid_scorers, fresh_img[0], args.fid_n_ref)
        base_row.update(fid_rows(fid_scorers, fid_reference,
                                 (("initial", decode(initial_particles).cpu()),
                                  ("floor", fresh_img[1]),
                                  ("roundtrip", obs_decode)), args.fid_n_ref))
        base_row["n_fid_ref"] = min(args.fid_n_ref, len(fresh_img[0]))
        base_row["n_fid_row"] = min(args.fid_n_ref, len(fresh_img[1]))
    else:
        fid_reference = {}
        base_row["n_fid_ref"] = base_row["n_fid_row"] = float("nan")
    sync()
    # The last of the three SHARED phases. `base_row` is copied into every estimator row,
    # so these columns are automatically identical across the rows of this repeat -- which
    # is exactly why they must not be summed across them. `seconds_shared` is built from
    # the already-ROUNDED parts so that it equals their sum on the numbers in the CSV.
    base_row["seconds_eval_shared"] = round(time.time() - t0, 1)
    base_row["seconds_shared"] = round(base_row["seconds_ae"]
                                       + base_row["seconds_nuisance"]
                                       + base_row["seconds_eval_shared"], 1)

    def estimator_row(estimator: str, v: torch.Tensor) -> dict:
        """One CSV row: the flow matched to the target weights `v`, and what follows it.

        The logged sample, the autoencoder, the nuisance fit, the initial particles,
        the flow budget and the reference draws are all fixed above, so two rows from
        the same repeat differ in `v` and in nothing else.

        The three timings written here -- `seconds_flow`, `seconds_decode` and
        `seconds_eval` -- are this estimator's OWN work and nothing else, so they can be
        summed across the rows of a repeat. In particular `seconds_eval` excludes the
        shared scoring, which lives in `seconds_eval_shared` on every row; until
        2026-10-03 the two were added together here, which triple-counted the shared half
        in a three-estimator run. `seconds_repeat` is NOT set here -- it is stamped on
        every row once the repeat is over, so it is a constant total rather than the
        cumulative clock it used to be.
        """
        t0 = time.time()
        particles, info = adamw_riesz_flow(
            initial_particles, flow_target,
            target_weights=v.float().to(device), sliced=False,
            n_iterations=args.flow_iterations, learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps, snapshot_every=args.snapshot_every,
            log_every=500, verbose=False)
        # `seconds_flow` covers this estimator's flow call alone. Until 2026-09-28 it
        # also contained the decode below, which is itself O(n_particles) and so
        # contaminated exactly the --n_particles axis; runs archived before that date
        # are not comparable with this column.
        sync()
        seconds_flow = time.time() - t0

        t0 = time.time()
        flow_images = decode(particles)
        sync()
        seconds_decode = time.time() - t0

        t0 = time.time()
        # The transport readback is billed to this row's own evaluation: it is a
        # diagnostic of the flow output, and keeping it inside a timed phase leaves no
        # untimed gap between the three per-estimator clocks.
        mean_transport = float((particles.cpu() - initial_particles.cpu())
                               .norm(dim=1).mean())
        if has_bank:
            flow_pos, flow_offmani = snap_to_bank(flow_images, bank_flat)
            flow_pos = flow_pos.cpu().float()
            snapped = {
                "offmani_flow_median": float(flow_offmani.median()),
                "offmani_flow_frac3": float((flow_offmani > 3).float().mean()),
                "pos_mean_flow_x": float(flow_pos[:, 0].mean()),
                "pos_mean_flow_y": float(flow_pos[:, 1].mean())}
        else:
            snapped = {"offmani_flow_median": nan, "offmani_flow_frac3": nan,
                       "pos_mean_flow_x": nan, "pos_mean_flow_y": nan}
        row = dict(base_row)
        row.update({
            "estimator": estimator,
            **weight_diagnostics(v),
            # The mass BEFORE the projection; `sum_v` above is the mass the flow saw.
            "sum_v_raw": raw_mass[estimator],
            "reweighted_pos_x": float((v.cpu() * pos_logged[:, 0]).sum()),
            "reweighted_pos_y": float((v.cpu() * pos_logged[:, 1]).sum()),
            "loss_initial": float(info["loss"][0]),
            "loss_final": float(info["loss"][-1]),
            "mean_transport": mean_transport,
            **snapped,
        })
        row.update(named_scores(score(flow_images.cpu(), fresh_img, device), "flow"))
        if has_bank:
            row.update(named_scores(score(flow_pos.double(), fresh_pos, device),
                                    "flow", "pos_"))
        # The latent row needs no encode: the particles ARE standardized latents, so this
        # scores the sampler alone, with the decoder taken back out of the measurement.
        row.update(latent_scores(particles, fresh_latent, "flow"))
        # This row's own FID, against the reference features the shared block built once.
        row.update(fid_rows(fid_scorers, fid_reference,
                            (("flow", flow_images.cpu()),), args.fid_n_ref))
        if fid_scorers:
            # The flow row is M images; smaller than --fid_n_ref it is scored whole and is
            # then NOT count-matched to the floor. Record the smaller of the two.
            row["n_fid_row"] = min(base_row["n_fid_row"], len(flow_images))
        sync()
        row.update({"seconds_flow": round(seconds_flow, 1),
                    "seconds_decode": round(seconds_decode, 1),
                    "seconds_eval": round(time.time() - t0, 1)})
        # Derived, and in neither identity: this estimator's own half of the nuisance fit
        # plus this row's own flow and decode, with no autoencoder and no measurement in it.
        row["seconds_sampling"] = sampling_seconds(estimator, base_row,
                                                  row["seconds_flow"],
                                                  row["seconds_decode"])
        return row

    rows = [estimator_row(name, target_weights[name]) for name in estimators]

    # -- wall-clock bookkeeping --------------------------------------------------------
    # `seconds_repeat` is the whole repeat and is stamped identically on every row, so it
    # can be averaged over rows or read off any one of them, but never summed. The parts
    # above do not quite add up to it -- the split, the latent standardisation and the
    # particle init are glue rather than named phases and sit inside none of them -- so
    # the shortfall is written out as `seconds_other` instead of being left for the reader
    # to discover. It is computed FROM THE ROUNDED columns, which makes
    #     seconds_repeat == seconds_shared + seconds_other
    #                       + sum_rows(seconds_flow + seconds_decode + seconds_eval)
    # an exact identity on the CSV values rather than one that holds up to rounding.
    sync()
    seconds_repeat = round(time.time() - t_start, 1)
    own = sum(row["seconds_flow"] + row["seconds_decode"] + row["seconds_eval"]
              for row in rows)
    # The trailing + 0.0 turns the -0.0 that float subtraction yields when the parts cancel
    # exactly into a plain 0.0, so the CSV reads "0.0" and not "-0.0".
    seconds_other = round(seconds_repeat - base_row["seconds_shared"] - own, 1) + 0.0
    for row in rows:
        row["seconds_other"] = seconds_other
        row["seconds_repeat"] = seconds_repeat
    return rows


def main() -> None:
    args = parse_args()
    warn_nonprojected_solver(resolve_solver(args), args.n_obs, args.n_landmarks)
    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = csv_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   output: {out_path}", flush=True)

    # One scorer per implementation for the whole process: loading Inception is a fixed
    # cost that belongs to no repeat, and the scorers are stateless across calls.
    fid_scorers = {}
    if args.fid in ("legacy", "both"):
        fid_scorers["legacy"] = LegacyFIDScorer(device=device, batch_size=64,
                                                distance_device=device)
    if args.fid in ("clean", "both"):
        fid_scorers["clean"] = CleanFIDScorer(device=device, batch_size=64,
                                              num_workers=args.fid_workers,
                                              allow_download=False)

    npz = Path(args.dsprites_npz)
    # The cache name encodes the factor levels, because both bank loaders return an
    # existing cache verbatim without checking them against their arguments.
    cache = npz.parent / ("confounded_bank_3x6x1.npy" if args.design == "confounded"
                          else "heart_sprite_bank_32x32.npy")
    if not npz.exists() and not cache.exists():
        sys.exit(f"dSprites data not found: neither {npz} nor {cache} exists — "
                 "copy the repo's data/dsprites folder (no download on compute nodes).")
    # The per-DATA-SEED setup clock. It starts HERE so the once-per-process sprite-bank
    # load is charged to the first data seed's seconds_data rather than to nothing at all;
    # it is restarted after each seed's repeats. See the module docstring.
    t_process = time.time()
    t_setup = t_process
    if args.design == "confounded":
        # (3, 6, 1, 32, 32, 64, 64) uint8, ~72 MB: the context sets shape and scale, so
        # the bank spans both, and the orientation is fixed at one level.
        sprite_bank = load_multifactor_bank(str(npz), cache_path=str(cache),
                                           shapes=(0, 1, 2),
                                           scale_ids=tuple(range(6)),
                                           orientation_ids=(0,))
        # There is no single-attribute (1024, 4096) position bank for this design, so the
        # snap, the probe, the clean-sprite floor and the position law have no object to
        # work on. `run_repeat` keys every one of those blocks on `bank_flat is not None`.
        bank_flat = None
        bank_positions = None
    else:
        sprite_bank = load_dsprite_bank(str(npz), cache_path=str(cache))
        bank_flat = torch.as_tensor(sprite_bank,
                                    dtype=torch.float32).reshape(1024, -1).to(device)
        bank_positions = torch.stack(torch.meshgrid(torch.arange(32), torch.arange(32),
                                                    indexing="ij"), dim=-1).reshape(1024, 2)

    rows = []
    accounted = 0.0
    # The progress line's floor: the clean-sprite row exists only under --design cpme, and
    # at --pixel_noise 0 the noisy row IS the clean oracle draw anyway.
    floor_key, floor_label = (("energy_floor_clean_mean", "clean") if has_bank_design(args)
                              else ("energy_floor_noisy_mean", "oracle"))
    for data_seed in args.data_seeds:
        data = sample_design(args, sprite_bank, args.n_obs, data_seed)
        # This seed's whole per-data-seed tier: data generation and nothing else here, so
        # seconds_setup equals seconds_data. Both sit outside every seconds_repeat.
        seconds_data = round(time.time() - t_setup, 1)
        setup = {"seconds_data": seconds_data, "seconds_setup": seconds_data}
        accounted += seconds_data
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(sprite_bank, bank_flat, bank_positions, data, setup,
                                  args, data_seed, repeat, device, fid_scorers)
            accounted += new_rows[-1]["seconds_repeat"]
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            for row in new_rows:
                print(f"[data_seed {data_seed} repeat {repeat} {row['estimator']}] "
                      f"pixel energy flow {row['energy_flow_mean']:.4f} "
                      f"(logged {row['energy_logged_mean']:.4f}, "
                      f"{floor_label} floor {row[floor_key]:.4f})   "
                      + (f"FID flow {row['fid_legacy_flow']:.1f} "
                         f"(initial {row['fid_legacy_initial']:.1f}, "
                         f"floor {row['fid_legacy_floor']:.1f})   "
                         if row.get('fid_legacy_flow') is not None else "")
                      + f"latent SW2 flow {row['latent_sw2_flow_mean']:.4f} "
                      f"(initial {row['latent_sw2_initial_mean']:.4f}, "
                      f"matched floor {row['latent_sw2_floor_matched_mean']:.4f})   "
                      f"pos mean flow ({row['pos_mean_flow_x']:.1f}, "
                      f"{row['pos_mean_flow_y']:.1f}) "
                      f"oracle ({row['pos_mean_oracle_x']:.1f}, "
                      f"{row['pos_mean_oracle_y']:.1f})   "
                      f"ESS {row['est_w_ess']:.0f}/{row['n_causal']} "
                      f"v_ESS {row['v_ess']:.0f}   "
                      # seconds_repeat is the whole repeat and so repeats verbatim on
                      # every estimator line; the split says which part is this row's.
                      f"{row['seconds_repeat']:.0f}s repeat "
                      f"= {row['seconds_shared']:.0f}s shared "
                      f"+ {row['seconds_flow']:.0f}s this flow + ...   "
                      f"[setup {row['seconds_setup']:.1f}s this seed]", flush=True)
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
