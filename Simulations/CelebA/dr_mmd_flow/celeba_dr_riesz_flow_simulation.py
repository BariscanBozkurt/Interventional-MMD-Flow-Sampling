"""
CelebA simulation: cross-fitted DR-CPME target + Riesz MMD flow (AdamW, warm-up + cosine).

Script version of
`NotebookExamples/Interventional_MMD_Flow/CelebA_DR_Riesz_MMD_Flow_AdamW_Female.ipynb`
(and its Male twin): the DecFM CelebA confounding design (binary A = Male, binary
X = Blond_Hair, cells sampled to P(X=1) = 0.3, P(A=1|X=0) = 0.6, P(A=1|X=1) = 0.1),
frozen pretrained SD-VAE latents, cross-fitted DR weights over the exact binary
kernels, and a dense Riesz flow under AdamW with linear warm-up and cosine decay on
standardized latents. No autoencoder split is needed: the encoder-decoder is an
external pretrained model that never sees this data at training time.

Each repeat runs BOTH intervention arms by default, sharing every arm-independent
stage (the cached data draw, the SD-VAE encodings, the cross-fitted propensity and
the per-fold ridge lambdas); only the importance weights, DR weight vector, particle
pick, flow and evaluation are arm-specific.

Estimators (--estimators, default dr): the target embedding is a weight vector over the
logged outcomes, chi_hat(pi) = sum_i v_i k_Y(y_i, .), and the three estimators of it are
algebraically nested -- v_DR = v_PI + v_correction, with v_PI the conditional-mean-
embedding term and v_IPW = w / n the correction's leading part. They therefore all come
out of ONE nuisance fit, and each requested one gets its own CSV row (the `estimator`
column) sharing the logged sample, that fit, the initial particles, the flow budget and
every reference, floor and evaluation subset with the others: the only difference between
the rows of one arm is the weight vector handed to the flow. One CSV row per (data_seed,
repeat, arm, estimator). The `sum_v*` and `v_*` columns describe that row's own weight
vector; `ess_on_arm` / `n_on_arm` / `max_weight` describe the arm's importance weights and
are shared by its rows. The IPW row is the Horvitz-Thompson form v = w / n by default --
the term that appears inside v_DR, and the reason the three are algebraically comparable,
but one with no unit mass; --ipw_form hajek switches it to the self-normalised w / sum(w),
a different estimator, and the form used is recorded in the ipw_form CSV column.

THE ESTIMATOR AXIS IS DEGENERATE ON THIS DESIGN: v_DR is v_IPW, not approximately but to
machine precision, for EVERY lambda. The kernels are saturated indicators -- K_X is
1{x_i = x_j} on the binary confounder and K_A is 1{a_i = a_j} on the binary action, so
K = K_X * K_A is 1{cell_i = cell_j} -- and the fitted logistic propensity on a binary X
reproduces the (A, X) cell frequencies. Then (K_pi 1)_i = 1{a_i = a} n_{x_i} while
(K w)_i = 1{a_i = a} n_{cell_i} / e(a | x_i) = 1{a_i = a} n_{x_i} as well, so K_pi 1 = K w
IDENTICALLY, the one-step correction is exactly the gap between the plug-in term and
w / n, and their sum is w / n. Measured on the n_obs = 2048 seed-0 cache with the exact
solver and --dr_fit nosplit: the fitted propensity matches the cell frequencies to
2.1e-10, ||K_pi 1 - K w||_inf is 1.8e-07 (arm 0) and 1.3e-06 (arm 1) against entries of
order n, and at lambda in {1e-6, 1e-3, 1e-2, 1, 30} both v_DR and v_IPW have sum_v
1.000000 with max|v_DR - v_IPW| never above 1.3e-10 of max|v_IPW|. So this design offers
TWO distinct estimators, not three, and a three-row CelebA table must NOT be read as "DR
buys nothing over IPW" -- the design admits no DR-vs-IPW contrast at all. The plug-in row
differs from the other two ONLY through the ridge: on that same run sum v_PI is 0.99998
at lambda = 1e-2, 0.9982 at lambda = 1 and 0.9497 at lambda = 30 (arm 0; arm 1 falls to
0.8785 at 30), rising to 1 as lambda -> 0, so "plug-in loses X per cent of the mass" here
reports the cross-validated lambda and not the estimator. A full three-estimator run of
this script bears both out: at n_obs = 2048, arm 0, --solver exact, --dr_fit nosplit and
the cross-validated lambda = 10, the IPW and DR rows came out identical on 109 of the 111
CSV columns -- the two that differ are `estimator` and a timing -- while the plug-in row
reported sum_v 0.9827, which is to four figures the saturated closed form
sum_x (n_x / n) n_cell(x, a) / (n_cell(x, a) + lambda). Relatedly, the (A, X) kernel has
exact rank 4, so the CME is all but determined and the plug-in row is expected to be
strong here -- a fact about a binary confounder with a binary action, not about plug-in
estimation in general. --dr_fit crossfit breaks the identity only weakly, the out-of-fold
propensity being still nearly cell-constant. The designs that do carry a DR-vs-IPW
contrast are the ones with a non-saturated kernel: ColorMNIST, DecFM_2D, dSprites and
Warfarin. Their stored no-split sum_v confirms it -- 0.9922 to 1.0054 (ColorMNIST),
0.9823 to 1.0115 (DecFM_2D), 0.9920 to 0.9989 (dSprites) -- against 0.9999999 to
1.0000001 here, the sum w / n a cell-frequency propensity forces.

Metrics per row: latent SW2 of the flowed particles against the encoded references
(with the reference split-half floor), and pixel-space SW2 / SW1 / RBF MMD^2 against
the arm's held-out interventional references for the observed arm, its VAE
round-trip, and the flow output — at the native resolution and at 64 x 64 (the
resolution of Dance et al., 2026, Figure 4).

FID (--fid, default both): each decoded row (observed, round-trip, flow) is scored
against the arm's full held-out reference set with the two scorers of
`metric_utils.fid` -- `legacy` reproduces the FID column of the Dance et al. (2026)
tables, `clean` is the standard Clean-FID -- at native 256 px (fid_*) and, with
--fid_64 1, after area-downsampling to 64 px (fid64_*). A reference split-half floor
is reported alongside.

Evaluation protocol (2026-09-15, matched to Dance et al., who also evaluate at
256 px): DR support = every logged row, Nystrom solver at every n, --n_particles
2000 per arm scored on --eval_particles 2000 against the fixed 2000-image held-out
reference set of the arm (--eval_max 2000 observed images for the base row), 3 data
seeds, no repeats. Every row is a 2000-vs-2000 number at every n_obs >= 5000. FID is biased upward for smaller samples,
so compare rows at matched counts: the flow row has n_particles images and the
others n_eval; the DecFM protocol scores 2000 generated against 2000 references.
The split-half floor is a 1000-vs-1000 number and therefore NOT comparable to a
2000-count row (the observed arm scores BELOW that floor at 2000 vs 2000); the
`*_half` columns score the first n_ref/2 images of each row against the second
reference half, which is count-matched to the floor and costs no extra Inception
passes (the row's feature matrix is simply sliced).

For the cheap distributional metrics (SW1/SW2, MMD^2, latent SW2) the same
mismatch is cured from the other side by the `*_floor_matched` columns: they
recompute each floor with BOTH reference halves randomly cut to the count at
which THIS row's flow sample was scored (`n_floor_matched`, and
`n_latent_floor_matched` in the latent space). Those metrics are strongly
sample-size biased -- on this data the SW2 between two independent draws of the
SAME law falls from 0.0266 at n = 250 to 0.0070 at n = 4000 -- and
`sliced_wasserstein` scores at min(len(x), len(y)), so without the twin a sweep
over --n_particles would move the flow rows for a purely statistical reason
while the floor stood still. FID gets no matched twin (an extra Inception pass
per row is too expensive); its `*_half` columns play that role instead.

Wall-clock accounting (2026-10-03): every `seconds_*` column is wall clock via
time.time(), rounded to 0.1 s, and each belongs to exactly ONE of three scopes. The
split matters in both directions: a repeat writes one row PER ESTIMATOR per arm, so a
column recording work SHARED by those rows is multiplied if it is summed down them, while
a cost paid once per DATA SEED is lost entirely by a sum of repeats -- and on this design
that lost term is the largest single phase of a run.

  PER DATA SEED -- identical on every row of every repeat of that data seed, and
  measured OUTSIDE seconds_repeat because it happens once before the repeat loop:
    seconds_data     preparing this seed's data: building the .pt cache from the raw
                     CelebA root when it is cold (minutes to tens of minutes at 256 px,
                     and a 16 GB file at n_obs = 20000) and loading it when it is warm.
                     The SD-VAE and the two Inception models are loaded once per PROCESS
                     and are charged to the FIRST data seed's seconds_data, so the
                     per-task identity below closes without a column a reader would have
                     to know to count once across seeds. Production tasks run one data
                     seed each, where the distinction does not arise.
    seconds_encode   the frozen SD-VAE forward pass over the logged sample and both
                     reference sets. NOT training: the VAE is an external pretrained
                     model that never sees this data, and the encodings are
                     pipeline-independent, which is exactly why they are hoisted out
                     of the repeats. Measured at 75.5 s on a production run -- the
                     largest single phase, which is why losing it was worth fixing
    seconds_fid_ref  the Inception reference features, once per (data seed, arm). This
                     is ALL of the shared FID reference work: no other column holds any
    seconds_setup    the per-data-seed TOTAL -- seconds_data + seconds_encode +
                     seconds_fid_ref -- and the ONE column to add once per data seed.
                     The three above are its breakdown, so never add them to it

  SHARED PER REPEAT -- identical on every row of the repeat (both arms, every
  estimator); NEVER sum these down the rows:
    seconds_ratio       the density-ratio half of the nuisance fit: the DR support
                        selection, the cross-fitting folds, the logistic propensity fit
                        and each arm's importance ratios w = 1{a_i = a} / e(a | x_i),
                        summed over arms. This is the whole of what the IPW estimator
                        needs. The support and the folds are clocked here because this
                        half consumes them first; the CME half reuses both
    seconds_cme         the conditional-mean-embedding half: the (A, X) cell Gram, the
                        per-fold ridge selection and each arm's single DR/CME weight
                        solve, summed over arms. This is the whole of what the plug-in
                        estimator needs; DR needs both halves. One solve serves every
                        estimator of the arm and returns all of its vectors, so the
                        sub-millisecond IPW rescaling v = w / n is clocked here too
    seconds_nuisance    seconds_ratio + seconds_cme, under the name it has always had.
                        It is their SUM: never add the three together, and never add the
                        two halves to seconds_shared. (Before 2026-10-03 this column held
                        only the propensity and ridge block and the CME solve was untimed,
                        so it is the one column whose CONTENT grew when the name did not.)
    seconds_eval_shared the arm-level baselines and floors: the observed and
                        round-trip decode, their pixel scores, every floor and every
                        count-matched floor, summed over arms
    seconds_fid_shared  the arm-level FID scoring (observed, round-trip and the
                        split-half floor), summed over arms
    seconds_shared      the SUM of seconds_nuisance, seconds_eval_shared and
                        seconds_fid_shared -- the one column to add for "what this repeat
                        spent on work its rows share"

  PER ESTIMATOR ROW -- this row's own work; safe to sum over rows:
    seconds_flow     this estimator's Riesz/AdamW iterations
    seconds_decode   SD-VAE decode of this row's flow particles
    seconds_eval     this row's OWN scoring, EXCLUDING anything shared: the latent SW2,
                     the pixel SW2/SW1/MMD^2 of the decoded row, and its FID. INCLUDES
                     seconds_fid
    seconds_fid      this row's OWN Inception passes and Frechet distances -- a BREAKDOWN
                     of seconds_eval, already inside it, so adding the two together double
                     counts. This is ColorMNIST's convention, adopted here on 2026-10-03
                     so that the own side of the identity is the same three columns in
                     every design that has them; before that date this column sat BESIDE
                     seconds_eval and an archived CSV's own sum needs it added back
    seconds_sampling DERIVED, and a term of NEITHER identity below: the isolated cost of
                     the proposed sampling method for this row, composed for its estimator
                     out of the columns above --
                         plugin -> seconds_cme + seconds_flow + seconds_decode
                         ipw    -> seconds_ratio + seconds_flow + seconds_decode
                         dr     -> seconds_ratio + seconds_cme + seconds_flow
                                   + seconds_decode
                     It EXCLUDES the frozen encoder (seconds_encode, in seconds_setup) and
                     all evaluation, because those are representation and measurement
                     rather than sampling, so an END-TO-END figure must add seconds_setup
                     back. (There is no seconds_ae here: this design trains no
                     autoencoder.) One caveat, a consequence of the shared columns it is
                     built from: the nuisance halves are repeat totals over BOTH arms, so
                     run one arm (--target_arms a) for one arm's figure. It reuses shared
                     columns, so never sum it down the rows of a repeat

  TOTAL:
    seconds_other    the unattributed remainder, defined so the per-repeat identity below
                     closes EXACTLY rather than approximately. It carries the small
                     per-arm bookkeeping not worth a column of its own -- the
                     starting-particle draw, the weight diagnostics, binary_stats and the
                     row assembly. If it is ever large, a phase has lost its clock. It can
                     also come out NEGATIVE: every phase is rounded to 0.1 s before the
                     residual is taken, so a repeat whose phases happen to round up
                     overshoots its own total by up to 0.05 s per rounded quantity -- 4 per
                     estimator row, the shared ones and seconds_repeat itself, about 1.4 s
                     for a 2-arm 3-estimator repeat. That ceiling is negligible against a
                     production repeat, whose flow and Inception passes each run for tens
                     of seconds, but it is a visible fraction of a toy one: --fid none at
                     n_obs 512 and 30 iterations measured -0.4 s out of 3.8 s. It is the
                     0.1 s grid, not missing time
    seconds_repeat   total wall clock of the WHOLE repeat, identical on every row of
                     that repeat. NOT cumulative down the rows

THE TWO IDENTITIES, both exact on the rounded values as written to the CSV:

    seconds_repeat == seconds_shared + seconds_other
                      + sum over the repeat's rows of
                        (seconds_flow + seconds_decode + seconds_eval)    [per repeat]

    task total     == sum over data seeds of seconds_setup
                      + sum over repeats of seconds_repeat                [per task]

reading seconds_setup once per data seed and seconds_repeat once per repeat (one row per
repeat), never once per row. What the task identity does not cover is named rather than
hidden: process start-up (the imports and the CUDA context) and the per-repeat CSV
rewrite. The run prints both the accounted total and the process wall clock on its last
line, so the gap is visible rather than inferred. With --fid none the FID columns are 0.0
rather than absent: the CSV schema is fixed, and whether FID runs is a flag, not a
property of the design.

Hardware provenance: every row also carries gpu_name, hostname, slurm_job_id,
slurm_array_task_id, slurm_nodelist, torch_version and cuda_version, so a published timing
can be tied to the pinned node that produced it. The SLURM fields are read from the
environment and are empty off the cluster.

Data caches: one .pt per (image_size, n_obs, n_ref, data_seed), generated on first
use from the raw CelebA root (readable NFS path) and written atomically so that
concurrent array tasks sharing a configuration cannot corrupt it. Generation
materialises every image; at 256 px keep n_obs modest (memory grows as
n_obs x 3 x S^2 floats — n_obs = 20000 at 256 px does NOT fit a 12 GB node; use
--image_size 64 for the DecFM-scale runs, whose defaults ARE the paper's settings).

Seed scheme: `data_seed` fixes the data draw (observed sample and both reference
sets); `pipeline_seed = 1000 * data_seed + repeat` fixes the propensity fit, the DR
subsample (active only when n_dr < n_obs), the particle pick and the flow. At the
default arguments the arm-0 row of (data_seed 0, repeat 0) follows the Female
notebook's seed path exactly.

Launched by `submit_celeba_dr_riesz_flow_parallel.sh`; single-run example:

    python celeba_dr_riesz_flow_simulation.py --n_obs 2048 --data_seeds 0 --n_repeats 3
"""

import argparse
import csv
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "src"))

from data_utils import binary_stats, load_celeba_paper
from autoencoder_utils import decode_latents, encode_images, load_sd_vae
from causality_utils import (cpme_dr_weights, cpme_dr_weights_crossfit,
                             cpme_dr_weights_crossfit_nystrom,
                             cpme_dr_weights_crossfit_projected_nystrom,
                             cpme_dr_weights_projected_nystrom, cpme_dr_weights_nystrom,
                             cpme_ipw_weights, lambda_n_half, make_crossfit_fold_ids,
                             select_regularization)
from dens_ratio_utils import LogisticPropensityModel, crossfit_propensity
from mmd_flow_utils import adamw_riesz_flow, squared_mmd
from metric_utils import CleanFIDScorer, LegacyFIDScorer, sliced_wasserstein
from kernel_utils import RBF

PIXEL_ROWS = ("observed", "roundtrip", "flow")

FID_IMPLS = ("legacy", "clean")

# Target-embedding estimators, in the order rows are written (see --estimators).
ESTIMATORS = ("plugin", "ipw", "dr")

COLUMNS = (
    ["tag", "device", "init", "data_seed", "repeat", "pipeline_seed", "target_a",
     "estimator",
     "image_size", "n_obs", "n_ref", "n_dr", "n_landmarks", "solver", "dr_fit",
     "lambda_rule", "ipw_form", "n_folds", "latent_dim",
     "n_arm", "n_particles_requested", "n_particles", "n_flow_eval", "n_eval",
     "flow_iterations", "learning_rate", "warmup_steps", "lr_schedule",
     "n_a0x0", "n_a0x1", "n_a1x0", "n_a1x1", "obs_blond_frac_arm",
     "ess_on_arm", "n_on_arm", "max_weight",
     "sum_v", "sum_abs_v", "sum_vX", "v_ess", "v_ess_on_arm", "v_max_abs",
     "fold_lambdas",
     "loss_initial", "loss_final", "mean_transport",
     "latent_sw2_initial", "latent_sw2_flow", "latent_sw2_floor",
     "latent_sw2_floor_matched", "n_latent_floor_matched"]
    + [f"{metric}_{row}" for row in PIXEL_ROWS for metric in ("sw2", "sw1", "mmd2")]
    + ["sw2_floor", "sw1_floor", "mmd2_floor"]
    + ["sw2_floor_matched", "sw1_floor_matched", "mmd2_floor_matched"]
    + [f"{metric}_64_{row}" for row in PIXEL_ROWS for metric in ("sw2", "sw1")]
    + ["sw2_64_floor", "sw1_64_floor"]
    + ["sw2_64_floor_matched", "sw1_64_floor_matched", "n_floor_matched"]
    + [f"fid_{impl}_{row}" for impl in FID_IMPLS for row in tuple(PIXEL_ROWS) + ("floor",)]
    + [f"fid64_{impl}_{row}" for impl in FID_IMPLS for row in tuple(PIXEL_ROWS) + ("floor",)]
    + [f"fid_{impl}_{row}_half" for impl in FID_IMPLS for row in PIXEL_ROWS]
    + [f"fid64_{impl}_{row}_half" for impl in FID_IMPLS for row in PIXEL_ROWS]
    # Timing columns, grouped by scope -- see "Wall-clock accounting" in the module
    # docstring. The per-estimator group may be summed down the rows of a repeat; the
    # shared and per-data-seed groups may NOT, being identical on every row. seconds_fid is
    # a BREAKDOWN of seconds_eval, seconds_ratio / seconds_cme of seconds_nuisance, and
    # seconds_data / seconds_encode / seconds_fid_ref of seconds_setup, so no breakdown may
    # be added alongside the column it decomposes; seconds_sampling is derived from several
    # of the others and belongs to no sum at all.
    + ["seconds_data", "seconds_encode", "seconds_fid_ref", "seconds_setup"]     # per seed
    + ["seconds_ratio", "seconds_cme", "seconds_nuisance", "seconds_eval_shared",
       "seconds_fid_shared", "seconds_shared"]                                  # per repeat
    + ["seconds_flow", "seconds_decode", "seconds_eval", "seconds_fid",
       "seconds_sampling"]                                                      # own
    + ["seconds_other", "seconds_repeat"]
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


def hardware_provenance(device: "torch.device") -> dict:
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
    the SD-VAE decode that turns the particles back into images. Representation and
    measurement are excluded by construction: the frozen encoder (`seconds_encode`, inside
    `seconds_setup`) and every evaluation column stay out, so an end-to-end figure must add
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


def phase_seconds(t0: float, device: "torch.device") -> float:
    """Wall-clock seconds since `t0`, with the CUDA queue drained first.

    CUDA kernel launches are asynchronous, so a bare time.time() at a phase boundary
    charges the NEXT phase for whatever the previous one left in flight. MEASURED on this
    design (RTX 4500 Ada, 2026-10-03), nothing meaningful ever is: every phase here ends
    in a blocking device-to-host read -- `encode_images` and `decode_latents` call .cpu()
    per chunk, the metric helpers end in .item(), both FID scorers move each feature batch
    to the host, and `adamw_riesz_flow` records its last step with loss.item() -- so
    draining afterwards takes at most 1e-4 s, on a 40.6 s flow, a 5.6 s decode, a 3.5 s
    Inception pass and a 3.6 s Frechet distance alike. That is three orders of magnitude
    below the 0.1 s these columns are rounded to, so the bare clocks were already right.

    The synchronisation is therefore insurance rather than a correction: it makes the
    split exact by construction instead of by accident, so a later refactor that leaves a
    phase returning a device tensor cannot quietly move seconds onto the next clock. It
    costs nothing it did not already cost and changes no number the run produces -- a
    synchronisation is a barrier, not an operation.

    Parameters
    ----------
    t0 : float
        A time.time() reading taken at the start of the phase.
    device : torch.device
        The device the phase ran on; only a CUDA device needs draining.

    Returns
    -------
    float
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time() - t0


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


def subsample_rows(rows: "torch.Tensor", n_keep: int, seed: int) -> "torch.Tensor":
    """Random `n_keep`-row subset of `rows`; the whole tensor when it is no larger.

    Parameters
    ----------
    rows : torch.Tensor of shape (n, d)
    n_keep : int
        Target count.
    seed : int
        Seed of a PRIVATE CPU generator, so the draw is reproducible and the global
        RNG stream -- and with it every pre-existing column -- is left untouched.

    Returns
    -------
    torch.Tensor of shape (min(n, n_keep), d)
    """
    if len(rows) <= n_keep:
        return rows
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))[:n_keep]
    return rows[pick.to(rows.device)]


def matched_halves(reference: "torch.Tensor", half: int, n_keep: int, seed: int) -> tuple:
    """The two reference halves, each cut to `n_keep` rows, for a count-matched floor.

    Parameters
    ----------
    reference : torch.Tensor of shape (n_ref, d)
    half : int
        Split point, n_ref // 2.
    n_keep : int
        Count the floor is matched to.
    seed : int
        Base seed; the two halves are drawn with `seed` and `seed + 1`.

    Returns
    -------
    tuple of two torch.Tensor
    """
    return (subsample_rows(reference[:half], n_keep, seed),
            subsample_rows(reference[half:], n_keep, seed + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n_obs", type=int, default=2048,
                        help="observed sample size")
    parser.add_argument("--n_ref", type=int, default=2000,
                        help="held-out interventional references per arm")
    parser.add_argument("--image_size", type=int, default=256,
                        help="image side length (64 = the DecFM paper's resolution)")
    parser.add_argument("--n_particles", type=int, default=2000,
                        help="flow particles M (capped at the observed arm size)")
    parser.add_argument("--n_dr", type=int, default=None,
                        help="optional cap on the DR support, kept for reproducing the "
                             "pre-2026-09 runs (--solver exact --n_dr 4096). Unset, the "
                             "support is every logged row.")
    parser.add_argument("--target_arms", type=int, nargs="+", default=[0, 1],
                        choices=(0, 1), help="intervention arms run in each repeat")
    parser.add_argument("--estimators", type=str, nargs="+", default=["dr"],
                        choices=ESTIMATORS,
                        help="which estimators of the target embedding to score; ONE CSV "
                             "ROW EACH, tagged by the `estimator` column. 'plugin' is the "
                             "conditional-mean-embedding term v_PI alone, 'ipw' the "
                             "importance weights w / n alone, 'dr' the plug-in term plus "
                             "the importance-weighted CME residual, so all three come out "
                             "of ONE nuisance fit. Every "
                             "requested estimator shares the logged sample, that fit, the "
                             "initial particles, the flow budget and every reference, "
                             "floor and evaluation subset -- only the weight vector handed "
                             "to the flow differs, which is what makes the comparison "
                             "paired rather than seed noise. Order and duplicates are "
                             "ignored (rows always follow plugin, ipw, dr). Default "
                             "['dr'], which leaves every historical run bitwise unchanged; "
                             "any other request appends _est<names> to the filename.")
    parser.add_argument("--eval_particles", type=int, default=2000,
                        help="score the pixel rows (SW, MMD, FID) of the flow on a fixed random "
                             "subset of this many particles. Lets the flow run with particles "
                             "matched to the arm (which removes the quantiser over-transport at "
                             "n=20000) while the evaluation keeps the DecFM count of 2000. The "
                             "latent metrics still use every particle. Default: all particles.")
    parser.add_argument("--eval_max", type=int, default=2000,
                        help="cap on observed/round-trip evaluation rows (None = the full "
                             "arm, the notebook protocol); set on large-n runs to bound "
                             "decode time and GPU memory, e.g. 2048 at n_obs = 16384")
    parser.add_argument("--fid", type=str, default="both",
                        choices=("none", "legacy", "clean", "both"),
                        help="which FID implementations to compute per decoded row (see "
                             "module docstring). 'clean' needs the Clean-FID weights in "
                             "~/.cache/torch/clean-fid (no download on compute nodes)")
    parser.add_argument("--fid_workers", type=int, default=8,
                        help="threads for Clean-FID's per-image resize (results identical)")
    parser.add_argument("--fid_64", type=int, default=0,
                        help="1 also scores FID after area-downsampling to 64 px (fid64_*); "
                             "off by default since the protocol evaluates at native 256 px")
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
                        choices=("observed", "normal", "moment", "normal_raw"),
                        help="initial particles: 'observed' = the logged arm (default), "
                             "'normal' = N(0, I), 'moment' = the empirical N(mu, Sigma) "
                             "of the flow's target support. Both Gaussians are drawn in "
                             "the coordinates the flow operates in. 'normal_raw' = N(0, I) "
                             "in the RAW encoder space, pushed through the same "
                             "standardisation; because this script standardises its latents, "
                             "'normal' is a DIAGONAL MOMENT MATCH rather than a standard "
                             "normal, and 'normal_raw' is the arm comparable with ColorMNIST "
                             "and DecFM_2D (which do not standardise). NOTE: the *_initial "
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
                             "The DR support is EVERY logged row under all of them, "
                             "so what the choice changes is the nuisance estimate, not "
                             "the sample; the threshold is keyed on n_obs, so runs "
                             "compared at the same n_obs always share a solver. NOTE: "
                             "this design's Gram is exactly rank 4 (X and A binary: "
                             "K[i,j] = 1{cell_i == cell_j} over four (A,X) cells), so any "
                             "landmark set touching all four cells spans the row space "
                             "and Nystrom is EXACT here -- verified to 0.00e+00 against "
                             "the exact solver -- which is the one design where the "
                             "generic warning overstates the risk.")
    parser.add_argument("--exact_max_n_obs", type=int, default=10000,
                        help="largest n_obs for which 'auto' solves exactly"
                             "; above it 'auto' refuses instead of choosing a solver "
                             "for you")
    parser.add_argument("--n_landmarks", type=int, default=1024,
                        help="Nystrom landmarks per fold (rank 4 needs far fewer; 1024 is "
                             "a safe default that still costs little)")
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
                             "fold; 'nosplit' fits both on ALL rows. The same-sample notes "
                             "license the latter for a FIXED kernel and feature map under "
                             "bounded nuisance norms and prediction consistency -- which is "
                             "the case here, since the SD-VAE is externally pretrained and "
                             "frozen and the (A, X) kernel is discrete, not data-selected.")
    parser.add_argument("--lambda_rule", type=str, default="cv",
                        choices=("cv", "n_half"),
                        help="ridge selection. 'cv' (default, historical) cross-validates per "
                             "training block; 'n_half' uses lambda = c * n^-1/2, the rate the "
                             "same-sample theory gives as sufficient, which also keeps the "
                             "penalty matched between --dr_fit arms (cross-validated lambdas "
                             "are chosen per block and differ systematically between a "
                             "full-sample and a k-fold fit, confounding that comparison).")
    parser.add_argument("--lambda_constant", type=float, default=1.0,
                        help="the constant c in --lambda_rule n_half")
    parser.add_argument("--snapshot_every", type=int, default=None,
                        help="store the particle cloud every this many iterations and score "
                             "the decoded images at each. Lets ONE run yield a whole "
                             "iteration curve instead of one run per step count. NOTE this is "
                             "equivalent to shorter runs only under a constant learning rate: "
                             "with the default cosine schedule the rate depends on the total "
                             "iteration count, so a snapshot at step s is NOT the same as a "
                             "run of s steps. Pair it with --lr_schedule const to sweep steps.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--flow_iterations", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=("const", "cosine"))
    parser.add_argument("--data_seeds", type=int, nargs="+", default=[0],
                        help="data-draw seeds run by this task")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="pipeline repeats per data seed")
    parser.add_argument("--tag", type=str, default="riesz")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--cache_dir", type=str,
                        default=str(REPO_ROOT / "data" / "celeba_dgp"))
    parser.add_argument("--celeba_root", type=str,
                        default="/nfs/ghome/live/danceh/data/celeba",
                        help="raw CelebA root; needed only when a cache must be generated")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu; default picks cuda when available")
    return parser.parse_args()



PARTICLE_INDICES: dict = {}


def particle_generator(args, seed: int, offset: int = 0) -> "torch.Generator":
    """CPU generator for the starting-particle draw; see --particle_seed."""
    base = seed if args.particle_seed is None else args.particle_seed + seed
    return torch.Generator().manual_seed(base + offset)


def write_particle_indices(out_path: Path) -> None:
    """Save the logged-sample rows used as starting particles, for cross-implementation reuse."""
    if PARTICLE_INDICES:
        np.savez(out_path.with_name(out_path.stem + "_particles.npz"), **PARTICLE_INDICES)


def requested_estimators(args: argparse.Namespace) -> list:
    """The requested estimators, deduplicated and in the canonical order of `ESTIMATORS`.

    Fixing the order makes `--estimators dr plugin` and `--estimators plugin dr` the same
    run: same filename, same rows, same order. See --estimators.
    """
    chosen = set(args.estimators)
    return [name for name in ESTIMATORS if name in chosen]


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


    NOTE for this design: X and A are binary, so K[i, j] = 1{cell_i == cell_j} over
    four (A, X) cells and any landmark set touching all four spans the row space. The
    truncation term is then zero and this solver coincides with the exact one (see
    --solver), so the warning is generic and none of the degradation above applies at
    the default budget.

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
          f"archived non-projected table.\n"
          f"  This design is the exception: its Gram has rank 4, so any landmark set "
          f"touching all four (A, X) cells spans the row space and the truncation term "
          f"is zero -- see --solver.", file=sys.stderr, flush=True)


def csv_path(args: argparse.Namespace) -> Path:
    """One CSV per task, its name carrying the swept constants and the seed block."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    suffix = "" if args.init == "observed" else f"_init{args.init}"
    # Tokens appended only off-default, so historical filenames are reproduced exactly.
    # The iteration token matters: write_rows rewrites the file atomically, so without it
    # an array sweep over step counts would silently keep only the last task to finish.
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
    # The estimator axis multiplies rows rather than replacing them, so a
    # three-estimator run must not land on a dr-only run's filename.
    estimators = requested_estimators(args)
    if estimators != ["dr"]:
        suffix += "_est" + "-".join(estimators)
    if args.ipw_form != "ht":
        suffix += f"_ipw{args.ipw_form}"
    if args.flow_iterations != 4000:
        suffix += f"_it{args.flow_iterations}"
    name = (f"celeba_{args.tag}_sz{args.image_size}_nobs{args.n_obs}"
            f"_M{args.n_particles}_arms{arms}_ds{seeds}{suffix}.csv")
    return Path(args.out_dir) / name


def ensure_cache(args: argparse.Namespace, data_seed: int) -> Path:
    """Return the cache path for this configuration, generating it atomically if absent."""
    cache_dir = Path(args.cache_dir)
    canonical = (cache_dir / f"celeba_paper_{args.image_size}_n{args.n_obs}"
                             f"_nref{args.n_ref}_seed{data_seed}.pt")
    if canonical.exists():
        return canonical
    legacy = cache_dir / f"celeba_paper_{args.image_size}_n{args.n_obs}_seed{data_seed}.pt"
    if args.n_ref == 2000 and legacy.exists():   # older caches omit the (then-default) n_ref
        return legacy
    if not Path(args.celeba_root).exists():
        sys.exit(f"cache {canonical.name} is missing and the raw CelebA root "
                 f"{args.celeba_root} is not readable — generate the cache on a machine "
                 "that can see the root, or fix --celeba_root.")
    # Generate under a task-unique name, then rename: os.replace is atomic on one
    # filesystem, so concurrent array tasks sharing this configuration can never
    # observe a half-written cache (the losing writer just overwrites with identical
    # bytes, since the draw is fully determined by the seed).
    expected_gb = (args.n_obs + 2 * args.n_ref) * 3 * args.image_size ** 2 * 4 / 1e9
    free_gb = shutil.disk_usage(cache_dir).free / 1e9
    if free_gb < 1.3 * expected_gb:   # statvfs cannot see NFS quotas, but it does
        sys.exit(f"cache dir {cache_dir} has {free_gb:.1f} GB free; generating "
                 f"{canonical.name} needs ~{expected_gb:.1f} GB (plus headroom). "
                 "Use a larger --cache_dir / CACHE_DIR.")   # catch small local scratch
    tmp = canonical.with_name(canonical.name + f".tmp{os.getpid()}")
    try:
        load_celeba_paper(tmp, root=args.celeba_root, seed=data_seed,
                          image_size=args.image_size, n_obs=args.n_obs, n_ref=args.n_ref,
                          verbose=False)
        os.replace(tmp, canonical)
    except BaseException as error:
        tmp.unlink(missing_ok=True)   # never leave a partial multi-GB tmp behind
        raise SystemExit(
            f"cache generation failed for {canonical.name} (~{expected_gb:.1f} GB needed "
            f"in {cache_dir}): {error!r}. If this is EDQUOT/ENOSPC, free space or point "
            "--cache_dir (CACHE_DIR in the submit script) at a filesystem with room."
        ) from error
    return canonical


def solve_dr_weights(target_a: int, shared: dict, args: argparse.Namespace, seed: int,
                     w_hat: "torch.Tensor", components: bool):
    """One DR solve for this arm, with the solver `--solver` x `--dr_fit` selects.

    Parameters
    ----------
    target_a : int
        Intervened arm, which enters only through K_A_pi and `w_hat`.
    shared : dict
        The repeat's shared state (`A_s`, `X_s`, `cell`, folds, lambdas, solver).
    seed : int
        Pipeline seed; seeds the private generator behind the Nystrom landmark draw.
    w_hat : torch.Tensor of shape (n_dr,)
        The arm's importance weights.
    components : bool
        Ask the solver for the pair (v_PI, v_correction) instead of their sum.

    Returns
    -------
    torch.Tensor of shape (n_dr,), or the pair (v_PI, v_correction) when `components`.
    """
    A_s, X_s = shared["A_s"], shared["X_s"]
    n_dr = len(A_s)
    if shared["solver"] in ("nystrom", "projected_nystrom"):
        # Gram blocks on demand: K[i,j] = 1{cell_i == cell_j}, K_pi[i,j] = 1{a_i == a}.
        # At n = 20000 the full Gram would be 3.2 GB and each fold a 16000^2 solve.
        cell = shared["cell"]
        X_dr = X_s.to(cell.device)
        on_arm_flag = (A_s == target_a).double().to(cell.device)
        # K = K_X * K_A = 1{cell_i == cell_j};  K_pi = K_X * K_A_pi = 1{x_i == x_j} 1{a_i == a}
        gram = lambda rows, cols: (cell[rows][:, None] == cell[cols][None, :]).double()
        gram_pi = lambda rows, cols: ((X_dr[rows][:, None] == X_dr[cols][None, :]).double()
                                      * on_arm_flag[rows][:, None])
        if args.dr_fit == "nosplit":
            solve = (cpme_dr_weights_projected_nystrom
                     if shared["solver"] == "projected_nystrom" else cpme_dr_weights_nystrom)
            return solve(
                gram, gram_pi, w_hat, shared["nosplit_lambda"],
                n_landmarks=args.n_landmarks,
                generator=torch.Generator().manual_seed(seed), components=components)
        solve = (cpme_dr_weights_crossfit_projected_nystrom
                 if shared["solver"] == "projected_nystrom" else cpme_dr_weights_crossfit_nystrom)
        return solve(
            gram, gram_pi, w_hat, shared["fold_ids"], shared["fold_lambdas"],
            n_landmarks=args.n_landmarks,
            generator=torch.Generator().manual_seed(seed), components=components)
    dev = shared["cell"].device
    K_x = (X_s[:, None] == X_s[None, :]).double().to(dev)
    K_a = (A_s[:, None] == A_s[None, :]).double().to(dev)
    K_a_target = ((A_s == target_a).double().to(dev)[:, None]
                  .expand(n_dr, n_dr).contiguous())
    if args.dr_fit == "nosplit":
        return cpme_dr_weights(K_x, K_a, K_a_target, w_hat, shared["nosplit_lambda"],
                               components=components)
    return cpme_dr_weights_crossfit(K_x, K_a, K_a_target, w_hat, shared["fold_ids"],
                                    shared["fold_lambdas"], components=components)


def arm_weight_vectors(target_a: int, shared: dict, args: argparse.Namespace, seed: int,
                       device: torch.device) -> tuple:
    """This arm's importance weights and one target weight vector per requested estimator.

    Every estimator of the interventional embedding is a weight vector over the logged
    outcomes, chi_hat(pi) = sum_i v_i k_Y(y_i, .), and the three here are algebraically
    nested in one nuisance fit:

        plug-in   v_PI  = (1/n) (K + lambda I)^{-1} K_pi 1
        IPW       v_IPW = w / n
        DR        v_DR  = v_PI + (1/n) [ w - (K + lambda I)^{-1} K w ].

    The DR solvers return the pair (v_PI, v_correction) under `components=True`, so the
    plug-in vector costs no second solve and no comparison can be confounded by a second
    nuisance fit. `components` is requested only when the plug-in row is wanted, which
    leaves every DR-only run -- including `--estimators dr ipw` -- on exactly the
    historical arithmetic. IPW needs no CME at all, so `--estimators ipw` skips the solve.

    Parameters
    ----------
    target_a : int
    shared : dict
    args : argparse.Namespace
    seed : int
    device : torch.device
        The flow's device; the returned vectors are float32 there, as the flow needs.

    Returns
    -------
    w_hat : torch.Tensor of shape (n_dr,)
        The arm's importance weights, in float64, whose diagnostics describe the nuisance
        rather than any one estimator.
    vectors : dict {str: torch.Tensor of shape (n_dr,)}
        Keyed by estimator name in the canonical order.
    seconds : dict
        This arm's two nuisance clocks, `seconds_ratio` (the importance ratios) and
        `seconds_cme` (the solve behind every vector), for the repeat to total over arms.
        Separated because the estimators do not need the same half: see `sampling_seconds`.
    """
    A_s, e1 = shared["A_s"], shared["e1"]
    requested = requested_estimators(args)
    t0 = time.time()
    w_hat = ((A_s == target_a).double()
             / (e1 if target_a == 1 else 1.0 - e1)).to(shared["cell"].device)
    seconds_ratio = phase_seconds(t0, device)

    t0 = time.time()
    vectors = {}
    if "plugin" in requested or "dr" in requested:
        components = "plugin" in requested
        solved = solve_dr_weights(target_a, shared, args, seed, w_hat, components)
        if components:
            vectors["plugin"], correction = solved
            vectors["dr"] = vectors["plugin"] + correction
        else:
            vectors["dr"] = solved
    if "ipw" in requested:
        vectors["ipw"] = cpme_ipw_weights(w_hat,
                                          normalize=args.ipw_form == "hajek")
    requested_vectors = {name: vectors[name].float().to(device) for name in requested}
    seconds_cme = phase_seconds(t0, device)
    return w_hat, requested_vectors, {"seconds_ratio": seconds_ratio,
                                      "seconds_cme": seconds_cme}


def weight_diagnostics(v: "torch.Tensor", on_arm_mask: "torch.Tensor",
                       X_s: "torch.Tensor") -> dict:
    """Mass and concentration of one estimator's target weight vector.

    The three estimators differ sharply here, which is most of the point of running them
    side by side: IPW puts all of its mass on the intervened arm with the raw ratio's
    tails undamped, while the plug-in vector spreads signed mass over every logged row.

    `sum_v`, `sum_abs_v`, `v_ess` and `v_max_abs` are the CANONICAL per-estimator
    diagnostics of this repository: the same functionals, under the same names, in the
    same units, in all five simulations, so an estimator table can be read across
    designs. `v_ess` is Kish's ratio of the vector ITSELF, (sum v)^2 / sum v^2 -- on a
    signed vector (sum |v|)^2 / sum v^2 is a different functional and not a rescaling of
    it -- and `v_max_abs` is the raw largest atom, never multiplied by n. On the
    non-negative IPW weights `v_ess` is an effective sample size in the usual sense; the
    plug-in and DR vectors carry negative entries, where it is a concentration index
    rather than a count -- still the quantity the target term's variance scales with, but
    not a sample size, and it may exceed n.

    Two columns beyond those four are specific to this design and named so: `v_ess_on_arm`
    is the same Kish ratio restricted to the rows logged under the intervened action
    (equal to `v_ess` for IPW, which is zero off the arm, and not for the estimators with
    a CME term), and `sum_vX` is the weights' projection on the confounder, whose target
    value is P(X = 1) = 0.3. `sum_v` is the total mass, 1 for a well-behaved estimator
    since chi(pi) is the embedding of a probability measure -- but see the module
    docstring before reading the IPW and plug-in rows of THIS design, where the axis is
    algebraically degenerate.

    Parameters
    ----------
    v : torch.Tensor of shape (n_dr,)
        The estimator's weight vector.
    on_arm_mask : torch.Tensor of shape (n_dr,)
        Boolean mask of the rows logged under the intervened action.
    X_s : torch.Tensor of shape (n_dr,)
        Binary confounder of the DR support.

    Returns
    -------
    dict
    """
    kish = lambda t: (float(t.sum() ** 2 / (t ** 2).sum())
                      if float((t ** 2).sum()) > 0.0 else 0.0)
    return {"sum_v": float(v.sum()), "sum_abs_v": float(v.abs().sum()),
            "sum_vX": float((v * X_s.float().to(v.device)).sum()),
            "v_ess": kish(v), "v_ess_on_arm": kish(v[on_arm_mask.to(v.device)]),
            "v_max_abs": float(v.abs().max())}


def starting_particles(target_a: int, shared: dict, args: argparse.Namespace,
                       seed: int) -> tuple:
    """The arm's initial particle cloud, drawn once and shared by every estimator.

    Parameters
    ----------
    target_a : int
    shared : dict
    args : argparse.Namespace
    seed : int

    Returns
    -------
    arm_indices : torch.Tensor
        Rows of the logged sample under the intervened action.
    initial_particles : torch.Tensor of shape (M, D)
    """
    A, Z_std, Z_std_s = shared["A"], shared["Z_std"], shared["Z_std_s"]
    arm_indices = torch.nonzero(A == target_a).ravel()
    n_particles = min(args.n_particles, len(arm_indices))
    pick = arm_indices[torch.randperm(len(arm_indices),
                                      generator=particle_generator(args, seed))[:n_particles]]
    PARTICLE_INDICES[f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"] = pick.cpu().numpy()
    if args.init == "observed":
        return arm_indices, Z_std[pick].clone()
    if args.init == "normal_raw":
        # A genuinely uninformative start. The flow runs on STANDARDISED latents, so a
        # torch.randn drawn there (--init normal) automatically carries the pooled latents'
        # mean and unit per-coordinate variance -- measured: per-coordinate std 0.997 and
        # mean norm 0.088/sqrt(D), against the target reference's 0.985 / 0.082. It is a
        # diagonal moment match, not a standard normal. This arm instead draws N(0, I) in
        # the RAW encoder output space and pushes it through the same standardising map,
        # which is also what --init normal already means for ColorMNIST and DecFM_2D (they
        # do not standardise), so it is the arm that lines up across datasets.
        #
        # NOT the VAE prior: encode_images returns latent_dist.mode() WITHOUT the diffusers
        # scaling_factor, so raw latents here have per-coordinate std ~4.4, not ~1. A unit
        # normal in that space is therefore under-dispersed -- in flow coordinates it lands
        # at std 0.225 with a 0.59/sqrt(D) mean offset, i.e. a shrunken, displaced ball.
        # That is still decisively far from the target (latent SW2 at iteration 0 is 0.962
        # vs 0.255 for --init normal, measured at n_obs=2048), which is what the arm is for,
        # but describe it as "unit normal in encoder space", never as the prior.
        raw = torch.randn(n_particles, Z_std_s.shape[1],
                          generator=torch.Generator(device=Z_std.device).manual_seed(seed + 3),
                          device=Z_std.device, dtype=Z_std.dtype)
        return arm_indices, shared["to_flow"](raw).to(Z_std.dtype)
    return arm_indices, gaussian_init(
        Z_std_s, n_particles, args.init,
        torch.Generator(device=Z_std.device).manual_seed(seed + 3))


def pixel_scores(name: str, images: "torch.Tensor", context: dict) -> dict:
    """SW2 / SW1 / RBF MMD^2 of one decoded row against the arm's references.

    Parameters
    ----------
    name : str
        Column suffix, one of `PIXEL_ROWS`.
    images : torch.Tensor of shape (n, 3, S, S)
    context : dict
        From `arm_evaluation`: the flattened references, the locked pixel kernel and the
        64 px downsampler.

    Returns
    -------
    dict
    """
    reference, reference_64 = context["reference"], context["reference_64"]
    kernel_pixel, flatten = context["kernel_pixel"], context["flatten"]
    rows_px = flatten(images)
    scores = {
        f"sw2_{name}": sliced_wasserstein(rows_px, reference, p=2, seed=0).item(),
        f"sw1_{name}": sliced_wasserstein(rows_px, reference, p=1, seed=0).item(),
        f"mmd2_{name}": squared_mmd(kernel_pixel, rows_px, reference, unbiased=True).item(),
    }
    rows_64 = flatten(context["to_64"](images))
    scores[f"sw2_64_{name}"] = sliced_wasserstein(rows_64, reference_64, p=2, seed=0).item()
    scores[f"sw1_64_{name}"] = sliced_wasserstein(rows_64, reference_64, p=1, seed=0).item()
    return scores


def fid_scores(name: str, images: "torch.Tensor", context: dict, shared: dict) -> dict:
    """FID of one decoded row against the arm's cached reference features.

    The reference features are computed once per (data seed, arm) in `main`, so every row
    of every estimator is scored against the same ones and only the row's own Inception
    pass is paid for here.

    Parameters
    ----------
    name : str
    images : torch.Tensor of shape (n, 3, S, S)
    context : dict
    shared : dict

    Returns
    -------
    dict
    """
    half = context["half"]
    scores = {}
    for impl, scorer in shared["fid_scorers"].items():
        for prefix, size_key, transform in context["fid_sizes"]:
            features_ref = shared["fid_ref_features"][(impl, context["target_a"])][size_key]
            features = scorer.features(transform(images))
            scores[f"{prefix}_{impl}_{name}"] = scorer.distance_from_features(features,
                                                                             features_ref)
            # count-matched to the split-half floor below (first half of the row vs
            # the second reference half); NaN when the row is smaller than a half
            scores[f"{prefix}_{impl}_{name}_half"] = (
                scorer.distance_from_features(features[:half], features_ref[half:])
                if len(features) >= half else float("nan"))
    return scores


def arm_evaluation(target_a: int, shared: dict, args: argparse.Namespace, seed: int,
                   device: torch.device, arm_indices: "torch.Tensor",
                   initial_particles: "torch.Tensor") -> tuple:
    """Everything the arm's estimator rows share: baselines, floors and the eval draws.

    The observed row, its VAE round-trip, every floor and the fixed evaluation subsets do
    not depend on the target embedding, so they are drawn, decoded and scored ONCE per
    arm and copied into each estimator's row. An extra estimator therefore costs one
    flow, one decode and one FID pass per row rather than a whole arm -- which matters
    because with `--fid both` the Clean-FID Frechet distance is the dominant cost of a
    row, and the baseline and floor distances would otherwise be recomputed identically.

    Parameters
    ----------
    target_a : int
    shared : dict
    args : argparse.Namespace
    seed : int
    device : torch.device
    arm_indices : torch.Tensor
    initial_particles : torch.Tensor of shape (M, D)
        The cloud every estimator starts from; its latent SW2 is an arm-level column.

    Returns
    -------
    context : dict
        Tensors, callables and this arm's SHARED evaluation timings
        (`seconds_eval_shared`, `seconds_fid_shared`) -- which belong to the arm, not to
        any estimator, and are therefore returned here for the repeat to total rather
        than added into a row's own `seconds_eval` / `seconds_fid`. Also
        `flow_eval_pick`, the ONE particle subset every estimator's pixel and FID rows
        are scored on.
    common : dict
        CSV columns identical across this arm's rows.
    """
    t0 = time.time()
    X, Y = shared["X"], shared["Y"]
    Y_ref, Z_ref_std = shared["Y_ref"][target_a], shared["Z_ref_std"][target_a]
    half = len(Y_ref) // 2
    n_particles = len(initial_particles)

    eval_indices = arm_indices
    if args.eval_max is not None and len(arm_indices) > args.eval_max:
        pick_eval = torch.randperm(len(arm_indices),
                                   generator=torch.Generator().manual_seed(seed + 7))[:args.eval_max]
        eval_indices = arm_indices[pick_eval]
    flow_eval_pick = None
    if args.eval_particles is not None and n_particles > args.eval_particles:
        pick_flow = torch.randperm(n_particles,
                                   generator=torch.Generator().manual_seed(seed + 11))[:args.eval_particles]
        flow_eval_pick = pick_flow.to(device)
    vae = shared["vae"]
    with torch.no_grad():
        observed_images = Y[eval_indices]
        roundtrip_images = decode_latents(vae, encode_images(vae, observed_images),
                                          args.image_size)

    flatten = lambda images: images.reshape(images.shape[0], -1).to(device)
    reference = flatten(Y_ref)
    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(reference)   # lock the bandwidth on the reference
    to_64 = lambda images: F.interpolate(images, size=64, mode="area").clamp(-1.0, 1.0)
    reference_64 = flatten(to_64(Y_ref))
    context = {
        "target_a": target_a, "half": half, "flatten": flatten, "to_64": to_64,
        "reference": reference, "reference_64": reference_64,
        "kernel_pixel": kernel_pixel, "Z_ref_std": Z_ref_std,
        "flow_eval_pick": flow_eval_pick,
        "fid_sizes": (("fid", "native", lambda images: images),)
                     + ((("fid64", "64", to_64),) if args.fid_64 else ()),
    }

    common = {
        "target_a": target_a, "n_arm": int(len(arm_indices)),
        "n_eval": int(len(eval_indices)),
        "n_particles_requested": args.n_particles, "n_particles": int(n_particles),
        "obs_blond_frac_arm": float(X[arm_indices].float().mean()),
        "latent_sw2_initial": sliced_wasserstein(initial_particles, Z_ref_std, p=2,
                                                 seed=0).item(),
        "latent_sw2_floor": sliced_wasserstein(Z_ref_std[:half], Z_ref_std[half:], p=2,
                                               seed=0).item(),
    }
    for name, images in (("observed", observed_images), ("roundtrip", roundtrip_images)):
        common.update(pixel_scores(name, images, context))
    common["sw2_floor"] = sliced_wasserstein(reference[:half], reference[half:], p=2, seed=0).item()
    common["sw1_floor"] = sliced_wasserstein(reference[:half], reference[half:], p=1, seed=0).item()
    common["mmd2_floor"] = squared_mmd(kernel_pixel, reference[:half], reference[half:],
                                       unbiased=True).item()
    common["sw2_64_floor"] = sliced_wasserstein(reference_64[:half], reference_64[half:],
                                                p=2, seed=0).item()
    common["sw1_64_floor"] = sliced_wasserstein(reference_64[:half], reference_64[half:],
                                                p=1, seed=0).item()

    # -- count-matched floors ----------------------------------------------------------
    # SW and MMD^2 are strongly sample-size biased: between two independent draws of the
    # SAME law the SW2 measured on this data falls from 0.0266 at n = 250 to 0.0070 at
    # n = 4000, and `sliced_wasserstein` scores at min(len(x), len(y)). The floors above
    # are half-vs-half numbers at n_ref // 2, whereas the flow row is scored at
    # k = min(n_flow_eval, n_ref), so a sweep over --n_particles would move the flow rows
    # for a purely statistical reason while the floor stood still. The twins below
    # recompute each floor with BOTH halves randomly cut (fixed seed off the pipeline
    # seed) to that same k -- capped at `half`, since a half cannot be enlarged, which is
    # what n_floor_matched reports.
    k_pixel = min(n_particles if flow_eval_pick is None else len(flow_eval_pick),
                  len(reference))
    k_latent = min(n_particles, len(Z_ref_std))
    ref_a, ref_b = matched_halves(reference, half, k_pixel, seed + 101)
    ref64_a, ref64_b = matched_halves(reference_64, half, k_pixel, seed + 101)
    lat_a, lat_b = matched_halves(Z_ref_std, half, k_latent, seed + 201)
    common["n_floor_matched"] = int(min(k_pixel, half))
    common["n_latent_floor_matched"] = int(min(k_latent, half))
    common["sw2_floor_matched"] = sliced_wasserstein(ref_a, ref_b, p=2, seed=0).item()
    common["sw1_floor_matched"] = sliced_wasserstein(ref_a, ref_b, p=1, seed=0).item()
    common["mmd2_floor_matched"] = squared_mmd(kernel_pixel, ref_a, ref_b,
                                               unbiased=True).item()
    common["sw2_64_floor_matched"] = sliced_wasserstein(ref64_a, ref64_b, p=2, seed=0).item()
    common["sw1_64_floor_matched"] = sliced_wasserstein(ref64_a, ref64_b, p=1, seed=0).item()
    common["latent_sw2_floor_matched"] = sliced_wasserstein(lat_a, lat_b, p=2, seed=0).item()
    context["seconds_eval_shared"] = phase_seconds(t0, device)

    t0 = time.time()
    for name, images in (("observed", observed_images), ("roundtrip", roundtrip_images)):
        common.update(fid_scores(name, images, context, shared))
    for impl, scorer in shared["fid_scorers"].items():
        for prefix, size_key, _ in context["fid_sizes"]:
            features_ref = shared["fid_ref_features"][(impl, target_a)][size_key]
            common[f"{prefix}_{impl}_floor"] = scorer.distance_from_features(
                features_ref[:half], features_ref[half:])
    context["seconds_fid_shared"] = phase_seconds(t0, device)
    return context, common


def flow_and_score(name: str, v: "torch.Tensor", shared: dict, args: argparse.Namespace,
                   seed: int, initial_particles: "torch.Tensor", context: dict) -> dict:
    """Flow one estimator's target weights to convergence and score the decoded output.

    The starting cloud, the target support, the iteration budget, the learning rate and
    the schedule are the arm's, so `target_weights` is the only thing that distinguishes
    this row from the arm's other estimator rows. NOTE that `loss_initial` / `loss_final`
    are values of THIS estimator's own Riesz objective, hence not comparable between
    estimators; the distributional columns are what the comparison rests on.

    Parameters
    ----------
    name : str
        Estimator name, written to the `estimator` column.
    v : torch.Tensor of shape (n_dr,)
        Target weight vector over the DR support.
    shared, args, context : dict, argparse.Namespace, dict
    seed : int
    initial_particles : torch.Tensor of shape (M, D)

    Returns
    -------
    dict
        This row's columns, including its OWN `seconds_flow`, `seconds_decode` and
        `seconds_eval`, with `seconds_fid` the FID breakdown already inside the last.
        None of them carries any of the arm's shared cost, which is what makes the three
        safe to sum over the repeat's rows.
    """
    # Three summable clocks, all of them THIS row's own work: flow, decode, and the scoring
    # that ends with its own FID. The arm's shared baselines and floors were timed once in
    # `arm_evaluation` and are NOT folded in here -- that is what made `seconds_eval` /
    # `seconds_fid` unsummable when the estimator axis arrived (three rows each reporting
    # the same shared cost).
    device = initial_particles.device
    t0 = time.time()
    torch.manual_seed(seed + 1)
    flow_particles, history = adamw_riesz_flow(
        initial_particles.clone(), shared["Z_std_s"], target_weights=v, sliced=False,
        n_iterations=args.flow_iterations, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, lr_schedule=args.lr_schedule,
        snapshot_every=args.snapshot_every, log_every=500, verbose=False)
    row = {"estimator": name, "seconds_flow": round(phase_seconds(t0, device), 1)}

    # own scoring, part 1: the latent metrics, which need no decode
    t0 = time.time()
    row.update({
        "loss_initial": float(history["loss"][0]),
        "loss_final": float(history["loss"][-1]),
        "mean_transport": float((flow_particles - initial_particles).norm(dim=1).mean()),
        "latent_sw2_flow": sliced_wasserstein(flow_particles, context["Z_ref_std"], p=2,
                                              seed=0).item(),
    })
    flow_eval = (flow_particles if context["flow_eval_pick"] is None
                 else flow_particles[context["flow_eval_pick"]])
    row["n_flow_eval"] = int(len(flow_eval))
    seconds_eval = phase_seconds(t0, device)

    t0 = time.time()
    with torch.no_grad():
        flow_images = decode_latents(shared["vae"], shared["from_flow"](flow_eval),
                                     args.image_size)
    row["seconds_decode"] = round(phase_seconds(t0, device), 1)

    # own scoring, part 2: the pixel metrics of the decoded row
    t0 = time.time()
    row.update(pixel_scores("flow", flow_images, context))
    seconds_eval += phase_seconds(t0, device)

    # own scoring, part 3: this row's FID. Timed into seconds_eval and reported separately
    # as a BREAKDOWN of it, which is ColorMNIST's convention: it keeps the own side of the
    # identity the same three columns (flow, decode, eval) in every design.
    t0 = time.time()
    row.update(fid_scores("flow", flow_images, context, shared))
    seconds_fid = phase_seconds(t0, device)
    row["seconds_eval"] = round(seconds_eval + seconds_fid, 1)
    row["seconds_fid"] = round(seconds_fid, 1)
    return row


def run_arm(target_a: int, shared: dict, args: argparse.Namespace, seed: int,
            device: torch.device) -> list:
    """Arm-specific half of one repeat: one CSV row per requested estimator.

    Everything upstream of the target weight vector is done once and shared by the rows
    -- the logged sample, the propensity, the single CME solve, the starting particles,
    the decoded observed and round-trip baselines, every floor and every evaluation
    subset -- so the rows differ only in which estimator's weights the flow was given.
    `ess_on_arm` / `n_on_arm` / `max_weight` describe the arm's importance weights and
    are therefore also shared; the `sum_v*` and `v_*` columns are the estimator's own.

    Returns
    -------
    rows : list of dict
        One row per requested estimator, in the canonical order. Each carries only its
        OWN timings (`seconds_flow`, `seconds_decode`, `seconds_eval`, with `seconds_fid`
        the breakdown inside the last).
    arm_seconds : dict
        This arm's SHARED timings, for `run_repeat` to total over arms and write
        identically onto every row: the arm's importance ratios and its DR/CME weight solve
        under `seconds_ratio` / `seconds_cme`, and the arm's baselines, floors and shared
        FID under `seconds_eval_shared` / `seconds_fid_shared`. Returned rather than copied
        into the rows precisely so that summing a column down the rows cannot multiply it.
    """
    # The weight solve is ONE nuisance fit for the whole arm (`components=True` hands back
    # the plug-in term and the correction from that single solve), so it belongs with the
    # repeat's propensity and ridge work in the nuisance columns, not in any row's own time.
    w_hat, vectors, arm_nuisance = arm_weight_vectors(target_a, shared, args, seed, device)
    on_arm_mask = shared["A_s"] == target_a
    on_arm = w_hat[on_arm_mask]
    arm_indices, initial_particles = starting_particles(target_a, shared, args, seed)
    context, common = arm_evaluation(target_a, shared, args, seed, device, arm_indices,
                                     initial_particles)
    common.update({"ess_on_arm": float(on_arm.sum() ** 2 / (on_arm ** 2).sum()),
                   "n_on_arm": int(len(on_arm)), "max_weight": float(w_hat.max())})
    rows = []
    for name, v in vectors.items():
        row = dict(common)
        row.update(weight_diagnostics(v, on_arm_mask, shared["X_s"]))
        row.update(flow_and_score(name, v, shared, args, seed, initial_particles, context))
        rows.append(row)
    arm_seconds = {"seconds_ratio": arm_nuisance["seconds_ratio"],
                   "seconds_cme": arm_nuisance["seconds_cme"],
                   "seconds_eval_shared": context["seconds_eval_shared"],
                   "seconds_fid_shared": context["seconds_fid_shared"]}
    return rows, arm_seconds


def run_repeat(shared: dict, args: argparse.Namespace, data_seed: int, repeat: int,
               device: torch.device) -> list:
    """One pipeline pass on a fixed data draw; one CSV row per arm and estimator.

    Also the one place the repeat's wall-clock accounting is assembled: the shared
    columns are totalled over arms and written identically onto every row, and
    `seconds_other` is set to the residual that makes the documented identity close
    exactly. See "Wall-clock accounting" in the module docstring.
    """
    seed = 1000 * data_seed + repeat
    t_start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -- the ratio half of the shared nuisance fit: the DR support, the folds and the
    # propensity. This is all the IPW estimator needs; the support and the folds are clocked
    # here because this half consumes them first, and the CME half reuses both.
    t0 = time.time()
    n = len(shared["A"])
    solver = resolve_solver(args)
    # The support is every logged row; --n_dr caps it only when asked for explicitly.
    if args.n_dr is not None and args.n_dr < n:
        pick_dr = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 5))[:args.n_dr]
    else:
        pick_dr = torch.arange(n)
    A_s, X_s = shared["A"][pick_dr], shared["X"][pick_dr]
    Z_std_s = shared["Z_std"][pick_dr]
    n_dr = len(pick_dr)

    fold_ids = make_crossfit_fold_ids(n_dr, n_folds=args.n_folds, seed=0)
    torch.manual_seed(seed)
    if args.dr_fit == "nosplit":
        # Fit once on every row and predict on those same rows. NOT crossfit_propensity
        # with one fold: that has no guard and silently returns a constant 0.5, which
        # would leave the two arms differing only in the CME nuisance.
        model = LogisticPropensityModel()
        model.fit(X_s.double().unsqueeze(1), A_s.double())
        e1 = model.predict(X_s.double().unsqueeze(1)).reshape(-1).double()
    else:
        e1 = crossfit_propensity(X_s.double().unsqueeze(1), A_s.double(), fold_ids)
    seconds_ratio = phase_seconds(t0, device)

    # -- the CME half: the (A, X) cell Gram and the per-fold ridge selection. This is all
    # the plug-in estimator needs, and DR needs both halves. Each arm's single weight solve
    # is added to this accumulator from the arm loop below.
    t0 = time.time()
    cell = (A_s * 2 + X_s).long().to(device)           # the 4 (A, X) cells
    cell_gram = lambda rows, cols: (cell[rows][:, None] == cell[cols][None, :]).double()
    if args.lambda_rule == "n_half":
        nosplit_lambda = lambda_n_half(n_dr, args.lambda_constant)
        fold_lambdas = {int(f): nosplit_lambda for f in np.unique(fold_ids)}
    else:
        fold_lambdas = {}
        for fold in np.unique(fold_ids):
            train = np.where(fold_ids != fold)[0]
            # lambda does not need the full block, and the full block is a 16000^2 solve
            pick = train[torch.randperm(len(train),
                                        generator=torch.Generator().manual_seed(0))[:2000].numpy()]
            fold_lambdas[int(fold)], _ = select_regularization(cell_gram(pick, pick),
                                                               Z_std_s[pick, 0].double().to(device),
                                                               seed=0)
        nosplit_lambda = None
        if args.dr_fit == "nosplit":
            # one training block (all rows) needs one scalar
            pick = torch.randperm(n_dr, generator=torch.Generator().manual_seed(0))[:2000].numpy()
            nosplit_lambda, _ = select_regularization(cell_gram(pick, pick),
                                                      Z_std_s[pick, 0].double().to(device), seed=0)
    seconds_cme = phase_seconds(t0, device)

    shared_repeat = dict(shared, A_s=A_s, X_s=X_s, Z_std_s=Z_std_s, e1=e1, cell=cell,
                         solver=solver, nosplit_lambda=nosplit_lambda,
                         fold_ids=fold_ids, fold_lambdas=fold_lambdas)
    stats = binary_stats(shared["A"], shared["X"])
    base = {
        "tag": args.tag, "device": str(device), "init": args.init,
        "data_seed": data_seed, "repeat": repeat, "pipeline_seed": seed,
        "image_size": args.image_size, "n_obs": args.n_obs, "n_ref": args.n_ref,
        "n_dr": int(n_dr),
        "n_landmarks": (args.n_landmarks
                        if solver in ("nystrom", "projected_nystrom") else 0),
        "solver": solver, "dr_fit": args.dr_fit, "lambda_rule": args.lambda_rule,
        "ipw_form": args.ipw_form,
        "n_folds": args.n_folds,
        "latent_dim": int(shared["Z_std"].shape[1]),
        "flow_iterations": args.flow_iterations, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps, "lr_schedule": args.lr_schedule,
        "n_a0x0": stats["cells"][(0, 0)], "n_a0x1": stats["cells"][(0, 1)],
        "n_a1x0": stats["cells"][(1, 0)], "n_a1x1": stats["cells"][(1, 1)],
        "fold_lambdas": ";".join(f"{fold}:{lam:g}" for fold, lam in sorted(fold_lambdas.items())),
        # per DATA SEED, measured in main() outside this repeat's clock: the second term of
        # the per-task identity, identical on every row of every repeat of this seed.
        **shared["setup"],
        **hardware_provenance(device),
    }
    # Shared-per-repeat clocks, accumulated over arms: the two halves of the nuisance fit
    # (the block above plus each arm's ratios and weight solve), and each arm's baselines,
    # floors and shared FID.
    shared_seconds = {"seconds_ratio": seconds_ratio, "seconds_cme": seconds_cme,
                      "seconds_eval_shared": 0.0, "seconds_fid_shared": 0.0}
    rows = []
    for target_a in args.target_arms:
        arm_rows, arm_seconds = run_arm(target_a, shared_repeat, args, seed, device)
        for key, value in arm_seconds.items():
            shared_seconds[key] += value
        for arm_row in arm_rows:
            row = dict(base)
            row.update(arm_row)
            rows.append(row)

    # -- close the wall-clock identity (see "Wall-clock accounting" in the module
    # docstring). Everything is rounded to 0.1 s FIRST and `seconds_other` is then the
    # residual of the rounded numbers, so the identity holds exactly on the CSV values a
    # reader actually adds up rather than only on the unrounded ones.
    seconds_repeat = round(phase_seconds(t_start, device), 1)
    shared_seconds = {key: round(value, 1) for key, value in shared_seconds.items()}
    # The two nuisance halves are a BREAKDOWN, so seconds_nuisance is their rounded sum and
    # only it enters seconds_shared -- adding all three would count the fit twice. Named
    # explicitly rather than summing the dict, so adding a shared column later has to be a
    # deliberate act: it must be listed here or the identity will not close.
    shared_seconds["seconds_nuisance"] = round(shared_seconds["seconds_ratio"]
                                               + shared_seconds["seconds_cme"], 1)
    shared_seconds["seconds_shared"] = round(shared_seconds["seconds_nuisance"]
                                             + shared_seconds["seconds_eval_shared"]
                                             + shared_seconds["seconds_fid_shared"], 1)
    # Three own columns, not four: seconds_fid is inside seconds_eval (see the docstring).
    own = sum(row["seconds_flow"] + row["seconds_decode"] + row["seconds_eval"]
              for row in rows)
    # The trailing + 0.0 turns the -0.0 that float subtraction yields when the parts cancel
    # exactly into a plain 0.0, so the CSV reads "0.0" and not "-0.0".
    seconds_other = round(seconds_repeat - shared_seconds["seconds_shared"] - own, 1) + 0.0
    for row in rows:
        row.update(shared_seconds)
        # Derived, and in neither identity: this estimator's own half of the nuisance fit
        # plus this row's own flow and decode, with no encoding and no measurement in it.
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

    # The per-DATA-SEED setup clock. It starts HERE so the once-per-process model loads --
    # the SD-VAE and the two Inception networks -- are charged to the first data seed's
    # seconds_data rather than to nothing at all; it is restarted after each seed's repeats.
    # See "Wall-clock accounting" in the module docstring.
    t_process = time.time()
    t_setup = t_process
    vae = load_sd_vae(device)
    fid_scorers = {}
    if args.fid in ("legacy", "both"):
        fid_scorers["legacy"] = LegacyFIDScorer(device=device, batch_size=32)
    if args.fid in ("clean", "both"):
        fid_scorers["clean"] = CleanFIDScorer(device=device, batch_size=32,
                                              num_workers=args.fid_workers, allow_download=False)
    rows = []
    accounted = 0.0
    for data_seed in args.data_seeds:
        cache = ensure_cache(args, data_seed)
        data = load_celeba_paper(cache, root=args.celeba_root, seed=data_seed,
                                 image_size=args.image_size, n_obs=args.n_obs,
                                 n_ref=args.n_ref, verbose=False)
        A = data["A"].view(-1).long()
        X = data["X"].view(-1).long()
        Y = data["Y"]
        # Building a cold cache materialises every image and can dominate a whole task, so
        # it gets a column of its own rather than sitting outside the accounting.
        seconds_data = round(phase_seconds(t_setup, device), 1)

        # Encodings are pipeline-independent (frozen VAE, fixed draw): once per data seed.
        # So this clock, like seconds_data above and seconds_fid_ref below, sits OUTSIDE
        # every seconds_repeat -- the cost is real but it is paid once per data seed, not
        # once per repeat and certainly not once per estimator row, which is why the
        # per-repeat identity excludes all three and seconds_setup totals them for the
        # per-task one.
        t0 = time.time()
        Z = encode_images(vae, Y)
        latent_mean = Z.mean(dim=0, keepdim=True)
        latent_std = Z.std(dim=0, keepdim=True).clamp_min(1e-3)
        Z_std = (Z - latent_mean) / latent_std
        Y_ref, Z_ref_std = {}, {}
        for arm in args.target_arms:
            Y_ref[arm] = data[f"Y{arm}_ref"]
            Z_ref_std[arm] = (encode_images(vae, Y_ref[arm]) - latent_mean) / latent_std
        seconds_encode = round(phase_seconds(t0, device), 1)
        # Reference features once per data seed and arm; every row is scored against them.
        t0 = time.time()
        fid_ref_features = {}
        for impl, scorer in fid_scorers.items():
            for arm in args.target_arms:
                fid_ref_features[(impl, arm)] = {"native": scorer.features(Y_ref[arm])}
                if args.fid_64:
                    small = F.interpolate(Y_ref[arm], size=64, mode="area").clamp(-1.0, 1.0)
                    fid_ref_features[(impl, arm)]["64"] = scorer.features(small)
        seconds_fid_ref = round(phase_seconds(t0, device), 1)

        # The per-data-seed tier as CSV columns: the three components and their total, the
        # one column to add once per data seed.
        setup = {"seconds_data": seconds_data, "seconds_encode": seconds_encode,
                 "seconds_fid_ref": seconds_fid_ref,
                 "seconds_setup": round(seconds_data + seconds_encode + seconds_fid_ref, 1)}
        accounted += setup["seconds_setup"]
        shared = {"vae": vae, "A": A, "X": X, "Y": Y, "Z_std": Z_std,
                  "Y_ref": Y_ref, "Z_ref_std": Z_ref_std,
                  "from_flow": lambda p: p * latent_std + latent_mean,
                  "to_flow": lambda z: (z - latent_mean) / latent_std,
                  "setup": setup,
                  "fid_scorers": fid_scorers, "fid_ref_features": fid_ref_features}
        for repeat in range(args.n_repeats):
            new_rows = run_repeat(shared, args, data_seed, repeat, device)
            accounted += new_rows[-1]["seconds_repeat"]
            rows.extend(new_rows)
            write_rows(out_path, rows)   # atomic rewrite per repeat: crash-safe
            write_particle_indices(out_path)
            for row in new_rows:
                fid_note = (f"FID legacy flow {row['fid_legacy_flow']:.1f} "
                            f"(observed {row['fid_legacy_observed']:.1f}, "
                            f"floor {row['fid_legacy_floor']:.1f})   "
                            if "fid_legacy_flow" in row else "")
                print(f"[data_seed {data_seed} repeat {repeat} arm {row['target_a']} "
                      f"{row['estimator']}] "
                      f"latent SW2 flow {row['latent_sw2_flow']:.4f} "
                      f"(initial {row['latent_sw2_initial']:.4f}, floor {row['latent_sw2_floor']:.4f})   "
                      f"pixel SW2 flow {row['sw2_flow']:.4f} (floor {row['sw2_floor']:.4f})   "
                      f"{fid_note}"
                      f"sum_v {row['sum_v']:.3f}   {row['seconds_repeat']:.0f}s repeat"
                      f" / {row['seconds_setup']:.0f}s setup this seed", flush=True)
        # The next data seed's setup clock starts once this seed's repeats are done.
        t_setup = time.time()
    # The per-task identity, printed so the gap it does not cover is visible rather than
    # inferred: process start-up and the per-repeat CSV rewrite.
    print(f"done: {len(rows)} rows -> {out_path}", flush=True)
    print(f"accounted {round(accounted, 1)}s = sum(seconds_setup) over data seeds "
          f"+ sum(seconds_repeat) over repeats, of "
          f"{phase_seconds(t_process, device):.1f}s process wall clock", flush=True)


if __name__ == "__main__":
    main()
