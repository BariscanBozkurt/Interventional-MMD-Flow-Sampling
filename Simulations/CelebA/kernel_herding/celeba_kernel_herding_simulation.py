"""Matched CelebA kernel-herding baseline.

This script reuses the current CelebA DR-IMMD simulation for the data draw,
frozen SD-VAE, projected-Nystrom DR embedding, intervention references, metrics,
and timing accounting.  It changes only the sampler: kernel herding greedily
selects standardized SD-VAE latents from all logged outcomes instead of moving
particles with the Riesz flow.

Kernel herding is a selection sampler.  Its primary outputs are therefore raw
logged faces, not newly synthesized images.  For a representation-matched
comparison with DR-IMMD, this script also decodes the selected latents through
the same frozen SD-VAE and scores those round-trip images separately.  The
standard ``*_flow`` columns contain the decoded herding scores, while the
explicit ``*_herding`` and ``*_herding_decoded`` columns retain both views.

The candidate set is the complete logged DR support, rather than an oracle
treatment-arm subset.  The CSV and companion ``*_selections.npz`` file record
which observations were selected, how many were unique, their multiplicities,
and their treatment/confounder composition.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DR_SIM_DIR = HERE.parent / "dr_mmd_flow"
sys.path.insert(0, str(DR_SIM_DIR))

import celeba_dr_riesz_flow_simulation as base

from autoencoder_utils import decode_latents
from causality_utils import herding_diagnostics, kernel_herding
from kernel_utils import RieszKernel
from metric_utils import sliced_wasserstein


HERDING_ROWS = ("herding", "herding_decoded")
EXTRA_COLUMNS = (
    [
        "sampler", "candidate_set", "output_source", "n_candidates",
        "n_herding_steps", "n_unique", "unique_fraction",
        "max_multiplicity", "candidate_coverage",
        "selected_target_arm_fraction", "selected_blond_fraction",
        "target_blond_mass", "seconds_herding", "seconds_herding_decode",
        "latent_sw2_herding",
    ]
    + [f"{metric}_{row}" for row in HERDING_ROWS
       for metric in ("sw2", "sw1", "mmd2")]
    + [f"{metric}_64_{row}" for row in HERDING_ROWS
       for metric in ("sw2", "sw1")]
    + [f"fid_{impl}_{row}" for impl in base.FID_IMPLS for row in HERDING_ROWS]
    + [f"fid_{impl}_{row}_half" for impl in base.FID_IMPLS for row in HERDING_ROWS]
    + [f"fid64_{impl}_{row}" for impl in base.FID_IMPLS for row in HERDING_ROWS]
    + [f"fid64_{impl}_{row}_half" for impl in base.FID_IMPLS for row in HERDING_ROWS]
)

HERDING_SELECTIONS = {}


def rename_row(scores: dict, old: str, new: str) -> dict:
    """Rename one metric-row suffix without recomputing the metric."""
    return {key.replace(f"_{old}", f"_{new}"): value
            for key, value in scores.items()}


def run_arm(target_a: int, shared: dict, args, seed: int,
            device: torch.device) -> tuple:
    """Build the matched DR target, herd logged latents, and score both outputs."""
    w_hat, vectors, arm_nuisance = base.arm_weight_vectors(
        target_a, shared, args, seed, device)
    v_target = vectors["dr"]
    on_arm_mask = shared["A_s"] == target_a
    on_arm = w_hat[on_arm_mask]

    # Retain exactly the same observed-arm baseline and evaluation draws as the
    # headline DR-IMMD run.  Herding itself uses all logged outcomes below.
    arm_indices, initial_particles = base.starting_particles(
        target_a, shared, args, seed)
    context, common = base.arm_evaluation(
        target_a, shared, args, seed, device, arm_indices, initial_particles)
    common.update({
        "ess_on_arm": float(on_arm.sum() ** 2 / (on_arm ** 2).sum()),
        "n_on_arm": int(len(on_arm)),
        "max_weight": float(w_hat.max()),
    })

    # The Riesz kernel and standardized SD-VAE coordinates match the practical
    # CelebA flow.  Support and candidates coincide, so every selected index maps
    # directly to a logged face (the parser below forbids a subsampled DR support).
    t0 = time.time()
    with torch.no_grad():
        herded_latents, selected = kernel_herding(
            v_target, RieszKernel(), shared["Z_std_s"], shared["Z_std_s"],
            len(initial_particles), return_indices=True)
    seconds_herding = base.phase_seconds(t0, device)

    selected_cpu = selected.detach().cpu()
    raw_images = shared["Y"][selected_cpu]
    selected_actions = shared["A_s"][selected.to(shared["A_s"].device)].cpu()
    selected_blond = shared["X_s"][selected.to(shared["X_s"].device)].cpu()
    diagnostics = herding_diagnostics(selected, n_candidates=len(shared["Z_std_s"]))

    eval_pick = context["flow_eval_pick"]
    if eval_pick is None:
        latent_eval = herded_latents
        raw_eval = raw_images
    else:
        latent_eval = herded_latents[eval_pick]
        raw_eval = raw_images[eval_pick.detach().cpu()]

    t0 = time.time()
    with torch.no_grad():
        decoded_eval = decode_latents(
            shared["vae"], shared["from_flow"](latent_eval), args.image_size)
    seconds_decode = base.phase_seconds(t0, device)

    key = f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"
    HERDING_SELECTIONS[f"{key}_candidate_positions"] = selected_cpu.numpy()
    # With --n_dr unset, candidate position and logged-row index are identical.
    HERDING_SELECTIONS[f"{key}_observation_indices"] = selected_cpu.numpy()

    # Pixel metrics for the raw retrieval baseline and the matched SD-VAE
    # round trip.  The latter is copied to *_flow for compatibility with the
    # existing aggregate-result code; no metric is evaluated twice.
    t0 = time.time()
    latent_sw2 = sliced_wasserstein(
        herded_latents, context["Z_ref_std"], p=2, seed=0).item()
    raw_scores = base.pixel_scores("herding", raw_eval, context)
    decoded_scores = base.pixel_scores("herding_decoded", decoded_eval, context)
    seconds_eval = base.phase_seconds(t0, device)

    t0 = time.time()
    raw_fid = base.fid_scores("herding", raw_eval, context, shared)
    decoded_fid = base.fid_scores(
        "herding_decoded", decoded_eval, context, shared)
    seconds_fid = base.phase_seconds(t0, device)
    seconds_eval += seconds_fid

    row = dict(common)
    row.update(base.weight_diagnostics(v_target, on_arm_mask, shared["X_s"]))
    row.update({
        "estimator": "dr",
        "sampler": "kernel_herding",
        "candidate_set": "all_logged_outcomes",
        "output_source": "raw_logged_outcome_and_sd_vae_roundtrip",
        "n_candidates": int(len(shared["Z_std_s"])),
        "n_herding_steps": int(len(herded_latents)),
        "n_unique": diagnostics["n_unique"],
        "unique_fraction": diagnostics["unique_fraction"],
        "max_multiplicity": diagnostics["max_multiplicity"],
        "candidate_coverage": diagnostics["candidate_coverage"],
        "selected_target_arm_fraction": float(
            (selected_actions == target_a).float().mean()),
        "selected_blond_fraction": float(selected_blond.float().mean()),
        "target_blond_mass": float(
            (v_target * shared["X_s"].float().to(v_target.device)).sum()),
        # Herding does not define a particle-to-particle transport pairing.  These
        # flow-specific columns are therefore deliberately not given a false value.
        "loss_initial": float("nan"),
        "loss_final": float("nan"),
        "mean_transport": float("nan"),
        "latent_sw2_flow": latent_sw2,
        "latent_sw2_herding": latent_sw2,
        "n_flow_eval": int(len(latent_eval)),
        "seconds_flow": round(seconds_herding, 1),
        "seconds_herding": round(seconds_herding, 1),
        "seconds_decode": round(seconds_decode, 1),
        "seconds_herding_decode": round(seconds_decode, 1),
        "seconds_eval": round(seconds_eval, 1),
        "seconds_fid": round(seconds_fid, 1),
    })
    row.update(raw_scores)
    row.update(decoded_scores)
    row.update(raw_fid)
    row.update(decoded_fid)
    row.update(rename_row(decoded_scores, "herding_decoded", "flow"))
    row.update(rename_row(decoded_fid, "herding_decoded", "flow"))

    arm_seconds = {
        "seconds_ratio": arm_nuisance["seconds_ratio"],
        "seconds_cme": arm_nuisance["seconds_cme"],
        "seconds_eval_shared": context["seconds_eval_shared"],
        "seconds_fid_shared": context["seconds_fid_shared"],
    }
    return [row], arm_seconds


ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args():
    """Use the shared CLI while guarding the matched herding contract."""
    args = ORIGINAL_PARSE_ARGS()
    if base.requested_estimators(args) != ["dr"]:
        raise SystemExit("kernel-herding headline runs require --estimators dr")
    if args.init != "observed":
        raise SystemExit("the matched kernel-herding baseline requires --init observed")
    if args.n_dr is not None:
        raise SystemExit("leave --n_dr unset so candidates map to all logged rows")
    if args.snapshot_every is not None:
        raise SystemExit("--snapshot_every is an MMD-flow option and does not apply here")
    args.flow_iterations = 0
    return args


def csv_path(args) -> Path:
    """Keep herding artifacts separate from the DR-IMMD result directory."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    name = (
        f"celeba_kernel_herding_{args.tag}_sz{args.image_size}_nobs{args.n_obs}"
        f"_M{args.n_particles}_arms{arms}_ds{seeds}_{args.dr_fit}_{args.solver}.csv"
    )
    return Path(args.out_dir) / name


def write_herding_indices(out_path: Path) -> None:
    """Save candidate positions and logged-row indices for an exact audit."""
    if HERDING_SELECTIONS:
        np.savez(out_path.with_name(out_path.stem + "_selections.npz"),
                 **HERDING_SELECTIONS)


def main() -> None:
    if "--out_dir" not in sys.argv:
        sys.argv.extend(["--out_dir", str(HERE / "results")])

    base.COLUMNS = base.COLUMNS + EXTRA_COLUMNS
    base.parse_args = parse_args
    base.csv_path = csv_path
    base.run_arm = run_arm
    base.write_particle_indices = write_herding_indices
    base.main()


if __name__ == "__main__":
    main()
