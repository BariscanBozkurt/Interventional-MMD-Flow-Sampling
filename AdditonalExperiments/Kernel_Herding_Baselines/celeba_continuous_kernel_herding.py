"""Matched finite and continuous latent-space kernel herding for CelebA.

The data draw, frozen SD-VAE, projected-Nystrom DR target, references, metric
randomness, and matched DR-IMMD run are shared.  The finite baseline selects
logged latents.  The continuous baseline implements the oracle of CPME
Algorithm 2 approximately over a declared continuous latent domain: its current
finite-bank maximizer is only an optimization seed, and gradient ascent may
return a new standardized latent.  Production runs use the unrestricted
standardized SD-VAE latent space; the new latent is decoded through the same
frozen SD-VAE.  A fixed random subset of decoded outputs is saved for qualitative
inspection.

The continuous score uses the displayed CPME Algorithm-2 convention: with
``p`` previously selected atoms its repulsion is divided by ``p+1``.  The
finite candidate baseline retains the released implementation's ``p``
denominator, and both conventions are written to the output table.

The supplied DR coefficients are never silently normalized.  Since the CelebA
sampler uses the conditionally positive-definite Riesz interaction, this wrapper
requires their mass to equal one within a declared numerical tolerance.  Under
this unit-mass condition, the displayed ``p+1`` denominator makes the unbounded
Riesz score coercive: it is ``-||z||/(p+1) + O(1)`` at infinity.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY_ROOT = HERE.parents[1]
DR_SIM_DIR = DEPLOY_ROOT / "Simulations" / "CelebA" / "dr_mmd_flow"
sys.path.insert(0, str(DR_SIM_DIR))

import celeba_dr_riesz_flow_simulation as base

from autoencoder_utils import decode_latents, encode_images
from causality_utils import herding_diagnostics, kernel_herding
from continuous_latent_herding import continuous_kernel_herding
from kernel_utils import RieszKernel, pairwise_squared_distance
from metric_utils import sliced_wasserstein
from qualitative_samples import (
    deterministic_indices,
    images_to_uint8,
    save_labeled_image_grid,
)


FINITE_ROWS = ("herding", "herding_decoded")
EXTRA_COLUMNS = (
    [
        "sampler", "optimization_domain", "output_source", "n_candidates",
        "n_herding_steps", "n_unique", "unique_fraction", "max_multiplicity",
        "candidate_coverage", "selected_target_arm_fraction",
        "selected_blond_fraction", "target_blond_mass",
        "seconds_herding", "seconds_herding_decode", "latent_sw2_herding",
        "herding_kernel", "herding_initializer", "herding_domain",
        "finite_repulsion_denominator", "continuous_repulsion_denominator",
        "n_continuous_samples", "continuous_opt_steps", "continuous_restarts",
        "continuous_learning_rate", "continuous_restart_scale",
        "continuous_support_chunk_size", "continuous_repulsion_chunk_size",
        "continuous_candidate_chunk_size", "continuous_box_quantile",
        "continuous_domain_margin",
        "continuous_target_mass",
        "continuous_target_abs_mass", "continuous_mean_seed_score",
        "continuous_mean_anchor_score",
        "continuous_mean_final_score", "continuous_mean_score_gain",
        "continuous_min_score_gain", "continuous_fraction_improved",
        "continuous_mean_gain_over_anchor", "continuous_min_gain_over_anchor",
        "continuous_fraction_improved_over_anchor",
        "continuous_fraction_candidate_restart", "continuous_fraction_anchor_restart",
        "continuous_fraction_random_restart",
        "continuous_mean_seed_displacement", "continuous_max_seed_displacement",
        "continuous_mean_optimizer_steps", "continuous_mean_gradient_norm",
        "continuous_max_gradient_norm", "continuous_mean_boundary_fraction",
        "continuous_n_unique", "continuous_unique_fraction",
        "continuous_norm_mean", "continuous_norm_median", "continuous_norm_max",
        "continuous_max_abs_coordinate",
        "continuous_nearest_support_mean", "continuous_nearest_support_median",
        "continuous_nearest_support_max", "continuous_support_match_fraction",
        "continuous_n_novel", "continuous_novel_fraction",
        "continuous_support_tolerance", "continuous_cycle_n",
        "continuous_cycle_mean", "continuous_cycle_median", "continuous_cycle_max",
        "seconds_continuous_herding", "seconds_continuous_decode",
        "seconds_continuous_diagnostics", "latent_sw2_continuous",
        "continuous_example_count", "continuous_example_seed",
        "continuous_example_selection", "continuous_example_rows",
        "continuous_example_file", "seconds_example_generation",
        "loss_initial_immd", "loss_final_immd", "mean_transport_immd",
        "latent_sw2_immd", "seconds_immd_flow", "seconds_immd_decode",
        "seconds_immd_eval", "seconds_immd_fid",
    ]
    + [f"{metric}_{row}" for row in FINITE_ROWS
       for metric in ("sw2", "sw1", "mmd2")]
    + [f"{metric}_64_{row}" for row in FINITE_ROWS
       for metric in ("sw2", "sw1")]
    + [f"fid_{impl}_{row}" for impl in base.FID_IMPLS for row in FINITE_ROWS]
    + [f"fid_{impl}_{row}_half" for impl in base.FID_IMPLS for row in FINITE_ROWS]
    + [f"fid64_{impl}_{row}" for impl in base.FID_IMPLS for row in FINITE_ROWS]
    + [f"fid64_{impl}_{row}_half" for impl in base.FID_IMPLS for row in FINITE_ROWS]
    + [f"{metric}_continuous" for metric in ("sw2", "sw1", "mmd2")]
    + [f"{metric}_64_continuous" for metric in ("sw2", "sw1")]
    + [f"fid_{impl}_continuous" for impl in base.FID_IMPLS]
    + [f"fid_{impl}_continuous_half" for impl in base.FID_IMPLS]
    + [f"fid64_{impl}_continuous" for impl in base.FID_IMPLS]
    + [f"fid64_{impl}_continuous_half" for impl in base.FID_IMPLS]
    + [f"{metric}_immd" for metric in ("sw2", "sw1", "mmd2")]
    + [f"{metric}_64_immd" for metric in ("sw2", "sw1")]
    + [f"fid_{impl}_immd" for impl in base.FID_IMPLS]
    + [f"fid_{impl}_immd_half" for impl in base.FID_IMPLS]
    + [f"fid64_{impl}_immd" for impl in base.FID_IMPLS]
    + [f"fid64_{impl}_immd_half" for impl in base.FID_IMPLS]
)

ARTIFACTS = {}
EXAMPLE_GRIDS = {}


def rename_row(scores: dict, old: str, new: str) -> dict:
    """Rename one metric-row suffix without evaluating the metric twice."""
    return {
        key.replace(f"_{old}", f"_{new}"): value
        for key, value in scores.items()
    }


@torch.no_grad()
def nearest_support_distances(points: torch.Tensor, support: torch.Tensor,
                              point_chunk: int = 16,
                              support_chunk: int = 2048) -> torch.Tensor:
    """Exact nearest-support distances without materializing an M-by-n matrix."""
    nearest = []
    for point_start in range(0, len(points), point_chunk):
        point_block = points[point_start:point_start + point_chunk]
        best_sq = torch.full(
            (len(point_block),), float("inf"),
            device=points.device, dtype=points.dtype)
        best_index = torch.zeros(
            len(point_block), device=points.device, dtype=torch.long)
        for support_start in range(0, len(support), support_chunk):
            support_block = support[support_start:support_start + support_chunk]
            chunk_sq, chunk_index = pairwise_squared_distance(
                point_block, support_block).min(dim=1)
            improved = chunk_sq < best_sq
            best_sq[improved] = chunk_sq[improved]
            best_index[improved] = support_start + chunk_index[improved]
        # Recompute the chosen pair directly. The matrix identity used above is
        # fast but loses precision when two high-dimensional vectors coincide.
        # This direct subtraction makes a true logged-support match exactly zero.
        direct = (point_block - support[best_index]).norm(dim=1)
        nearest.append(direct.cpu())
    return torch.cat(nearest)


def run_arm(target_a: int, shared: dict, args, seed: int,
            device: torch.device) -> tuple:
    """Fit one DR target, run all three samplers, and score them together."""
    key = f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"
    w_hat, vectors, arm_nuisance = base.arm_weight_vectors(
        target_a, shared, args, seed, device)
    v_target = vectors["dr"]
    on_arm_mask = shared["A_s"] == target_a
    on_arm = w_hat[on_arm_mask]

    arm_indices, initial_particles = base.starting_particles(
        target_a, shared, args, seed)
    context, common = base.arm_evaluation(
        target_a, shared, args, seed, device, arm_indices, initial_particles)
    common.update({
        "ess_on_arm": float(on_arm.sum() ** 2 / (on_arm ** 2).sum()),
        "n_on_arm": int(len(on_arm)),
        "max_weight": float(w_hat.max()),
    })

    immd = base.flow_and_score(
        "dr", v_target, shared, args, seed,
        initial_particles.clone(), context)

    kernel = RieszKernel()

    # Existing finite-candidate baseline. It is repeated in this process so its
    # representation, target, references, and evaluation projections are exact
    # matches for both continuous samplers.
    t0 = time.time()
    with torch.no_grad():
        finite_latents, selected = kernel_herding(
            v_target, kernel, shared["Z_std_s"], shared["Z_std_s"],
            len(initial_particles), return_indices=True)
    seconds_finite = base.phase_seconds(t0, device)
    selected_cpu = selected.detach().cpu()
    raw_images = shared["Y"][selected_cpu]
    selected_actions = shared["A_s"][selected.to(shared["A_s"].device)].cpu()
    selected_blond = shared["X_s"][selected.to(shared["X_s"].device)].cpu()
    finite_diagnostics = herding_diagnostics(
        selected, n_candidates=len(shared["Z_std_s"]))

    target_mass = float(v_target.sum())
    target_abs_mass = float(v_target.abs().sum())
    if abs(target_mass - 1.0) > args.continuous_mass_tolerance:
        raise RuntimeError(
            "continuous Riesz herding requires unit target mass; "
            f"received sum(v)={target_mass:.12g}. No normalization was applied."
        )

    herding_generator = torch.Generator(device=device)
    herding_generator.manual_seed(seed + 7000 + 101 * target_a)
    t0 = time.time()
    continuous_latents, continuous_history = continuous_kernel_herding(
        weights=v_target,
        kernel=kernel,
        target_support=shared["Z_std_s"],
        num_samples=len(initial_particles),
        initial_candidates=shared["Z_std_s"],
        anchor_points=finite_latents,
        n_restarts=args.continuous_restarts,
        n_steps=args.continuous_opt_steps,
        learning_rate=args.continuous_learning_rate,
        restart_scale=args.continuous_restart_scale,
        support_chunk_size=(None if args.continuous_support_chunk_size == 0
                            else args.continuous_support_chunk_size),
        repulsion_chunk_size=(None if args.continuous_repulsion_chunk_size == 0
                              else args.continuous_repulsion_chunk_size),
        candidate_chunk_size=args.continuous_candidate_chunk_size,
        repulsion_denominator="algorithm2",
        domain=args.continuous_domain,
        domain_margin=args.continuous_domain_margin,
        lower_quantile=args.continuous_box_quantile,
        upper_quantile=1.0 - args.continuous_box_quantile,
        generator=herding_generator,
        require_unit_mass=True,
        verbose=False,
    )
    seconds_continuous = base.phase_seconds(t0, device)

    eval_pick = context["flow_eval_pick"]
    if eval_pick is None:
        finite_eval = finite_latents
        raw_eval = raw_images
        continuous_eval = continuous_latents
    else:
        finite_eval = finite_latents[eval_pick]
        raw_eval = raw_images[eval_pick.detach().cpu()]
        continuous_eval = continuous_latents[eval_pick]

    t0 = time.time()
    with torch.no_grad():
        finite_decoded = decode_latents(
            shared["vae"], shared["from_flow"](finite_eval), args.image_size)
    seconds_finite_decode = base.phase_seconds(t0, device)
    t0 = time.time()
    with torch.no_grad():
        continuous_decoded = decode_latents(
            shared["vae"], shared["from_flow"](continuous_eval), args.image_size)
    seconds_continuous_decode = base.phase_seconds(t0, device)
    seconds_decode = seconds_finite_decode + seconds_continuous_decode

    example_seed = args.example_seed + 1009 * seed + 101 * target_a
    reference_images = shared["Y_ref"][target_a]
    example_count = min(
        args.example_count, len(continuous_decoded), len(reference_images))
    seconds_examples = 0.0
    if example_count > 0:
        example_positions = deterministic_indices(
            len(continuous_decoded), example_count, example_seed)
        oracle_positions = deterministic_indices(
            len(reference_images), example_count, example_seed + 1)
        example_tensor = torch.as_tensor(
            example_positions, device=continuous_decoded.device,
            dtype=torch.long)
        oracle_tensor = torch.as_tensor(
            oracle_positions, device=reference_images.device,
            dtype=torch.long)
        t0 = time.time()
        EXAMPLE_GRIDS[key] = {
            "continuous": images_to_uint8(
                continuous_decoded[example_tensor],
                image_range="minus_one_one"),
            "finite_decoded": images_to_uint8(
                finite_decoded[example_tensor],
                image_range="minus_one_one"),
            "oracle": images_to_uint8(
                reference_images[oracle_tensor],
                image_range="minus_one_one"),
            "herding_positions": example_positions,
            "oracle_positions": oracle_positions,
            "seed": example_seed,
            "n_columns": args.example_grid_columns,
        }
        seconds_examples = base.phase_seconds(t0, device)

    # Continuous-output diagnostics: exact floating-point uniqueness, distance
    # from the logged latent bank, and encoder-decoder cycle error. They are
    # evaluated on the same fixed subset as pixel metrics when one is requested.
    t0 = time.time()
    nearest = nearest_support_distances(
        continuous_eval, shared["Z_std_s"],
        point_chunk=args.nearest_point_chunk,
        support_chunk=args.nearest_support_chunk,
    )
    continuous_cpu = continuous_eval.detach().cpu()
    n_unique_continuous = int(torch.unique(continuous_cpu, dim=0).shape[0])
    continuous_norms = torch.linalg.vector_norm(continuous_cpu, dim=1)
    cycle_latents = shared["to_flow"](
        encode_images(shared["vae"], continuous_decoded))
    cycle = (cycle_latents - continuous_eval).norm(dim=1).detach().cpu()
    seconds_continuous_diagnostics = base.phase_seconds(t0, device)

    ARTIFACTS[f"{key}_finite_candidate_positions"] = selected_cpu.numpy()
    ARTIFACTS[f"{key}_continuous_latents"] = (
        continuous_latents.detach().cpu().float().numpy())
    ARTIFACTS[f"{key}_continuous_nearest_support"] = nearest.numpy()
    ARTIFACTS[f"{key}_continuous_cycle_error"] = cycle.numpy()
    seed_indices = continuous_history.get("seed_candidate_indices")
    if seed_indices is not None:
        ARTIFACTS[f"{key}_continuous_seed_indices"] = (
            torch.as_tensor(seed_indices).detach().cpu().numpy())
    for name, value in continuous_history.items():
        if torch.is_tensor(value):
            ARTIFACTS[f"{key}_optimizer_{name}"] = value.detach().cpu().numpy()

    # Distribution metrics. The raw finite retrieval and its round trip stay
    # separate; the continuous result has only a decoded outcome-space view.
    t0 = time.time()
    finite_latent_sw2 = sliced_wasserstein(
        finite_latents, context["Z_ref_std"], p=2, seed=0).item()
    continuous_latent_sw2 = sliced_wasserstein(
        continuous_latents, context["Z_ref_std"], p=2, seed=0).item()
    raw_scores = base.pixel_scores("herding", raw_eval, context)
    finite_decoded_scores = base.pixel_scores(
        "herding_decoded", finite_decoded, context)
    continuous_scores = base.pixel_scores(
        "continuous", continuous_decoded, context)
    seconds_eval = base.phase_seconds(t0, device)

    t0 = time.time()
    raw_fid = base.fid_scores("herding", raw_eval, context, shared)
    finite_decoded_fid = base.fid_scores(
        "herding_decoded", finite_decoded, context, shared)
    continuous_fid = base.fid_scores(
        "continuous", continuous_decoded, context, shared)
    seconds_fid = base.phase_seconds(t0, device)
    seconds_eval += seconds_fid

    tolerance = args.continuous_support_tolerance
    support_matches = nearest <= tolerance
    n_novel = int((~support_matches).sum())
    score_gains = torch.as_tensor(
        continuous_history["score_improvements"]).detach().cpu().float()
    anchor_gains = torch.as_tensor(
        continuous_history["score_improvements_over_anchor"]).detach().cpu().float()
    initial_scores = torch.as_tensor(
        continuous_history["initial_scores"]).detach().cpu().float()
    anchor_scores = torch.as_tensor(
        continuous_history["anchor_scores"]).detach().cpu().float()
    final_scores = torch.as_tensor(
        continuous_history["final_scores"]).detach().cpu().float()
    distances_from_seed = torch.as_tensor(
        continuous_history["distances_from_seed"]).detach().cpu().float()
    optimizer_steps = torch.as_tensor(
        continuous_history["optimizer_steps"]).detach().cpu().float()
    gradient_norms = torch.as_tensor(
        continuous_history["final_gradient_norms"]).detach().cpu().float()
    boundary_fractions = torch.as_tensor(
        continuous_history["boundary_fractions"]).detach().cpu().float()
    selected_restarts = torch.as_tensor(
        continuous_history["selected_restarts"]).detach().cpu().long()

    row = dict(common)
    row.update(base.weight_diagnostics(v_target, on_arm_mask, shared["X_s"]))
    row.update({
        "estimator": "dr",
        "sampler": "finite_and_continuous_kernel_herding",
        "optimization_domain": "standardized_sd_vae_latent",
        "output_source": "logged_retrieval_roundtrip_and_continuous_decode",
        "n_candidates": int(len(shared["Z_std_s"])),
        "n_herding_steps": int(len(finite_latents)),
        "n_unique": finite_diagnostics["n_unique"],
        "unique_fraction": finite_diagnostics["unique_fraction"],
        "max_multiplicity": finite_diagnostics["max_multiplicity"],
        "candidate_coverage": finite_diagnostics["candidate_coverage"],
        "selected_target_arm_fraction": float(
            (selected_actions == target_a).float().mean()),
        "selected_blond_fraction": float(selected_blond.float().mean()),
        "target_blond_mass": float(
            (v_target * shared["X_s"].float().to(v_target.device)).sum()),
        "seconds_herding": round(seconds_finite, 4),
        "seconds_herding_decode": round(seconds_finite_decode, 4),
        "latent_sw2_herding": finite_latent_sw2,
        "herding_kernel": "riesz_s1",
        "herding_initializer": "current_finite_candidate_oracle",
        "herding_domain": args.continuous_domain,
        "finite_repulsion_denominator": "previous_count",
        "continuous_repulsion_denominator": continuous_history[
            "repulsion_denominator"],
        "n_continuous_samples": int(len(continuous_latents)),
        "continuous_opt_steps": args.continuous_opt_steps,
        "continuous_restarts": args.continuous_restarts,
        "continuous_learning_rate": args.continuous_learning_rate,
        "continuous_restart_scale": args.continuous_restart_scale,
        "continuous_support_chunk_size": args.continuous_support_chunk_size,
        "continuous_repulsion_chunk_size": args.continuous_repulsion_chunk_size,
        "continuous_candidate_chunk_size": args.continuous_candidate_chunk_size,
        "continuous_box_quantile": args.continuous_box_quantile,
        "continuous_domain_margin": args.continuous_domain_margin,
        "continuous_target_mass": target_mass,
        "continuous_target_abs_mass": target_abs_mass,
        "continuous_mean_seed_score": float(initial_scores.mean()),
        "continuous_mean_anchor_score": float(anchor_scores.mean()),
        "continuous_mean_final_score": float(final_scores.mean()),
        "continuous_mean_score_gain": float(score_gains.mean()),
        "continuous_min_score_gain": float(score_gains.min()),
        "continuous_fraction_improved": float(
            (score_gains > 1e-10).float().mean()),
        "continuous_mean_gain_over_anchor": float(anchor_gains.mean()),
        "continuous_min_gain_over_anchor": float(anchor_gains.min()),
        "continuous_fraction_improved_over_anchor": float(
            (anchor_gains > 1e-10).float().mean()),
        "continuous_fraction_candidate_restart": float(
            (selected_restarts == 0).float().mean()),
        "continuous_fraction_anchor_restart": float(
            (selected_restarts == 1).float().mean()),
        "continuous_fraction_random_restart": float(
            (selected_restarts >= 2).float().mean()),
        "continuous_mean_seed_displacement": float(distances_from_seed.mean()),
        "continuous_max_seed_displacement": float(distances_from_seed.max()),
        "continuous_mean_optimizer_steps": float(optimizer_steps.mean()),
        "continuous_mean_gradient_norm": float(gradient_norms.mean()),
        "continuous_max_gradient_norm": float(gradient_norms.max()),
        "continuous_mean_boundary_fraction": float(boundary_fractions.mean()),
        "continuous_n_unique": n_unique_continuous,
        "continuous_unique_fraction": n_unique_continuous / len(continuous_eval),
        "continuous_norm_mean": float(continuous_norms.mean()),
        "continuous_norm_median": float(continuous_norms.median()),
        "continuous_norm_max": float(continuous_norms.max()),
        "continuous_max_abs_coordinate": float(continuous_cpu.abs().max()),
        "continuous_nearest_support_mean": float(nearest.mean()),
        "continuous_nearest_support_median": float(nearest.median()),
        "continuous_nearest_support_max": float(nearest.max()),
        "continuous_support_match_fraction": float(support_matches.float().mean()),
        "continuous_n_novel": n_novel,
        "continuous_novel_fraction": n_novel / len(nearest),
        "continuous_support_tolerance": tolerance,
        "continuous_cycle_n": int(len(cycle)),
        "continuous_cycle_mean": float(cycle.mean()),
        "continuous_cycle_median": float(cycle.median()),
        "continuous_cycle_max": float(cycle.max()),
        "seconds_continuous_herding": round(seconds_continuous, 4),
        "seconds_continuous_decode": round(seconds_continuous_decode, 4),
        "seconds_continuous_diagnostics": round(seconds_continuous_diagnostics, 4),
        "latent_sw2_continuous": continuous_latent_sw2,
        "continuous_example_count": example_count,
        "continuous_example_seed": example_seed,
        "continuous_example_selection": "fixed_random_without_replacement",
        "continuous_example_rows": "continuous|finite_decoded|oracle",
        "continuous_example_file": (
            f"{csv_path(args).stem}_{key}_continuous_examples.png"
            if example_count > 0 else ""),
        "seconds_example_generation": round(seconds_examples, 4),
        "loss_initial": float("nan"),
        "loss_final": float("nan"),
        "mean_transport": float("nan"),
        # Base compatibility columns refer to the genuinely continuous output.
        "latent_sw2_flow": continuous_latent_sw2,
        "loss_initial_immd": immd["loss_initial"],
        "loss_final_immd": immd["loss_final"],
        "mean_transport_immd": immd["mean_transport"],
        "latent_sw2_immd": immd["latent_sw2_flow"],
        "seconds_immd_flow": immd["seconds_flow"],
        "seconds_immd_decode": immd["seconds_decode"],
        "seconds_immd_eval": immd["seconds_eval"],
        "seconds_immd_fid": immd["seconds_fid"],
        "n_flow_eval": int(len(continuous_eval)),
        # One CSV row carries all three paired samplers. Charge every phase once
        # so the base repeat-level wall-clock identity remains exact.
        "seconds_flow": round(
            seconds_finite + seconds_continuous
            + seconds_continuous_diagnostics + immd["seconds_flow"], 1),
        "seconds_decode": round(seconds_decode + immd["seconds_decode"], 1),
        "seconds_eval": round(
            seconds_eval + seconds_examples + immd["seconds_eval"], 1),
        "seconds_fid": round(seconds_fid + immd["seconds_fid"], 1),
    })
    for scores in (
        raw_scores, finite_decoded_scores, continuous_scores,
        raw_fid, finite_decoded_fid, continuous_fid,
    ):
        row.update(scores)
    row.update(rename_row(continuous_scores, "continuous", "flow"))
    row.update(rename_row(continuous_fid, "continuous", "flow"))
    for metric in ("sw2", "sw1", "mmd2"):
        row[f"{metric}_immd"] = immd[f"{metric}_flow"]
    for metric in ("sw2", "sw1"):
        row[f"{metric}_64_immd"] = immd[f"{metric}_64_flow"]
    for impl in base.FID_IMPLS:
        for target, source in (
            (f"fid_{impl}_immd", f"fid_{impl}_flow"),
            (f"fid_{impl}_immd_half", f"fid_{impl}_flow_half"),
            (f"fid64_{impl}_immd", f"fid64_{impl}_flow"),
            (f"fid64_{impl}_immd_half", f"fid64_{impl}_flow_half"),
        ):
            if source in immd:
                row[target] = immd[source]

    arm_seconds = {
        "seconds_ratio": arm_nuisance["seconds_ratio"],
        "seconds_cme": arm_nuisance["seconds_cme"],
        "seconds_eval_shared": context["seconds_eval_shared"],
        "seconds_fid_shared": context["seconds_fid_shared"],
    }
    return [row], arm_seconds


ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args():
    """Parse continuous-herding controls, then the shared CelebA CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--continuous_opt_steps", type=int, default=25)
    parser.add_argument("--continuous_restarts", type=int, default=8)
    parser.add_argument("--continuous_learning_rate", type=float, default=0.01)
    parser.add_argument("--continuous_restart_scale", type=float, default=0.05)
    parser.add_argument("--continuous_support_chunk_size", type=int, default=0,
                        help="0 evaluates the complete target support in one call")
    parser.add_argument("--continuous_repulsion_chunk_size", type=int, default=0,
                        help="0 evaluates all previous atoms in one call")
    parser.add_argument("--continuous_candidate_chunk_size", type=int, default=512)
    parser.add_argument("--continuous_domain", type=str,
                        default="unbounded",
                        choices=("unbounded", "support_quantile_box"))
    parser.add_argument("--continuous_domain_margin", type=float, default=0.05)
    parser.add_argument("--continuous_box_quantile", type=float, default=0.0,
                        help="symmetric tail fraction; 0 uses coordinate min/max")
    parser.add_argument("--continuous_mass_tolerance", type=float, default=1e-5)
    parser.add_argument("--continuous_support_tolerance", type=float, default=1e-4)
    parser.add_argument("--nearest_point_chunk", type=int, default=16)
    parser.add_argument("--nearest_support_chunk", type=int, default=2048)
    parser.add_argument("--example_count", type=int, default=16)
    parser.add_argument("--example_seed", type=int, default=20261120)
    parser.add_argument("--example_grid_columns", type=int, default=8)
    continuous, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    for key, value in vars(continuous).items():
        setattr(args, key, value)

    if base.requested_estimators(args) != ["dr"]:
        raise SystemExit("the matched herding ablation requires --estimators dr")
    if args.init != "observed":
        raise SystemExit("the matched herding ablation requires --init observed")
    if args.n_dr is not None:
        raise SystemExit("leave --n_dr unset so the complete logged support is used")
    if args.snapshot_every is not None:
        raise SystemExit("--snapshot_every does not apply to sequential herding")
    if args.continuous_opt_steps < 0 or args.continuous_restarts < 2:
        raise SystemExit(
            "continuous steps must be nonnegative and at least two restarts are "
            "required to include the finite-herding anchor")
    if not 0.0 <= args.continuous_box_quantile < 0.5:
        raise SystemExit("--continuous_box_quantile must lie in [0, 0.5)")
    if args.example_count < 0 or args.example_grid_columns < 1:
        raise SystemExit(
            "--example_count must be nonnegative and grid columns positive")
    return args


def csv_path(args) -> Path:
    """Keep continuous-herding artifacts separate from existing baselines."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    name = (
        f"celeba_continuous_kernel_herding_{args.tag}_sz{args.image_size}"
        f"_nobs{args.n_obs}_M{args.n_particles}_arms{arms}_ds{seeds}"
        f"_{args.dr_fit}_{args.solver}_domain{args.continuous_domain}"
        f"_steps{args.continuous_opt_steps}"
        f"_r{args.continuous_restarts}.csv"
    )
    return Path(args.out_dir) / name


def write_artifacts(out_path: Path) -> None:
    """Save finite indices and generated continuous latents for audit."""
    if ARTIFACTS:
        np.savez_compressed(
            out_path.with_name(out_path.stem + "_artifacts.npz"), **ARTIFACTS)
    if EXAMPLE_GRIDS:
        arrays = {}
        for key, examples in EXAMPLE_GRIDS.items():
            for name in (
                    "continuous", "finite_decoded", "oracle",
                    "herding_positions", "oracle_positions"):
                arrays[f"{key}_{name}"] = examples[name]
            arrays[f"{key}_row_order"] = np.asarray(
                ["continuous", "finite_decoded", "oracle"])
            save_labeled_image_grid(
                [("continuous", examples["continuous"]),
                 ("finite decoded", examples["finite_decoded"]),
                 ("oracle", examples["oracle"])],
                out_path.with_name(
                    f"{out_path.stem}_{key}_continuous_examples.png"),
                n_columns=examples["n_columns"],
            )
        np.savez_compressed(
            out_path.with_name(out_path.stem + "_examples.npz"), **arrays)


def main() -> None:
    if "--out_dir" not in sys.argv:
        sys.argv.extend(["--out_dir", str(HERE / "results" / "CelebA")])

    base.COLUMNS = base.COLUMNS + EXTRA_COLUMNS
    base.parse_args = parse_args
    base.csv_path = csv_path
    base.run_arm = run_arm
    base.write_particle_indices = write_artifacts
    base.main()


if __name__ == "__main__":
    main()
