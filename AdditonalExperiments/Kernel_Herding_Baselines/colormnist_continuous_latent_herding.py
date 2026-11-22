"""Matched continuous latent-space kernel herding for ColorMNIST.

The process produces three outputs from one fitted representation and one DR
target: DR--IMMD Flow, finite-candidate latent herding, and continuous latent
herding followed by decoding.  Continuous herding implements the CPME paper's
Algorithm 2 over the unrestricted autoencoder latent space.  The finite-candidate
atoms are used as numerical initializers and fallbacks, but do not constrain the
returned continuous atoms.  The runner also saves a fixed random subset of
decoded examples for qualitative inspection.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY_ROOT = HERE.parents[1]
DR_SIM_DIR = DEPLOY_ROOT / "Simulations" / "ColorMNIST" / "dr_mmd_flow"
sys.path.insert(0, str(DR_SIM_DIR))

import colormnist_dr_mmd_flow_simulation as base
import colormnist_kernel_herding as finite

from autoencoder_utils import encode_in_batches
from continuous_latent_herding import continuous_kernel_herding
from data_utils import sample_interventional, to_image_range, to_model_range
from kernel_utils import RBF
from metric_utils import sliced_wasserstein
from qualitative_samples import (
    deterministic_indices,
    images_to_uint8,
    save_labeled_image_grid,
)


EXTRA_COLUMNS = [
    "herding_n_restarts", "herding_opt_steps", "herding_lr",
    "herding_restart_scale", "continuous_domain", "herding_box_quantile",
    "herding_box_padding",
    "herding_target_chunk", "herding_repulsion_chunk",
    "herding_candidate_chunk", "herding_call_seed",
    "continuous_repulsion_denominator", "finite_repulsion_denominator",
    "seconds_continuous_herding", "seconds_continuous_decode",
    "seconds_continuous_eval", "seconds_continuous_fid",
    "objective_final_continuous", "latent_sw2_continuous",
    "colour_mean_continuous", "colour_std_continuous",
    "colour_w2_continuous", "sw1_continuous", "sw2_continuous",
    "mmd2_continuous", "fid_legacy_continuous", "fid_clean_continuous",
    "continuous_unique_count", "continuous_unique_fraction",
    "continuous_norm_mean", "continuous_norm_median", "continuous_norm_max",
    "continuous_max_abs_coordinate",
    "nearest_logged_mean", "nearest_logged_median", "nearest_logged_max",
    "fraction_within_logged_tolerance", "logged_tolerance",
    "cycle_error_mean", "cycle_error_median", "cycle_error_max",
    "score_initial_mean", "score_anchor_mean", "score_final_mean",
    "score_improvement_mean", "score_improvement_min", "score_improvement_max",
    "score_improvement_over_anchor_mean",
    "score_improvement_over_anchor_min",
    "score_improvement_over_anchor_max",
    "optimizer_steps_mean", "optimizer_gradient_norm_mean",
    "optimizer_boundary_fraction", "selected_restart_mean",
    "continuous_example_count", "continuous_example_seed",
    "continuous_example_selection", "continuous_example_rows",
    "continuous_example_file", "seconds_example_generation",
]

CONTINUOUS_ARTIFACTS = {}
CONTINUOUS_EXAMPLES = {}
CALL_INDEX = 0
BASE_WRITE_PARTICLES = base.write_particle_indices
ORIGINAL_PARSE_ARGS = finite.ORIGINAL_PARSE_ARGS


def nearest_logged_distances(atoms: torch.Tensor, support: torch.Tensor,
                             chunk_size: int = 256) -> torch.Tensor:
    """Nearest logged-latent distance without materializing the full matrix."""
    distances = []
    for start in range(0, len(atoms), chunk_size):
        stop = min(start + chunk_size, len(atoms))
        distances.append(torch.cdist(atoms[start:stop], support).amin(dim=1))
    return torch.cat(distances)


def _diagnostic_tensor(diagnostics: dict, *names: str):
    for name in names:
        value = diagnostics.get(name)
        if torch.is_tensor(value):
            return value
    return None


def run_arm(target_a: int, ae, pools: list, data: torch.Tensor,
            causal_idx: torch.Tensor, A_obs: torch.Tensor, A_s: torch.Tensor,
            Z_s: torch.Tensor, e1: torch.Tensor, K_x, K_a, gram, gram_x,
            solver: str, fold_ids: np.ndarray, fold_lambdas: dict, args,
            seed: int, device: torch.device, fid_scorers: dict = None,
            nosplit_lambda: float = None) -> tuple:
    """Run the existing paired comparison, then add continuous latent herding."""
    global CALL_INDEX
    captured = {}
    original_herding = finite.kernel_herding
    original_target_norm = finite.target_embedding_norm

    def capture_herding(*call_args, **call_kwargs):
        output = original_herding(*call_args, **call_kwargs)
        captured["weights"] = call_args[0]
        captured["kernel"] = call_args[1]
        captured["support"] = call_args[2]
        captured["finite_atoms"] = output[0] if isinstance(output, tuple) else output
        return output

    def capture_target_norm(*call_args, **call_kwargs):
        value = original_target_norm(*call_args, **call_kwargs)
        captured["target_norm"] = value
        return value

    finite.kernel_herding = capture_herding
    finite.target_embedding_norm = capture_target_norm
    try:
        rows, arm_seconds = finite.run_arm(
            target_a, ae, pools, data, causal_idx, A_obs, A_s, Z_s, e1,
            K_x, K_a, gram, gram_x, solver, fold_ids, fold_lambdas, args,
            seed, device, fid_scorers, nosplit_lambda)
    finally:
        finite.kernel_herding = original_herding
        finite.target_embedding_norm = original_target_norm

    if len(rows) != 1 or rows[0]["estimator"] != "dr":
        raise RuntimeError("continuous-herding comparison expects one DR row per arm")
    row = rows[0]
    v_target = captured["weights"]
    kernel = captured["kernel"]
    finite_atoms = captured["finite_atoms"]
    n_particles = int(row["n_particles"])

    call_seed = args.herding_seed + CALL_INDEX
    call_key = f"ds{seed // 1000}_r{seed % 1000}_a{target_a}"
    CALL_INDEX += 1

    t0 = time.time()
    atoms, diagnostics = continuous_kernel_herding(
        v_target, kernel, Z_s, n_particles,
        initial_candidates=Z_s,
        anchor_points=finite_atoms,
        n_restarts=args.herding_n_restarts,
        n_steps=args.herding_opt_steps,
        learning_rate=args.herding_lr,
        restart_scale=args.herding_restart_scale,
        support_chunk_size=args.herding_target_chunk,
        repulsion_chunk_size=args.herding_repulsion_chunk,
        candidate_chunk_size=args.herding_candidate_chunk,
        domain=args.herding_domain,
        domain_margin=args.herding_box_padding,
        lower_quantile=args.herding_box_quantile,
        upper_quantile=1.0 - args.herding_box_quantile,
        require_unit_mass=False,
        repulsion_denominator=args.herding_repulsion_denominator,
        gradient_tolerance=args.herding_gradient_tolerance,
        generator=torch.Generator(device=device).manual_seed(call_seed),
    )
    base.sync_device(device)
    seconds_continuous = time.time() - t0

    eval_idx = torch.arange(n_particles, device=device)
    if args.eval_particles is not None and n_particles > args.eval_particles:
        eval_idx = torch.randperm(
            n_particles,
            generator=torch.Generator(device=device).manual_seed(seed + 11),
            device=device)[:args.eval_particles]

    t0 = time.time()
    with torch.no_grad():
        decoded_model = ae.decode(atoms[eval_idx])
        decoded = to_image_range(decoded_model)
        recycled = encode_in_batches(ae.encode, decoded_model)
        cycle = torch.linalg.vector_norm(recycled - atoms[eval_idx], dim=1)
    base.sync_device(device)
    seconds_decode = time.time() - t0

    # Recreate the same deterministic oracle draw used inside finite.run_arm.
    _, Y_oracle = sample_interventional(
        pools, args.n_ref, a=target_a, tau=args.tau, alpha=args.alpha,
        k_mask=args.k_mask,
        generator=torch.Generator(device=device).manual_seed(seed + 9))
    with torch.no_grad():
        Z_oracle = encode_in_batches(ae.encode, to_model_range(Y_oracle))
    kernel_pixel = RBF(use_length_scale_heuristic=True)
    _ = kernel_pixel(Y_oracle.reshape(len(Y_oracle), -1))

    example_seed = args.example_seed + 1009 * seed + 101 * target_a
    example_count = min(args.example_count, n_particles, len(Y_oracle))
    seconds_examples = 0.0
    if example_count > 0:
        example_positions = deterministic_indices(
            n_particles, example_count, example_seed)
        oracle_positions = deterministic_indices(
            len(Y_oracle), example_count, example_seed + 1)
        example_tensor = torch.as_tensor(
            example_positions, device=device, dtype=torch.long)
        oracle_tensor = torch.as_tensor(
            oracle_positions, device=Y_oracle.device, dtype=torch.long)
        t_examples = time.time()
        with torch.no_grad():
            continuous_examples = to_image_range(ae.decode(
                atoms[example_tensor]))
            finite_examples = to_image_range(ae.decode(
                finite_atoms[example_tensor]))
            oracle_examples = Y_oracle[oracle_tensor]
        base.sync_device(device)
        seconds_examples = time.time() - t_examples
        CONTINUOUS_EXAMPLES[call_key] = {
            "continuous": images_to_uint8(
                continuous_examples, image_range="zero_one"),
            "finite_decoded": images_to_uint8(
                finite_examples, image_range="zero_one"),
            "oracle": images_to_uint8(
                oracle_examples, image_range="zero_one"),
            "herding_positions": example_positions,
            "oracle_positions": oracle_positions,
            "seed": example_seed,
            "n_columns": args.example_grid_columns,
        }

    t0 = time.time()
    with torch.no_grad():
        objective = finite.embedding_objective(
            kernel, atoms, Z_s, v_target, captured["target_norm"],
            chunk_size=args.herding_target_chunk)
        nearest = nearest_logged_distances(atoms, Z_s)
    continuous_metrics = base.evaluate_images(decoded, Y_oracle, kernel_pixel)
    seconds_eval = time.time() - t0

    t_fid = time.time()
    for impl, scorer in (fid_scorers or {}).items():
        oracle_features = scorer.features(Y_oracle, image_range="zero_one")
        row[f"fid_{impl}_continuous"] = scorer.distance_from_features(
            scorer.features(decoded, image_range="zero_one"), oracle_features)
    base.sync_device(device)
    seconds_fid = time.time() - t_fid
    seconds_eval += seconds_fid

    unique_count = int(torch.unique(atoms, dim=0).shape[0])
    atom_norms = torch.linalg.vector_norm(atoms, dim=1)
    row.update({
        "herding_n_restarts": args.herding_n_restarts,
        "herding_opt_steps": args.herding_opt_steps,
        "herding_lr": args.herding_lr,
        "herding_restart_scale": args.herding_restart_scale,
        "continuous_domain": args.herding_domain,
        "herding_box_quantile": args.herding_box_quantile,
        "herding_box_padding": args.herding_box_padding,
        "herding_target_chunk": args.herding_target_chunk,
        "herding_repulsion_chunk": args.herding_repulsion_chunk,
        "herding_candidate_chunk": args.herding_candidate_chunk,
        "herding_call_seed": call_seed,
        "continuous_repulsion_denominator": diagnostics[
            "repulsion_denominator"],
        "finite_repulsion_denominator": "previous_count",
        "seconds_continuous_herding": round(seconds_continuous, 4),
        "seconds_continuous_decode": round(seconds_decode, 4),
        "seconds_continuous_eval": round(seconds_eval, 4),
        "seconds_continuous_fid": round(seconds_fid, 4),
        "objective_final_continuous": float(objective),
        "latent_sw2_continuous": sliced_wasserstein(
            atoms, Z_oracle, p=2, seed=0).item(),
        "continuous_unique_count": unique_count,
        "continuous_unique_fraction": unique_count / len(atoms),
        "continuous_norm_mean": float(atom_norms.mean()),
        "continuous_norm_median": float(atom_norms.median()),
        "continuous_norm_max": float(atom_norms.max()),
        "continuous_max_abs_coordinate": float(atoms.abs().max()),
        "nearest_logged_mean": float(nearest.mean()),
        "nearest_logged_median": float(nearest.median()),
        "nearest_logged_max": float(nearest.max()),
        "fraction_within_logged_tolerance": float(
            (nearest <= args.logged_tolerance).float().mean()),
        "logged_tolerance": args.logged_tolerance,
        "cycle_error_mean": float(cycle.mean()),
        "cycle_error_median": float(cycle.median()),
        "cycle_error_max": float(cycle.max()),
    })
    row.update({f"{key}_continuous": value
                for key, value in continuous_metrics.items()})

    initial_score = _diagnostic_tensor(
        diagnostics, "initial_scores", "initial_score", "seed_score")
    anchor_score = _diagnostic_tensor(diagnostics, "anchor_scores")
    final_score = _diagnostic_tensor(
        diagnostics, "final_scores", "final_score", "selected_score")
    improvement = _diagnostic_tensor(
        diagnostics, "score_improvements", "score_improvement", "improvement")
    improvement_over_anchor = _diagnostic_tensor(
        diagnostics, "score_improvements_over_anchor")
    n_steps = _diagnostic_tensor(
        diagnostics, "optimizer_steps", "n_steps", "steps")
    gradient_norm = _diagnostic_tensor(
        diagnostics, "final_gradient_norms", "final_gradient_norm", "gradient_norm")
    selected_restart = _diagnostic_tensor(
        diagnostics, "selected_restarts", "selected_restart", "restart_index")
    if initial_score is not None:
        row["score_initial_mean"] = float(initial_score.float().mean())
    if anchor_score is not None:
        row["score_anchor_mean"] = float(anchor_score.float().mean())
    if final_score is not None:
        row["score_final_mean"] = float(final_score.float().mean())
    if improvement is not None:
        row.update({
            "score_improvement_mean": float(improvement.float().mean()),
            "score_improvement_min": float(improvement.float().min()),
            "score_improvement_max": float(improvement.float().max()),
        })
    if improvement_over_anchor is not None:
        row.update({
            "score_improvement_over_anchor_mean": float(
                improvement_over_anchor.float().mean()),
            "score_improvement_over_anchor_min": float(
                improvement_over_anchor.float().min()),
            "score_improvement_over_anchor_max": float(
                improvement_over_anchor.float().max()),
        })
    if n_steps is not None:
        row["optimizer_steps_mean"] = float(n_steps.float().mean())
    if gradient_norm is not None:
        row["optimizer_gradient_norm_mean"] = float(
            gradient_norm.float().mean())
    boundary = _diagnostic_tensor(
        diagnostics, "boundary_fractions", "boundary_fraction")
    row["optimizer_boundary_fraction"] = (
        float(boundary.float().mean()) if boundary is not None else float("nan"))
    if selected_restart is not None:
        row["selected_restart_mean"] = float(selected_restart.float().mean())
    row.update({
        "continuous_example_count": example_count,
        "continuous_example_seed": example_seed,
        "continuous_example_selection": "fixed_random_without_replacement",
        "continuous_example_rows": "continuous|finite_decoded|oracle",
        "continuous_example_file": (
            f"{csv_path(args).stem}_{call_key}_continuous_examples.png"
            if example_count > 0 else ""),
        "seconds_example_generation": round(seconds_examples, 4),
    })

    # The compatibility clocks represent every method executed in this one row.
    row["seconds_flow"] = round(
        row["seconds_flow"] + seconds_continuous, 1)
    row["seconds_decode"] = round(row["seconds_decode"] + seconds_decode, 1)
    row["seconds_eval"] = round(
        row["seconds_eval"] + seconds_eval + seconds_examples, 1)
    row["seconds_fid"] = round(row["seconds_fid"] + seconds_fid, 1)

    CONTINUOUS_ARTIFACTS[f"{call_key}_atoms"] = atoms.detach().cpu().numpy()
    CONTINUOUS_ARTIFACTS[f"{call_key}_finite_initializers"] = (
        finite_atoms.detach().cpu().numpy())
    CONTINUOUS_ARTIFACTS[f"{call_key}_nearest_logged"] = nearest.cpu().numpy()
    CONTINUOUS_ARTIFACTS[f"{call_key}_cycle_error"] = cycle.cpu().numpy()
    for key, value in diagnostics.items():
        if torch.is_tensor(value):
            CONTINUOUS_ARTIFACTS[f"{call_key}_{key}"] = value.detach().cpu().numpy()
    return rows, arm_seconds


def parse_args() -> argparse.Namespace:
    """Parse new optimizer controls before delegating the shared experiment CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--herding_n_restarts", type=int, default=2)
    parser.add_argument("--herding_opt_steps", type=int, default=10)
    parser.add_argument("--herding_lr", type=float, default=0.02)
    parser.add_argument("--herding_restart_scale", type=float, default=0.05)
    parser.add_argument(
        "--herding_domain", type=str, default="unbounded",
        choices=("support_quantile_box", "unbounded"))
    parser.add_argument(
        "--herding_repulsion_denominator", type=str, default="algorithm2",
        choices=("algorithm2", "previous_count"))
    parser.add_argument("--herding_box_quantile", type=float, default=0.0)
    parser.add_argument("--herding_box_padding", type=float, default=0.05)
    parser.add_argument("--herding_target_chunk", type=int, default=2048)
    parser.add_argument("--herding_repulsion_chunk", type=int, default=2048)
    parser.add_argument("--herding_candidate_chunk", type=int, default=512)
    parser.add_argument("--herding_gradient_tolerance", type=float, default=0.0)
    parser.add_argument("--herding_seed", type=int, default=20261120)
    parser.add_argument("--logged_tolerance", type=float, default=1e-6)
    parser.add_argument("--example_count", type=int, default=16)
    parser.add_argument("--example_seed", type=int, default=20261120)
    parser.add_argument("--example_grid_columns", type=int, default=8)
    custom, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    for key, value in vars(custom).items():
        setattr(args, key, value)

    if base.resolve_estimators(args) != ["dr"]:
        raise SystemExit("continuous-herding comparisons require --estimators dr")
    if args.init != "observed":
        raise SystemExit("the matched comparison requires --init observed")
    if args.n_dr is not None:
        raise SystemExit("leave --n_dr unset so every method uses the full DR support")
    if args.snapshot_every is not None:
        raise SystemExit("--snapshot_every does not apply to sequential herding")
    if not 0.0 <= args.herding_box_quantile < 0.5:
        raise SystemExit("--herding_box_quantile must lie in [0, 0.5)")
    if args.example_count < 0 or args.example_grid_columns < 1:
        raise SystemExit(
            "--example_count must be nonnegative and grid columns positive")
    return args


def csv_path(args) -> Path:
    """A filename that cannot collide with finite-only or flow artifacts."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    arms = "".join(str(arm) for arm in args.target_arms)
    name = (
        f"colormnist_continuous_herding_{args.tag}_nobs{args.n_obs}"
        f"_M{args.n_particles}_w{args.w_conf:g}_a{args.alpha:g}"
        f"_k{args.k_mask:g}_tau{args.tau:g}_arms{arms}_ds{seeds}"
        f"_{args.dr_fit}_{args.solver}_domain{args.herding_domain}"
        f"_r{args.herding_n_restarts}"
        f"_it{args.herding_opt_steps}.csv")
    return Path(args.out_dir) / name


def write_artifacts(out_path: Path) -> None:
    """Retain initial-particle, finite-selection, and continuous-atom artifacts."""
    BASE_WRITE_PARTICLES(out_path)
    finite.write_herding_indices(out_path)
    if CONTINUOUS_ARTIFACTS:
        np.savez(out_path.with_name(out_path.stem + "_continuous_atoms.npz"),
                 **CONTINUOUS_ARTIFACTS)
    if CONTINUOUS_EXAMPLES:
        arrays = {}
        for key, examples in CONTINUOUS_EXAMPLES.items():
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
        sys.argv.extend(["--out_dir", str(HERE / "results" / "ColorMNIST")])
    base.COLUMNS = base.COLUMNS + finite.EXTRA_COLUMNS + EXTRA_COLUMNS
    base.parse_args = parse_args
    base.csv_path = csv_path
    base.run_arm = run_arm
    base.write_particle_indices = write_artifacts
    base.main()


if __name__ == "__main__":
    main()
