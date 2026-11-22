"""Matched kernel-herding baselines for the paper's two-dimensional DGP.

The data, projected-Nystrom DR coefficients, references, initial particles and
metrics are delegated to the paper simulation.  Only its two particle samplers
are replaced by finite-candidate greedy selection from all logged outcomes.

The RBF run uses the estimated DR coefficients verbatim.  The Riesz kernel
``-||y-y'||`` is conditionally positive definite, so its energy-discrepancy
interpretation requires the target and empirical measures to have equal mass.
We therefore divide the Riesz coefficients by their (nonzero) total mass and
record both masses in the CSV.  This is an explicit probability normalization,
not clipping, truncation, or a change to the fitted nuisances.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY_ROOT = HERE.parents[1]
DR_SIM_DIR = DEPLOY_ROOT / "Simulations" / "DecFM_2D" / "dr_mmd_flow"
sys.path.insert(0, str(DR_SIM_DIR))

import decfm_2d_dr_flow_simulation as base

from causality_utils import herding_diagnostics
from kernel_utils import RieszKernel


CHUNK_SIZE = int(os.environ.get("KERNEL_HERDING_CHUNK_SIZE", "512"))
EXTRA_COLUMNS = [
    "base_sampler", "candidate_set", "output_source", "n_candidates",
    "n_herding_steps", "n_selected", "n_unique", "unique_fraction", "max_multiplicity",
    "candidate_coverage", "selected_target_arm_fraction",
    "target_mass_handling", "sum_v_raw", "sum_v_used", "sum_abs_v_used",
    "embedding_chunk_size", "target_quadratic_objective",
    "seconds_embedding_evaluation", "seconds_greedy_selection",
    "seconds_herding",
]

SELECTION_QUEUE = []
HERDING_SELECTIONS = {}


def kernel_herding_chunked(weights, kernel, support, candidates, num_samples):
    """Greedy herding without materialising the n-by-n kernel matrix at once."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if len(weights) != len(support):
        raise ValueError("weights and support must have the same length")

    t0 = time.time()
    chi = torch.empty(len(candidates), dtype=support.dtype, device=support.device)
    with torch.no_grad():
        for start in range(0, len(candidates), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, len(candidates))
            chi[start:stop] = kernel(support, candidates[start:stop]).T @ weights
    seconds_embedding = time.time() - t0

    t0 = time.time()
    repulsion = torch.zeros_like(chi)
    indices = torch.empty(num_samples, dtype=torch.long, device=candidates.device)
    with torch.no_grad():
        for step in range(num_samples):
            objective = chi if step == 0 else chi - repulsion / step
            index = int(objective.argmax())
            indices[step] = index
            repulsion.add_(kernel(candidates[index:index + 1], candidates)[0])
        # This is an RKHS squared discrepancy for RBF.  For Riesz it is the
        # corresponding equal-mass energy quadratic (the weights are normalized).
        target_norm = weights @ chi
        empirical_norm = kernel(candidates[indices], candidates[indices]).mean()
        target_quadratic = empirical_norm - 2.0 * chi[indices].mean() + target_norm
    seconds_greedy = time.time() - t0
    return candidates[indices], indices, {
        "seconds_embedding_evaluation": seconds_embedding,
        "seconds_greedy_selection": seconds_greedy,
        "target_quadratic_objective": float(target_quadratic),
    }


def run_herding(base_sampler, kernel, support, initial_particles, target_weights,
                mass_handling):
    """Run one kernel variant and append its selection audit to the call queue."""
    if target_weights is None:
        raise ValueError("kernel herding requires target_weights")
    raw_mass = target_weights.sum()
    if mass_handling == "unit_mass":
        if not torch.isfinite(raw_mass) or float(raw_mass) <= 1e-12:
            raise ValueError("cannot normalize Riesz target with nonpositive/nonfinite mass")
        used_weights = target_weights / raw_mass
    else:
        used_weights = target_weights

    t0 = time.time()
    selected, indices, timing = kernel_herding_chunked(
        used_weights, kernel, support, support, len(initial_particles))
    seconds_total = time.time() - t0
    diagnostics = herding_diagnostics(indices, n_candidates=len(support))
    SELECTION_QUEUE.append({
        "base_sampler": base_sampler,
        "indices": indices.detach().cpu().numpy(),
        "n_candidates": int(len(support)),
        "target_mass_handling": mass_handling,
        "sum_v_raw": float(raw_mass),
        "sum_v_used": float(used_weights.sum()),
        "sum_abs_v_used": float(used_weights.abs().sum()),
        "seconds_herding": seconds_total,
        **diagnostics,
        **timing,
    })
    return selected, {}


def rbf_herding(kernel, target_samples, initial_particles, target_weights=None,
                **kwargs):
    """RBF herding against the exact estimated DR Hilbert element."""
    return run_herding("rbf_euler", kernel, target_samples, initial_particles,
                       target_weights, "raw_dr_coefficients")


def riesz_herding(initial_particles, target, target_weights=None, **kwargs):
    """Equal-mass Riesz-energy herding against normalized DR coefficients."""
    return run_herding("riesz_adamw", RieszKernel(), target, initial_particles,
                       target_weights, "unit_mass")


ORIGINAL_PARSE_ARGS = base.parse_args
ORIGINAL_RUN_REPEAT = base.run_repeat
ORIGINAL_WRITE_PARTICLE_INDICES = base.write_particle_indices


def parse_args():
    """Retain the shared CLI and enforce the finite-candidate comparison contract."""
    args = ORIGINAL_PARSE_ARGS()
    if args.estimators != ["dr"]:
        raise SystemExit("kernel-herding headline runs require --estimators dr")
    if args.init != "observed":
        raise SystemExit("kernel herding requires --init observed for this comparison")
    if args.n_dr is not None:
        raise SystemExit("leave --n_dr unset so candidates are all logged outcomes")
    return args


def run_repeat(data, setup, args, data_seed, repeat):
    """Enrich the shared simulation rows with selection-specific diagnostics."""
    SELECTION_QUEUE.clear()
    rows = ORIGINAL_RUN_REPEAT(data, setup, args, data_seed, repeat)
    if len(rows) != len(SELECTION_QUEUE):
        raise RuntimeError("herding call/row count mismatch")

    actions = data["A_logged"].reshape(-1).cpu().numpy()
    for row, record in zip(rows, SELECTION_QUEUE):
        if row["sampler"] != record["base_sampler"]:
            raise RuntimeError("herding call order does not match simulation row order")
        indices = record.pop("indices")
        target_a = int(row["target_a"])
        sampler = ("kernel_herding_rbf" if row["sampler"] == "rbf_euler"
                   else "kernel_herding_riesz")
        key = f"ds{data_seed}_r{repeat}_a{target_a}_{sampler}"
        HERDING_SELECTIONS[key] = indices
        row.update(record)
        row.update({
            "sampler": sampler,
            "candidate_set": "all_logged_outcomes",
            "output_source": "raw_logged_outcome_no_encoder",
            "n_herding_steps": int(len(indices)),
            "selected_target_arm_fraction": float(
                np.mean(actions[indices] == target_a)),
            "embedding_chunk_size": CHUNK_SIZE,
            "seconds_herding": round(record["seconds_herding"], 1),
            # Herding selects an unordered empirical measure; there is no
            # particle-wise transport pairing or optimizer step count.
            "flow_iterations": 0,
            "step_or_lr": float("nan"),
            "mean_transport": float("nan"),
        })
    return rows


def csv_path(args):
    """Separate the rebuttal artifacts from the paper simulation results."""
    seeds = "-".join(str(seed) for seed in args.data_seeds)
    kernels = "-".join("rbf" if s == "rbf_euler" else "riesz"
                       for s in args.samplers)
    name = (
        f"decfm2d_kernel_herding_{args.tag}_nobs{args.n_obs}_M{args.n_particles}"
        f"_kernels{kernels}_ds{seeds}_{args.dr_fit}_{args.solver}.csv"
    )
    return Path(args.out_dir) / name


def write_indices(out_path):
    """Write both matched starting indices and kernel-herding selections."""
    ORIGINAL_WRITE_PARTICLE_INDICES(out_path)
    if HERDING_SELECTIONS:
        np.savez(out_path.with_name(out_path.stem + "_selections.npz"),
                 **HERDING_SELECTIONS)


def has_flag(flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def add_default(flag, value):
    if not has_flag(flag):
        sys.argv.extend([flag, str(value)])


def main():
    # Standalone defaults are the paper/rebuttal cell; every value remains
    # overrideable for a smoke test.
    for flag, value in (("--n_obs", 20000), ("--n_particles", 2000),
                        ("--n_ref", 2000), ("--n_repeats", 1),
                        ("--solver", "projected_nystrom"),
                        ("--n_landmarks", 1024), ("--dr_fit", "nosplit"),
                        ("--init", "observed"), ("--tag", "rebuttal")):
        add_default(flag, value)
    if not has_flag("--out_dir"):
        sys.argv.extend(["--out_dir", str(HERE / "results" / "DecFM_2D")])

    base.COLUMNS = base.COLUMNS + EXTRA_COLUMNS
    base.parse_args = parse_args
    base.csv_path = csv_path
    base.run_repeat = run_repeat
    base.write_particle_indices = write_indices
    base.mmd_gradient_flow = rbf_herding
    base.adamw_riesz_flow = riesz_herding
    base.main()


if __name__ == "__main__":
    main()
