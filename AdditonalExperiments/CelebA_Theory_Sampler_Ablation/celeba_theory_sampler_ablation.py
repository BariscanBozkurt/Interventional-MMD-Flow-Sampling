"""Paired CelebA sampler ablation: Riesz--AdamW versus IMQ--Euler.

Every sampler in an arm receives the same full-sample projected-Nystrom DR
embedding, observed-arm initial particles, held-out references, and evaluation
subsets.  This isolates the practical Riesz--AdamW particle optimizer from the
smooth positive-definite IMQ--Euler construction used by the paper's analysis.

The two IMQ rows differ only in whether the analysis-form injected Gaussian
query noise is enabled.  Their practical Euler step was selected once from the
seed-0 arm-0 screening pilot and is then frozen across both arms and all three
data seeds.  This experiment tests the theory-matched *algorithmic form*; it
does not claim that the chosen finite step verifies every conservative
sufficient condition of the finite-particle theorem.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parents[1]
SRC = DEPLOY / "src"
BASE_PATH = (
    DEPLOY
    / "Simulations"
    / "CelebA"
    / "dr_mmd_flow"
    / "celeba_dr_riesz_flow_simulation.py"
)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kernel_utils import IMQKernel, imq_c_from_median
from mmd_flow_utils import witness_gradient

from imq_euler import (
    euler_flow,
    scaled_witness_gradient,
    target_self_inner_product,
)


def _load_base_module():
    spec = importlib.util.spec_from_file_location("celeba_headline_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import the CelebA runner from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


N_ITERATIONS = 4000
IMQ_C_MULTIPLIER = 1.0
IMQ_ETA = 0.5
NOISE_ALPHA = 0.25
N_QUALITATIVE = 8

SAMPLER_COLUMNS = [
    "sampler",
    "kernel",
    "optimizer",
    "theory_relation",
    "imq_c_multiplier",
    "imq_c",
    "imq_eta",
    "imq_raw_euler_step",
    "noise_beta0",
    "noise_alpha",
    "noise_schedule",
    "gradient_check_relative_error",
    "seconds_imq_bandwidth",
    "seconds_imq_target_self",
]

column_position = base.COLUMNS.index("estimator") + 1
base.COLUMNS = (
    base.COLUMNS[:column_position]
    + SAMPLER_COLUMNS
    + base.COLUMNS[column_position:]
)


# Small decoded panels only.  Full 2,000-image clouds are scored and discarded.
QUALITATIVE: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
GRADIENT_RELATIVE_ERROR = float("nan")


def _validate_protocol(args) -> None:
    """Refuse silent departures from the paired headline configuration."""
    expected = {
        "n_obs": 20_000,
        "n_ref": 2_000,
        "image_size": 256,
        "n_particles": 2_000,
        "eval_particles": 2_000,
        "eval_max": 2_000,
        "n_landmarks": 1_024,
        "dr_fit": "nosplit",
        "init": "observed",
        "flow_iterations": N_ITERATIONS,
        "learning_rate": 0.001,
        "warmup_steps": 100,
        "lr_schedule": "cosine",
    }
    mismatches = [
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if base.resolve_solver(args) != "projected_nystrom":
        mismatches.append(
            f"solver={base.resolve_solver(args)!r} (expected 'projected_nystrom')"
        )
    if base.requested_estimators(args) != ["dr"]:
        mismatches.append(
            f"estimators={base.requested_estimators(args)!r} (expected ['dr'])"
        )
    if list(args.target_arms) != [0, 1]:
        mismatches.append(f"target_arms={args.target_arms!r} (expected [0, 1])")
    if args.fid != "both":
        mismatches.append(f"fid={args.fid!r} (expected 'both')")
    if mismatches:
        raise ValueError(
            "This rebuttal runner is intentionally locked to the headline protocol:\n  "
            + "\n  ".join(mismatches)
        )


def _deterministic_imq_bandwidth(points: torch.Tensor) -> float:
    """Median heuristic on one fixed 5,000-row subset."""
    n_subset = min(5_000, len(points))
    rows = torch.randperm(
        len(points), generator=torch.Generator().manual_seed(0)
    )[:n_subset].to(points.device)
    # imq_c_from_median does not subsample when handed at most subset_limit rows.
    return imq_c_from_median(points[rows], subset_limit=n_subset)


def _verify_gradient(device: torch.device) -> float:
    """Check the analytic GEMM expression against the repository autograd path."""
    generator = torch.Generator(device=device).manual_seed(2718)
    particles = torch.randn(13, 11, generator=generator, device=device)
    target = torch.randn(17, 11, generator=generator, device=device)
    eval_points = particles + 0.1 * torch.randn(
        particles.shape, generator=generator, device=device
    )
    weights = torch.randn(17, generator=generator, device=device)
    weights /= weights.sum()
    c = 2.7
    kernel = IMQKernel(c=c, beta=-0.5)
    automatic = (c**3) * witness_gradient(
        kernel,
        particles,
        target,
        eval_points,
        target_weights=weights,
    )
    analytic = scaled_witness_gradient(
        particles,
        target,
        weights,
        c,
        eval_points=eval_points,
    )
    relative_error = float(
        (analytic - automatic).norm()
        / automatic.norm().clamp_min(1e-12)
    )
    if relative_error > 1e-5:
        raise AssertionError(
            f"analytic IMQ gradient check failed: relative error {relative_error:.3e}"
        )
    return relative_error


def _sampler_metadata(
    sampler: str,
    c: float,
    noise_beta0: float,
    seconds_bandwidth: float,
    seconds_target_self: float,
) -> dict:
    if sampler == "riesz_adamw":
        return {
            "sampler": sampler,
            "kernel": "riesz_s1_energy",
            "optimizer": "adamw_weight_decay_0",
            "theory_relation": "practical_headline_sampler",
            "imq_c_multiplier": float("nan"),
            "imq_c": float("nan"),
            "imq_eta": float("nan"),
            "imq_raw_euler_step": float("nan"),
            "noise_beta0": 0.0,
            "noise_alpha": float("nan"),
            "noise_schedule": "none",
            "gradient_check_relative_error": GRADIENT_RELATIVE_ERROR,
            "seconds_imq_bandwidth": 0.0,
            "seconds_imq_target_self": 0.0,
        }
    noisy = sampler == "imq_euler_injected_noise"
    return {
        "sampler": sampler,
        "kernel": "normalized_imq_beta_minus_half",
        "optimizer": "forward_euler",
        "theory_relation": (
            "smooth_pd_euler_with_analysis_noise_schedule"
            if noisy
            else "smooth_pd_euler_deterministic_special_case"
        ),
        "imq_c_multiplier": IMQ_C_MULTIPLIER,
        "imq_c": c,
        "imq_eta": IMQ_ETA,
        "imq_raw_euler_step": IMQ_ETA * c**3,
        "noise_beta0": noise_beta0,
        "noise_alpha": NOISE_ALPHA,
        "noise_schedule": (
            "beta_r=beta0*(r+1)^(-alpha)" if noisy else "none"
        ),
        "gradient_check_relative_error": GRADIENT_RELATIVE_ERROR,
        "seconds_imq_bandwidth": round(seconds_bandwidth, 1),
        "seconds_imq_target_self": round(seconds_target_self, 1),
    }


def _score_flow(
    sampler: str,
    flow_particles: torch.Tensor,
    loss_initial: float,
    loss_final: float,
    seconds_flow: float,
    shared: dict,
    args,
    seed: int,
    initial_particles: torch.Tensor,
    context: dict,
    metadata: dict,
) -> dict:
    """Apply the unchanged headline metrics to one sampler endpoint."""
    device = initial_particles.device
    row = {
        "estimator": "dr",
        **metadata,
        "seconds_flow": round(seconds_flow, 1),
        "flow_iterations": N_ITERATIONS,
    }
    if sampler == "riesz_adamw":
        row.update(
            {
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "lr_schedule": args.lr_schedule,
            }
        )
    else:
        row.update(
            {
                "learning_rate": float("nan"),
                "warmup_steps": 0,
                "lr_schedule": "constant_euler_step",
            }
        )

    started = time.time()
    row.update(
        {
            "loss_initial": loss_initial,
            "loss_final": loss_final,
            "mean_transport": float(
                (flow_particles - initial_particles).norm(dim=1).mean()
            ),
            "latent_sw2_flow": base.sliced_wasserstein(
                flow_particles, context["Z_ref_std"], p=2, seed=0
            ).item(),
        }
    )
    flow_eval = (
        flow_particles
        if context["flow_eval_pick"] is None
        else flow_particles[context["flow_eval_pick"]]
    )
    row["n_flow_eval"] = int(len(flow_eval))
    seconds_eval = base.phase_seconds(started, device)

    started = time.time()
    with torch.no_grad():
        flow_images = base.decode_latents(
            shared["vae"], shared["from_flow"](flow_eval), args.image_size
        )
    row["seconds_decode"] = round(base.phase_seconds(started, device), 1)

    started = time.time()
    row.update(base.pixel_scores("flow", flow_images, context))
    seconds_eval += base.phase_seconds(started, device)

    started = time.time()
    row.update(base.fid_scores("flow", flow_images, context, shared))
    seconds_fid = base.phase_seconds(started, device)
    row["seconds_eval"] = round(seconds_eval + seconds_fid, 1)
    row["seconds_fid"] = round(seconds_fid, 1)

    key = (seed // 1000, context["target_a"])
    QUALITATIVE.setdefault(key, {})[sampler] = (
        flow_images[:N_QUALITATIVE].detach().cpu()
    )
    return row


def _run_riesz(
    v: torch.Tensor,
    shared: dict,
    args,
    seed: int,
    initial_particles: torch.Tensor,
    context: dict,
    c: float,
    seconds_bandwidth: float,
) -> dict:
    device = initial_particles.device
    started = time.time()
    torch.manual_seed(seed + 1)
    particles, history = base.adamw_riesz_flow(
        initial_particles.clone(),
        shared["Z_std_s"],
        target_weights=v,
        sliced=False,
        n_iterations=N_ITERATIONS,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        log_every=500,
        verbose=False,
    )
    seconds_flow = base.phase_seconds(started, device)
    metadata = _sampler_metadata(
        "riesz_adamw", c, 0.0, seconds_bandwidth, 0.0
    )
    return _score_flow(
        "riesz_adamw",
        particles,
        float(history["loss"][0]),
        float(history["loss"][-1]),
        seconds_flow,
        shared,
        args,
        seed,
        initial_particles,
        context,
        metadata,
    )


def _run_imq(
    sampler: str,
    v: torch.Tensor,
    shared: dict,
    args,
    seed: int,
    initial_particles: torch.Tensor,
    context: dict,
    c: float,
    target_self: float,
    seconds_bandwidth: float,
    seconds_target_self: float,
) -> dict:
    latent_dim = initial_particles.shape[1]
    noise_beta0 = (
        1.0 / (2.0 * math.sqrt(latent_dim))
        if sampler == "imq_euler_injected_noise"
        else 0.0
    )
    particles, info = euler_flow(
        initial_particles,
        shared["Z_std_s"],
        v,
        c=c,
        eta=IMQ_ETA,
        n_iterations=N_ITERATIONS,
        target_self=target_self,
        noise_beta0=noise_beta0,
        noise_alpha=NOISE_ALPHA,
        seed=seed + 1,
        log_every=500,
        sample_rows=N_QUALITATIVE,
    )
    if info["diverged"] or info["steps_completed"] != N_ITERATIONS:
        raise RuntimeError(
            f"{sampler} failed after {info['steps_completed']} iterations"
        )
    metadata = _sampler_metadata(
        sampler,
        c,
        noise_beta0,
        seconds_bandwidth,
        seconds_target_self,
    )
    return _score_flow(
        sampler,
        particles,
        info["mmd2_initial"],
        info["mmd2_final"],
        info["seconds_flow"],
        shared,
        args,
        seed,
        initial_particles,
        context,
        metadata,
    )


def run_arm(target_a: int, shared: dict, args, seed: int, device: torch.device):
    """Run all three samplers against one frozen arm-specific DR target."""
    _validate_protocol(args)

    weights, vectors, arm_nuisance = base.arm_weight_vectors(
        target_a, shared, args, seed, device
    )
    v = vectors["dr"]
    on_arm_mask = shared["A_s"] == target_a
    on_arm = weights[on_arm_mask]
    arm_indices, initial_particles = base.starting_particles(
        target_a, shared, args, seed
    )
    context, common = base.arm_evaluation(
        target_a,
        shared,
        args,
        seed,
        device,
        arm_indices,
        initial_particles,
    )
    common.update(
        {
            "ess_on_arm": float(on_arm.sum() ** 2 / (on_arm**2).sum()),
            "n_on_arm": int(len(on_arm)),
            "max_weight": float(weights.max()),
        }
    )

    # One deterministic bandwidth per repeat, reused by both arms and both IMQ rows.
    if "_theory_imq_c" not in shared:
        started = time.time()
        shared["_theory_imq_c"] = (
            IMQ_C_MULTIPLIER * _deterministic_imq_bandwidth(shared["Z_std_s"])
        )
        shared["_seconds_theory_imq_bandwidth"] = base.phase_seconds(
            started, device
        )
    c = shared["_theory_imq_c"]
    seconds_bandwidth = shared["_seconds_theory_imq_bandwidth"]

    started = time.time()
    target_self = target_self_inner_product(shared["Z_std_s"], v, c)
    seconds_target_self = base.phase_seconds(started, device)

    key = (seed // 1000, target_a)
    if "start" not in QUALITATIVE.setdefault(key, {}):
        with torch.no_grad():
            QUALITATIVE[key]["start"] = base.decode_latents(
                shared["vae"],
                shared["from_flow"](initial_particles[:N_QUALITATIVE]),
                args.image_size,
            ).cpu()

    diagnostics = base.weight_diagnostics(v, on_arm_mask, shared["X_s"])
    sampler_rows = [
        _run_riesz(
            v,
            shared,
            args,
            seed,
            initial_particles,
            context,
            c,
            seconds_bandwidth,
        ),
        _run_imq(
            "imq_euler_deterministic",
            v,
            shared,
            args,
            seed,
            initial_particles,
            context,
            c,
            target_self,
            seconds_bandwidth,
            seconds_target_self,
        ),
        _run_imq(
            "imq_euler_injected_noise",
            v,
            shared,
            args,
            seed,
            initial_particles,
            context,
            c,
            target_self,
            seconds_bandwidth,
            seconds_target_self,
        ),
    ]

    rows = []
    for sampler_row in sampler_rows:
        row = dict(common)
        row.update(diagnostics)
        row.update(sampler_row)
        rows.append(row)

    arm_seconds = {
        "seconds_ratio": arm_nuisance["seconds_ratio"],
        "seconds_cme": arm_nuisance["seconds_cme"],
        "seconds_eval_shared": context["seconds_eval_shared"],
        "seconds_fid_shared": context["seconds_fid_shared"],
    }
    return rows, arm_seconds


def _uint8(images: torch.Tensor) -> np.ndarray:
    return (
        ((images.clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .byte()
        .numpy()
    )


def save_qualitative_outputs(out_dir: Path) -> None:
    """Save small exact-particle panels and machine-readable uint8 arrays."""
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    order = [
        ("start", "Start"),
        ("riesz_adamw", "Riesz--AdamW"),
        ("imq_euler_deterministic", "IMQ--Euler"),
        ("imq_euler_injected_noise", "IMQ--Euler + noise"),
    ]
    for (data_seed, arm), panels in sorted(QUALITATIVE.items()):
        if not all(key in panels for key, _ in order):
            continue
        arrays = {key: _uint8(panels[key]) for key, _ in order}
        np.savez_compressed(
            figures / f"celeba_sampler_samples_ds{data_seed}_a{arm}.npz",
            **arrays,
        )

        fig, axes = plt.subplots(
            len(order), N_QUALITATIVE, figsize=(1.65 * N_QUALITATIVE, 7.0)
        )
        for row, (key, label) in enumerate(order):
            for column in range(N_QUALITATIVE):
                image = arrays[key][column].transpose(1, 2, 0)
                axes[row, column].imshow(image)
                axes[row, column].axis("off")
                if column == 0:
                    axes[row, column].set_ylabel(
                        label, rotation=0, ha="right", va="center", labelpad=55
                    )
        fig.suptitle(
            f"CelebA paired sampler ablation: data seed {data_seed}, a={arm}",
            fontsize=12,
        )
        fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.97))
        fig.savefig(
            figures / f"celeba_sampler_samples_ds{data_seed}_a{arm}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    global GRADIENT_RELATIVE_ERROR

    # The base runner owns parsing, data generation, the nuisance fit, references,
    # metrics, atomic CSV writing, timing accounting, and hardware provenance.
    # Only its arm-level sampler loop is replaced here.
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    GRADIENT_RELATIVE_ERROR = _verify_gradient(torch.device(device_name))
    print(
        f"analytic IMQ gradient check: relative error "
        f"{GRADIENT_RELATIVE_ERROR:.3e}",
        flush=True,
    )
    base.run_arm = run_arm
    base.main()

    # All production array tasks pass --out_dir explicitly.  Parse it without
    # calling the base parser a second time (which would only repeat validation).
    out_dir = Path("results")
    if "--out_dir" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out_dir") + 1])
    save_qualitative_outputs(out_dir)
    protocol = {
        "samplers": [
            "riesz_adamw",
            "imq_euler_deterministic",
            "imq_euler_injected_noise",
        ],
        "imq_c_rule": "median distance on a fixed seed-0 subset of at most 5000 logged standardized latents",
        "imq_c_multiplier": IMQ_C_MULTIPLIER,
        "imq_eta": IMQ_ETA,
        "imq_iterations": N_ITERATIONS,
        "noise_beta0": "1/(2*sqrt(latent_dim))",
        "noise_alpha": NOISE_ALPHA,
        "gradient_check_relative_error": GRADIENT_RELATIVE_ERROR,
        "interpretation": (
            "The IMQ rows match the smooth-PD Euler algorithmic form. The chosen "
            "finite step is empirical and is not asserted to verify every sufficient "
            "condition of the finite-particle theorem."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
