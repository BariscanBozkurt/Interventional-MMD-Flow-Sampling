"""Exact analytic IMQ witness gradients and Euler particle updates.

The implementation uses the normalized inverse-multiquadric kernel

    k(x, y) = (1 + ||x-y||^2 / c^2)^(-1/2).

Multiplying the witness gradient by ``c^2`` gives a dimensionless update and
avoids very small gradients when the median-heuristic bandwidth is large.  The
result is algebraically identical to using the repository's unnormalised
``IMQKernel(c, beta=-1/2)`` with raw Euler step ``eta * c^3``.

This file is deliberately small and independent of the CelebA runner so that
the analytic gradient can be unit tested on CPU.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import torch


def _squared_distances(
    x: torch.Tensor,
    y: torch.Tensor,
    x_norm: Optional[torch.Tensor] = None,
    y_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the clamped matrix ``||x_i-y_j||^2``."""
    if x_norm is None:
        x_norm = x.square().sum(dim=-1, keepdim=True)
    if y_norm is None:
        y_norm = y.square().sum(dim=-1, keepdim=True)
    return (x_norm + y_norm.T - 2.0 * (x @ y.T)).clamp_min_(0.0)


@torch.no_grad()
def scaled_witness_gradient(
    particles: torch.Tensor,
    target: torch.Tensor,
    target_weights: torch.Tensor,
    c: float,
    eval_points: Optional[torch.Tensor] = None,
    target_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``c^2 grad f(z)`` for the normalized IMQ witness.

    ``eval_points`` implements injected noise: the witness is queried at
    ``z``, while its empirical particle term is still formed from the
    unperturbed ``particles``.  The returned gradient is subsequently applied
    to those unperturbed particles.
    """
    if c <= 0.0:
        raise ValueError("c must be positive")
    if target_weights.ndim != 1 or len(target_weights) != len(target):
        raise ValueError("target_weights must have one entry per target point")

    z = particles if eval_points is None else eval_points
    n_particles = particles.shape[0]
    inv_c2 = 1.0 / (c * c)

    z_norm = z.square().sum(dim=-1, keepdim=True)
    particle_norm = (
        z_norm
        if eval_points is None
        else particles.square().sum(dim=-1, keepdim=True)
    )
    if target_norm is None:
        target_norm = target.square().sum(dim=-1, keepdim=True)

    distance_particle = _squared_distances(
        z, particles, z_norm, particle_norm
    )
    weight_particle = (1.0 + distance_particle * inv_c2).pow_(-1.5)
    weight_particle.mul_(1.0 / n_particles)
    sum_particle = weight_particle.sum(dim=1, keepdim=True)
    mean_particle = weight_particle @ particles

    distance_target = _squared_distances(z, target, z_norm, target_norm)
    weight_target = (1.0 + distance_target * inv_c2).pow_(-1.5)
    weight_target.mul_(target_weights[None, :])
    sum_target = weight_target.sum(dim=1, keepdim=True)
    mean_target = weight_target @ target

    return (
        (sum_target - sum_particle) * z
        + mean_particle
        - mean_target
    )


@torch.no_grad()
def target_self_inner_product(
    target: torch.Tensor,
    target_weights: torch.Tensor,
    c: float,
    chunk_size: int = 1024,
) -> float:
    """Compute the exact normalized-IMQ target self inner product in chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    target_norm = target.square().sum(dim=-1, keepdim=True)
    weights64 = target_weights.double()
    total = torch.zeros((), dtype=torch.float64, device=target.device)
    inv_c2 = 1.0 / (c * c)
    for start in range(0, len(target), chunk_size):
        stop = min(len(target), start + chunk_size)
        distance = _squared_distances(
            target[start:stop], target, target_norm[start:stop], target_norm
        )
        kernel = (1.0 + distance * inv_c2).rsqrt_()
        total += (
            weights64[start:stop] * (kernel.double() @ weights64)
        ).sum()
    return float(total.item())


@torch.no_grad()
def squared_mmd_to_target(
    particles: torch.Tensor,
    target: torch.Tensor,
    target_weights: torch.Tensor,
    c: float,
    target_self: float,
) -> float:
    """Compute ``||mu_particles - sum_i v_i k(target_i, .)||^2``."""
    inv_c2 = 1.0 / (c * c)
    particle_norm = particles.square().sum(dim=-1, keepdim=True)
    target_norm = target.square().sum(dim=-1, keepdim=True)

    distance_xx = _squared_distances(
        particles, particles, particle_norm, particle_norm
    )
    term_xx = (1.0 + distance_xx * inv_c2).rsqrt_().double().mean()

    distance_xy = _squared_distances(
        particles, target, particle_norm, target_norm
    )
    kernel_xy = (1.0 + distance_xy * inv_c2).rsqrt_()
    term_xy = (kernel_xy.double() @ target_weights.double()).mean()
    return float((term_xx - 2.0 * term_xy + target_self).item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def euler_flow(
    initial_particles: torch.Tensor,
    target: torch.Tensor,
    target_weights: torch.Tensor,
    c: float,
    eta: float,
    n_iterations: int,
    target_self: float,
    noise_beta0: float = 0.0,
    noise_alpha: float = 0.25,
    seed: int = 1,
    log_every: int = 500,
    sample_rows: int = 8,
    sample_steps: Sequence[int] = (),
) -> Tuple[torch.Tensor, Dict]:
    """Run forward Euler with optional injected Gaussian query noise.

    The update at iteration ``r`` is

    ``x <- x - eta * c^2 grad f(x + beta_r u)``,

    where ``beta_r = noise_beta0 * (r+1)^(-noise_alpha)``.  Noise changes only
    the query point; it is never added to the particle state.
    """
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    if n_iterations < 0:
        raise ValueError("n_iterations must be nonnegative")
    if noise_beta0 < 0.0:
        raise ValueError("noise_beta0 must be nonnegative")
    if not 0.0 <= noise_alpha < 0.5:
        raise ValueError("noise_alpha must lie in [0, 1/2)")

    particles = initial_particles.detach().clone()
    target = target.detach()
    target_weights = target_weights.detach()
    target_norm = target.square().sum(dim=-1, keepdim=True)
    generator = torch.Generator(device=particles.device).manual_seed(seed)
    requested_steps = set(int(step) for step in sample_steps)
    requested_steps.add(0)
    requested_steps.add(n_iterations)
    snapshots = {}
    if 0 in requested_steps:
        snapshots[0] = particles[:sample_rows].cpu().clone()

    initial_mmd2 = squared_mmd_to_target(
        particles, target, target_weights, c, target_self
    )
    curve = [{"step": 0, "mmd2": initial_mmd2, "mean_transport": 0.0}]
    first_gradient_rms = float("nan")
    diverged = False

    _sync(particles.device)
    started = time.perf_counter()
    for step in range(n_iterations):
        eval_points = None
        if noise_beta0 > 0.0:
            beta = noise_beta0 * ((step + 1) ** (-noise_alpha))
            noise = torch.randn(
                particles.shape,
                generator=generator,
                dtype=particles.dtype,
                device=particles.device,
            )
            eval_points = particles + beta * noise

        gradient = scaled_witness_gradient(
            particles,
            target,
            target_weights,
            c,
            eval_points=eval_points,
            target_norm=target_norm,
        )
        if step == 0:
            first_gradient_rms = float(gradient.square().mean().sqrt().item())
        particles.add_(gradient, alpha=-eta)
        completed = step + 1

        if not torch.isfinite(particles).all():
            diverged = True
            break
        if completed in requested_steps:
            snapshots[completed] = particles[:sample_rows].cpu().clone()
        if (
            completed % log_every == 0
            or completed == n_iterations
        ):
            curve.append(
                {
                    "step": completed,
                    "mmd2": squared_mmd_to_target(
                        particles, target, target_weights, c, target_self
                    ),
                    "mean_transport": float(
                        (particles - initial_particles).norm(dim=1).mean().item()
                    ),
                }
            )

    _sync(particles.device)
    seconds = time.perf_counter() - started
    final_mmd2 = squared_mmd_to_target(
        particles, target, target_weights, c, target_self
    )
    mean_transport = float(
        (particles - initial_particles).norm(dim=1).mean().item()
    )
    completed = step + 1 if n_iterations > 0 else 0
    info = {
        "diverged": diverged,
        "steps_completed": completed,
        "seconds_flow": seconds,
        "mmd2_initial": initial_mmd2,
        "mmd2_final": final_mmd2,
        "mean_transport": mean_transport,
        "first_gradient_rms": first_gradient_rms,
        "curve": curve,
        "snapshots": snapshots,
    }
    return particles, info
