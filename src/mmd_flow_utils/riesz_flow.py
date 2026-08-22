"""
Riesz (energy-distance) MMD flow minimised with AdamW.

The optimiser-based counterpart of `mmd_flow.mmd_gradient_flow`: instead of a
fixed-step forward-Euler update of the kernel witness, the (weighted) energy
distance is minimised directly with AdamW over the particle positions. The
adaptive-scale normalisation and the sliced variant follow H. Chen's generative
Riesz-flow implementation of

    J. Hertrich, C. Wald, F. Altekrueger, P. Hagemann,
    "Generative Sliced MMD Flows with Riesz Kernels", ICLR 2024.

The Riesz kernel k(x, y) = -||x - y|| is only conditionally positive definite, so
the objective is written as the weighted energy distance

    D(Q, target) = 2 E||G - P|| - E||G - G'||,

with G the particles and P the target support; the P-P' term is constant in the
particles and dropped. `target_weights` reweights the target term exactly as in
`mmd_flow`, e.g. with doubly robust counterfactual weights.

AdamW rescales each coordinate's step by the running RMS of its gradient, which
removes the step-size tuning of the Euler flow in high-dimensional spaces where
the raw Riesz gradient is small and anisotropic. Its learning rate must be
scheduled: a constant post-warm-up rate prevents the particle cloud from settling
(the per-coordinate normalisation keeps the update speed near the learning rate
even at equilibrium), so in low-dimensional outcome spaces the cosine schedule,
which decays the rate to zero, is required to reach the Euler flow's accuracy.
"""

import math

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from tqdm.auto import trange


def _adaptive_scale(particles: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Isotropic coordinate scale: the mean particle-to-point pairwise distance divided
    by sqrt(feature dimension), floored at `eps`. Rescaling every coordinate by the
    same constant conditions the loss magnitude without changing the minimiser.
    """
    with torch.no_grad():
        all_points = torch.cat([particles, target], dim=0)
        scale = torch.cdist(particles, all_points).mean()
        return (scale / particles.shape[1] ** 0.5).clamp_min(eps)


def riesz_energy_loss(particles: torch.Tensor,
                      target: torch.Tensor,
                      target_weights: Optional[torch.Tensor] = None,
                      sliced: bool = False,
                      num_projections: int = 128,
                      normalize_scale: bool = True,
                      eps: float = 1e-3,
                      ) -> torch.Tensor:
    """
    Weighted Riesz energy distance between particles G and a target support P,

        loss = 2 E||G - P||_v - E||G - G'||,

    where the attraction term weights the target points by v (uniform 1/N when
    `target_weights` is None), so the minimiser matches the weighted target
    embedding sum_j v_j k(P_j, .).

    Parameters
    ----------
    particles : torch.Tensor of shape (M, D)
        Particle positions (the optimised variable).
    target : torch.Tensor of shape (N, D)
        Target support points.
    target_weights : torch.Tensor of shape (N,), optional
        Weights on the target support, possibly signed.
    sliced : bool
        If True, averages one-dimensional energy distances over `num_projections`
        random unit directions (fresh directions each call); pairwise Euclidean
        distances concentrate in high dimension, one-dimensional projections do
        not. If False, uses the full-dimensional distance.
    num_projections : int
        Number of random directions when `sliced` is True.
    normalize_scale : bool
        Divide all coordinates by the adaptive scale before the distances.
    eps : float
        Floor of the adaptive scale.

    Returns
    -------
    torch.Tensor (scalar)
    """
    target = target.detach()
    n_target = target.shape[0]
    if target_weights is None:
        weights = particles.new_full((n_target,), 1.0 / n_target)
    else:
        weights = target_weights.detach().to(particles)

    if normalize_scale:
        scale = _adaptive_scale(particles, target, eps)
        gen, pos = particles / scale, target / scale
    else:
        gen, pos = particles, target

    if not sliced:
        attraction = (torch.cdist(gen, pos) * weights[None, :]).sum(dim=1).mean()
        self_repulsion = torch.cdist(gen, gen).mean()
        return 2.0 * attraction - self_repulsion

    directions = F.normalize(torch.randn(gen.shape[1], num_projections,
                                         dtype=gen.dtype, device=gen.device), dim=0)
    gen_proj = gen @ directions
    pos_proj = pos @ directions
    attraction = (torch.abs(gen_proj[:, None, :] - pos_proj[None, :, :])
                  * weights[None, :, None]).sum(dim=1).mean(dim=0)
    self_repulsion = torch.abs(gen_proj[:, None, :] - gen_proj[None, :, :]).mean(dim=(0, 1))
    return (2.0 * attraction - self_repulsion).mean()


def warmup_const_lr(step: int, learning_rate: float, warmup_steps: int) -> float:
    """Linear warm-up from ~0 to `learning_rate` over `warmup_steps`, then constant."""
    if warmup_steps > 0 and step < warmup_steps:
        return 1e-6 + (learning_rate - 1e-6) * (step / warmup_steps)
    return learning_rate


def warmup_cosine_lr(step: int, learning_rate: float, warmup_steps: int,
                     n_iterations: int) -> float:
    """
    Linear warm-up, then cosine decay from `learning_rate` to ~0 at `n_iterations`.

    The decay is what lets Adam-style optimisers settle: their per-coordinate RMS
    normalisation keeps the update speed near the learning rate regardless of the
    gradient magnitude, so with a constant rate the particle cloud keeps jittering
    at equilibrium with amplitude set by the rate.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return 1e-6 + (learning_rate - 1e-6) * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, n_iterations - warmup_steps)
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def adamw_riesz_flow(initial_particles: torch.Tensor,
                     target: torch.Tensor,
                     target_weights: Optional[torch.Tensor] = None,
                     sliced: bool = False,
                     num_projections: int = 128,
                     n_iterations: int = 2500,
                     learning_rate: float = 0.03,
                     warmup_steps: int = 300,
                     lr_schedule: str = "const",
                     betas: Tuple[float, float] = (0.9, 0.95),
                     weight_decay: float = 0.0,
                     max_grad_norm: Optional[float] = None,
                     normalize_scale: bool = True,
                     log_every: int = 100,
                     snapshot_every: Optional[int] = None,
                     verbose: bool = True,
                     ) -> Tuple[torch.Tensor, Dict]:
    """
    Minimises the (sliced) Riesz energy distance over the particle positions with
    AdamW. Each iteration sets the scheduled learning rate, evaluates the loss at
    the current particles, backpropagates, optionally clips the gradient, and steps.

    Parameters
    ----------
    initial_particles : torch.Tensor of shape (M, D)
        Starting particles; copied, not modified in place.
    target : torch.Tensor of shape (N, D)
        Target support points.
    target_weights : torch.Tensor of shape (N,), optional
        Weights on the target support (None = uniform).
    sliced, num_projections, normalize_scale
        Passed to `riesz_energy_loss`.
    n_iterations : int
    learning_rate : float
        Peak learning rate after warm-up.
    warmup_steps : int
        Linear warm-up length.
    lr_schedule : {"const", "cosine"}
        Post-warm-up schedule: "const" holds the peak rate; "cosine" decays it to
        ~0 at `n_iterations`, which is required for the particles to settle in
        low-dimensional outcome spaces (see the module docstring).
    betas, weight_decay
        AdamW parameters. Weight decay on free particles pulls them toward the
        origin and defaults to 0.
    max_grad_norm : float, optional
        Gradient-norm clip; None disables it.
    log_every : int
        Records the loss every `log_every` iterations and at the last one. The
        recorded value is the loss evaluated at the pre-update particles of that
        iteration.
    snapshot_every : int, optional
        If given, stores a copy of the particles every `snapshot_every` iterations.
    verbose : bool
        Shows a progress bar.

    Returns
    -------
    particles : torch.Tensor of shape (M, D)
    history : dict with keys "iterations", "loss", "snapshots".
    """
    if lr_schedule not in ("const", "cosine"):
        raise ValueError("lr_schedule must be 'const' or 'cosine'")
    particles = initial_particles.detach().clone().requires_grad_(True)
    target = target.detach()
    optimizer = torch.optim.AdamW([particles], lr=learning_rate, betas=betas,
                                  weight_decay=weight_decay)

    history: Dict = {"iterations": [], "loss": [], "snapshots": {}}
    steps = trange(n_iterations, desc="Riesz flow") if verbose else range(n_iterations)
    for step in steps:
        if lr_schedule == "cosine":
            lr = warmup_cosine_lr(step, learning_rate, warmup_steps, n_iterations)
        else:
            lr = warmup_const_lr(step, learning_rate, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss = riesz_energy_loss(particles, target, target_weights, sliced,
                                 num_projections, normalize_scale)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([particles], max_grad_norm)
        optimizer.step()

        if snapshot_every is not None and step % snapshot_every == 0:
            history["snapshots"][step] = particles.detach().clone()
        if step % log_every == 0 or step == n_iterations - 1:
            history["iterations"].append(step)
            history["loss"].append(loss.item())

    return particles.detach(), history
