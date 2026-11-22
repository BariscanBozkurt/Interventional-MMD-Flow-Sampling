"""Continuous latent-space kernel herding.

For a target embedding ``chi(z) = sum_i weights[i] k(support[i], z)``, CPME
Algorithm 2 adds atoms by maximizing ``chi(z)`` minus a kernel interaction with
the previously selected atoms.  The displayed Algorithm 2 divides that sum by
``t`` when selecting atom ``t``, whereas Appendix 14.2.1 and the released code
divide by the number ``t-1`` of previous atoms.  Both conventions are supported
explicitly below.  This module approximates the resulting continuous argmax with
multistart Adam ascent.  It does not restrict returned atoms to the observed data
and never clips or normalizes the supplied target coefficients.  The default
domain is unbounded; a projected support box remains available only for
reproducing the earlier pilot.

A finite candidate bank is used only to initialize each non-convex argmax.  The
current best bank point is retained as a fallback while local restarts are
optimized, so the returned atom cannot have a worse herding score than that
finite-bank initializer.
"""

from typing import Dict, Optional, Tuple, Union

import torch

from kernel_utils import IMQKernel, RieszKernel
from kernel_utils.linear_algebra import pairwise_squared_distance


TensorOrFloat = Union[torch.Tensor, float]


def _validate_inputs(weights: torch.Tensor,
                     target_support: torch.Tensor,
                     queries: torch.Tensor) -> None:
    if weights.ndim != 1:
        raise ValueError("weights must have shape (n,)")
    if target_support.ndim != 2 or queries.ndim != 2:
        raise ValueError("target_support and queries must be two-dimensional")
    if len(weights) != len(target_support):
        raise ValueError("weights and target_support must have the same length")
    if target_support.shape[1] != queries.shape[1]:
        raise ValueError("target_support and queries must have the same dimension")
    if weights.device != target_support.device or queries.device != target_support.device:
        raise ValueError("weights, target_support, and queries must share a device")
    if weights.dtype != target_support.dtype or queries.dtype != target_support.dtype:
        raise ValueError("weights, target_support, and queries must share a dtype")
    if not target_support.is_floating_point():
        raise ValueError("continuous herding requires floating-point tensors")
    if not torch.isfinite(weights).all():
        raise ValueError("weights must be finite")
    if not torch.isfinite(target_support).all() or not torch.isfinite(queries).all():
        raise ValueError("support and query points must be finite")


def _kernel_values_and_radial_derivative(kernel,
                                         queries: torch.Tensor,
                                         support: torch.Tensor
                                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return k(q,s) and c(q,s) in grad_q k(q,s)=c(q,s)(q-s)."""
    distance_sq = pairwise_squared_distance(queries, support)
    if isinstance(kernel, IMQKernel):
        base = kernel.c ** 2 + distance_sq
        return base.pow(kernel.beta), 2.0 * kernel.beta * base.pow(kernel.beta - 1.0)
    if isinstance(kernel, RieszKernel):
        if kernel.eps <= 0.0:
            raise ValueError("continuous Riesz herding requires kernel.eps > 0")
        base = distance_sq + kernel.eps ** 2
        return (-base.pow(kernel.s / 2.0),
                -kernel.s * base.pow(kernel.s / 2.0 - 1.0))
    raise TypeError("continuous herding currently supports IMQKernel and RieszKernel")


def _weighted_score_and_gradient(kernel,
                                 queries: torch.Tensor,
                                 support: torch.Tensor,
                                 coefficients: torch.Tensor,
                                 chunk_size: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a weighted radial-kernel expansion and its exact query gradient."""
    score = torch.zeros(len(queries), dtype=queries.dtype, device=queries.device)
    gradient = torch.zeros_like(queries)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(support)
    for start in range(0, len(support), chunk_size):
        stop = min(start + chunk_size, len(support))
        support_chunk = support[start:stop]
        coefficient_chunk = coefficients[start:stop]
        values, derivative = _kernel_values_and_radial_derivative(
            kernel, queries, support_chunk)
        score.add_(values @ coefficient_chunk)
        weighted_derivative = derivative * coefficient_chunk.view(1, -1)
        gradient.add_(
            queries * weighted_derivative.sum(dim=1, keepdim=True)
            - weighted_derivative @ support_chunk)
    return score, gradient


def herding_score_and_gradient(weights: torch.Tensor,
                               kernel,
                               target_support: torch.Tensor,
                               queries: torch.Tensor,
                               previous_atoms: Optional[torch.Tensor] = None,
                               support_chunk_size: Optional[int] = 2048,
                               repulsion_chunk_size: Optional[int] = 2048,
                               repulsion_denominator: str = "algorithm2",
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a CPME herding score and its continuous gradient.

    ``algorithm2`` uses denominator ``p+1`` for ``p`` previous atoms, matching
    the displayed Algorithm 2.  ``previous_count`` uses denominator ``p``,
    matching Appendix 14.2.1 and the released implementation.
    """
    _validate_inputs(weights, target_support, queries)
    score, gradient = _weighted_score_and_gradient(
        kernel, queries, target_support, weights, support_chunk_size)
    if previous_atoms is None or len(previous_atoms) == 0:
        return score, gradient
    if previous_atoms.ndim != 2 or previous_atoms.shape[1] != queries.shape[1]:
        raise ValueError("previous_atoms must have shape (t,d)")
    if previous_atoms.device != queries.device or previous_atoms.dtype != queries.dtype:
        raise ValueError("previous_atoms must share query device and dtype")
    if repulsion_denominator == "algorithm2":
        denominator = len(previous_atoms) + 1
    elif repulsion_denominator == "previous_count":
        denominator = len(previous_atoms)
    else:
        raise ValueError(
            "repulsion_denominator must be 'algorithm2' or 'previous_count'")
    repulsion_weights = torch.full(
        (len(previous_atoms),), 1.0 / denominator,
        dtype=queries.dtype, device=queries.device)
    repulsion_score, repulsion_gradient = _weighted_score_and_gradient(
        kernel, queries, previous_atoms, repulsion_weights,
        repulsion_chunk_size)
    return score - repulsion_score, gradient - repulsion_gradient


def latent_quantile_bounds(reference: torch.Tensor,
                           lower_quantile: float = 0.001,
                           upper_quantile: float = 0.999,
                           padding_fraction: float = 0.05,
                           minimum_width: float = 1e-3,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a continuous coordinate box from an observed latent cloud."""
    if reference.ndim != 2 or len(reference) < 2:
        raise ValueError("reference must have shape (n,d) with n>=2")
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if padding_fraction < 0.0 or minimum_width <= 0.0:
        raise ValueError("padding_fraction must be nonnegative and minimum_width positive")
    # The production image runs use the full coordinate range.  Avoid sorting
    # every coordinate of the 20,000 by 4,096 CelebA latent matrix in that case.
    lower = (reference.amin(dim=0) if lower_quantile == 0.0 else
             torch.quantile(reference, lower_quantile, dim=0))
    upper = (reference.amax(dim=0) if upper_quantile == 1.0 else
             torch.quantile(reference, upper_quantile, dim=0))
    width = (upper - lower).clamp_min(minimum_width)
    return lower - padding_fraction * width, upper + padding_fraction * width


def _coerce_bound(value: Optional[TensorOrFloat], reference: torch.Tensor,
                  name: str) -> Optional[torch.Tensor]:
    if value is None:
        return None
    bound = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if bound.ndim == 0:
        bound = bound.repeat(reference.shape[1])
    if bound.shape != (reference.shape[1],):
        raise ValueError(f"{name} must be scalar or have shape (d,)")
    if not torch.isfinite(bound).all():
        raise ValueError(f"{name} must be finite")
    return bound


def _embedding_on_candidates(weights: torch.Tensor,
                             kernel,
                             target_support: torch.Tensor,
                             candidates: torch.Tensor,
                             candidate_chunk_size: int) -> torch.Tensor:
    values = torch.empty(len(candidates), dtype=candidates.dtype,
                         device=candidates.device)
    if candidate_chunk_size <= 0:
        candidate_chunk_size = len(candidates)
    for start in range(0, len(candidates), candidate_chunk_size):
        stop = min(start + candidate_chunk_size, len(candidates))
        values[start:stop] = kernel(
            target_support, candidates[start:stop]).T @ weights
    return values


def _project(points: torch.Tensor,
             lower: Optional[torch.Tensor],
             upper: Optional[torch.Tensor]) -> torch.Tensor:
    if lower is None:
        return points
    return torch.maximum(torch.minimum(points, upper), lower)


def continuous_kernel_herding(
        weights: torch.Tensor,
        kernel,
        target_support: torch.Tensor,
        num_samples: int,
        *,
        initial_candidates: Optional[torch.Tensor] = None,
        initial_points: Optional[torch.Tensor] = None,
        anchor_points: Optional[torch.Tensor] = None,
        n_restarts: int = 2,
        n_steps: int = 10,
        learning_rate: float = 0.01,
        restart_scale: float = 0.05,
        support_chunk_size: Optional[int] = 2048,
        repulsion_chunk_size: Optional[int] = 2048,
        candidate_chunk_size: int = 512,
        domain: str = "unbounded",
        domain_margin: float = 0.05,
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
        require_unit_mass: bool = False,
        repulsion_denominator: str = "algorithm2",
        gradient_tolerance: float = 0.0,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Approximately solve the continuous argmax at every herding iteration.

    Parameters
    ----------
    weights, kernel, target_support
        Fixed estimated target embedding.  Weights may be signed and are never
        modified.
    num_samples
        Number of equal-weight atoms to return.
    initial_candidates
        Continuous-optimization seed bank.  At each iteration the exact best
        bank point under the current continuous-atom repulsion is used as
        restart zero.  Defaults to ``target_support``.
    initial_points
        Alternative fixed sequence of one seed per atom.  It is mutually
        exclusive with ``initial_candidates``.
    anchor_points
        Optional fixed sequence, usually the already computed finite-candidate
        herding output.  At step ``t`` it is included as an additional restart;
        it does not constrain the returned atom.
    domain
        ``"unbounded"`` or ``"support_quantile_box"``.  The latter is a
        continuous coordinate box derived from ``target_support`` and padded by
        ``domain_margin`` times its coordinatewise quantile width.
    repulsion_denominator
        ``"algorithm2"`` uses ``1/t`` in the displayed paper algorithm;
        ``"previous_count"`` uses the appendix/released-code ``1/(t-1)``.

    Returns
    -------
    atoms, diagnostics
        ``atoms`` has shape ``(num_samples,d)``.  Diagnostics include the
        per-atom seed/final score, chosen restart, movement, gradient norm,
        boundary fraction, seed candidate index, and unchanged target-mass
        summaries.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if n_restarts < 1 or n_steps < 0:
        raise ValueError("n_restarts must be positive and n_steps nonnegative")
    if learning_rate <= 0.0 or restart_scale < 0.0:
        raise ValueError("learning_rate must be positive and restart_scale nonnegative")
    if gradient_tolerance < 0.0:
        raise ValueError("gradient_tolerance must be nonnegative")
    if repulsion_denominator not in {"algorithm2", "previous_count"}:
        raise ValueError(
            "repulsion_denominator must be 'algorithm2' or 'previous_count'")
    if initial_candidates is not None and initial_points is not None:
        raise ValueError("initial_candidates and initial_points are mutually exclusive")
    if initial_candidates is None and initial_points is None:
        initial_candidates = target_support
    if initial_points is not None and len(initial_points) != num_samples:
        raise ValueError("initial_points must contain exactly num_samples rows")
    seed_reference = initial_points if initial_points is not None else initial_candidates
    if seed_reference is None or len(seed_reference) == 0:
        raise ValueError("the initialization set cannot be empty")
    _validate_inputs(weights, target_support, seed_reference)
    if anchor_points is not None:
        _validate_inputs(weights, target_support, anchor_points)
        if anchor_points.shape != (num_samples, target_support.shape[1]):
            raise ValueError("anchor_points must have shape (num_samples,d)")
        if n_restarts < 2:
            raise ValueError("anchor_points requires n_restarts >= 2")

    mass = float(weights.sum())
    mass_tolerance = 1e-4
    if require_unit_mass and abs(mass - 1.0) > mass_tolerance:
        raise ValueError(
            f"target mass must equal one within {mass_tolerance}; got {mass:.8g}")
    if domain == "unbounded":
        lower = upper = None
    elif domain == "support_quantile_box":
        lower, upper = latent_quantile_bounds(
            target_support, lower_quantile, upper_quantile, domain_margin)
    else:
        raise ValueError("domain must be 'unbounded' or 'support_quantile_box'")
    if isinstance(kernel, RieszKernel) and lower is None:
        if abs(mass - 1.0) > mass_tolerance:
            raise ValueError("unbounded Riesz herding requires a unit-mass target")
        if repulsion_denominator != "algorithm2":
            raise ValueError(
                "unbounded Riesz herding requires the Algorithm-2 denominator")
        # With p previous atoms, unit target mass and the displayed p+1
        # denominator, the score is -||z||/(p+1) + O(1) at infinity.  The
        # objective is therefore coercive.  The alternate p denominator
        # cancels this leading radial term and is deliberately rejected above.

    device = target_support.device
    dtype = target_support.dtype
    dimension = target_support.shape[1]
    atoms = torch.empty((num_samples, dimension), dtype=dtype, device=device)
    initial_scores = torch.empty(num_samples, dtype=torch.float64)
    anchor_scores = torch.full((num_samples,), float("nan"), dtype=torch.float64)
    final_scores = torch.empty(num_samples, dtype=torch.float64)
    selected_restarts = torch.empty(num_samples, dtype=torch.long)
    steps_used = torch.empty(num_samples, dtype=torch.long)
    final_gradient_norms = torch.empty(num_samples, dtype=torch.float64)
    boundary_fractions = torch.zeros(num_samples, dtype=torch.float64)
    distances_from_seed = torch.empty(num_samples, dtype=torch.float64)
    seed_candidate_indices = torch.full((num_samples,), -1, dtype=torch.long)

    candidate_attraction = None
    candidate_repulsion = None
    if initial_candidates is not None:
        candidate_attraction = _embedding_on_candidates(
            weights, kernel, target_support, initial_candidates,
            candidate_chunk_size)
        candidate_repulsion = torch.zeros_like(candidate_attraction)

    latent_scale = target_support.std(dim=0, unbiased=False).clamp_min(1e-6)
    beta1, beta2, adam_epsilon = 0.9, 0.999, 1e-8
    report_every = max(1, num_samples // 10)

    with torch.no_grad():
        for t in range(num_samples):
            previous = atoms[:t]
            if initial_candidates is None:
                seed = initial_points[t]
            else:
                candidate_score = candidate_attraction
                if t > 0:
                    denominator = t + 1 if repulsion_denominator == "algorithm2" else t
                    candidate_score = candidate_score - candidate_repulsion / denominator
                seed_index = int(candidate_score.argmax())
                seed_candidate_indices[t] = seed_index
                seed = initial_candidates[seed_index]

            points = seed.view(1, -1).repeat(n_restarts, 1)
            next_restart = 1
            if anchor_points is not None:
                points[1] = anchor_points[t]
                next_restart = 2
            if next_restart < n_restarts:
                random_indices = torch.randint(
                    len(target_support), (n_restarts - next_restart,),
                    device=device, generator=generator)
                points[next_restart:] = target_support[random_indices]
            if next_restart < n_restarts and restart_scale > 0.0:
                noise = torch.randn(
                    (n_restarts - next_restart, dimension), dtype=dtype,
                    device=device, generator=generator)
                points[next_restart:].add_(noise * latent_scale, alpha=restart_scale)
            points.copy_(_project(points, lower, upper))

            score, gradient = herding_score_and_gradient(
                weights, kernel, target_support, points, previous,
                support_chunk_size, repulsion_chunk_size,
                repulsion_denominator)
            seed_score = score[0].clone()
            if anchor_points is not None:
                anchor_scores[t] = float(score[1])
            best_points = points.clone()
            best_scores = score.clone()
            moment = torch.zeros_like(points)
            second_moment = torch.zeros_like(points)
            used = 0

            for step in range(1, n_steps + 1):
                if gradient_tolerance > 0.0:
                    if float(gradient.norm(dim=1).max()) <= gradient_tolerance:
                        break
                moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2)
                moment_hat = moment / (1.0 - beta1 ** step)
                second_hat = second_moment / (1.0 - beta2 ** step)
                points.addcdiv_(moment_hat, second_hat.sqrt().add_(adam_epsilon),
                               value=learning_rate)
                points.copy_(_project(points, lower, upper))
                score, gradient = herding_score_and_gradient(
                    weights, kernel, target_support, points, previous,
                    support_chunk_size, repulsion_chunk_size,
                    repulsion_denominator)
                improved = score > best_scores
                best_scores[improved] = score[improved]
                best_points[improved] = points[improved]
                used = step

            restart = int(best_scores.argmax())
            atom = best_points[restart]
            if not torch.isfinite(atom).all():
                raise RuntimeError(
                    f"continuous herding produced a non-finite atom at step {t}")
            atom_score, atom_gradient = herding_score_and_gradient(
                weights, kernel, target_support, atom.view(1, -1), previous,
                support_chunk_size, repulsion_chunk_size,
                repulsion_denominator)
            atoms[t] = atom
            initial_scores[t] = float(seed_score)
            final_scores[t] = float(atom_score[0])
            selected_restarts[t] = restart
            steps_used[t] = used
            final_gradient_norms[t] = float(atom_gradient[0].norm())
            distances_from_seed[t] = float((atom - seed).norm())
            if lower is not None:
                tolerance = 10.0 * torch.finfo(dtype).eps
                at_boundary = ((atom - lower).abs() <= tolerance
                               ) | ((atom - upper).abs() <= tolerance)
                boundary_fractions[t] = float(at_boundary.float().mean())
            if initial_candidates is not None:
                candidate_repulsion.add_(
                    kernel(atom.view(1, -1), initial_candidates)[0])
            if verbose and ((t + 1) % report_every == 0 or t + 1 == num_samples):
                print(f"continuous herding: {t + 1}/{num_samples}")

    if not torch.isfinite(atoms).all():
        raise RuntimeError("continuous herding returned non-finite atoms")

    negative_weights = weights[weights < 0]
    diagnostics: Dict[str, object] = {
        "initial_scores": initial_scores,
        "anchor_scores": anchor_scores,
        "final_scores": final_scores,
        "score_improvements": final_scores - initial_scores,
        "score_improvements_over_anchor": final_scores - anchor_scores,
        "selected_restarts": selected_restarts,
        "optimizer_steps": steps_used,
        "final_gradient_norms": final_gradient_norms,
        "boundary_fractions": boundary_fractions,
        "distances_from_seed": distances_from_seed,
        "seed_candidate_indices": seed_candidate_indices,
        "target_mass": mass,
        "target_absolute_mass": float(weights.abs().sum()),
        "target_negative_mass": float(-negative_weights.sum()),
        "n_negative_weights": int((weights < 0).sum()),
        "n_target_support": int(len(target_support)),
        "n_initial_candidates": int(len(initial_candidates)) if initial_candidates is not None else 0,
        "n_samples": int(num_samples),
        "n_restarts": int(n_restarts),
        "n_steps": int(n_steps),
        "learning_rate": float(learning_rate),
        "restart_scale": float(restart_scale),
        "domain": domain,
        "domain_margin": float(domain_margin),
        "repulsion_denominator": repulsion_denominator,
    }
    return atoms, diagnostics
