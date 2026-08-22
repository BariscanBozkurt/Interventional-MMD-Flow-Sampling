"""
MMD gradient flow with noise injection.

Implements Algorithm 1 (noisy gradient flow of the MMD) from

    M. Arbel, A. Korba, A. Salim, A. Gretton,
    "Maximum Mean Discrepancy Gradient Flow", NeurIPS 2019.

Let mu be the target distribution, available through samples y_1, ..., y_M, and let
nu_n be the empirical distribution of the particles x_1, ..., x_N at iteration n.
The flow minimizes F(nu) = 1/2 MMD^2(mu, nu) by moving each particle along the
negative gradient of the witness function between nu_n and mu,

    f(z)      = (1/N) sum_j k(x_j, z) - (1/M) sum_m k(y_m, z),
    x_i(n+1)  = x_i(n) - gamma grad f(x_i(n) + beta_n u_i(n)),   u_i(n) ~ N(0, I),

where beta_n = 0 recovers plain gradient descent on the sample-based MMD. The
gradient is evaluated at a blurred position but applied to the original particle,
which helps escape local optima without changing the global optimum.

Counterfactual variant. The target may instead be a weighted kernel expansion
chi_hat = sum_i v_i k(y_i, .), such as a doubly robust counterfactual policy mean
embedding (Zenati, Bozkurt, Gretton, 2025) whose weights can be signed. Passing
`target_weights` replaces the uniform target term by sum_i v_i k(y_i, z) in the
witness, so the flow minimizes F(Q) = 1/2 ||mu_Q - chi_hat||^2.

The kernel bandwidth must be fixed before the flow: if the kernel uses the length
scale heuristic, evaluate kernel(target_samples) once beforehand to lock the scale.
"""

import torch
from typing import Callable, Dict, Optional, Tuple
from tqdm.auto import trange


def squared_mmd(kernel,
                X: torch.Tensor,
                Y: torch.Tensor,
                unbiased: bool = False) -> torch.Tensor:
    """
    Squared maximum mean discrepancy between the empirical distributions of X and Y,

        MMD^2 = E[k(x, x')] - 2 E[k(x, y)] + E[k(y, y')].

    Parameters
    ----------
    kernel : Kernel
    X : torch.Tensor of shape (n, d)
    Y : torch.Tensor of shape (m, d)
    unbiased : bool
        If True, uses the unbiased U-statistic (diagonal terms removed), which can be
        slightly negative. Default False (biased V-statistic).

    Returns
    -------
    torch.Tensor (scalar)
    """
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)
    if unbiased:
        n, m = X.shape[0], Y.shape[0]
        term_XX = (K_XX.sum() - K_XX.diagonal().sum()) / (n * (n - 1))
        term_YY = (K_YY.sum() - K_YY.diagonal().sum()) / (m * (m - 1))
    else:
        term_XX = K_XX.mean()
        term_YY = K_YY.mean()
    return term_XX - 2.0 * K_XY.mean() + term_YY


def embedding_squared_distance(kernel,
                               particles: torch.Tensor,
                               target_points: torch.Tensor,
                               target_weights: torch.Tensor) -> torch.Tensor:
    """
    Squared RKHS distance between the empirical embedding of the particles and a
    weighted kernel expansion chi_hat = sum_i v_i k(y_i, .),

        ||mu_Q - chi_hat||^2 = (1/N^2) sum_{j,l} k(x_j, x_l)
                               - (2/N) sum_{j,i} v_i k(x_j, y_i)
                               + sum_{i,i'} v_i v_{i'} k(y_i, y_{i'}).

    For uniform weights v_i = 1/M this equals the biased squared MMD.

    Parameters
    ----------
    kernel : Kernel
    particles : torch.Tensor of shape (N, d)
    target_points : torch.Tensor of shape (M, d)
    target_weights : torch.Tensor of shape (M,)
        Expansion weights, possibly signed.

    Returns
    -------
    torch.Tensor (scalar)
    """
    K_XX = kernel(particles, particles)
    K_XY = kernel(particles, target_points)
    K_YY = kernel(target_points, target_points)
    return K_XX.mean() - 2.0 * (K_XY @ target_weights).mean() \
        + target_weights @ (K_YY @ target_weights)


def witness_gradient(kernel,
                     particles: torch.Tensor,
                     target_samples: torch.Tensor,
                     eval_points: torch.Tensor,
                     target_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gradient of the witness function evaluated at `eval_points`, computed with
    autograd:

        grad f(z) = (1/N) sum_j grad_2 k(x_j, z) - (1/M) sum_m grad_2 k(y_m, z).

    With `target_weights` v, the target term becomes sum_m v_m grad_2 k(y_m, z).

    Parameters
    ----------
    kernel : Kernel
    particles : torch.Tensor of shape (N, d)
    target_samples : torch.Tensor of shape (M, d)
    eval_points : torch.Tensor of shape (N, d)
        Points z at which the gradient is evaluated (the particles, or their noisy
        versions x + beta u).
    target_weights : torch.Tensor of shape (M,), optional
        Expansion weights of the target embedding. None means uniform 1/M.

    Returns
    -------
    torch.Tensor of shape (N, d)
    """
    z = eval_points.detach().requires_grad_(True)
    if target_weights is None:
        target_term = kernel(target_samples.detach(), z).mean(dim=0)
    else:
        target_term = target_weights.detach() @ kernel(target_samples.detach(), z)
    f = kernel(particles.detach(), z).mean(dim=0) - target_term
    # Each f_i depends only on z_i, so the gradient of the sum yields all rows at once.
    grad, = torch.autograd.grad(f.sum(), z)
    return grad


def mmd_gradient_flow(kernel,
                      target_samples: torch.Tensor,
                      initial_particles: torch.Tensor,
                      step_size: float = 0.1,
                      n_iterations: int = 10000,
                      noise_schedule: Optional[Callable[[int], float]] = None,
                      target_weights: Optional[torch.Tensor] = None,
                      log_every: int = 100,
                      snapshot_every: Optional[int] = None,
                      unbiased_mmd: bool = False,
                      generator: Optional[torch.Generator] = None,
                      verbose: bool = True,
                      ) -> Tuple[torch.Tensor, Dict]:
    """
    Runs the (noisy) MMD gradient flow.

    Parameters
    ----------
    kernel : Kernel
        Kernel with a fixed bandwidth (see the module docstring).
    target_samples : torch.Tensor of shape (M, d)
        Samples of the target mu, or expansion points of the target embedding.
    initial_particles : torch.Tensor of shape (N, d)
    step_size : float
        Step size gamma of the forward-Euler update.
    n_iterations : int
    noise_schedule : callable n -> beta_n, optional
        Noise level per iteration (see the schedule helpers below). None disables
        noise injection.
    target_weights : torch.Tensor of shape (M,), optional
        Expansion weights of the target embedding. When given, the recorded
        objective is ||mu_Q - chi_hat||^2 instead of MMD^2.
    log_every : int
        Records the objective every `log_every` iterations and at the last one.
    snapshot_every : int, optional
        If given, stores a copy of the particles every `snapshot_every` iterations.
    unbiased_mmd : bool
        Whether the recorded MMD^2 uses the unbiased estimator (ignored when
        `target_weights` is given).
    generator : torch.Generator, optional
        Source of the injected noise, for reproducibility.
    verbose : bool
        Shows a progress bar.

    Returns
    -------
    particles : torch.Tensor of shape (N, d)
    history : dict with keys "iterations", "squared_mmd", "noise_levels", "snapshots".
    """
    particles = initial_particles.detach().clone()
    target_samples = target_samples.detach()
    history = {"iterations": [], "squared_mmd": [], "noise_levels": [], "snapshots": {}}

    iterations = trange(n_iterations, desc="MMD flow") if verbose else range(n_iterations)
    for n in iterations:
        beta_n = float(noise_schedule(n)) if noise_schedule is not None else 0.0
        if beta_n > 0.0:
            noise = torch.randn(particles.shape, generator=generator,
                                dtype=particles.dtype, device=particles.device)
            eval_points = particles + beta_n * noise
        else:
            eval_points = particles

        gradient = witness_gradient(kernel, particles, target_samples, eval_points,
                                    target_weights=target_weights)
        particles = particles - step_size * gradient

        if snapshot_every is not None and n % snapshot_every == 0:
            history["snapshots"][n] = particles.clone()
        if n % log_every == 0 or n == n_iterations - 1:
            with torch.no_grad():
                if target_weights is None:
                    objective = squared_mmd(kernel, particles, target_samples,
                                            unbiased=unbiased_mmd)
                else:
                    objective = embedding_squared_distance(kernel, particles,
                                                           target_samples, target_weights)
            history["iterations"].append(n)
            history["squared_mmd"].append(objective.item())
            history["noise_levels"].append(beta_n)

    return particles, history


def constant_noise(beta0: float) -> Callable[[int], float]:
    """beta_n = beta0 for all n."""
    return lambda n: beta0


def noise_until(beta0: float, cutoff: int) -> Callable[[int], float]:
    """beta_n = beta0 for n < cutoff, then 0 (Arbel et al., 2019, Appendix G.2)."""
    return lambda n: beta0 if n < cutoff else 0.0


def halving_noise(beta0: float, halve_every: int) -> Callable[[int], float]:
    """beta_n = beta0 * 0.5^(n // halve_every)."""
    return lambda n: beta0 * 0.5 ** (n // halve_every)
