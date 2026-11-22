"""Fixed-IMQ random Fourier features and deterministic particle updates.

For k(x,y)=(c**2+||x-y||**2)**(-1/2), draw independent standard normals
g_l and u_l in R^d, and set omega_l=g_l*u_l/c. This is the Gaussian--Gamma
spectral mixture: g_l**2/(2*c**2) ~ Gamma(1/2, rate=c**2). Paired features
[cos(x omega.T), sin(x omega.T)] / sqrt(c*L) have constant diagonal 1/c.

The finite feature map is not characteristic over unrestricted distributions.
A cached target is valid only with its original frequencies, bandwidth, latent
representation, support, and coefficient vector. The experiment runner records
those identities. Target coefficients may be signed and are never normalized.
"""

import math

import numpy as np
import torch


def _points(name, values):
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name} must be a nonempty matrix")
    if values.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")


def _same_type(reference, *values):
    if any(v.dtype != reference.dtype or v.device != reference.device for v in values):
        raise ValueError("all tensors must share dtype and device")


def _bandwidth(c):
    if not math.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")


def _feature_inputs(points, omega, c):
    _points("points", points)
    _points("omega", omega)
    _same_type(points, omega)
    _bandwidth(c)
    if points.shape[1] != omega.shape[1]:
        raise ValueError("point and frequency dimensions must agree")


def sample_imq_frequencies(latent_dim, n_features, c, seed=0, *,
                           device=None, dtype=torch.float32):
    """Draw IMQ frequencies with deterministic nested prefixes.

    Parameters
    ----------
    latent_dim : int
        Input dimension.
    n_features : int
        Even total dimension D=2L, counting both sine and cosine coordinates.
    c : float
        Scale in the unnormalized IMQ kernel.
    seed : int
        Nonnegative seed for a local CPU NumPy PCG64 generator.
    device, dtype
        Device and precision of the returned frequency matrix.

    Returns
    -------
    torch.Tensor, shape (n_features // 2, latent_dim)
        Frequencies. For fixed dimension, scale, and seed, smaller draws are
        exact row prefixes of larger draws before device/dtype conversion.
    """
    if not isinstance(latent_dim, int) or latent_dim < 1:
        raise ValueError("latent_dim must be a positive integer")
    if not isinstance(n_features, int) or n_features < 2 or n_features % 2:
        raise ValueError("n_features must be a positive even integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    _bandwidth(c)
    # Each row consumes the mixing scalar and its d normal directions together.
    draws = np.random.default_rng(seed).standard_normal(
        (n_features // 2, latent_dim + 1))
    omega = draws[:, :1] * draws[:, 1:] / c
    return torch.as_tensor(omega, dtype=dtype, device=device)


def imq_features(points, omega, c):
    """Return paired features, ordered as all cosines followed by all sines.

    This function retains autograd support for independent derivative checks.
    """
    _feature_inputs(points, omega, c)
    phase = points @ omega.T
    scale = 1.0 / math.sqrt(c * len(omega))
    return torch.cat((phase.cos(), phase.sin()), dim=1) * scale


@torch.no_grad()
def cache_target_mean(targets, weights, omega, c, block_size=2048):
    """Compute sum_i weights_i * psi(targets_i) without normalizing weights.

    Blocks limit temporary feature storage. Coefficients and points must
    already have the same dtype/device as the frequency matrix.
    """
    _feature_inputs(targets, omega, c)
    _same_type(targets, weights)
    if weights.ndim != 1 or len(weights) != len(targets):
        raise ValueError("weights must have one entry per target")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not bool(torch.isfinite(targets).all() and torch.isfinite(weights).all()
                and torch.isfinite(omega).all()):
        raise ValueError("targets, weights and frequencies must be finite")
    total = targets.new_zeros(2 * len(omega))
    for start in range(0, len(targets), block_size):
        stop = start + block_size
        total += weights[start:stop] @ imq_features(targets[start:stop], omega, c)
    return total


def _rff_gradient(particles, target_mean, omega, c, eval_points=None):
    """Analytic witness gradient; caller has validated tensor compatibility."""
    scale = 1.0 / math.sqrt(c * len(omega))
    phase = particles @ omega.T
    cosine, sine = phase.cos(), phase.sin()
    residual_cos = scale * cosine.mean(0) - target_mean[:len(omega)]
    residual_sin = scale * sine.mean(0) - target_mean[len(omega):]
    if eval_points is not None:
        phase = eval_points @ omega.T
        cosine, sine = phase.cos(), phase.sin()
    return scale * (-sine * residual_cos + cosine * residual_sin) @ omega


@torch.no_grad()
def rff_witness_gradient(particles, target_mean, omega, c, eval_points=None):
    """Evaluate J_psi(x)^T(mean_j psi(particle_j) - cached_target).

    The source mean is fixed while differentiating the query. There is no
    extra 1/M factor in this Wasserstein-scaled witness gradient.
    """
    _feature_inputs(particles, omega, c)
    _same_type(particles, target_mean)
    if target_mean.shape != (2 * len(omega),):
        raise ValueError("target_mean must contain 2*len(omega) coordinates")
    if eval_points is not None:
        _feature_inputs(eval_points, omega, c)
    return _rff_gradient(particles, target_mean, omega, c, eval_points)


def _flow_arguments(initial, step_size, n_iterations):
    _points("initial", initial)
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not isinstance(n_iterations, int) or n_iterations < 0:
        raise ValueError("n_iterations must be a nonnegative integer")


@torch.no_grad()
def rff_euler_flow(initial, target_mean, omega, c, step_size, n_iterations):
    """Move equal-weight interacting particles against one cached target.

    Returns only final particles. Timing, diagnostics, and synchronization
    belong to the experiment runner; there are none inside the update loop.
    """
    _flow_arguments(initial, step_size, n_iterations)
    _feature_inputs(initial, omega, c)
    _same_type(initial, target_mean)
    if target_mean.shape != (2 * len(omega),):
        raise ValueError("target_mean must contain 2*len(omega) coordinates")
    particles = initial.detach().clone()
    for _ in range(n_iterations):
        gradient = _rff_gradient(particles, target_mean, omega, c)
        particles.add_(gradient, alpha=-step_size)
    return particles

def _imq_term_gradient(query, support, weights, c, block_size):
    """Derivative in query of a weighted IMQ expansion."""
    gradient = torch.zeros_like(query)
    query_norm = query.square().sum(1, keepdim=True)
    for start in range(0, len(support), block_size):
        block = support[start:start + block_size]
        distance = query_norm + block.square().sum(1)[None, :] - 2 * (query @ block.T)
        power = distance.clamp_min_(0).add_(c * c).pow_(-1.5)
        if weights is None:
            power.div_(len(support))
        else:
            power.mul_(weights[start:start + block_size][None, :])
        gradient += power @ block - query * power.sum(1, keepdim=True)
    return gradient


def _exact_inputs(particles, target_points, target_weights, c, eval_points, block_size):
    _points("particles", particles)
    _points("target_points", target_points)
    _same_type(particles, target_points, target_weights)
    _bandwidth(c)
    if particles.shape[1] != target_points.shape[1]:
        raise ValueError("particle and target dimensions must agree")
    if target_weights.shape != (len(target_points),):
        raise ValueError("target_weights must have one entry per target")
    if eval_points is not None:
        _points("eval_points", eval_points)
        _same_type(particles, eval_points)
        if eval_points.shape[1] != particles.shape[1]:
            raise ValueError("query and particle dimensions must agree")
    if block_size is not None and (not isinstance(block_size, int) or block_size < 1):
        raise ValueError("block_size must be a positive integer or None")


def _exact_gradient(particles, target_points, target_weights, c, eval_points, block_size):
    query = particles if eval_points is None else eval_points
    block_size = max(len(particles), len(target_points)) if block_size is None else block_size
    source = _imq_term_gradient(query, particles, None, c, block_size)
    target = _imq_term_gradient(query, target_points, target_weights, c, block_size)
    return source - target


@torch.no_grad()
def exact_imq_witness_gradient(particles, target_points, target_weights, c,
                               eval_points=None, block_size=None):
    """Exact IMQ witness derivative with raw signed target coefficients.

    This analytic implementation uses the same kernel and particle time scale
    as the deployed autograd witness, but avoids differentiating kernel graphs.
    """
    _exact_inputs(particles, target_points, target_weights, c, eval_points, block_size)
    return _exact_gradient(particles, target_points, target_weights, c,
                           eval_points, block_size)


@torch.no_grad()
def exact_imq_euler_flow(initial, target_points, target_weights, c,
                         step_size, n_iterations, block_size=None):
    """Exact analytic IMQ Euler updates without in-loop diagnostics."""
    _flow_arguments(initial, step_size, n_iterations)
    _exact_inputs(initial, target_points, target_weights, c, None, block_size)
    particles = initial.detach().clone()
    for _ in range(n_iterations):
        gradient = _exact_gradient(particles, target_points, target_weights, c,
                                   None, block_size)
        particles.add_(gradient, alpha=-step_size)
    return particles


__all__ = [
    "cache_target_mean",
    "exact_imq_euler_flow",
    "exact_imq_witness_gradient",
    "imq_features",
    "rff_euler_flow",
    "rff_witness_gradient",
    "sample_imq_frequencies",
]
