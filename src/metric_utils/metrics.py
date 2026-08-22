"""
Distribution comparison metrics for evaluating generated samples.

The sliced Wasserstein estimator follows the evaluation code of Dance et al. (2026)
so that reported numbers are directly comparable with that paper: random directions
on the unit sphere, projections truncated to the smaller sample size, and the
one-dimensional distance computed from sorted projections (the exact empirical
Wasserstein distance for equal sample sizes). That paper reports Wasserstein-1 for
the low-dimensional synthetic and benchmark tables and Sliced Wasserstein-2 for the
image experiments, so both orders are supported.
"""

import torch
from typing import Optional


@torch.no_grad()
def sliced_wasserstein(x,
                       y,
                       n_projections: int = 1000,
                       p: int = 1,
                       generator: Optional[torch.Generator] = None,
                       seed: Optional[int] = None) -> torch.Tensor:
    """
    Sliced Wasserstein distance of order p between empirical measures on R^d.

    Both samples are projected onto `n_projections` random directions; each
    projection is truncated to the smaller sample size and sorted, and the sorted
    differences are averaged (p = 1) or root-mean-squared (p = 2) jointly over
    samples and directions.

    Parameters
    ----------
    x : torch.Tensor or array-like of shape (n, d)
    y : torch.Tensor or array-like of shape (m, d)
        Converted to a tensor of x's dtype.
    n_projections : int
        Number of random directions.
    p : int
        Wasserstein order, 1 or 2.
    generator : torch.Generator, optional
        Source of the random directions, for reproducibility.
    seed : int, optional
        Convenience alternative to `generator`: seeds a fresh generator on x's
        device. Ignored when `generator` is given.

    Returns
    -------
    torch.Tensor (scalar)
    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    if generator is None and seed is not None:
        generator = torch.Generator(device=x.device.type).manual_seed(seed)

    theta = torch.randn(n_projections, x.shape[1], generator=generator,
                        dtype=x.dtype, device=x.device)
    theta = theta / theta.norm(dim=1, keepdim=True)

    proj_x = x @ theta.T
    proj_y = y @ theta.T

    n = min(proj_x.shape[0], proj_y.shape[0])
    proj_x = proj_x[:n].sort(dim=0).values
    proj_y = proj_y[:n].sort(dim=0).values

    if p == 1:
        return (proj_x - proj_y).abs().mean()
    if p == 2:
        return ((proj_x - proj_y) ** 2).mean().sqrt()
    raise ValueError("only p = 1 or p = 2 is supported")
