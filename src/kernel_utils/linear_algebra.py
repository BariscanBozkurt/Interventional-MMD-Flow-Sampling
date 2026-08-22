"""Pairwise distance computations used by the kernel classes."""

import torch
from typing import Optional


@torch.jit.script
def pairwise_squared_distance(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Pairwise squared Euclidean distances ||X_i - Y_j||^2.

    Computed as ||x||^2 + ||y||^2 - 2 x^T y and clamped at zero for numerical
    stability.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
    Y : torch.Tensor of shape (m, d), optional
        Defaults to X.

    Returns
    -------
    torch.Tensor of shape (n, m)
    """
    if Y is None:
        Y = X
    X_norm_sq = (X ** 2).sum(dim=-1, keepdim=True)
    Y_norm_sq = (Y ** 2).sum(dim=-1, keepdim=True).T
    dist_sq = X_norm_sq + Y_norm_sq - 2.0 * (X @ Y.T)
    return torch.clamp(dist_sq, min=0.0)


@torch.jit.script
def pairwise_distance(X: torch.Tensor, Y: Optional[torch.Tensor] = None,
                      eps: float = 1e-12) -> torch.Tensor:
    """
    Pairwise Euclidean distances ||X_i - Y_j||.

    The squared distances are clamped below by `eps` before the square root so that
    autograd never differentiates sqrt at exactly zero, which would produce NaN
    gradients on the diagonal of a self-distance matrix.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
    Y : torch.Tensor of shape (m, d), optional
        Defaults to X.
    eps : float
        Lower clamp on the squared distances.

    Returns
    -------
    torch.Tensor of shape (n, m)
    """
    return torch.sqrt(torch.clamp(pairwise_squared_distance(X, Y), min=eps))
