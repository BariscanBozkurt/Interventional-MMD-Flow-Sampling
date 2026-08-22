"""
Kernel classes for maximum mean discrepancy computations and MMD gradient flows.

All kernels are differentiable with respect to their inputs for a fixed length
scale, as required by the particle updates of an MMD gradient flow.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from .linear_algebra import pairwise_squared_distance


class Kernel(nn.Module, ABC):
    """Base class for all kernels."""

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self,
                X: torch.Tensor,
                Y: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Computes the kernel matrix K(X, Y) of shape (n, m)."""

    def get_params(self) -> Dict[str, Any]:
        """Returns the kernel parameters."""
        return {}

    def __repr__(self) -> str:
        param_str = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({param_str})"


class LengthScaleKernel(Kernel):
    """
    Base class for kernels governed by a single length scale.

    Implements the one-time, self-locking quantile heuristic: on the first
    self-kernel call K(X, X), a quantile of the pairwise distances of X is
    converted into a length scale, the parameter is fixed, and the heuristic is
    switched off. Subsequent calls reuse the fixed value, so the heuristic must be
    triggered on the reference data before any cross-kernel evaluation.

    Parameters
    ----------
    length_scale : float or torch.Tensor
        Initial length scale (ignored once the heuristic runs).
    use_length_scale_heuristic : bool
        Whether to set the length scale from the first self-kernel call.
    length_scale_heuristic_quantile : float
        Quantile of the pairwise distances used by the heuristic (0.5 gives the
        median heuristic).
    heuristic_subset_limit : int
        Maximum number of rows used to evaluate the heuristic.
    trainable : bool
        Whether the log-length-scale is a trainable parameter.
    """

    def __init__(self,
                 length_scale: Union[float, torch.Tensor] = 1.0,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 heuristic_subset_limit: int = 5000,
                 trainable: bool = False,
                 **kwargs) -> None:
        super().__init__()
        if isinstance(length_scale, (float, int)):
            length_scale = torch.as_tensor(float(length_scale))
        self.log_length_scale = nn.Parameter(torch.log(length_scale), requires_grad=trainable)
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.heuristic_subset_limit = heuristic_subset_limit
        self._is_initialized = False

    @property
    def length_scale(self) -> torch.Tensor:
        """The length scale (exponential of the stored log-parameter)."""
        return torch.exp(self.log_length_scale)

    @abstractmethod
    def _heuristic_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Pairwise distance matrix, in the kernel's own metric, for the heuristic."""

    @abstractmethod
    def _scale_from_quantile(self, quantile_value: torch.Tensor) -> torch.Tensor:
        """Maps a distance quantile to a length scale."""

    def calculate_and_fix_length_scale(self, X: torch.Tensor):
        """Evaluates the quantile heuristic on X, sets the length scale, and locks it."""
        with torch.no_grad():
            n = X.shape[0]
            if n > self.heuristic_subset_limit:
                indices = torch.randperm(n, device=X.device)[:self.heuristic_subset_limit]
                X = X[indices]
            distances = self._heuristic_distances(X)
            n_sub = X.shape[0]
            upper = torch.triu_indices(n_sub, n_sub, offset=1, device=X.device)
            off_diagonal = distances[upper[0], upper[1]]
            if off_diagonal.numel() > 0:
                quantile_value = torch.quantile(off_diagonal, self.length_scale_heuristic_quantile)
                self.log_length_scale.data.copy_(torch.log(self._scale_from_quantile(quantile_value)))
        self.use_length_scale_heuristic = False
        self._is_initialized = True

    def _maybe_apply_heuristic(self, X: torch.Tensor, Y: Optional[torch.Tensor]):
        is_self_kernel = (Y is None) or (X.data_ptr() == Y.data_ptr())
        if self.use_length_scale_heuristic and is_self_kernel and not self._is_initialized:
            self.calculate_and_fix_length_scale(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            'length_scale': self.length_scale.item(),
            'use_length_scale_heuristic': self.use_length_scale_heuristic,
            'length_scale_heuristic_quantile': self.length_scale_heuristic_quantile,
            'trainable': self.log_length_scale.requires_grad,
        }


class RBF(LengthScaleKernel):
    """
    Gaussian (radial basis function) kernel

        K(x, y) = exp(-||x - y||^2 / (2 l^2)).

    The heuristic sets l = sqrt(q / 2) where q is the chosen quantile of the pairwise
    squared distances; the default quantile 0.5 gives the median heuristic.
    """

    def _heuristic_distances(self, X: torch.Tensor) -> torch.Tensor:
        return pairwise_squared_distance(X)

    def _scale_from_quantile(self, quantile_value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(quantile_value / 2.0)

    def forward(self,
                X: torch.Tensor,
                Y: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        self._maybe_apply_heuristic(X, Y)
        if Y is None:
            Y = X
        return torch.exp(-pairwise_squared_distance(X, Y) / (2.0 * self.length_scale ** 2))


class RieszKernel(Kernel):
    """
    Riesz (energy-distance) kernel

        K(x, y) = -||x - y||^s,     0 < s <= 2.

    The default s = 1 induces the multivariate energy distance, which is
    characteristic: the MMD vanishes if and only if the two distributions are equal.
    The kernel is only conditionally positive definite, which is sufficient for MMD
    gradient flows, and its witness gradient does not vanish for far-apart points.
    The boundary case s = 2 is degenerate: it compares only the means of the two
    distributions and the induced flow is a rigid translation of the particle cloud.

    The Euclidean distance is singular at x = y, so the kernel differentiates the
    eps-guarded form -(||x - y||^2 + eps^2)^(s/2), keeping autograd through particle
    positions NaN-free. The kernel is scale-free: there is no length scale to set.
    """

    def __init__(self, s: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if not (0.0 < s <= 2.0):
            raise ValueError(f"the Riesz kernel requires 0 < s <= 2, got s = {s}")
        self.s = float(s)
        self.eps = float(eps)

    def forward(self,
                X: torch.Tensor,
                Y: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        if Y is None:
            Y = X
        distances_sq = pairwise_squared_distance(X, Y)
        return -(distances_sq + self.eps ** 2).pow(self.s / 2.0)

    def get_params(self) -> Dict[str, Any]:
        return {"s": self.s, "eps": self.eps}


class IMQKernel(Kernel):
    """
    Inverse multiquadric kernel

        K(x, y) = (c^2 + ||x - y||^2)^beta,     beta < 0,  c > 0.

    The characteristic choice beta = -1/2 gives K = 1 / sqrt(c^2 + ||x - y||^2).
    Unlike the Gaussian kernel, the IMQ kernel has heavy tails: its witness
    gradient does not vanish for far-apart points, so flow particles are still
    pulled toward distant target mass. Because c > 0 the kernel and its gradient
    are smooth everywhere and need no eps-guard for autograd through particle
    positions. c and beta are fixed at construction; `imq_c_from_median` gives a
    data-driven c.
    """

    def __init__(self, c: float = 1.0, beta: float = -0.5):
        super().__init__()
        if beta >= 0.0:
            raise ValueError(f"the IMQ kernel requires beta < 0, got beta = {beta}")
        if c <= 0.0:
            raise ValueError(f"the IMQ kernel requires c > 0, got c = {c}")
        self.c = float(c)
        self.beta = float(beta)

    def forward(self,
                X: torch.Tensor,
                Y: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        if Y is None:
            Y = X
        distances_sq = pairwise_squared_distance(X, Y)
        return (self.c ** 2 + distances_sq).pow(self.beta)

    def get_params(self) -> Dict[str, Any]:
        return {"c": self.c, "beta": self.beta}


def imq_c_from_median(X: torch.Tensor, subset_limit: int = 5000) -> float:
    """
    Median-heuristic scale for the IMQ kernel: the median off-diagonal pairwise
    Euclidean distance of X, computed on a subsample when X is large.
    """
    with torch.no_grad():
        n = X.shape[0]
        if n > subset_limit:
            X = X[torch.randperm(n, device=X.device)[:subset_limit]]
        d2 = pairwise_squared_distance(X)
        m = X.shape[0]
        upper = torch.triu_indices(m, m, offset=1, device=X.device)
        return float(d2[upper[0], upper[1]].clamp_min(0.0).sqrt().median())
