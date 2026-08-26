"""
Conditional action policies pi(a | x) for logged-bandit data-generating processes.

Holding the policy as an object keeps the three places it is needed consistent:
data generation samples from it, the counterfactual policy mean embedding draws
its resampling actions a~_j ~ pi(. | x_j) from it (Zenati et al., 2025,
arXiv:2506.02793), and importance weights are the exact density ratio of two
policies evaluated in log space. The target policy pi is the intervention chosen
by design, so it is known exactly at estimation time; only the logging policy
pi_0 has to be estimated.

Arrays are NumPy; `sample` takes an explicit `numpy.random.Generator` so a
data-generating process keeps one reproducible random stream.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np


class Policy(ABC):
    """
    A conditional action distribution pi(a | x).

    Subclasses implement `sample` (used to generate data and to draw the
    estimator's resampling actions) and `log_prob` (used for importance
    weights). A deterministic intervention do(a*) has no density, so it cannot
    implement `log_prob` meaningfully -- density-ratio weights degenerate there.
    """

    @abstractmethod
    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one action per context: X of shape (n, d_x) -> actions (n, d_a)."""

    @abstractmethod
    def log_prob(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """log pi(a_i | x_i) for paired rows: X (n, d_x), A (n, d_a) -> (n,)."""

    def prob(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """pi(a_i | x_i) for paired rows."""
        return np.exp(self.log_prob(X, A))


class GaussianPolicy(Policy):
    """
    Diagonal Gaussian policy pi(a | x) = N(mean_fn(x), diag(scale^2)).

    Parameters
    ----------
    mean_fn : callable
        Maps contexts (n, d_x) to action means (n, d_a). This is the
        confounding channel: the action depends on the context through it.
    scale : float or array of shape (d_a,)
        Per-dimension standard deviation (a scalar gives an isotropic policy).
    """

    def __init__(self, mean_fn: Callable[[np.ndarray], np.ndarray],
                 scale: Union[float, np.ndarray]):
        self.mean_fn = mean_fn
        self.scale = np.asarray(scale, dtype=float)

        if self.scale.ndim > 1:
            raise ValueError("scale must be a scalar or a one-dimensional array")
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0.0):
            raise ValueError("scale must contain finite, positive values")

    def _mean_and_scale(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite matrix")

        mean = np.asarray(self.mean_fn(X), dtype=float)
        if mean.ndim == 1:
            mean = mean[:, None]
        if mean.ndim != 2 or mean.shape[0] != X.shape[0]:
            raise ValueError("mean_fn(X) must have shape (n, d_a)")
        if not np.isfinite(mean).all():
            raise ValueError("mean_fn(X) returned non-finite values")

        try:
            scale = np.broadcast_to(self.scale, (mean.shape[1],))
        except ValueError as error:
            raise ValueError("scale is incompatible with the action dimension") from error
        return mean, scale

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean, scale = self._mean_and_scale(X)
        return mean + scale * rng.standard_normal(mean.shape)

    def log_prob(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        mean, scale = self._mean_and_scale(X)
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A[:, None]
        if A.shape != mean.shape or not np.isfinite(A).all():
            raise ValueError("A must be finite and have the same shape as the policy mean")

        z = (A - mean) / scale
        d = mean.shape[1]
        log_scale = np.log(scale).sum()
        return -0.5 * (z ** 2).sum(axis=1) - log_scale - 0.5 * d * np.log(2.0 * np.pi)


def context_shift_gaussian(shift: float, scale: float) -> GaussianPolicy:
    """
    The policy family of the dSprites design (arXiv:2506.02793, Sec. 14.1.5):

        a | x ~ N((x - 1/2) + shift, scale^2 I).

    The logging and target policies are this family at -shift and +shift, so
    `shift` controls the counterfactual gap (and the overlap) and `scale` the
    policy spread.
    """
    return GaussianPolicy(lambda X: (X - 0.5) + shift, scale)


def importance_weights(target_policy: Policy,
                       logging_policy: Policy,
                       X: np.ndarray,
                       A: np.ndarray) -> np.ndarray:
    """
    Density-ratio weights w_i = pi(a_i | x_i) / pi_0(a_i | x_i) at the logged
    actions, computed in log space. Pass an estimated logging policy to get
    pi / pi_0_hat; the target policy stays exact because it is known by design.

    Parameters
    ----------
    target_policy, logging_policy : Policy
    X : np.ndarray of shape (n, d_x)
    A : np.ndarray of shape (n, d_a)
        The logged actions.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    log_weights = log_importance_weights(target_policy, logging_policy, X, A)
    smallest = np.nextafter(0.0, 1.0)
    if log_weights.max(initial=-np.inf) > np.log(np.finfo(float).max):
        raise FloatingPointError("importance weights overflow; inspect policy overlap")
    if log_weights.min(initial=np.inf) < np.log(smallest):
        raise FloatingPointError("importance weights underflow; inspect policy overlap")
    return np.exp(log_weights)


def log_importance_weights(target_policy: Policy,
                           logging_policy: Policy,
                           X: np.ndarray,
                           A: np.ndarray) -> np.ndarray:
    """Log density ratios log pi(a | x) - log pi_0(a | x)."""
    log_target = np.asarray(target_policy.log_prob(X, A), dtype=float)
    log_logging = np.asarray(logging_policy.log_prob(X, A), dtype=float)

    if log_target.ndim != 1 or log_target.shape != log_logging.shape:
        raise ValueError("policy log_prob methods must return matching vectors")
    log_weights = log_target - log_logging
    if not np.isfinite(log_weights).all():
        raise ValueError("policy log density ratio contains non-finite values")
    return log_weights
