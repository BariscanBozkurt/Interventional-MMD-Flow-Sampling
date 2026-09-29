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


class GaussianMixturePolicy(Policy):
    """
    Mixture of diagonal Gaussians sharing a base mean:

        pi(a | x) = sum_k w_k N(mean_fn(x) + shift_k, diag(scale^2)).

    The point of this family is the MEAN-PRESERVING case. With two equally
    weighted components at shifts -delta and +delta the conditional action mean
    is exactly mean_fn(x) -- the same as the unmixed policy -- while the
    conditional variance rises by delta^2 and the law becomes bimodal. An
    intervention of that form is invisible to any procedure that reports only
    E[Y(pi)], because the mean action (and, for an outcome linear in the action,
    the mean outcome) is unchanged; the outcome DISTRIBUTION nevertheless moves.
    That is the sharpest available motivation for sampling the whole law rather
    than estimating a scalar functional of it.

    Parameters
    ----------
    mean_fn : callable
        Maps contexts (n, d_x) to the shared base mean (n, d_a).
    shifts : array of shape (n_components, d_a) or (n_components,)
        Per-component offset added to the base mean. A one-dimensional array is
        read as one scalar offset per component, broadcast across the action.
    scale : float or array of shape (d_a,)
        Per-dimension standard deviation, shared by the components.
    weights : array of shape (n_components,), optional
        Mixture weights; defaults to uniform. Must be positive and sum to one.
    """

    def __init__(self, mean_fn, shifts, scale, weights=None):
        self.mean_fn = mean_fn
        shifts = np.asarray(shifts, dtype=float)
        # A 1-D array is one SCALAR offset per component, i.e. (n_components, 1) --
        # not a single component with a d_a-dimensional offset, which is what
        # np.atleast_2d would produce.
        self.shifts = shifts[:, None] if shifts.ndim == 1 else shifts
        if self.shifts.ndim != 2:
            raise ValueError("shifts must be 1- or 2-dimensional")
        self.scale = np.asarray(scale, dtype=float)

        n_components = self.shifts.shape[0]
        if weights is None:
            weights = np.full(n_components, 1.0 / n_components)
        self.weights = np.asarray(weights, dtype=float)

        if self.weights.shape != (n_components,):
            raise ValueError("weights must have one entry per mixture component")
        if np.any(self.weights <= 0.0) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must be positive and sum to one")
        if self.scale.ndim > 1 or np.any(self.scale <= 0.0) or not np.isfinite(self.scale).all():
            raise ValueError("scale must be a positive scalar or one-dimensional array")
        if not np.isfinite(self.shifts).all():
            raise ValueError("shifts must be finite")

    def _base_mean(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        mean = np.asarray(self.mean_fn(X), dtype=float)
        if mean.ndim == 1:
            mean = mean[:, None]
        if mean.ndim != 2 or mean.shape[0] != X.shape[0] or not np.isfinite(mean).all():
            raise ValueError("mean_fn(X) must return a finite (n, d_a) matrix")
        return mean

    def _component_means(self, X: np.ndarray) -> np.ndarray:
        """(n_components, n, d_a): the base mean offset by each component shift."""
        mean = self._base_mean(X)
        shifts = np.broadcast_to(self.shifts, (self.shifts.shape[0], mean.shape[1]))
        return mean[None, :, :] + shifts[:, None, :]

    def mean(self, X: np.ndarray) -> np.ndarray:
        """E[a | x], the weight-averaged component mean. Mean-preserving iff this
        equals mean_fn(X), i.e. iff the weights and shifts satisfy sum_k w_k s_k = 0."""
        return np.tensordot(self.weights, self._component_means(X), axes=(0, 0))

    def sample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        means = self._component_means(X)
        n, d = means.shape[1], means.shape[2]
        scale = np.broadcast_to(self.scale, (d,))
        which = rng.choice(len(self.weights), size=n, p=self.weights)
        return means[which, np.arange(n), :] + scale * rng.standard_normal((n, d))

    def log_prob(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        means = self._component_means(X)
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A[:, None]
        if A.shape != means.shape[1:] or not np.isfinite(A).all():
            raise ValueError("A must be finite and match the policy's action shape")

        d = A.shape[1]
        scale = np.broadcast_to(self.scale, (d,))
        z = (A[None, :, :] - means) / scale
        log_norm = np.log(scale).sum() + 0.5 * d * np.log(2.0 * np.pi)
        per_component = -0.5 * (z ** 2).sum(axis=2) - log_norm      # (n_components, n)
        return _log_sum_exp(per_component + np.log(self.weights)[:, None], axis=0)


def _log_sum_exp(values: np.ndarray, axis: int) -> np.ndarray:
    """Stable log-sum-exp, so a far-off mixture component cannot underflow the sum."""
    largest = np.max(values, axis=axis, keepdims=True)
    largest = np.where(np.isfinite(largest), largest, 0.0)
    return (largest + np.log(np.exp(values - largest).sum(axis=axis, keepdims=True))).squeeze(axis)


def mean_preserving_mixture(base_policy: GaussianPolicy,
                            delta: float) -> GaussianMixturePolicy:
    """
    The 50/50 mean-preserving mixture around a Gaussian policy: components at
    -delta and +delta with equal weight, so E[a | x] is unchanged and
    Var[a | x] rises by delta^2. Used as the target policy in the stochastic
    policy benchmark; see `GaussianMixturePolicy`.
    """
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    return GaussianMixturePolicy(base_policy.mean_fn, [-delta, delta], base_policy.scale)


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
