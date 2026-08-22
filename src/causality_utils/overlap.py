"""Diagnostics for policy overlap and importance-weight behaviour."""

import numpy as np


def effective_sample_size(weights) -> float:
    """
    Kish effective sample size of a set of importance weights,

        ESS = (sum_i w_i)^2 / sum_i w_i^2.

    Equal weights give ESS equal to the number of observations; a few dominant
    weights drive it toward one. Under weak overlap the raw density ratios develop
    heavy upper tails, so the ESS is far below the nominal sample size and the
    variance of any importance-weighted estimator is correspondingly inflated.

    Parameters
    ----------
    weights : array-like of shape (n,)
        Non-negative importance weights.

    Returns
    -------
    float
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = weights.sum()
    if total == 0.0:
        return 0.0
    return float(total ** 2 / np.dot(weights, weights))
