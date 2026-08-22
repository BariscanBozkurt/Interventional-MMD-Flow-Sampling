"""Cross-fitting utilities."""

import numpy as np
from typing import Optional


def make_crossfit_fold_ids(n_samples: int,
                           n_folds: int = 5,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Returns a shuffled integer fold assignment for every observation.

    One shared assignment should drive every cross-fitted nuisance (propensity or
    density ratio, conditional mean embedding, regularization selection), so that
    all nuisances used for an observation are fit without that observation.

    Parameters
    ----------
    n_samples : int
    n_folds : int
    seed : int, optional

    Returns
    -------
    np.ndarray of shape (n_samples,) with values in {0, ..., n_folds - 1}.
    """
    if n_samples < 3:
        raise ValueError("cross-fitting requires at least three observations")
    if n_folds < 2 or n_folds > n_samples:
        raise ValueError("n_folds must be between 2 and the number of observations")
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(n_samples, dtype=int)
    for fold, held_out in enumerate(np.array_split(rng.permutation(n_samples), n_folds)):
        fold_ids[held_out] = fold
    return fold_ids
