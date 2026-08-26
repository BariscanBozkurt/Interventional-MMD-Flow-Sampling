"""
Cross-fitted importance-weight estimation for continuous actions.

The doubly robust CPME estimator needs the raw density ratio

    w(x, a) = pi(a | x) / pi_0(a | x),

where the target policy pi is known and the logging policy pi_0 is estimated
from logged pairs (X, A). Two estimators are implemented:

1. classify logged pairs against pairs sampled from the target policy;
2. fit the logging-policy density and divide the known target density by it.

Both use cross-fitting: every reported weight is evaluated by a model that did
not train on that logged observation. Arrays are NumPy, matching
`causality_utils.policies`.
"""

from typing import Callable, Dict, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from causality_utils.crossfit import make_crossfit_fold_ids
from causality_utils.policies import GaussianPolicy, Policy


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Converts a one- or two-dimensional array to a finite matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_logged_data(X: np.ndarray, A: np.ndarray):
    """Checks and returns paired context and action matrices."""
    X = _as_matrix(X, "X")
    A = _as_matrix(A, "A")
    if X.shape[0] != A.shape[0]:
        raise ValueError("X and A must have the same number of rows")
    if X.shape[0] < 2:
        raise ValueError("at least two logged observations are required")
    return X, A


def _validate_fold_ids(fold_ids: np.ndarray, n_samples: int) -> np.ndarray:
    """Checks user-supplied cross-fitting fold assignments."""
    fold_ids = np.asarray(fold_ids)
    if fold_ids.shape != (n_samples,):
        raise ValueError("fold_ids must have one entry per logged observation")
    if not np.issubdtype(fold_ids.dtype, np.integer):
        raise ValueError("fold_ids must contain integers")
    if len(np.unique(fold_ids)) < 2:
        raise ValueError("fold_ids must contain at least two folds")
    largest_fold = np.unique(fold_ids, return_counts=True)[1].max()
    if n_samples - largest_fold < 2:
        raise ValueError("every fold must leave at least two training observations")
    return fold_ids.astype(int, copy=False)


def _pair_features(X: np.ndarray,
                   A: np.ndarray,
                   feature_fn: Optional[Callable]) -> np.ndarray:
    """Constructs classifier features from paired contexts and actions."""
    if feature_fn is None:
        features = np.concatenate([X, A], axis=1)
    else:
        features = np.asarray(feature_fn(X, A), dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2 or features.shape[0] != X.shape[0]:
        raise ValueError("feature_fn(X, A) must return a matrix with one row per pair")
    if not np.isfinite(features).all():
        raise ValueError("classifier features must be finite")
    return features


def _fit_logistic_classifier(features: np.ndarray,
                             labels: np.ndarray,
                             sample_weights: np.ndarray,
                             reg_lambda: float,
                             max_iter: int):
    """Fits a standardized weighted logistic regression with SciPy."""
    total_weight = sample_weights.sum()
    mean = (sample_weights[:, None] * features).sum(axis=0) / total_weight
    variance = (sample_weights[:, None] * (features - mean) ** 2).sum(axis=0)
    variance = variance / total_weight
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    features_std = (features - mean) / scale

    n_features = features.shape[1]

    def objective(parameters):
        coefficients = parameters[:n_features]
        intercept = parameters[-1]
        logits = features_std @ coefficients + intercept
        losses = np.logaddexp(0.0, logits) - labels * logits
        loss = np.dot(sample_weights, losses) / total_weight
        loss = loss + 0.5 * reg_lambda * np.dot(coefficients, coefficients)

        errors = sample_weights * (expit(logits) - labels) / total_weight
        gradient = np.empty_like(parameters)
        gradient[:n_features] = features_std.T @ errors + reg_lambda * coefficients
        gradient[-1] = errors.sum()
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(n_features + 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"logistic density-ratio fit failed: {result.message}")

    coefficients = result.x[:n_features]
    intercept = float(result.x[-1])
    return coefficients, intercept, mean, scale


def _predict_logit(features: np.ndarray, fitted) -> np.ndarray:
    """Evaluates a fitted standardized logistic classifier."""
    coefficients, intercept, mean, scale = fitted
    return ((features - mean) / scale) @ coefficients + intercept


def importance_weights_from_log_ratio(log_ratio: np.ndarray) -> np.ndarray:
    """Exponentiates log ratios without silently clipping raw DR weights."""
    log_ratio = np.asarray(log_ratio, dtype=float)
    if log_ratio.ndim != 1 or len(log_ratio) == 0 or not np.isfinite(log_ratio).all():
        raise ValueError("log density ratios must be a nonempty finite vector")

    smallest = np.nextafter(0.0, 1.0)
    if log_ratio.max(initial=-np.inf) > np.log(np.finfo(float).max):
        raise FloatingPointError("importance weights overflow; inspect policy overlap")
    if log_ratio.min(initial=np.inf) < np.log(smallest):
        raise FloatingPointError("importance weights underflow; inspect policy overlap")
    return np.exp(log_ratio)


class LinearGaussianPolicy(GaussianPolicy):
    """Diagonal Gaussian policy with a linear conditional mean."""

    def __init__(self,
                 intercept: np.ndarray,
                 coefficients: np.ndarray,
                 scale: np.ndarray):
        self.intercept = np.asarray(intercept, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.intercept.ndim != 1 or self.coefficients.ndim != 2:
            raise ValueError("intercept and coefficients must be a vector and a matrix")
        if self.coefficients.shape[1] != self.intercept.shape[0]:
            raise ValueError("coefficient and intercept action dimensions must agree")
        super().__init__(self._linear_mean, scale)

    def _linear_mean(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.coefficients.shape[0]:
            raise ValueError("X has the wrong context dimension")
        return self.intercept + X @ self.coefficients


def fit_linear_gaussian_policy(X: np.ndarray,
                               A: np.ndarray,
                               ridge: float = 1e-6,
                               min_scale: float = 1e-3) -> LinearGaussianPolicy:
    """
    Fits A | X as a diagonal Gaussian with a linear mean.

    This is a deliberately small default logging-policy model. A more flexible
    conditional density estimator can be supplied to
    `crossfit_logging_policy_log_prob` through `policy_fitter`.

    Parameters
    ----------
    X : np.ndarray of shape (n, d_x)
    A : np.ndarray of shape (n, d_a)
    ridge : float
        Ridge penalty on the slopes. The intercept is not penalized.
    min_scale : float
        Lower bound on each fitted residual standard deviation.
    """
    X, A = _validate_logged_data(X, A)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if not np.isfinite(min_scale) or min_scale <= 0.0:
        raise ValueError("min_scale must be finite and positive")

    design = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ A)

    residuals = A - design @ fitted
    scale = np.sqrt(np.mean(residuals ** 2, axis=0))
    scale = np.maximum(scale, min_scale)
    return LinearGaussianPolicy(fitted[0], fitted[1:], scale)


def crossfit_classifier_log_ratio(
        X: np.ndarray,
        A: np.ndarray,
        target_policy: Policy,
        n_folds: int = 5,
        n_target_actions: int = 1,
        reg_lambda: float = 1e-4,
        max_iter: int = 500,
        feature_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        fold_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimates log pi(a | x) - log pi_0(a | x) by cross-fitted classification.

    Logged pairs have class zero. Pairs formed by drawing actions from the known
    target policy at training-fold contexts have class one. The two classes have
    equal total training weight, so the held-out classifier odds estimate the
    required density ratio directly.

    When `n_target_actions > 1`, every target pair receives sample weight
    `1 / n_target_actions`. This keeps the class prior balanced instead of making
    the ratio depend on an arbitrary number of target draws.

    The default classifier is linear logistic regression on `[X, A]`. Pass a
    fixed, deterministic, row-wise `feature_fn(X, A)` to add nonlinear or
    problem-specific features. Do not fit, rank, or batch-standardize inside this
    function because logged, target, and held-out rows are evaluated separately.
    Returned log ratios are in the original row order.
    """
    X, A = _validate_logged_data(X, A)
    if not isinstance(n_target_actions, (int, np.integer)) or n_target_actions < 1:
        raise ValueError("n_target_actions must be a positive integer")
    if not np.isfinite(reg_lambda) or reg_lambda < 0.0:
        raise ValueError("reg_lambda must be finite and nonnegative")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    rng = np.random.default_rng(seed)
    if fold_ids is None:
        fold_ids = make_crossfit_fold_ids(len(X), n_folds=n_folds, seed=seed)
    else:
        fold_ids = _validate_fold_ids(fold_ids, len(X))
    log_ratio = np.empty(len(X), dtype=float)

    all_indices = np.arange(len(X))
    for fold in np.unique(fold_ids):
        held_out = all_indices[fold_ids == fold]
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[held_out] = False
        train = all_indices[train_mask]
        X_train = X[train]
        A_train = A[train]

        target_actions = []
        for _ in range(n_target_actions):
            draw = _as_matrix(target_policy.sample(X_train, rng), "target actions")
            if draw.shape != A_train.shape:
                raise ValueError("target-policy actions must match the logged action shape")
            target_actions.append(draw)

        X_target = np.tile(X_train, (n_target_actions, 1))
        A_target = np.concatenate(target_actions, axis=0)
        logged_features = _pair_features(X_train, A_train, feature_fn)
        target_features = _pair_features(X_target, A_target, feature_fn)
        features = np.concatenate([logged_features, target_features], axis=0)

        labels = np.concatenate([
            np.zeros(len(X_train)),
            np.ones(len(X_target)),
        ])
        sample_weights = np.concatenate([
            np.ones(len(X_train)),
            np.full(len(X_target), 1.0 / n_target_actions),
        ])
        fitted = _fit_logistic_classifier(
            features, labels, sample_weights, reg_lambda, max_iter
        )

        held_out_features = _pair_features(X[held_out], A[held_out], feature_fn)
        log_ratio[held_out] = _predict_logit(held_out_features, fitted)

    return log_ratio


def estimate_classifier_importance_weights(
        X: np.ndarray,
        A: np.ndarray,
        target_policy: Policy,
        n_folds: int = 5,
        n_target_actions: int = 1,
        reg_lambda: float = 1e-4,
        max_iter: int = 500,
        feature_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        fold_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns raw cross-fitted classifier density ratios for the DR estimator.

    Ratios are not normalized or clipped. Call `crossfit_classifier_log_ratio`
    directly when log-scale evaluation is needed.
    """
    log_ratio = crossfit_classifier_log_ratio(
        X,
        A,
        target_policy,
        n_folds=n_folds,
        n_target_actions=n_target_actions,
        reg_lambda=reg_lambda,
        max_iter=max_iter,
        feature_fn=feature_fn,
        seed=seed,
        fold_ids=fold_ids,
    )
    return importance_weights_from_log_ratio(log_ratio)


def crossfit_logging_policy_log_prob(
        X: np.ndarray,
        A: np.ndarray,
        n_folds: int = 5,
        policy_fitter: Optional[Callable] = None,
        ridge: float = 1e-6,
        min_scale: float = 1e-3,
        seed: Optional[int] = None,
        fold_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimates held-out log pi_0(A_i | X_i) with cross-fitting.

    `policy_fitter(X_train, A_train)` may return any object with a `log_prob`
    method. The default is `fit_linear_gaussian_policy`.
    """
    X, A = _validate_logged_data(X, A)
    if fold_ids is None:
        fold_ids = make_crossfit_fold_ids(len(X), n_folds=n_folds, seed=seed)
    else:
        fold_ids = _validate_fold_ids(fold_ids, len(X))
    log_prob = np.empty(len(X), dtype=float)

    if policy_fitter is None:
        def policy_fitter(X_train, A_train):
            return fit_linear_gaussian_policy(
                X_train, A_train, ridge=ridge, min_scale=min_scale
            )

    all_indices = np.arange(len(X))
    for fold in np.unique(fold_ids):
        held_out = all_indices[fold_ids == fold]
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[held_out] = False
        train = all_indices[train_mask]
        fitted_policy = policy_fitter(X[train], A[train])
        if not hasattr(fitted_policy, "log_prob"):
            raise TypeError("policy_fitter must return an object with a log_prob method")

        values = np.asarray(fitted_policy.log_prob(X[held_out], A[held_out]), dtype=float)
        if values.shape != (len(held_out),) or not np.isfinite(values).all():
            raise ValueError("fitted logging policy returned invalid held-out log densities")
        log_prob[held_out] = values

    return log_prob


def importance_weights_from_logging_log_prob(
        target_policy: Policy,
        X: np.ndarray,
        A: np.ndarray,
        logging_log_prob: np.ndarray) -> np.ndarray:
    """Combines a known target policy with held-out logging log densities."""
    X, A = _validate_logged_data(X, A)
    logging_log_prob = np.asarray(logging_log_prob, dtype=float)
    if logging_log_prob.shape != (len(X),) or not np.isfinite(logging_log_prob).all():
        raise ValueError("logging_log_prob must be a finite vector aligned with X and A")

    log_target = np.asarray(target_policy.log_prob(X, A), dtype=float)
    if log_target.shape != (len(X),) or not np.isfinite(log_target).all():
        raise ValueError("target policy returned invalid log densities")
    return importance_weights_from_log_ratio(log_target - logging_log_prob)


def estimate_logging_policy_importance_weights(
        X: np.ndarray,
        A: np.ndarray,
        target_policy: Policy,
        n_folds: int = 5,
        policy_fitter: Optional[Callable] = None,
        ridge: float = 1e-6,
        min_scale: float = 1e-3,
        seed: Optional[int] = None,
        fold_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimates raw ratios using a cross-fitted logging-policy density.

    For reuse with several target policies, call
    `crossfit_logging_policy_log_prob` once and subtract its result from each
    target policy's log density before exponentiating.
    """
    X, A = _validate_logged_data(X, A)
    log_logging = crossfit_logging_policy_log_prob(
        X,
        A,
        n_folds=n_folds,
        policy_fitter=policy_fitter,
        ridge=ridge,
        min_scale=min_scale,
        seed=seed,
        fold_ids=fold_ids,
    )
    return importance_weights_from_logging_log_prob(
        target_policy, X, A, log_logging
    )


def importance_weight_diagnostics(weights: np.ndarray,
                                  oracle_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Summarizes raw importance weights and optional oracle log-ratio error.

    Returns the empirical mean, ESS, ESS fraction, maximum, and upper quantiles.
    When oracle weights are supplied, also returns RMSE on the log-weight scale.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("importance weights must be finite and positive")

    weight_scale = weights.max()
    scaled_weights = weights / weight_scale
    scaled_total = scaled_weights.sum()
    scaled_sum_squares = np.dot(scaled_weights, scaled_weights)
    ess = scaled_total ** 2 / scaled_sum_squares
    diagnostics = {
        "n": int(len(weights)),
        "mean": float(weight_scale * scaled_weights.mean()),
        "std": float(weight_scale * scaled_weights.std()),
        "ess": float(ess),
        "ess_fraction": float(ess / len(weights)),
        "min": float(weights.min()),
        "median": float(np.quantile(weights, 0.50)),
        "q90": float(np.quantile(weights, 0.90)),
        "q95": float(np.quantile(weights, 0.95)),
        "q99": float(np.quantile(weights, 0.99)),
        "max": float(weights.max()),
    }

    if oracle_weights is not None:
        oracle_weights = np.asarray(oracle_weights, dtype=float)
        if oracle_weights.shape != weights.shape:
            raise ValueError("oracle_weights must have the same shape as weights")
        if not np.isfinite(oracle_weights).all() or np.any(oracle_weights <= 0.0):
            raise ValueError("oracle_weights must be finite and positive")
        diagnostics["log_rmse"] = float(
            np.sqrt(np.mean((np.log(weights) - np.log(oracle_weights)) ** 2))
        )

    return diagnostics
