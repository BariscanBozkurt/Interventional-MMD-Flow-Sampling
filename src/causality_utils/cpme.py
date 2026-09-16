"""
Counterfactual policy mean embedding (CPME) estimators.

Implements the plug-in and one-step doubly robust estimators of

    H. Zenati, B. Bozkurt, A. Gretton,
    "Doubly-Robust Estimation of Counterfactual Policy Mean Embeddings", 2025.

Logged bandit data (x_i, a_i, y_i) ~ P_0 are collected under a logging policy
pi_0(a | x). The counterfactual outcome distribution nu(pi) of a target policy pi
is represented by its mean embedding chi(pi) = E_{nu(pi)}[phi_Y(Y)] in the outcome
RKHS. Every estimator below is a finite expansion over the logged outcomes,

    chi_hat(pi) = sum_i v_i k_Y(y_i, .),

so an estimator is a weight vector v in R^n. With K = K_X * K_A (Hadamard product
of the context and action Gram matrices), K_A_pi[i, j] = k_A(a_i, a~_j) for target
actions a~_j ~ pi(. | x_j), and importance weights w_i = pi(a_i | x_i) / pi_0(a_i | x_i):

    plug-in:       v_PI = (1/n) (K + lambda I)^{-1} (K_X * K_A_pi) 1
    doubly robust: v_DR = v_PI + (1/n) [ w - (K + lambda I)^{-1} K w ]

The regularization convention is (K + lambda I).
"""

import torch
from typing import Sequence, Tuple, Optional


def cpme_plugin_weights(K_x: torch.Tensor,
                        K_a: torch.Tensor,
                        K_a_target: torch.Tensor,
                        reg_lambda: float) -> torch.Tensor:
    """
    Plug-in CPME weights v_PI = (1/n) (K + lambda I)^{-1} (K_X * K_A_pi) 1.

    Parameters
    ----------
    K_x : torch.Tensor of shape (n, n)
        Gram matrix k_X(x_i, x_j) of the logged contexts.
    K_a : torch.Tensor of shape (n, n)
        Gram matrix k_A(a_i, a_j) of the logged actions.
    K_a_target : torch.Tensor of shape (n, n)
        Cross Gram matrix k_A(a_i, a~_j) between logged and target-policy actions,
        with a~_j ~ pi(. | x_j) drawn at the j-th logged context.
    reg_lambda : float
        Ridge regularization of the conditional mean embedding.

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    n = K_x.shape[0]
    K = K_x * K_a
    eye = torch.eye(n, dtype=K.dtype, device=K.device)
    rhs = (K_x * K_a_target).mean(dim=1)
    return torch.linalg.solve(K + reg_lambda * eye, rhs)


def cpme_dr_weights(K_x: torch.Tensor,
                    K_a: torch.Tensor,
                    K_a_target: torch.Tensor,
                    importance_weights: torch.Tensor,
                    reg_lambda: float) -> torch.Tensor:
    """
    Doubly robust (one-step) CPME weights

        v_DR = v_PI + (1/n) [ w - (K + lambda I)^{-1} K w ].

    The correction term adds the importance-weighted residuals
    w_i (phi_Y(y_i) - mu_hat(Y | a_i, x_i)) to the plug-in estimate.

    Parameters
    ----------
    Same as `cpme_plugin_weights`, plus:
    importance_weights : torch.Tensor of shape (n,)
        w_i = pi(a_i | x_i) / pi_0(a_i | x_i).

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    n = K_x.shape[0]
    K = K_x * K_a
    eye = torch.eye(n, dtype=K.dtype, device=K.device)
    v_plugin = torch.linalg.solve(K + reg_lambda * eye, (K_x * K_a_target).mean(dim=1))
    mu_hat_weights = torch.linalg.solve(K + reg_lambda * eye, K @ importance_weights)
    return v_plugin + (importance_weights - mu_hat_weights) / n


def cpme_dr_weights_crossfit(K_x: torch.Tensor,
                             K_a: torch.Tensor,
                             K_a_target: torch.Tensor,
                             importance_weights: torch.Tensor,
                             fold_ids,
                             reg_lambda) -> torch.Tensor:
    """
    Cross-fitted doubly robust CPME weights.

    The same one-step estimator as `cpme_dr_weights`, but the conditional mean
    embedding nuisance behind both the plug-in term and the residual correction is
    fit only on the observations outside each fold. For fold k with held-out rows H
    and training rows T, each held-out pseudo-outcome

        Gamma_i = mu_hat_pi^(-k)(x_i) + w_i (phi(y_i) - mu_hat^(-k)(x_i, a_i)),  i in H,

    contributes expansion weights over the training outcomes through the two ridge
    solves on K_TT, plus w_i on its own outcome. Averaging all n pseudo-outcomes
    gives the fold-size-weighted aggregate

        v[T] += (1/n) (K_TT + lambda_k I)^{-1} ((K_X * K_A_pi)[T, H] 1 - K[T, H] w_H),
        v[H] += (1/n) w_H.

    When the training block is extended to all observations this formula reduces
    exactly to `cpme_dr_weights` (an algebraic property; an actual single-fold
    assignment would leave every training block empty and is rejected). Pass the
    fold assignment used to cross-fit the importance weights so both nuisances share
    the same folds.

    Parameters
    ----------
    K_x, K_a, K_a_target, importance_weights
        As in `cpme_dr_weights`; use cross-fitted importance weights evaluated
        out-of-fold with the same `fold_ids`.
    fold_ids : array-like of shape (n,)
        Integer fold assignment per observation. Degenerate assignments (fewer than
        two folds, or a fold whose training block has fewer than two rows) raise a
        ValueError.
    reg_lambda : float or dict {fold: float}
        Ridge regularization; a dict gives each fold its own value, selected on that
        fold's training block.

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    n = K_x.shape[0]
    fold_ids = torch.as_tensor(fold_ids, device=K_x.device).reshape(n)
    folds = fold_ids.unique()
    if len(folds) < 2:
        raise ValueError("fold_ids must contain at least two folds")
    for fold in folds:
        if int((fold_ids != fold).sum()) < 2:
            raise ValueError("every fold must leave at least two training observations")
    K = K_x * K_a
    K_pi = K_x * K_a_target
    v = torch.zeros(n, dtype=K.dtype, device=K.device)

    for fold in folds:
        held_out = fold_ids == fold
        train = ~held_out
        lam = reg_lambda[int(fold)] if isinstance(reg_lambda, dict) else reg_lambda
        K_tt = K[train][:, train]
        eye = torch.eye(K_tt.shape[0], dtype=K.dtype, device=K.device)
        rhs = K_pi[train][:, held_out].sum(dim=1) - K[train][:, held_out] @ importance_weights[held_out]
        v[train] += torch.linalg.solve(K_tt + lam * eye, rhs) / n
        v[held_out] += importance_weights[held_out] / n
    return v


def select_regularization(K: torch.Tensor,
                          Y: torch.Tensor,
                          lambda_grid: Sequence[float] = (1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4),
                          n_folds: int = 3,
                          seed: Optional[int] = None) -> Tuple[float, torch.Tensor]:
    """
    Selects the ridge parameter by k-fold cross-validated kernel ridge regression of
    an outcome on the joint Gram matrix.

    Parameters
    ----------
    K : torch.Tensor of shape (n, n)
        Gram matrix of the logged (x_i, a_i) pairs, e.g. K_x * K_a.
    Y : torch.Tensor of shape (n,) or (n, 1)
        Outcome used for the validation loss.
    lambda_grid : sequence of float
    n_folds : int
    seed : int, optional
        Seed for the fold shuffling.

    Returns
    -------
    best_lambda : float
    cv_errors : torch.Tensor of shape (len(lambda_grid),)
        Mean squared held-out error for each candidate.
    """
    n = K.shape[0]
    y = Y.reshape(n)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    folds = [perm[f::n_folds] for f in range(n_folds)]

    cv_errors = []
    for reg_lambda in lambda_grid:
        fold_errors = []
        for f in range(n_folds):
            test_idx = folds[f]
            train_idx = torch.cat([folds[g] for g in range(n_folds) if g != f])
            K_tr = K[train_idx][:, train_idx]
            eye = torch.eye(len(train_idx), dtype=K.dtype, device=K.device)
            alpha = torch.linalg.solve(K_tr + reg_lambda * eye, y[train_idx])
            pred = K[test_idx][:, train_idx] @ alpha
            fold_errors.append(((pred - y[test_idx]) ** 2).mean())
        cv_errors.append(torch.stack(fold_errors).mean())

    cv_errors = torch.stack(cv_errors)
    best_lambda = float(lambda_grid[int(cv_errors.argmin())])
    return best_lambda, cv_errors


def _blocked_rhs(gram, gram_pi, train, held_out, importance_weights, block: int = 2048):
    """(K_pi[T, H] 1 - K[T, H] w_H), accumulated over column blocks of H.

    Materialising the whole (|T|, |H|) block is what makes the exact solver run out
    of memory at n = 20000; the right-hand side only ever needs column sums, so it
    is accumulated a slice at a time.
    """
    total = None
    for start in range(0, len(held_out), block):
        columns = held_out[start:start + block]
        piece = (gram_pi(train, columns).sum(dim=1)
                 - gram(train, columns) @ importance_weights[columns])
        total = piece if total is None else total + piece
    return total


def cpme_dr_weights_crossfit_nystrom(gram,
                                     gram_pi,
                                     importance_weights: torch.Tensor,
                                     fold_ids,
                                     reg_lambda,
                                     n_landmarks: int = 1024,
                                     generator: Optional[torch.Generator] = None,
                                     jitter: float = 1e-8) -> torch.Tensor:
    """
    Nystrom-approximated cross-fitted doubly robust CPME weights.

    The estimator is the one of `cpme_dr_weights_crossfit`; only each fold's ridge
    solve is approximated. With m landmarks L drawn from that fold's training block
    T and U = K[T, L],

        K_TT ~= U K_LL^{-1} U^T,
        (U K_LL^{-1} U^T + lambda I)^{-1} r
            = (1/lambda) [ r - U (lambda K_LL + U^T U)^{-1} U^T r ]

    by Woodbury, so a fold costs O(|T| m^2 + m^3) instead of O(|T|^3) and no
    |T| x |T| matrix is formed. That is what makes the DR support the whole logged
    sample rather than a few thousand rows.

    Taking L = T recovers the exact solver up to floating point.

    Parameters
    ----------
    gram, gram_pi : callable
        `gram(rows, cols)` returns the (len(rows), len(cols)) block of
        K = K_X * K_A; `gram_pi` the corresponding block of K_pi = K_X * K_A_pi.
        Blocks are computed on demand so the full Gram never exists.
    importance_weights : torch.Tensor of shape (n,)
        Cross-fitted weights, evaluated out-of-fold with the same `fold_ids`.
    fold_ids : array-like of shape (n,)
    reg_lambda : float or dict {fold: float}
    n_landmarks : int
        Landmarks per fold, capped at the training-block size.
    generator : torch.Generator, optional
        Draws the landmark subsets.
    jitter : float
        Relative eigenvalue floor for the landmark Gram: components below
        `jitter * max eigenvalue` are dropped when whitening.

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    import numpy as np

    fold_ids = np.asarray(fold_ids).reshape(-1)
    n = len(fold_ids)
    if importance_weights.shape[0] != n:
        raise ValueError("importance_weights and fold_ids disagree on n")
    folds = np.unique(fold_ids)
    if len(folds) < 2:
        raise ValueError("fold_ids must contain at least two folds")

    weights = importance_weights
    v = torch.zeros(n, dtype=weights.dtype, device=weights.device)
    for fold in folds:
        held_out = np.where(fold_ids == fold)[0]
        train = np.where(fold_ids != fold)[0]
        if len(train) < 2:
            raise ValueError("every fold must leave at least two training observations")
        lam = reg_lambda[int(fold)] if isinstance(reg_lambda, dict) else reg_lambda

        rhs = _blocked_rhs(gram, gram_pi, train, held_out, weights)
        m = min(n_landmarks, len(train))
        order = torch.randperm(len(train), generator=generator)[:m].numpy()
        landmarks = train[np.sort(order)]

        U = gram(train, landmarks)                       # (|T|, m)
        K_ll = gram(landmarks, landmarks)                # (m, m)

        # Whitened Nystrom features. The naive Woodbury form
        # (1/lam)[r - U (lam K_LL + U^T U)^{-1} U^T r] is algebraically right but
        # numerically poor: it squares the Gram's condition number and the 1/lam
        # prefactor amplifies whatever survives, so a small selected lambda
        # (select_regularization searches down to 1e-4) can destroy the weights
        # entirely. Factoring K_LL = V diag(s) V^T and setting Phi = U V s^{-1/2}
        # gives K_TT ~= Phi Phi^T with the SAME approximation but an inner system
        # lam I + Phi^T Phi that stays well conditioned for any lambda.
        eigenvalues, eigenvectors = torch.linalg.eigh(K_ll)
        keep = eigenvalues > jitter * float(eigenvalues.max().clamp_min(0.0))
        if not bool(keep.any()):
            raise ValueError("landmark Gram is numerically zero")
        eigenvalues, eigenvectors = eigenvalues[keep], eigenvectors[:, keep]
        phi = U @ (eigenvectors / eigenvalues.sqrt())    # (|T|, k)
        inner = (phi.T @ phi
                 + lam * torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device))
        correction = phi @ torch.linalg.solve(inner, phi.T @ rhs)
        v[train] += (rhs - correction) / (lam * n)
        v[held_out] += weights[held_out] / n
    return v
