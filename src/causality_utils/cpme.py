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


def cpme_dr_weights_nystrom(gram,
                            gram_pi,
                            importance_weights: torch.Tensor,
                            reg_lambda: float,
                            n_landmarks: int = 1024,
                            generator: Optional[torch.Generator] = None,
                            jitter: float = 1e-8) -> torch.Tensor:
    """
    Full-sample (no-split) DR weights with a Nystrom ridge solve.

    The estimator is the one of `cpme_dr_weights` -- every row is used both to fit the
    conditional mean embedding and to evaluate its own residual -- but the two n x n
    solves are replaced by a Nystrom approximation on `n_landmarks` columns, so the
    Gram is never materialised. Without this there is no memory-feasible no-split path
    at the sizes the simulations use: at n = 20000 a dense Gram is 3.2 GB in float64 and
    the solve is cubic.

    The interface matches `cpme_dr_weights_crossfit_nystrom`: `gram(rows, cols)` and
    `gram_pi(rows, cols)` return blocks of K = K_X * K_A and K_pi = K_X * K_A_target on
    demand, so the caller can exploit whatever structure its kernels have.

    Whitened features are used for the same reason as in the cross-fitted solver: the
    naive Woodbury form squares the Gram's condition number and its 1/lambda prefactor
    amplifies the result, which destroys the weights at the small lambdas the selector
    can return.

    Parameters
    ----------
    gram, gram_pi : callable
        ``gram(rows, cols) -> (len(rows), len(cols))`` blocks of K and K_pi.
    importance_weights : torch.Tensor of shape (n,)
    reg_lambda : float
        A single scalar: the no-split fit has no folds. NOTE that the penalty is not
        scaled by the training-block size, so a no-split fit at training block n and a
        k-fold fit at block (1 - 1/k) n are NOT at matched effective regularisation for
        the same value -- see `select_regularization` and `lambda_n_half`.
    n_landmarks : int
    generator : torch.Generator, optional
        Controls the landmark draw.
    jitter : float
        Relative eigenvalue floor for the landmark Gram.

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    import numpy as np

    weights = importance_weights.reshape(-1).double()
    n = len(weights)
    rows = np.arange(n)
    lam = float(reg_lambda)

    # rhs = K_pi 1 - K w, accumulated in column blocks exactly as the cross-fitted
    # solver does, so neither n x n matrix is ever held.
    rhs = _blocked_rhs(gram, gram_pi, rows, rows, weights)

    m = min(n_landmarks, n)
    order = torch.randperm(n, generator=generator)[:m].numpy()
    landmarks = rows[np.sort(order)]
    U = gram(rows, landmarks)
    K_ll = gram(landmarks, landmarks)
    eigenvalues, eigenvectors = torch.linalg.eigh(K_ll)
    keep = eigenvalues > jitter * float(eigenvalues.max().clamp_min(0.0))
    if not bool(keep.any()):
        raise ValueError("landmark Gram is numerically zero")
    eigenvalues, eigenvectors = eigenvalues[keep], eigenvectors[:, keep]
    phi = U @ (eigenvectors / eigenvalues.sqrt())
    inner = (phi.T @ phi
             + lam * torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device))
    correction = phi @ torch.linalg.solve(inner, phi.T @ rhs)
    return (rhs - correction) / (lam * n) + weights / n


def lambda_n_half(n: int, constant: float = 1.0) -> float:
    """
    The ridge regime the same-sample theory gives as sufficient: lambda_n ~ n^(-1/2).

    Zenati et al. and the same-sample notes state the outcome-side bound for a
    kernel-ridge fit whose penalty multiplies the AVERAGED empirical loss, so the
    corresponding Gram-matrix regularisation is n * lambda_n -- which is what the solves
    in this module apply when handed this value. Under it the fitted operator has
    ||C_hat||_HS = O_p(1) and prediction error O_p(n^(-1/4)), which is what the
    equicontinuity argument needs; cross-validated selection targets prediction error
    instead and gives no such norm control.

    Use it to hold the effective penalty matched when comparing a cross-fitted fit with
    a no-split one: cross-validated lambdas are selected per training block and so differ
    systematically between the two.

    Parameters
    ----------
    n : int
        Number of observations the fit averages over.
    constant : float
        Multiplier on n^(-1/2); the theory fixes only the rate, not the constant.

    Returns
    -------
    float
    """
    if n < 1:
        raise ValueError("n must be positive")
    return float(constant) * float(n) ** -0.5


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


def _retained_whitening(K_ll: torch.Tensor, jitter: float):
    """
    Whitening map G = V_r D_r^{-1/2} of the landmark Gram, on its positive eigenspace.

    Parameters
    ----------
    K_ll : torch.Tensor of shape (m, m)
        Landmark Gram W = K_S[L, L].
    jitter : float
        Relative eigenvalue floor: eigenpairs at or below `jitter * max eigenvalue`
        are treated as numerical zeros and dropped. Keeping roundoff-positive
        directions would divide by their square root and destabilise the whitening;
        dropping genuinely positive ones is a real extra subspace approximation, so
        the floor should stay a roundoff-level constant.

    Returns
    -------
    torch.Tensor of shape (m, r)
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(K_ll)
    keep = eigenvalues > jitter * float(eigenvalues.max().clamp_min(0.0))
    if not bool(keep.any()):
        raise ValueError("landmark Gram is numerically zero")
    return eigenvectors[:, keep] / eigenvalues[keep].sqrt()


def _projected_nystrom_coefficients(U: torch.Tensor,
                                    K_ll: torch.Tensor,
                                    rhs_landmarks: torch.Tensor,
                                    gram_ridge: float,
                                    jitter: float) -> torch.Tensor:
    """
    Projected Nystrom coefficients b_T = U (U^T U + rho W)^+ r_L, evaluated whitened.

    With W = V_r D_r V_r^T on the retained positive eigenspace, G = V_r D_r^{-1/2},
    Phi = U G and q = G^T r_L, restricting the normal matrix to Range(W) gives the
    numerically stable equivalent

        b_T = Phi (Phi^T Phi + rho I_r)^{-1} q,

    which is what is computed. Whitening is only a coordinate change inside that
    subspace; it needs no length-|T| right-hand side, which is the whole point --
    only the m-vector r_L enters.

    Parameters
    ----------
    U : torch.Tensor of shape (N, m)
        K_S[T, L].
    K_ll : torch.Tensor of shape (m, m)
        W = K_S[L, L].
    rhs_landmarks : torch.Tensor of shape (m,)
        r_L = K_pi[L, H] 1 - K_S[L, H] w_H.
    gram_ridge : float
        The ridge rho of the unnormalised matrix system. For the averaged-loss
        penalty lambda_N of the theory, rho = N * lambda_N with N = |T|.
    jitter : float
        Relative eigenvalue floor, see `_retained_whitening`.

    Returns
    -------
    torch.Tensor of shape (N,)
    """
    G = _retained_whitening(K_ll, jitter)
    phi = U @ G                                          # (N, r)
    q = G.T @ rhs_landmarks                              # (r,)
    inner = (phi.T @ phi
             + gram_ridge * torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device))
    return phi @ torch.linalg.solve(inner, q)


def _landmark_indices(train, n_landmarks: int, generator, given, fold=None):
    """
    Landmarks for one training block: either the supplied indices or a uniform draw.

    Parameters
    ----------
    train : numpy.ndarray
        Training-block row indices.
    n_landmarks : int
        Landmark budget, capped at the training-block size.
    generator : torch.Generator or None
        Drives the uniform draw. A generator is STATEFUL: handing the same object to
        two sequential calls does not reproduce a draw unless its seed is reset.
    given : array-like, dict, or None
        Precomputed landmarks; a dict is keyed by `int(fold)`. Must be a subset of
        `train`.
    fold : optional
        Fold label used to index a dict.

    Returns
    -------
    numpy.ndarray
    """
    import numpy as np

    if given is not None:
        chosen = given[int(fold)] if isinstance(given, dict) else given
        chosen = np.unique(np.asarray(chosen).reshape(-1))
        if not np.all(np.isin(chosen, train)):
            raise ValueError("landmarks must be a subset of the training block")
        return chosen
    m = min(n_landmarks, len(train))
    order = torch.randperm(len(train), generator=generator)[:m].numpy()
    return train[np.sort(order)]


def cpme_dr_weights_crossfit_projected_nystrom(gram,
                                               gram_pi,
                                               importance_weights: torch.Tensor,
                                               fold_ids,
                                               reg_lambda,
                                               n_landmarks: int = 1024,
                                               generator: Optional[torch.Generator] = None,
                                               jitter: float = 1e-8,
                                               landmarks=None) -> torch.Tensor:
    """
    Cross-fitted DR CPME weights with a PROJECTED Nystrom conditional mean embedding.

    Same estimand, same folds, same length-n coefficient vector, and the same
    aggregation as `cpme_dr_weights_crossfit_nystrom` -- but a different estimator of
    the CME nuisance, and the one the consistency rate is stated for. Here the ridge
    regression is restricted to the landmark-generated subspace

        G_L = { s -> sum_{l in L} k_S(s, S_l) c_l },

    so with U = K_S[T, L], W = K_S[L, L] and rho the matrix ridge the coefficients on
    the training outcomes are

        b_T = U (U^T U + rho W)^+ r_L,     r_L = K_pi[L, H] 1 - K_S[L, H] w_H.

    Contrast `cpme_dr_weights_crossfit_nystrom`, which applies an approximate Gram
    inverse to the EXACT full training right-hand side r_T,

        b_T = (Phi Phi^T + rho I_N)^{-1} r_T,

    an exact Woodbury computation of a different object. The two coincide exactly when
    r_T = U W_r^+ r_L on the retained eigenspace, and need not otherwise. Both are
    kept: this function adds the projected estimator, it does not replace the other.

    Two practical consequences of projecting. The right-hand side is built only at the
    landmarks, so the fold touches an m x |H| cross-kernel rather than an |T| x |H|
    one, and no length-|T| right-hand side is ever formed. The per-fold cost is
    O(|T| m^2 + m^3 + |H| m).

    Unlike `cpme_dr_weights_nystrom` this does not cast to float64; pass float64
    inputs and float64 gram blocks, as the ridge solves need them.

    Parameters
    ----------
    gram, gram_pi : callable
        `gram(rows, cols)` returns the (len(rows), len(cols)) block of K = K_X * K_A;
        `gram_pi` the corresponding block of K_pi = K_X * K_A_pi. Only the blocks
        (T, L), (L, L) and (L, H) are requested.
    importance_weights : torch.Tensor of shape (n,)
        Cross-fitted weights, evaluated out-of-fold with the same `fold_ids`.
    fold_ids : array-like of shape (n,)
        Integer fold assignment; at least two folds, each leaving at least two
        training rows.
    reg_lambda : float or dict {fold: float}
        The ridge rho of the matrix system, exactly as the other solvers in this
        module take it. The theory's averaged-loss penalty lambda_N corresponds to
        rho = |T| * lambda_N, so `lambda_n_half` must be rescaled by the
        training-block size before it matches that regime.
    n_landmarks : int
        Landmarks per fold, capped at the training-block size. The consistency
        statement needs m to grow with |T|; a fixed budget is a computational choice.
    generator : torch.Generator, optional
        Draws the landmark subsets, uniformly without replacement. Stateful -- reseed
        it to reproduce a draw, or pass `landmarks`.
    jitter : float
        Relative eigenvalue floor for the landmark Gram, see `_retained_whitening`.
    landmarks : dict {fold: array-like} or array-like, optional
        Precomputed landmark rows, bypassing the draw. A dict gives each fold its own
        set; a single array is used for every fold and must lie in each training
        block. Each set must be a subset of that fold's training rows.

    Returns
    -------
    torch.Tensor of shape (n,)
        v with v[T] += b_T / n and v[H] += w_H / n accumulated over folds.
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
        rho = reg_lambda[int(fold)] if isinstance(reg_lambda, dict) else reg_lambda
        L = _landmark_indices(train, n_landmarks, generator, landmarks, fold)

        # r_L: the DR right-hand side evaluated at the landmark rows only. This is
        # _blocked_rhs called with landmarks in place of train -- an m-vector.
        r_L = _blocked_rhs(gram, gram_pi, L, held_out, weights)
        U = gram(train, L)                               # (|T|, m)
        K_ll = gram(L, L)                                # (m, m)

        v[train] += _projected_nystrom_coefficients(U, K_ll, r_L, float(rho), jitter) / n
        v[held_out] += weights[held_out] / n
    return v


def cpme_dr_weights_projected_nystrom(gram,
                                      gram_pi,
                                      importance_weights: torch.Tensor,
                                      reg_lambda: float,
                                      n_landmarks: int = 1024,
                                      generator: Optional[torch.Generator] = None,
                                      jitter: float = 1e-8,
                                      landmarks=None) -> torch.Tensor:
    """
    Full-sample (no-split) DR weights with a PROJECTED Nystrom CME.

    The no-split analogue of `cpme_dr_weights_crossfit_projected_nystrom`, standing to
    `cpme_dr_weights_nystrom` as that function stands to
    `cpme_dr_weights_crossfit_nystrom`: every row both fits the CME and evaluates its
    own residual, so with T = H = all rows

        v = U (U^T U + rho W)^+ r_L / n + w / n,
        r_L = K_pi[L, :] 1 - K_S[L, :] w.

    No consistency claim is made for this path -- it has no cross-fitting, so the
    doubly robust remainder argument does not apply. It exists as the matched
    no-split comparison the ablations use.

    Parameters
    ----------
    gram, gram_pi : callable
        As in `cpme_dr_weights_crossfit_projected_nystrom`; only the blocks
        (all, L), (L, L) and (L, all) are requested.
    importance_weights : torch.Tensor of shape (n,)
    reg_lambda : float
        The matrix ridge rho. Not scaled by the training-block size, so a no-split
        fit at block n and a k-fold fit at block (1 - 1/k) n are NOT at matched
        effective regularisation for the same value.
    n_landmarks : int
    generator : torch.Generator, optional
        Stateful, see `_landmark_indices`.
    jitter : float
    landmarks : array-like, optional
        Precomputed landmark rows, bypassing the draw.

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    import numpy as np

    weights = importance_weights.reshape(-1)
    n = len(weights)
    rows = np.arange(n)

    L = _landmark_indices(rows, n_landmarks, generator, landmarks, fold=0)
    r_L = _blocked_rhs(gram, gram_pi, L, rows, weights)
    U = gram(rows, L)
    K_ll = gram(L, L)
    b = _projected_nystrom_coefficients(U, K_ll, r_L, float(reg_lambda), jitter)
    return b / n + weights / n
