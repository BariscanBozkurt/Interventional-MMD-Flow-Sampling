"""
Propensity estimation for binary treatments.

The importance weights of the doubly robust CPME estimator require the logging
propensity e(x) = P(A = 1 | x). This module provides a plug-in logistic model and
a cross-fitting wrapper that evaluates every observation out of fold.
"""

import numpy as np
import torch
import torch.nn.functional as F


class LogisticPropensityModel:
    """
    Plug-in logistic propensity model

        e(x) = P(A = 1 | x) = sigmoid(x^T beta + b),

    fit by maximum likelihood with Adam.

    Parameters
    ----------
    n_iterations : int
        Number of gradient steps.
    learning_rate : float
        Adam learning rate.
    """

    def __init__(self, n_iterations: int = 400, learning_rate: float = 0.1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.slope = None
        self.bias = None

    def fit(self, X: torch.Tensor, A: torch.Tensor) -> "LogisticPropensityModel":
        """
        Fits the model on logged pairs.

        Parameters
        ----------
        X : torch.Tensor of shape (n, d)
            Logged contexts.
        A : torch.Tensor of shape (n,)
            Logged binary treatments in {0, 1}.

        Returns
        -------
        self
        """
        slope = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device, requires_grad=True)
        bias = torch.zeros(1, dtype=X.dtype, device=X.device, requires_grad=True)
        optimizer = torch.optim.Adam([slope, bias], lr=self.learning_rate)
        target = A.to(X.dtype)
        for _ in range(self.n_iterations):
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(X @ slope + bias, target).backward()
            optimizer.step()
        self.slope = slope.detach()
        self.bias = bias.detach()
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts e(x) = P(A = 1 | x).

        Parameters
        ----------
        X : torch.Tensor of shape (n, d)

        Returns
        -------
        torch.Tensor of shape (n,)
        """
        if self.slope is None:
            raise RuntimeError("the model must be fit before predicting")
        with torch.no_grad():
            return torch.sigmoid(X @ self.slope + self.bias)

    def predict_arm(self, X: torch.Tensor, a: int) -> torch.Tensor:
        """Predicts e_a(x) = P(A = a | x) for a in {0, 1}."""
        e1 = self.predict(X)
        return e1 if a == 1 else 1.0 - e1


def crossfit_propensity(X: torch.Tensor,
                        A: torch.Tensor,
                        fold_ids: np.ndarray,
                        model_factory=LogisticPropensityModel) -> torch.Tensor:
    """
    Out-of-fold propensities e_hat(x_i) = P(A = 1 | x_i).

    For each fold, a fresh model from `model_factory` is fit on the remaining folds
    and evaluated on the held-out fold, so no observation is used to train its own
    propensity prediction. Use the same fold assignment for every cross-fitted
    nuisance of the doubly robust estimator.

    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        Logged contexts.
    A : torch.Tensor of shape (n,)
        Logged binary treatments in {0, 1}.
    fold_ids : np.ndarray of shape (n,)
        Integer fold assignment, e.g. from `causality_utils.make_crossfit_fold_ids`.
    model_factory : callable
        Zero-argument constructor of a model with `fit(X, A)` and `predict(X)`.

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    propensity = torch.empty(len(X), dtype=X.dtype, device=X.device)
    for fold in np.unique(fold_ids):
        held_out = torch.as_tensor(fold_ids == fold, device=X.device)
        train = ~held_out
        model = model_factory().fit(X[train], A[train])
        propensity[held_out] = model.predict(X[held_out])
    return propensity
