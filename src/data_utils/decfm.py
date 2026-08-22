"""
Synthetic confounded designs of

    H. Dance, R. Xi, P. Orbanz, B. Bloem-Reddy,
    "Debiased Counterfactual Generation via Flow Matching from Observations", 2026.
"""

import numpy as np
import torch
from typing import Dict, Optional


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.get_default_dtype())


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sample_decfm_2d_data(n_samples: int = 1000,
                         confounding_strength: float = 3.0,
                         mixture_delta: float = 2.0,
                         noise_std: float = 0.2,
                         seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Two-dimensional mixture-of-normals design (Dance et al., 2026, Appendix B.2.2):

        X ~ N(0, I_2),
        A | x ~ Bernoulli(sigmoid(gamma w^T x)),      w = (1, -1) / sqrt(2),
        Y = X + eps,   eps ~ 1/2 N(0, sigma^2 I) + 1/2 N((delta, delta), sigma^2 I).

    The outcome does not depend on A, so both interventional laws equal the marginal
    P_Y, while P_{Y | A = a} differs from P_Y under confounding. The task is to
    deconfound an observed arm back to the marginal.

    Parameters
    ----------
    n_samples : int
    confounding_strength : float
        gamma in the propensity.
    mixture_delta : float
        Separation (delta, delta) between the two mixture components.
    noise_std : float
        Within-component standard deviation sigma.
    seed : int, optional

    Returns
    -------
    dict of torch.Tensor with keys
        "X", "Y_logged", "Y_oracle" : (n, 2); Y_oracle is a fresh draw from the
                                      interventional law P_Y
        "A_logged"                  : (n, 1)
        "propensity"                : (n,)  pi_0(A = 1 | x)
        "importance_weights_a0/1"   : (n,)  deterministic-intervention weights
    """
    rng = np.random.default_rng(seed)
    w = np.array([1.0, -1.0]) / np.sqrt(2.0)

    def draw_outcomes(X):
        n = X.shape[0]
        eps = noise_std * rng.standard_normal(size=(n, 2))
        shift = rng.binomial(1, 0.5, size=n).astype(float)
        return X + eps + mixture_delta * shift[:, None]

    X = rng.standard_normal(size=(n_samples, 2))
    probs = _sigmoid(confounding_strength * (X @ w))
    A_logged = rng.binomial(1, probs).astype(float)
    Y_logged = draw_outcomes(X)

    X_fresh = rng.standard_normal(size=(n_samples, 2))
    Y_oracle = draw_outcomes(X_fresh)

    return {
        "X": _to_tensor(X),
        "A_logged": _to_tensor(A_logged).reshape(-1, 1),
        "Y_logged": _to_tensor(Y_logged),
        "Y_oracle": _to_tensor(Y_oracle),
        "propensity": _to_tensor(probs),
        "importance_weights_a0": _to_tensor((A_logged == 0) / np.clip(1.0 - probs, 1e-8, None)),
        "importance_weights_a1": _to_tensor((A_logged == 1) / np.clip(probs, 1e-8, None)),
    }
