"""
Continuous-action dSprites counterfactual design (Zenati et al., 2025,
arXiv:2506.02793, Sec. 14.1.5).

A latent context draws a 2D continuous action from a Gaussian logging policy;
the action places a heart sprite in a 64x64 image, which is the outcome. The
target policy shifts the action mean, so the interventional image law differs
visibly from the logged one while the density-ratio weights stay analytic.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch

from causality_utils.policies import (Policy, context_shift_gaussian,
                                      importance_weights)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.get_default_dtype())


def load_dsprite_bank(npz_path: str,
                      cache_path: Optional[str] = None,
                      shape: int = 2,
                      scale_id: int = 3,
                      orientation_id: int = 0) -> np.ndarray:
    """
    Builds (and caches) the 32x32 grid of dSprites images at a fixed shape /
    scale / orientation, varying only the sprite position (posX, posY). Loading
    only these 1024 images avoids materialising the full dSprites array.

    Parameters
    ----------
    npz_path : str
        Path to `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` (the dSprites
        release of Matthey et al., https://github.com/deepmind/dsprites-dataset).
    cache_path : str, optional
        Where to read/write the small (32, 32, 64, 64) uint8 cache. If it
        exists it is loaded directly; otherwise it is built from `npz_path` and
        saved here. If None, the bank is always rebuilt from the npz.
    shape : int
        dSprites shape index (0 square, 1 ellipse, 2 heart).
    scale_id, orientation_id : int
        Fixed scale and orientation indices.

    Returns
    -------
    np.ndarray of shape (32, 32, 64, 64), uint8
        bank[posX, posY] is the 64x64 image of the sprite at that position.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    z = np.load(npz_path, allow_pickle=True, encoding="bytes")
    latents_sizes = z["metadata"][()][b"latents_sizes"]            # [1, 3, 6, 40, 32, 32]
    bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1])))
    idx = np.array([[0, shape, scale_id, orientation_id, px, py]
                    for px in range(32) for py in range(32)]) @ bases
    bank = z["imgs"][idx].reshape(32, 32, 64, 64).astype(np.uint8)

    if cache_path is not None:
        np.save(cache_path, bank)
    return bank


def _action_to_position(A: np.ndarray, pos_gain: float) -> np.ndarray:
    """Maps a 2D action to integer sprite positions in {0, ..., 31}^2 via a sigmoid."""
    pos = np.round(31.0 * _sigmoid(pos_gain * A)).astype(int)
    return np.clip(pos, 0, 31)


def sample_cpme_dsprite_data(sprite_bank: np.ndarray,
                             n_samples: int = 500,
                             sigma_policy: float = 0.5,
                             policy_shift: float = 0.2,
                             pos_gain: float = 3.0,
                             pixel_noise: float = 0.05,
                             seed: Optional[int] = None,
                             logging_policy: Optional[Policy] = None,
                             target_policy: Optional[Policy] = None) -> Dict[str, torch.Tensor]:
    """
    Structured (image) outcome counterfactual DGP of arXiv:2506.02793, Sec. 14.1.5.

    Latent context x ~ U([0, 1]^2). A 2D continuous action a is drawn from a
    Gaussian logging / target policy whose mean depends on x (the confounding
    channel):

        logging pi_0 : a | x ~ N( (x - 1/2) - policy_shift, sigma_policy^2 I_2 )
        target  pi   : a | x ~ N( (x - 1/2) + policy_shift, sigma_policy^2 I_2 )

    The action sets the sprite position (posX, posY) = round(31 * sigmoid(pos_gain * a)),
    and the outcome is the rendered 64x64 dSprites image at that position
    (flattened to 4096-d) plus small pixel noise. The two policies place the
    sprite in different regions, so the counterfactual image law nu(pi) differs
    visibly from the logged one. Importance weights w_i = pi(a_i | x_i) / pi_0(a_i | x_i)
    are analytic (ratio of the two Gaussians).

    Both `policy_shift` and `sigma_policy` control policy overlap (effective
    sample size); `pos_gain` independently amplifies the visual position shift
    without changing overlap.

    Parameters
    ----------
    sprite_bank : np.ndarray of shape (32, 32, 64, 64)
        Position grid of sprites from `load_dsprite_bank`.
    n_samples : int
    sigma_policy : float
        Standard deviation of both Gaussian policies (larger => more overlap).
    policy_shift : float
        Half the action-mean gap between logging and target.
    pos_gain : float
        Gain of the action-to-position sigmoid map.
    pixel_noise : float
        Standard deviation of the additive pixel noise.
    seed : int, optional
    logging_policy, target_policy : Policy, optional
        Override the default `context_shift_gaussian` policies at -/+
        `policy_shift`. Both policies are returned so the *known* target policy
        can be reused at estimation time -- to draw the resampling actions
        a~_j ~ pi(. | x_j) and to form pi / pi_0_hat once the logging policy is
        estimated.

    Returns
    -------
    dict of torch.Tensor with keys
        "X"                  : (n, 2) contexts
        "A_logged"           : (n, 2) actions under pi_0
        "Y_logged"           : (n, 4096) images under pi_0
        "A_target"           : (n, 2) oracle actions under pi (same contexts)
        "Y_target"           : (n, 4096) oracle images under pi
        "importance_weights" : (n,)    pi(a_i | x_i) / pi_0(a_i | x_i)
        "pos_logged"         : (n, 2) integer sprite positions under pi_0
        "pos_target"         : (n, 2) integer sprite positions under pi
        "logging_policy"     : Policy  pi_0 (to be replaced by an estimate downstream)
        "target_policy"      : Policy  pi   (known by design)
    """
    rng = np.random.default_rng(seed)
    bank = sprite_bank.astype(np.float64)

    if logging_policy is None:
        logging_policy = context_shift_gaussian(-policy_shift, sigma_policy)
    if target_policy is None:
        target_policy = context_shift_gaussian(+policy_shift, sigma_policy)

    X = rng.uniform(size=(n_samples, 2))

    def render(A):
        pos = _action_to_position(A, pos_gain)
        imgs = bank[pos[:, 0], pos[:, 1]]                          # (n, 64, 64)
        imgs = imgs + pixel_noise * rng.standard_normal(imgs.shape)
        return imgs.reshape(n_samples, -1), pos

    A_logged = logging_policy.sample(X, rng)
    A_target = target_policy.sample(X, rng)
    Y_logged, pos_logged = render(A_logged)
    Y_target, pos_target = render(A_target)

    weights = importance_weights(target_policy, logging_policy, X, A_logged)

    return {
        "X": _to_tensor(X),
        "A_logged": _to_tensor(A_logged),
        "Y_logged": _to_tensor(Y_logged),
        "A_target": _to_tensor(A_target),
        "Y_target": _to_tensor(Y_target),
        "importance_weights": _to_tensor(weights),
        "pos_logged": _to_tensor(pos_logged),
        "pos_target": _to_tensor(pos_target),
        "logging_policy": logging_policy,
        "target_policy": target_policy,
    }
