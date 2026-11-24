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

from data_utils.dsprites_multifactor import render_bilinear
from causality_utils.policies import (GaussianMixturePolicy, GaussianPolicy, Policy,
                                      context_shift_gaussian,
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


def _context_attributes(X: np.ndarray, n_shapes: int, n_scales: int):
    """Shape from X1 and scale from X2 -- the direct context-to-outcome channel."""
    shape = np.minimum(n_shapes - 1, np.floor(n_shapes * X[:, 0]).astype(int))
    scale = np.minimum(n_scales - 1, np.floor(n_scales * X[:, 1]).astype(int))
    return shape, scale


def rotated_context_gaussian(gain: float, angle: float, scale: float) -> GaussianPolicy:
    """
    Logging family of the confounded design:

        a | x ~ N(gain * R_angle (x - 1/2), scale^2 I_2),

    with `R_angle` the 2x2 rotation. The rotation mixes the two context
    coordinates into both action coordinates, so an observed action is
    informative about BOTH the shape (set by x1) and the scale (set by x2).
    """
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return GaussianPolicy(lambda X: gain * (X - 0.5) @ rotation.T, scale)


def sample_confounded_dsprite_data(attribute_bank: np.ndarray,
                                   n_samples: int = 4000,
                                   policy_gain: float = 1.8,
                                   policy_angle: float = np.pi / 6,
                                   sigma_policy: float = 0.6,
                                   target_shift: float = 0.45,
                                   mean_preserving: bool = False,
                                   pos_gain: float = 1.5,
                                   pixel_noise: float = 0.0,
                                   seed: Optional[int] = None,
                                   logging_policy: Optional[Policy] = None,
                                   target_policy: Optional[Policy] = None) -> Dict[str, torch.Tensor]:
    """
    Confounded image-outcome design with a continuous action.

    The design of `sample_cpme_dsprite_data` renders from the ACTION alone, so
    the outcome is independent of the context given the action: p(y | do(a))
    equals p(y | a), no adjustment is needed, and reweighting by the MARGINAL
    action ratio is already consistent. This design closes that gap by letting
    the context set visual attributes as well as the action:

        x ~ U([0, 1]^2),
        shape  = floor(n_shapes * x1),      scale = floor(n_scales * x2),
        a | x ~ pi_0(. | x) = N(gain * R_angle (x - 1/2), sigma^2 I_2),
        position(a) = 31 * sigmoid(pos_gain * a)  in [0, 31]^2,
        y = bilinear blend of the sprite with that shape and scale at that
            position.

    So x -> a, x -> y and a -> y: the backdoor path x is open, and recovering
    nu(pi) genuinely requires adjusting for it. The policy shift moves only the
    position, leaving the shape and scale MARGINALS untouched; what changes is
    their COUPLING with position, which is why a marginal summary of the outcome
    cannot detect the difference and a distributional one can.

    The default target is the shifted mixture

        pi(. | x) = 1/2 N(mu_0(x), sigma^2 I) + 1/2 N(mu_0(x) + delta, sigma^2 I),
        delta = (target_shift, -target_shift),

    which contains the logging policy as a component, so the density ratio is
    bounded below by 1/2 and overlap is benign. With `mean_preserving=True` the
    components sit at -delta and +delta instead: the conditional action mean is
    then exactly the logging mean, so any procedure reporting only E[A | X] --
    or only the mean outcome -- is blind to the intervention, while the outcome
    law is visibly bimodal.

    Parameters
    ----------
    attribute_bank : np.ndarray of shape (n_shapes, n_scales, n_orientations, 32, 32, 64, 64)
        From `load_multifactor_bank`; pass `orientation_ids=(0,)` for the fixed
        orientation this design uses.
    n_samples : int
    policy_gain, policy_angle, sigma_policy : float
        Logging policy; see `rotated_context_gaussian`. `policy_gain` relative to
        `sigma_policy` sets how strongly the action reveals the context, i.e. the
        confounding strength.
    target_shift : float
        Half-gap `delta = (target_shift, -target_shift)` of the target mixture.
    mean_preserving : bool
        Use the mean-preserving mixture instead of the shifted one.
    pos_gain : float
        Gain of the action-to-position sigmoid.
    pixel_noise : float
        Std of additive pixel noise, clipped back to [0, 1]. DEFAULT 0: the
        bilinear renderer already keeps the outcome continuous in the action, so
        noise is not needed to avoid atoms, and it is what makes FID on this
        design measure the missing noise rather than the sampler.
    seed : int, optional
    logging_policy, target_policy : Policy, optional
        Override the defaults. Both are returned, so the known target policy can
        be reused at estimation time.

    Returns
    -------
    dict of torch.Tensor
        Same keys as `sample_cpme_dsprite_data` -- "X", "A_logged", "Y_logged",
        "A_target", "Y_target", "importance_weights", "pos_logged",
        "pos_target", "logging_policy", "target_policy" -- plus "shape" and
        "scale" (the context-determined attributes, oracle knowledge, for
        diagnostics only).
    """
    rng = np.random.default_rng(seed)
    n_shapes, n_scales = attribute_bank.shape[0], attribute_bank.shape[1]

    if logging_policy is None:
        logging_policy = rotated_context_gaussian(policy_gain, policy_angle, sigma_policy)
    if target_policy is None:
        delta = np.array([target_shift, -target_shift])
        shifts = [-delta, delta] if mean_preserving else [np.zeros(2), delta]
        target_policy = GaussianMixturePolicy(logging_policy.mean_fn, shifts, sigma_policy)

    X = rng.uniform(size=(n_samples, 2))
    shape, scale = _context_attributes(X, n_shapes, n_scales)

    orientation = np.zeros(n_samples, dtype=int)

    def render(A):
        images, position = render_bilinear(attribute_bank, shape, scale, orientation,
                                           A, pos_gain)
        images = images.reshape(n_samples, -1)
        if pixel_noise > 0.0:
            images = np.clip(images + pixel_noise * rng.standard_normal(images.shape).astype(
                np.float32), 0.0, 1.0)
        return images, position

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
        "shape": torch.as_tensor(shape, dtype=torch.long),
        "scale": torch.as_tensor(scale, dtype=torch.long),
        "logging_policy": logging_policy,
        "target_policy": target_policy,
    }
