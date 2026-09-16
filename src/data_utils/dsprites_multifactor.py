"""
Multi-factor continuous-action dSprites design.

A harder variant of the single-sprite design of `dsprites.py`. There, shape,
scale and orientation are frozen, so the outcome manifold is the 1024 positions
of one heart and the interventional law is a reweighting of those 1024 images.
That makes the benchmark insensitive to particle initialisation: a flow started
from Gaussian noise reaches the same discrepancy as one started from the logged
images.

This design keeps the same estimation structure -- context, continuous 2D
action, Gaussian logging/target policies, analytic density ratios -- and changes
two things:

* **Content varies.** Shape, scale and orientation are drawn per observation, so
  the outcome manifold is the full multi-factor sprite set rather than one
  sprite's position grid.
* **The intervention edits only position, in a shape-dependent direction.** The
  target policy shifts the action mean by `delta * v_C` where `v_C` is one of
  three unit directions selected by the *shape*. Shape, scale and orientation
  are untouched by the intervention, so the logged images already carry the
  correct content and the counterfactual differs from them by a structured local
  edit.

The coupling A_pi = A_0 + delta * v_C over a shared Gaussian innovation bounds
the outcome-law distance by the renderer's Lipschitz constant times delta, so
the logged law is provably close to the target law while Gaussian noise is not.
Whether that closeness translates into a better *sampler* initialisation is the
empirical question the design exists to ask.

Positions are rendered by **bilinear interpolation** between the four
neighbouring grid sprites rather than rounding to the nearest, so the outcome
varies continuously with the continuous action and a policy shift produces a
smooth edit instead of a jump between discrete images.

Design after a specification by the user (2026-09-12); the estimation interface
matches `dsprites.py` so the same nuisance and flow code applies unchanged.
"""

import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from causality_utils.policies import GaussianPolicy, Policy, importance_weights

SHAPES = (0, 1, 2)
SCALE_IDS = (0, 2, 4)
ORIENTATION_IDS = (0, 5, 10, 15, 20, 25, 30, 35)
N_ORIENTATIONS_TOTAL = 40
GRID = 32
IMAGE_SIDE = 64

# Unit shift directions, one per shape: right, up-left, down-left.
SHAPE_DIRECTIONS = np.array([[1.0, 0.0],
                             [-0.5, np.sqrt(3.0) / 2.0],
                             [-0.5, -np.sqrt(3.0) / 2.0]])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.get_default_dtype())


def load_multifactor_bank(npz_path: str,
                          cache_path: Optional[str] = None,
                          shapes: Sequence[int] = SHAPES,
                          scale_ids: Sequence[int] = SCALE_IDS,
                          orientation_ids: Sequence[int] = ORIENTATION_IDS
                          ) -> np.ndarray:
    """
    Builds (and caches) the multi-factor sprite bank.

    Parameters
    ----------
    npz_path : str
        Path to `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`
        (Matthey et al., https://github.com/deepmind/dsprites-dataset).
    cache_path : str, optional
        Where to read/write the uint8 cache (~300 MB at the defaults). Loaded
        directly when present.
    shapes, scale_ids, orientation_ids : sequence of int
        Factor levels to include. The defaults give 3 x 3 x 8 = 72 content
        configurations, each on the full 32 x 32 position grid.

    Returns
    -------
    np.ndarray of shape (n_shapes, n_scales, n_orientations, 32, 32, 64, 64), uint8
        bank[c, s, o, px, py] is the rendered sprite.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    z = np.load(npz_path, allow_pickle=True, encoding="bytes")
    latents_sizes = z["metadata"][()][b"latents_sizes"]            # [1, 3, 6, 40, 32, 32]
    bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1])))
    index = np.array([[0, c, s, o, px, py]
                      for c in shapes for s in scale_ids for o in orientation_ids
                      for px in range(GRID) for py in range(GRID)]) @ bases
    bank = z["imgs"][index].reshape(len(shapes), len(scale_ids), len(orientation_ids),
                                    GRID, GRID, IMAGE_SIDE, IMAGE_SIDE).astype(np.uint8)
    if cache_path is not None:
        np.save(cache_path, bank)
    return bank


def context_features(shape_idx: np.ndarray, scale_idx: np.ndarray,
                     orientation_idx: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Numeric context features X for the policies and the nuisance estimators.

    Columns are `[u1, u2, cos(2 pi O / 40), sin(2 pi O / 40), onehot(C) (3), S / 4]`,
    chosen so that BOTH policy means are exactly linear in X: the logging mean is
    `0.35 (u1, u2) + 0.15 (cos, sin)` and the target adds `delta v_C`, which is
    linear in the shape one-hot. A cross-fitted linear-Gaussian logging-density
    fit is therefore well specified, so an initialisation comparison run on this
    design is not confounded by nuisance misspecification.

    Parameters
    ----------
    shape_idx : np.ndarray of shape (n,)
        Position within `shapes` (not the raw dSprites shape id).
    scale_idx, orientation_idx : np.ndarray of shape (n,)
        Positions within `scale_ids` / `orientation_ids`.
    U : np.ndarray of shape (n, 2)

    Returns
    -------
    np.ndarray of shape (n, 8)
    """
    angle = 2.0 * np.pi * np.asarray(ORIENTATION_IDS)[orientation_idx] / N_ORIENTATIONS_TOTAL
    one_hot = np.zeros((len(shape_idx), len(SHAPES)))
    one_hot[np.arange(len(shape_idx)), shape_idx] = 1.0
    return np.column_stack([U, np.cos(angle), np.sin(angle), one_hot,
                            np.asarray(SCALE_IDS)[scale_idx] / 4.0])


def _logging_mean(X: np.ndarray) -> np.ndarray:
    """m_0(X) = 0.35 (u1, u2) + 0.15 (cos q, sin q)."""
    X = np.asarray(X, dtype=float)
    return 0.35 * X[:, 0:2] + 0.15 * X[:, 2:4]


def multifactor_policies(policy_shift: float, sigma_policy: float):
    """
    The logging and target policies of this design.

    Returns
    -------
    (Policy, Policy)
        `pi_0(. | x) = N(m_0(x), sigma^2 I_2)` and
        `pi(. | x) = N(m_0(x) + policy_shift * v_C, sigma^2 I_2)`, with `v_C`
        read off the shape one-hot in columns 4:7 of the context features.
    """
    def target_mean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _logging_mean(X) + policy_shift * (X[:, 4:7] @ SHAPE_DIRECTIONS)

    return (GaussianPolicy(_logging_mean, sigma_policy),
            GaussianPolicy(target_mean, sigma_policy))


def render_bilinear(bank: np.ndarray, shape_idx: np.ndarray, scale_idx: np.ndarray,
                    orientation_idx: np.ndarray, A: np.ndarray,
                    pos_gain: float) -> np.ndarray:
    """
    Bilinear rendering of the sprite at a continuous action.

    The action maps to continuous grid coordinates `p_j = 31 sigmoid(gain a_j)`,
    and the image is the bilinear combination of the four neighbouring grid
    sprites. Rounding to the nearest grid point instead would make the outcome a
    step function of the action, so a policy shift would move probability between
    1024 fixed images rather than deform one continuously.

    Returns
    -------
    np.ndarray of shape (n, 64, 64), float32
    """
    position = 31.0 * _sigmoid(pos_gain * np.asarray(A, dtype=float))
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, GRID - 1)
    frac = position - low

    out = np.zeros((len(A), IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)
    for take_high_x, weight_x in ((False, 1.0 - frac[:, 0]), (True, frac[:, 0])):
        rows = high[:, 0] if take_high_x else low[:, 0]
        for take_high_y, weight_y in ((False, 1.0 - frac[:, 1]), (True, frac[:, 1])):
            columns = high[:, 1] if take_high_y else low[:, 1]
            weight = (weight_x * weight_y).astype(np.float32)[:, None, None]
            out += weight * bank[shape_idx, scale_idx, orientation_idx, rows, columns]
    return out, position


def sample_multifactor_dsprite_data(bank: np.ndarray,
                                    n_samples: int = 2000,
                                    sigma_policy: float = 0.35,
                                    policy_shift: float = 0.35,
                                    pos_gain: float = 2.5,
                                    pixel_noise: float = 0.02,
                                    seed: Optional[int] = None
                                    ) -> Dict[str, object]:
    """
    Multi-factor dSprites counterfactual DGP.

    Content factors are drawn uniformly and are UNCHANGED by the intervention;
    the target policy shifts the action mean by `policy_shift * v_C` in a
    shape-dependent direction, so the counterfactual image law differs from the
    logged one by a structured position edit on identical content.

    The logged and target actions share their Gaussian innovation
    (`A_pi = A_0 + policy_shift * v_C`), which is what makes the two outcome laws
    provably close: with an `L`-Lipschitz renderer,
    `W_2(P_Y, P_Y(pi)) <= L * policy_shift`.

    Parameters
    ----------
    bank : np.ndarray
        From `load_multifactor_bank`.
    n_samples : int
    sigma_policy : float
        Standard deviation of both policies (larger => more overlap).
    policy_shift : float
        Magnitude `delta` of the shape-dependent action-mean shift.
    pos_gain : float
        Gain of the action-to-position sigmoid.
    pixel_noise : float
        Standard deviation of the additive pixel noise; images are clipped to
        [0, 1] afterwards.
    seed : int, optional

    Returns
    -------
    dict with the keys of `sample_cpme_dsprite_data` plus
        "shape_idx", "scale_idx", "orientation_idx" : (n,) content factors
        "position_logged", "position_target"        : (n, 2) CONTINUOUS positions
    """
    rng = np.random.default_rng(seed)
    n_shapes, n_scales, n_orientations = bank.shape[:3]
    bank_float = bank  # gathered rows are cast during rendering

    shape_idx = rng.integers(n_shapes, size=n_samples)
    scale_idx = rng.integers(n_scales, size=n_samples)
    orientation_idx = rng.integers(n_orientations, size=n_samples)
    U = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    X = context_features(shape_idx, scale_idx, orientation_idx, U)

    logging_policy, target_policy = multifactor_policies(policy_shift, sigma_policy)
    innovation = rng.standard_normal((n_samples, 2))
    A_logged = logging_policy._mean_and_scale(X)[0] + sigma_policy * innovation
    A_target = A_logged + policy_shift * (X[:, 4:7] @ SHAPE_DIRECTIONS)

    def render(A):
        images, position = render_bilinear(bank_float, shape_idx, scale_idx,
                                           orientation_idx, A, pos_gain)
        images = images + pixel_noise * rng.standard_normal(images.shape).astype(np.float32)
        return np.clip(images, 0.0, 1.0).reshape(n_samples, -1), position

    Y_logged, position_logged = render(A_logged)
    Y_target, position_target = render(A_target)
    weights = importance_weights(target_policy, logging_policy, X, A_logged)

    return {
        "X": _to_tensor(X),
        "A_logged": _to_tensor(A_logged),
        "Y_logged": _to_tensor(Y_logged),
        "A_target": _to_tensor(A_target),
        "Y_target": _to_tensor(Y_target),
        "importance_weights": _to_tensor(weights),
        "shape_idx": torch.as_tensor(shape_idx, dtype=torch.long),
        "scale_idx": torch.as_tensor(scale_idx, dtype=torch.long),
        "orientation_idx": torch.as_tensor(orientation_idx, dtype=torch.long),
        "position_logged": _to_tensor(position_logged),
        "position_target": _to_tensor(position_target),
        "logging_policy": logging_policy,
        "target_policy": target_policy,
    }


# --------------------------------------------------------------------------------
# Coloured variant: the intervention moves an appearance statistic as well as a
# geometric one.
# --------------------------------------------------------------------------------

def colour_ratio(Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relative red-blue intensity R / (R + B) summed over pixels, per image.

    The coloured design sets the sprite hue from a single action coordinate, and
    this ratio inverts that map, so it recovers the hue coordinate almost exactly.
    It is the dSprites analogue of `colormnist.color_ratio`: a scalar readout of
    the intervention's appearance channel with a known oracle value, which makes
    the appearance part of the shift measurable without a probe.

    Parameters
    ----------
    Y : torch.Tensor of shape (n, 3, 64, 64) or (n, 3 * 64 * 64), values in [0, 1]

    Returns
    -------
    torch.Tensor of shape (n,)
    """
    if Y.dim() == 2:
        Y = Y.reshape(len(Y), 3, IMAGE_SIDE, IMAGE_SIDE)
    red, blue = Y[:, 0].sum((1, 2)), Y[:, 2].sum((1, 2))
    return red / (red + blue + eps)


def coloured_policies(policy_shift: float, colour_shift: float, sigma_policy: float):
    """
    Logging and target policies over the 3D action `(a_1, a_2, a_3)`.

    `(a_1, a_2)` place the sprite as in the grayscale design; `a_3` sets its hue.
    The target shifts position by `policy_shift * v_C` -- shape-dependent, so the
    geometric channel tests a conditional policy -- and hue by `colour_shift`,
    the same for every shape, so the appearance channel has a single scalar
    oracle value. Both channels move under one intervention.
    """
    def logging_mean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([_logging_mean(X), np.zeros(len(X))])

    def target_mean(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        position = _logging_mean(X) + policy_shift * (X[:, 4:7] @ SHAPE_DIRECTIONS)
        return np.column_stack([position, np.full(len(X), colour_shift)])

    return (GaussianPolicy(logging_mean, sigma_policy),
            GaussianPolicy(target_mean, sigma_policy))


def sample_coloured_dsprite_data(bank: np.ndarray,
                                 n_samples: int = 2000,
                                 sigma_policy: float = 0.5,
                                 policy_shift: float = 0.35,
                                 colour_shift: float = 0.3,
                                 pos_gain: float = 2.5,
                                 colour_gain: float = 6.0,
                                 alpha: float = 0.0,
                                 pixel_noise: float = 0.02,
                                 seed: Optional[int] = None) -> Dict[str, object]:
    """
    Multi-factor dSprites whose intervention moves BOTH position and colour.

    Motivation. Across the project's designs, an initialisation advantage
    survives to convergence exactly where the intervention targets a global
    APPEARANCE statistic (ColorMNIST colour, CelebA hair colour) and vanishes
    where it targets a GEOMETRIC one (dSprites position): a decoder will move a
    blob from any starting latent, but reaching a prescribed global colour
    requires landing in a particular region of latent space. This design adds the
    appearance channel to dSprites while keeping everything else, so that
    hypothesis can be tested rather than assumed.

    The sprite intensity `S` is rendered bilinearly from `(a_1, a_2)` exactly as
    in the grayscale design; the hue comes from `a_3` through
    `u = sigmoid(colour_gain * a_3)` and the ColorMNIST foreground map
    `c_fg(u) = (u, alpha, 1 - u)`, giving `Y = c_fg(u) * S` per channel.

    Parameters
    ----------
    bank : np.ndarray
        From `load_multifactor_bank`.
    n_samples : int
    sigma_policy : float
        Standard deviation of both policies, shared by all three coordinates.
    policy_shift : float
        Magnitude of the shape-dependent POSITION shift.
    colour_shift : float
        Shift of the hue coordinate `a_3` under the target policy. Zero recovers
        a position-only intervention on coloured images, which is the control.
    pos_gain, colour_gain : float
        Gains of the action-to-position and action-to-hue sigmoids. These set how
        much VISIBLE change a given action shift produces and cost nothing in
        overlap, so they are the right dial: the defaults were chosen by sweeping
        (colour_shift, colour_gain, sigma_policy) for estimated-weight ESS. A
        naive colour_shift = 0.8 at sigma = 0.35 is a 2.3-sigma shift in the hue
        coordinate and collapses ESS to 4 of 2000; the defaults give 674 of 2000,
        comparable to the grayscale design's 773, while still moving the colour
        ratio 0.50 -> 0.64.
    alpha : float
        Fixed green channel, as in the ColorMNIST design.
    pixel_noise : float
    seed : int, optional

    Returns
    -------
    dict
        As `sample_multifactor_dsprite_data`, with "Y_logged"/"Y_target" of width
        3 * 64 * 64, plus "hue_logged"/"hue_target" (the scalar u per row).
    """
    rng = np.random.default_rng(seed)
    n_shapes, n_scales, n_orientations = bank.shape[:3]

    shape_idx = rng.integers(n_shapes, size=n_samples)
    scale_idx = rng.integers(n_scales, size=n_samples)
    orientation_idx = rng.integers(n_orientations, size=n_samples)
    U = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    X = context_features(shape_idx, scale_idx, orientation_idx, U)

    logging_policy, target_policy = coloured_policies(policy_shift, colour_shift,
                                                      sigma_policy)
    innovation = rng.standard_normal((n_samples, 3))
    A_logged = logging_policy._mean_and_scale(X)[0] + sigma_policy * innovation
    A_target = (target_policy._mean_and_scale(X)[0] + sigma_policy * innovation)

    def render(A):
        intensity, position = render_bilinear(bank, shape_idx, scale_idx,
                                              orientation_idx, A[:, :2], pos_gain)
        hue = _sigmoid(colour_gain * A[:, 2])
        foreground = np.stack([hue, np.full(n_samples, alpha), 1.0 - hue], axis=1)
        images = foreground[:, :, None, None] * intensity[:, None, :, :]
        images = images + pixel_noise * rng.standard_normal(images.shape)
        return np.clip(images, 0.0, 1.0).reshape(n_samples, -1), position, hue

    Y_logged, position_logged, hue_logged = render(A_logged)
    Y_target, position_target, hue_target = render(A_target)
    weights = importance_weights(target_policy, logging_policy, X, A_logged)

    return {
        "X": _to_tensor(X),
        "A_logged": _to_tensor(A_logged),
        "Y_logged": _to_tensor(Y_logged),
        "A_target": _to_tensor(A_target),
        "Y_target": _to_tensor(Y_target),
        "importance_weights": _to_tensor(weights),
        "shape_idx": torch.as_tensor(shape_idx, dtype=torch.long),
        "scale_idx": torch.as_tensor(scale_idx, dtype=torch.long),
        "orientation_idx": torch.as_tensor(orientation_idx, dtype=torch.long),
        "position_logged": _to_tensor(position_logged),
        "position_target": _to_tensor(position_target),
        "hue_logged": _to_tensor(hue_logged),
        "hue_target": _to_tensor(hue_target),
        "logging_policy": logging_policy,
        "target_policy": target_policy,
    }
