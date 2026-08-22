"""
ColorMNIST data-generating process (Dance et al., 2026, Appendix B.2.4).

A scalar confounder X in [0, 1] sets the foreground colour of an MNIST digit and
also drives the treatment A, which selects the digit class, so colour, treatment
and digit content are entangled in the observational law:

    X ~ U[0, 1],
    A | X = x ~ Bernoulli(sigmoid(w (x - 1/2))),
    Y | X, A  = a random digit of class d_A, recoloured by c_fg(X),

with the recolouring built from a soft foreground mask on the grayscale digit S:

    M = sigmoid(k (S - tau)),   c_fg(x) = (x, alpha, 1 - x),
    Y = M (c_fg(X) S) + (1 - M) c_bg.

All design parameters are explicit function arguments so that a notebook's
configuration is visible at the call site.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torchvision import datasets


def load_digit_pools(root: str,
                     digits: Sequence[int] = (1, 6),
                     device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """
    Grayscale MNIST digits S in [0, 1]^(n, 28, 28), one pool per class in `digits`.

    Parameters
    ----------
    root : str
        torchvision MNIST root (downloads on first use).
    digits : sequence of int
        (d_0, d_1): the digit class selected by each treatment arm.
    device : torch.device, optional

    Returns
    -------
    list of torch.Tensor, one (n_d, 28, 28) pool per requested digit class.
    """
    mnist = datasets.MNIST(root=root, train=True, download=True)
    S = mnist.data.to(device).float() / 255.0
    targets = mnist.targets.to(device)
    return [S[targets == d] for d in digits]


def recolor(S: torch.Tensor,
            x: torch.Tensor,
            alpha: float = 0.2,
            tau: float = 0.08,
            k_mask: float = 20.0) -> torch.Tensor:
    """
    Recolours grayscale digits with foreground colour c_fg(x) = (x, alpha, 1 - x)
    on a black background.

    Parameters
    ----------
    S : torch.Tensor of shape (n, 28, 28), values in [0, 1]
    x : torch.Tensor of shape (n,)
        Confounder values.
    alpha : float
        Fixed green channel.
    tau, k_mask : float
        Threshold and sharpness of the soft foreground mask.

    Returns
    -------
    torch.Tensor of shape (n, 3, 28, 28) in [0, 1].
    """
    M = torch.sigmoid(k_mask * (S - tau)).unsqueeze(1)
    c_fg = torch.stack([x, torch.full_like(x, alpha), 1.0 - x], 1)
    fg = c_fg[:, :, None, None] * S.unsqueeze(1)
    return M * fg


def sample_images(pools: Sequence[torch.Tensor],
                  x: torch.Tensor,
                  a: torch.Tensor,
                  alpha: float = 0.2,
                  tau: float = 0.08,
                  k_mask: float = 20.0,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Y | X = x, A = a: draws a digit of class d_a uniformly from its pool and
    recolours it by x.
    """
    device = x.device
    S = torch.empty(x.shape[0], 28, 28, device=device, dtype=pools[0].dtype)
    for j in range(len(pools)):
        mask = a == j
        if mask.any():
            idx = torch.randint(pools[j].shape[0], (int(mask.sum()),),
                                generator=generator, device=device)
            S[mask] = pools[j][idx]
    return recolor(S, x, alpha=alpha, tau=tau, k_mask=k_mask)


def sample_observed(pools: Sequence[torch.Tensor],
                    n: int,
                    w_conf: float = 10.0,
                    alpha: float = 0.2,
                    tau: float = 0.08,
                    k_mask: float = 20.0,
                    generator: Optional[torch.Generator] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Factual logged data (X, A, Y = Y(A)) under the confounded propensity
    e(1 | x) = sigmoid(w_conf (x - 1/2)).

    Returns
    -------
    x : (n,)   confounder
    a : (n,)   treatment in {0, 1}
    y : (n, 3, 28, 28) images in [0, 1]
    """
    device = pools[0].device
    x = torch.rand(n, generator=generator, device=device)
    e = torch.sigmoid(w_conf * (x - 0.5))
    a = (torch.rand(n, generator=generator, device=device) < e).long()
    return x, a, sample_images(pools, x, a, alpha=alpha, tau=tau, k_mask=k_mask,
                               generator=generator)


def sample_interventional(pools: Sequence[torch.Tensor],
                          n: int,
                          a: int = 1,
                          alpha: float = 0.2,
                          tau: float = 0.08,
                          k_mask: float = 20.0,
                          generator: Optional[torch.Generator] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct draws from the true P(Y(a)): the digit class is fixed to d_a and X is
    resampled independently, keeping the colour variation but removing the
    treatment-selection bias. For evaluation only.
    """
    device = pools[0].device
    x = torch.rand(n, generator=generator, device=device)
    a_vec = torch.full((n,), a, device=device, dtype=torch.long)
    return x, sample_images(pools, x, a_vec, alpha=alpha, tau=tau, k_mask=k_mask,
                            generator=generator)


def color_ratio(Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relative red-blue intensity R / (R + B) summed over pixels. With the black
    background this recovers the confounder X exactly, making it the natural
    scalar summary for checking whether generated images carry the correct
    colour law.
    """
    R, B = Y[:, 0].sum((1, 2)), Y[:, 2].sum((1, 2))
    return R / (R + B + eps)


def to_model_range(Y: torch.Tensor) -> torch.Tensor:
    """[0, 1] images -> [-1, 1]."""
    return Y * 2.0 - 1.0


def to_image_range(Y: torch.Tensor) -> torch.Tensor:
    """[-1, 1] images -> [0, 1], clamped, for plotting and colour statistics."""
    return ((Y + 1.0) / 2.0).clamp(0.0, 1.0)
