"""
Denoising convolutional autoencoder for the dSprites position images.

The continuous-action pipeline encodes the image outcome Y with a learned
feature map, runs the DR-targeted MMD flow in the latent space, and decodes the
transported particles back to images. Pixel-space flow fails on this family:
valid sprites form a thin nonlinear manifold in R^4096, and straight-line pixel
steps leave it. The decoder's job is to keep the flow on the sprite manifold.

The sprites vary only in their 2D position, so the image manifold is
intrinsically 2-dimensional and a small latent suffices. Training is
*denoising*: the input is a clean bank sprite plus the same Gaussian pixel
noise the data-generating process adds, the target is the clean sprite. The
encoder therefore absorbs the pixel noise and the decoder always renders clean
sprites. Images are 1x64x64 in [0, 1].
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 8 groups, valid for any channel count."""
    return nn.GroupNorm(math.gcd(8, channels), channels)


class DspriteAE(nn.Module):
    """
    Symmetric conv autoencoder for 1x64x64 dSprites images in [0, 1].

    Encoder: four stride-2 convs (64 -> 32 -> 16 -> 8 -> 4) then a linear
    bottleneck to `latent_dim`. Decoder mirrors it. GroupNorm + SiLU
    throughout; a final sigmoid puts the reconstruction in [0, 1] (clean
    sprites are binary).

    Parameters
    ----------
    latent_dim : int
        Bottleneck width. The sprite manifold is 2-dimensional (position only),
        so even latent 8 reconstructs perfectly -- but the *linear* organisation
        of the latent improves with width (position-probe R^2 at equal
        reconstruction: ~0.5 at 8, ~0.8 at 16, ~0.95 at 32), and the linear
        probe is what predicts kernel feature quality. Default 32.
    base_channels : int
        Channel width of the first conv (doubled at the next two stages).
    in_channels : int
        1 for the grayscale design, 3 for the coloured variant whose sprite hue
        is set by the action.
    """

    def __init__(self, latent_dim: int = 32, base_channels: int = 32,
                 in_channels: int = 1):
        super().__init__()
        c = base_channels
        self.base_channels = c
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1), _group_norm(c), nn.SiLU(),      # 64 -> 32
            nn.Conv2d(c, 2 * c, 4, 2, 1), _group_norm(2 * c), nn.SiLU(),        # 32 -> 16
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), _group_norm(4 * c), nn.SiLU(),    # 16 -> 8
            nn.Conv2d(4 * c, 4 * c, 4, 2, 1), _group_norm(4 * c), nn.SiLU(),    # 8 -> 4
        )
        self.to_latent = nn.Linear(4 * c * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 4 * c * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 4 * c, 4, 2, 1), _group_norm(4 * c), nn.SiLU(),  # 4 -> 8
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), _group_norm(2 * c), nn.SiLU(),  # 8 -> 16
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), _group_norm(c), nn.SiLU(),          # 16 -> 32
            nn.ConvTranspose2d(c, in_channels, 4, 2, 1), nn.Sigmoid(),                  # 32 -> 64
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Images (n, in_channels, 64, 64) -> latent features (n, latent_dim)."""
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent features (n, latent_dim) -> images (n, in_channels, 64, 64) in [0, 1]."""
        h = self.from_latent(z).view(-1, 4 * self.base_channels, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_dsprite_ae(clean_images: torch.Tensor,
                     latent_dim: int = 32,
                     base_channels: int = 32,
                     in_channels: Optional[int] = None,
                     noise_std: float = 0.05,
                     n_iterations: int = 4000,
                     batch_size: int = 128,
                     learning_rate: float = 1e-3,
                     device: Optional[torch.device] = None,
                     log_every: int = 500,
                     verbose: bool = True) -> Tuple[DspriteAE, list]:
    """
    Train a denoising `DspriteAE` on clean sprites: minimise MSE between the
    clean image and the reconstruction of its noise-corrupted copy (fresh noise
    each batch).

    Parameters
    ----------
    clean_images : float torch.Tensor of shape (n, 1, 64, 64) in [0, 1]
        Clean sprites, e.g. the full position bank from `load_dsprite_bank`
        (cast the uint8 bank to float first). Moved to `device` internally.
    noise_std : float
        Std of the additive Gaussian corruption. Match the data-generating
        process's `pixel_noise` so the encoder sees exactly the corruption the
        logged images carry.

    Returns
    -------
    (net, history) : the trained (eval-mode) net and the list of (iter, mse) logs.
    """
    if device is None:
        device = clean_images.device
    clean_images = clean_images.to(device)
    channels = in_channels if in_channels is not None else clean_images.shape[1]
    net = DspriteAE(latent_dim=latent_dim, base_channels=base_channels,
                    in_channels=channels).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    n = clean_images.shape[0]
    history = []
    net.train()
    for it in range(n_iterations):
        idx = torch.randint(n, (batch_size,), device=device)
        clean = clean_images[idx]
        noisy = clean + noise_std * torch.randn_like(clean)
        loss = F.mse_loss(net(noisy), clean)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == n_iterations - 1:
            mse = float(loss.detach())
            history.append((it, mse))
            if verbose:
                print(f"iter {it:5d}  mse {mse:.5f}")
    net.eval()
    return net, history


@torch.no_grad()
def snap_to_bank(images: torch.Tensor, bank_flat: torch.Tensor):
    """
    Snap each image to its nearest clean bank sprite (L2 over all 1024).

    The dSprites manifold is a finite grid, so "which sprite is this?" has an
    exact answer. The distance to that sprite is the **off-manifold** score: it
    is small for a genuine sprite (clean or reconstructed) and large for a
    particle that the flow has left between grid points, which decodes to a
    blurred superposition of several sprites.

    Parameters
    ----------
    images : torch.Tensor of shape (n, 4096) or (n, 1, 64, 64)
    bank_flat : torch.Tensor of shape (1024, 4096)
        The flattened clean bank, rows ordered as (posX * 32 + posY).

    Returns
    -------
    (positions, distances) : (n, 2) integer (posX, posY) of the nearest sprite,
    and the (n,) L2 distance to it.
    """
    flat = images.reshape(images.shape[0], -1)
    distances, nearest = torch.cdist(flat, bank_flat).min(dim=1)
    return torch.stack([nearest // 32, nearest % 32], dim=1), distances


@torch.no_grad()
def nearest_bank_position(images: torch.Tensor, bank_flat: torch.Tensor) -> torch.Tensor:
    """Positions only, from `snap_to_bank` -- the position-decoding diagnostic."""
    return snap_to_bank(images, bank_flat)[0]


def position_probe(features: torch.Tensor, positions: torch.Tensor, n_train: int = 700) -> dict:
    """
    Fit a linear regression from the latent to the sprite position (posX, posY)
    on a train split and score R^2 on the held-out rest.

    Position is the single factor of variation of this image family, so a
    feature map that lets a *linear* readout recover it carries the information
    the kernel needs; judge the feature map by this probe, never by
    reconstruction alone.

    Rows are shuffled internally (fixed seed) before the split: the natural
    bank order is sorted by posX, so a sequential split would score posX by
    extrapolation.

    Parameters
    ----------
    features : torch.Tensor of shape (n, d)
    positions : torch.Tensor of shape (n, 2)
    n_train : int
        Clamped to at most 70% of the rows so the test split is never empty.

    Returns
    -------
    dict with keys "dim", "posX_r2", "posY_r2".
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np

    Z = features.detach().cpu().numpy()
    P = positions.detach().cpu().numpy()
    order = np.random.default_rng(0).permutation(len(Z))
    Z, P = Z[order], P[order]
    n_train = min(n_train, int(0.7 * len(Z)))
    tr, te = slice(0, n_train), slice(n_train, len(Z))
    Zs = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)   # standardise on the train split
    r2x = LinearRegression().fit(Zs[tr], P[tr, 0]).score(Zs[te], P[te, 0])
    r2y = LinearRegression().fit(Zs[tr], P[tr, 1]).score(Zs[te], P[te, 1])
    return {"dim": Z.shape[1], "posX_r2": float(r2x), "posY_r2": float(r2y)}
