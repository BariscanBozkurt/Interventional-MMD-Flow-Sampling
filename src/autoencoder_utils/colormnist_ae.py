"""
Convolutional autoencoder feature map for ColorMNIST.

The latent-space counterfactual sampler encodes the image outcome Y with a learned
feature map, runs the MMD flow in the latent space, and decodes the transported
particles: the encoder defines the kernel's feature space and the decoder maps any
latent back onto the image manifold.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    """
    Symmetric conv autoencoder for 3x28x28 images in [-1, 1].

    Encoder: three stride-2 convs (28 -> 14 -> 7 -> 4) then a linear bottleneck to
    `latent_dim`. The decoder mirrors it; a final tanh keeps the reconstruction in
    [-1, 1]. GroupNorm and SiLU throughout.

    Parameters
    ----------
    latent_dim : int
        Bottleneck width, the feature dimension of the latent kernel.
    base_channels : int
        Channel width of the first conv (doubled at each later stage).
    """

    def __init__(self, latent_dim: int = 32, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.base_channels = c
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.GroupNorm(8, c), nn.SiLU(),              # 28 -> 14
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.GroupNorm(8, 2 * c), nn.SiLU(),      # 14 -> 7
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.GroupNorm(8, 4 * c), nn.SiLU(),  # 7 -> 4
        )
        self.to_latent = nn.Linear(4 * c * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 4 * c * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 3, 2, 1, output_padding=0),          # 4 -> 7
            nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.GroupNorm(8, c), nn.SiLU(), # 7 -> 14
            nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Tanh(),                         # 14 -> 28
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Images (n, 3, 28, 28) -> latent features (n, latent_dim)."""
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent features (n, latent_dim) -> images (n, 3, 28, 28) in [-1, 1]."""
        h = self.from_latent(z).view(-1, 4 * self.base_channels, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_conv_ae(data: torch.Tensor,
                  latent_dim: int = 32,
                  base_channels: int = 32,
                  n_iterations: int = 3000,
                  batch_size: int = 128,
                  learning_rate: float = 1e-3,
                  device: Optional[torch.device] = None,
                  generator: Optional[torch.Generator] = None,
                  log_every: int = 500,
                  verbose: bool = True) -> Tuple[ConvAE, list]:
    """
    Trains a `ConvAE` to minimise reconstruction MSE on `data` (n, 3, 28, 28) in
    [-1, 1]. The data is moved to `device` internally.

    Returns
    -------
    (net, history) : the trained (eval-mode) net and the list of (iter, mse) logs.
    """
    if device is None:
        device = data.device
    data = data.to(device)
    net = ConvAE(latent_dim=latent_dim, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    n = data.shape[0]
    history = []
    net.train()
    for it in range(n_iterations):
        idx = torch.randint(n, (batch_size,), device=device, generator=generator)
        x = data[idx]
        loss = F.mse_loss(net(x), x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if it % log_every == 0 or it == n_iterations - 1:
            mse = float(loss.detach())
            history.append((it, mse))
            if verbose:
                print(f"iter {it:5d}  mse {mse:.4f}")
    net.eval()
    return net, history


@torch.no_grad()
def encode_in_batches(encode_fn, data: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Applies a feature map `encode_fn` over `data` in batches and concatenates."""
    return torch.cat([encode_fn(data[i:i + batch_size])
                      for i in range(0, data.shape[0], batch_size)])
