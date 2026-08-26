"""
Pretrained Stable-Diffusion VAE as a fixed image feature map.

The frozen encoder maps face images to a flattened latent (a factor-8
convolutional code: a 256 x 256 image gives 4 x 32 x 32 = 4096 coordinates) and
the frozen decoder maps transported latents back to images, keeping flow
particles on the image manifold.
"""

from typing import Tuple

import torch


def load_sd_vae(device: torch.device,
                model_name: str = "stabilityai/sd-vae-ft-mse"):
    """Loads the pretrained VAE (frozen, eval mode)."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_images(vae, images: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
    """
    Encodes images in [-1, 1] to flattened latents.

    Parameters
    ----------
    vae : AutoencoderKL
    images : torch.Tensor of shape (n, 3, S, S)
    chunk_size : int

    Returns
    -------
    torch.Tensor of shape (n, 4 (S/8)^2), on the VAE's device.
    """
    device = next(vae.parameters()).device
    latents = []
    for start in range(0, images.shape[0], chunk_size):
        chunk = images[start:start + chunk_size].to(device)
        latents.append(vae.encode(chunk).latent_dist.mode().flatten(1).cpu())
    return torch.cat(latents).to(device)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, image_size: int,
                   chunk_size: int = 16) -> torch.Tensor:
    """
    Decodes flattened latents back to images in [-1, 1] (clamped), on CPU.

    Parameters
    ----------
    vae : AutoencoderKL
    latents : torch.Tensor of shape (n, 4 (S/8)^2)
    image_size : int
        Side length S of the decoded images.
    chunk_size : int

    Returns
    -------
    torch.Tensor of shape (n, 3, S, S)
    """
    device = next(vae.parameters()).device
    side = image_size // 8
    images = []
    for start in range(0, latents.shape[0], chunk_size):
        chunk = latents[start:start + chunk_size].reshape(-1, 4, side, side).to(device)
        images.append(vae.decode(chunk).sample.clamp(-1.0, 1.0).cpu())
    return torch.cat(images)
