"""
Kernels pulled back through a feature map.

For an encoder E and a base kernel k on the encoder's feature space, the
pullback kernel is

    k_E(y, y') = k(E(y), E(y')).

Its RKHS feature map is the composition of k's feature map with E, so a mean
embedding under k_E is the embedding of the *encoded* law under k. Running the
MMD flow with k_E therefore optimises the same objective as the latent-space
flow, but with the images themselves as the particles: the witness gradient at
an image y is J_E(y)^T applied to the latent witness gradient at E(y), which
autograd produces automatically as long as the encoder is evaluated inside the
kernel call. Two consequences follow. The flow moves an image only along
directions the encoder is sensitive to — directions in the Jacobian's row
space — and leaves the encoder's blind directions untouched. And nothing
constrains the images to remain on any decoder manifold, since no decoder is
involved.
"""

import torch


class EncoderPullbackKernel:
    """
    k_E(y, y') = k(E(y), E(y')) for a frozen, differentiable encoder.

    Parameters
    ----------
    kernel : Kernel
        Base kernel on the encoder's feature space, with its bandwidth already
        fixed (lock any median heuristic on encoded data beforehand).
    encoder : callable
        Maps image tensors (n, *image_shape) to features (n, d). Must be
        differentiable — do not wrap it in torch.no_grad — and should be frozen
        (parameters with requires_grad False, module in eval mode).
    image_shape : tuple of int
        Shape of one image, e.g. (3, 28, 28). Inputs arrive flattened as
        (n, prod(image_shape)) rows, the particle convention of the flow.
    """

    def __init__(self, kernel, encoder, image_shape):
        self.kernel = kernel
        self.encoder = encoder
        self.image_shape = tuple(image_shape)

    def _encode(self, rows: torch.Tensor) -> torch.Tensor:
        return self.encoder(rows.reshape(rows.shape[0], *self.image_shape))

    def __call__(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        Z_X = self._encode(X)
        Z_Y = Z_X if Y is None else self._encode(Y)
        return self.kernel(Z_X, Z_Y)
