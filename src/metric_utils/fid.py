"""
Frechet Inception Distance (FID) for decoded RGB images.

FID (Heusel et al. 2017) embeds each image with an ImageNet Inception-v3 network
and reports the Frechet (2-Wasserstein) distance between Gaussians fitted to the
two feature clouds,

    FID = ||mu_g - mu_r||^2 + tr(S_g + S_r - 2 (S_g S_r)^{1/2}).

Unlike sliced Wasserstein in pixel space, the Inception features respond to
blur, texture loss and off-manifold artefacts, so FID separates samples that
pixel-space metrics score alike (see the CelebA notebooks: latent and pixel SW2
at the reference floor while decoded faces are visibly degraded).

Two implementations are provided because the published comparison tables of
Dance et al. (2026) were produced with the second:

`CleanFIDScorer`
    Clean-FID (Parmar, Zhang and Zhu, CVPR 2022): images are quantised to uint8,
    resized to 299 x 299 with PIL bicubic, and embedded by the official
    TensorFlow-ported Inception TorchScript model. This is the standard,
    cross-repository comparable number and the one to report for new work.
`LegacyFIDScorer`
    TorchVision's `Inception_V3_Weights.IMAGENET1K_V1` backbone with antialiased
    bilinear resizing and ImageNet normalisation, and an EXACT finite-sample
    Frechet distance through the low-rank identity of `legacy_frechet_distance`.
    It reproduces the historical FID column of those tables. The two scorers are
    close but not interchangeable; never mix them within one comparison.

Both take decoded images shaped (N, 3, H, W) in [-1, 1] (or [0, 1] with
`image_range="zero_one"`) and refuse anything else, so latent vectors or flow
particles cannot be scored by mistake. Following the protocol of Dance et al.,
each intervention arm is scored separately against its full held-out reference
set and the arm scores are averaged.

Implementation after H. Dance's evaluation code for Dance et al. (2026).
"""

import hashlib
import importlib.metadata
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLEAN_FID_WEIGHTS = "inception-2015-12-05.pt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PathLike = Union[str, Path]


def default_clean_fid_weights_dir() -> Path:
    """Directory where Clean-FID keeps its Inception TorchScript weights."""
    return Path(torch.hub.get_dir()).parent / "clean-fid"


def _check_rgb_batch(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Expected decoded RGB images shaped (N, 3, H, W), got {tuple(images.shape)}. "
            "Decode flow particles before computing FID; latent vectors are not images.")


def _to_unit_range(images: torch.Tensor, image_range: str) -> torch.Tensor:
    """Detached float CPU copy of the images mapped into [0, 1]."""
    _check_rgb_batch(images)
    images = images.detach().float().cpu()
    if not torch.isfinite(images).all():
        raise ValueError("Images contain non-finite values")
    if image_range == "minus_one_one":
        return (images + 1.0) / 2.0
    if image_range == "zero_one":
        return images
    raise ValueError(f"Unsupported image range: {image_range!r}")


def quantize_images_uint8(images: torch.Tensor, image_range: str = "minus_one_one"
                          ) -> torch.Tensor:
    """
    Round images to the uint8 values a lossless PNG export would hold.

    Clean-FID is defined on 8-bit images; quantising tensors the same way keeps
    tensor and folder inputs bitwise comparable.

    Parameters
    ----------
    images : torch.Tensor of shape (n, 3, h, w)
    image_range : {"minus_one_one", "zero_one"}

    Returns
    -------
    torch.Tensor of shape (n, 3, h, w), uint8, on CPU
    """
    unit = _to_unit_range(images, image_range)
    return unit.mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8)


def image_files(folder: PathLike) -> List[Path]:
    """Sorted image paths under `folder`, searched recursively."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    files = sorted(path for path in folder.rglob("*")
                   if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No supported image files found under {folder}")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_frechet_distance(generated: torch.Tensor, reference: torch.Tensor,
                            device: Union[torch.device, str] = "cpu") -> float:
    """
    Exact finite-sample Frechet distance between two feature clouds.

    With centred factors G = (g - mean g) / sqrt(n_g - 1) and R likewise,
    tr((S_g S_r)^{1/2}) = ||G R^T||_*, the nuclear norm of an n_g x n_r matrix.
    That avoids the 2048 x 2048 matrix square root and its numerical
    regularisation, so the result is the exact empirical quantity.

    Parameters
    ----------
    generated, reference : torch.Tensor of shape (n, d)
        Inception features (flattened if higher-dimensional).
    device : torch.device or str
        Where the SVD runs; CPU reproduces the CelebA tables of Dance et al.
        (2026), CUDA their ColorMNIST tables.

    Returns
    -------
    float
    """
    generated = generated.detach().flatten(1).to(device=device, dtype=torch.float64)
    reference = reference.detach().flatten(1).to(device=device, dtype=torch.float64)
    if len(generated) < 2 or len(reference) < 2:
        raise ValueError("FID requires at least two generated and two reference images")
    if generated.shape[1] != reference.shape[1]:
        raise ValueError("Generated and reference features must have equal width")
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("FID features contain non-finite values")
    mean_g, mean_r = generated.mean(0), reference.mean(0)
    factor_g = (generated - mean_g) / math.sqrt(len(generated) - 1)
    factor_r = (reference - mean_r) / math.sqrt(len(reference) - 1)
    mean_term = (mean_g - mean_r).square().sum()
    trace_term = factor_g.square().sum() + factor_r.square().sum()
    cross_term = torch.linalg.svdvals(factor_g @ factor_r.T).sum()
    return float((mean_term + trace_term - 2.0 * cross_term).clamp_min(0.0).cpu())


class CleanFIDScorer:
    """
    Clean-FID (`clean` mode of clean-fid 0.1.35) with the official Inception weights.

    Parameters
    ----------
    device : torch.device or str
    batch_size : int
        Images per Inception forward pass.
    weights_dir : path, optional
        Where the TorchScript weights live; defaults to
        `default_clean_fid_weights_dir()`.
    allow_download : bool
        Fetch the weights on first use. Compute nodes without network access
        need the file pre-populated and `allow_download=False`.
    num_workers : int
        Threads for the per-image PIL resize, the dominant cost. Each image is
        resized by the same call regardless of thread count, so features are
        bitwise identical for any value; the default 1 is the reference path.

    Notes
    -----
    The TorchScript Inception is not run-to-run deterministic on GPU: identical
    inputs give features differing by ~3e-3, which is far below the resolution
    of any reported FID but means GPU feature matrices are not bitwise
    reproducible. CPU evaluation is exact. `LegacyFIDScorer` is deterministic on
    both.
    """

    def __init__(self, device: Union[torch.device, str], batch_size: int = 32,
                 weights_dir: Optional[PathLike] = None, allow_download: bool = True,
                 num_workers: int = 1) -> None:
        if batch_size < 1 or num_workers < 1:
            raise ValueError("batch_size and num_workers must be positive")
        try:
            from cleanfid import fid
            from cleanfid.inception_torchscript import InceptionV3W
            from cleanfid.resize import build_resizer
        except ImportError as error:
            raise RuntimeError("Install clean-fid==0.1.35 to use CleanFIDScorer") from error
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.weights_dir = Path(weights_dir or default_clean_fid_weights_dir()).expanduser()
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = self.weights_dir / CLEAN_FID_WEIGHTS
        if not allow_download and not self.weights_path.is_file():
            raise FileNotFoundError(f"Clean-FID weights not found: {self.weights_path}")
        self._model = InceptionV3W(str(self.weights_dir), download=allow_download,
                                   resize_inside=False).to(self.device).eval()
        self._resize = build_resizer("clean")
        self._frechet_distance = fid.frechet_distance
        self.package_version = importlib.metadata.version("clean-fid")

    def _resize_one(self, image: torch.Tensor) -> torch.Tensor:
        """uint8 (3, h, w) -> float32 (3, 299, 299) through Clean-FID's PIL resizer."""
        clean = self._resize(image.permute(1, 2, 0).cpu().numpy())
        return torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1)))

    @torch.inference_mode()
    def _features_uint8(self, images: torch.Tensor) -> np.ndarray:
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8:
            raise ValueError("Expected uint8 RGB images shaped (N, 3, H, W)")
        if len(images) == 0:
            raise ValueError("At least one image is required")
        chunks = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            for start in range(0, len(images), self.batch_size):
                batch = list(pool.map(self._resize_one, images[start:start + self.batch_size]))
                batch = torch.stack(batch).to(self.device, dtype=torch.float32)
                chunks.append(self._model(batch).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def features(self, images: torch.Tensor, image_range: str = "minus_one_one"
                 ) -> np.ndarray:
        """Inception features, shape (n, 2048), of decoded images."""
        return self._features_uint8(quantize_images_uint8(images, image_range))

    def features_from_folder(self, folder: PathLike) -> np.ndarray:
        """Inception features of every image file under `folder`."""
        chunks = []
        files = image_files(folder)
        for start in range(0, len(files), self.batch_size):
            batch = []
            for path in files[start:start + self.batch_size]:
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                batch.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
            chunks.append(self._features_uint8(torch.stack(batch)))
        return np.concatenate(chunks, axis=0)

    def distance_from_features(self, left: np.ndarray, right: np.ndarray) -> float:
        """Clean-FID's Frechet distance between two feature clouds."""
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("Expected compatible two-dimensional feature matrices")
        if len(left) < 2 or len(right) < 2:
            raise ValueError("FID requires at least two images in each sample")
        return float(self._frechet_distance(np.mean(left, axis=0), np.cov(left, rowvar=False),
                                            np.mean(right, axis=0), np.cov(right, rowvar=False)))

    def score_tensors(self, generated: torch.Tensor, reference: torch.Tensor,
                      image_range: str = "minus_one_one") -> float:
        return self.distance_from_features(self.features(generated, image_range),
                                           self.features(reference, image_range))

    def score_folders(self, generated: PathLike, reference: PathLike) -> float:
        return self.distance_from_features(self.features_from_folder(generated),
                                           self.features_from_folder(reference))

    def metadata(self) -> dict:
        return {"implementation": "clean-fid", "version": self.package_version,
                "mode": "clean", "feature_dimension": 2048,
                "weights_sha256": _sha256(self.weights_path),
                "tensor_quantization": "round-half-up uint8 before Clean-FID resize",
                "resize": "Clean-FID PIL bicubic 299x299"}


class LegacyFIDScorer:
    """
    TorchVision Inception-v3 FID reproducing the tables of Dance et al. (2026).

    Parameters
    ----------
    device : torch.device or str
        Where Inception runs.
    batch_size : int
    distance_device : torch.device or str
        Where `legacy_frechet_distance` runs its SVD (CPU for the CelebA
        tables, CUDA for the ColorMNIST tables of that paper).
    """

    def __init__(self, device: Union[torch.device, str], batch_size: int = 32,
                 distance_device: Union[torch.device, str] = "cpu") -> None:
        from torchvision.models import Inception_V3_Weights, inception_v3
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device)
        self.distance_device = torch.device(distance_device)
        self.batch_size = int(batch_size)
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model.dropout = torch.nn.Identity()
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def features(self, images: torch.Tensor, image_range: str = "minus_one_one"
                 ) -> torch.Tensor:
        """Pooled Inception features, shape (n, 2048), on CPU."""
        images = _to_unit_range(images, image_range).clamp(0.0, 1.0)
        chunks = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size].to(self.device)
            chunk = F.interpolate(chunk, size=(299, 299), mode="bilinear",
                                  align_corners=False, antialias=True)
            output = self.model((chunk - self._mean) / self._std)
            if not isinstance(output, torch.Tensor):
                output = output.logits
            chunks.append(output.flatten(1).cpu())
        if not chunks:
            raise ValueError("At least one image is required")
        return torch.cat(chunks)

    def features_from_folder(self, folder: PathLike) -> torch.Tensor:
        chunks = []
        files = image_files(folder)
        for start in range(0, len(files), self.batch_size):
            batch = []
            for path in files[start:start + self.batch_size]:
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                batch.append(torch.from_numpy(array).permute(2, 0, 1))
            chunks.append(self.features(torch.stack(batch), image_range="zero_one"))
        return torch.cat(chunks)

    def distance_from_features(self, generated: torch.Tensor, reference: torch.Tensor
                               ) -> float:
        return legacy_frechet_distance(generated, reference, device=self.distance_device)

    def score_tensors(self, generated: torch.Tensor, reference: torch.Tensor,
                      image_range: str = "minus_one_one") -> float:
        return self.distance_from_features(self.features(generated, image_range),
                                           self.features(reference, image_range))

    def score_folders(self, generated: PathLike, reference: PathLike) -> float:
        return self.distance_from_features(self.features_from_folder(generated),
                                           self.features_from_folder(reference))

    @staticmethod
    def metadata() -> dict:
        return {"implementation": "TorchVision Inception-v3 FID (Dance et al. 2026 tables)",
                "backbone": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
                "resize": "F.interpolate bilinear 299x299, antialias=True",
                "normalization": "ImageNet mean/std",
                "distance": "exact empirical low-rank Frechet identity"}
