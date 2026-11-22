"""Small deterministic image grids for kernel-herding diagnostics."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def deterministic_indices(size: int, count: int, seed: int) -> np.ndarray:
    """Choose a fixed, non-cherry-picked subset without replacement."""
    if size < 1:
        raise ValueError("size must be positive")
    if count < 0:
        raise ValueError("count must be nonnegative")
    count = min(count, size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=count, replace=False)).astype(np.int64)


def images_to_uint8(images: torch.Tensor, image_range: str) -> np.ndarray:
    """Convert an ``NCHW`` tensor to compact uint8 images."""
    images = torch.as_tensor(images).detach().cpu().float()
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("images must have shape (N,1,H,W) or (N,3,H,W)")
    if image_range == "minus_one_one":
        images = (images + 1.0) / 2.0
    elif image_range != "zero_one":
        raise ValueError("image_range must be 'zero_one' or 'minus_one_one'")
    return images.clamp(0.0, 1.0).mul(255.0).round().to(
        torch.uint8).numpy()


def save_labeled_image_grid(rows: list, output_path: Path,
                            n_columns: int = 8, padding: int = 2) -> None:
    """Save labeled rows of uint8 ``NCHW`` images as one PNG.

    Each item in ``rows`` is ``(label, images)``.  When a row contains more
    than ``n_columns`` images, it wraps within the same labeled block.
    """
    if not rows:
        raise ValueError("at least one image row is required")
    if n_columns < 1 or padding < 0:
        raise ValueError("n_columns must be positive and padding nonnegative")

    normalized = []
    height = width = None
    for label, images in rows:
        images = np.asarray(images)
        if images.ndim != 4 or images.shape[1] not in (1, 3):
            raise ValueError("grid images must be uint8 NCHW arrays")
        if images.dtype != np.uint8:
            raise ValueError("grid images must use uint8 storage")
        if images.shape[1] == 1:
            images = np.repeat(images, 3, axis=1)
        if height is None:
            height, width = images.shape[2:]
        elif images.shape[2:] != (height, width):
            raise ValueError("all grid rows must use the same image size")
        normalized.append((str(label), images))

    label_width = max(96, 7 * max(len(label) for label, _ in normalized) + 12)
    grid_columns = min(n_columns, max(len(images) for _, images in normalized))
    canvas_width = label_width + padding + grid_columns * (width + padding)
    block_heights = [
        padding + math.ceil(len(images) / grid_columns) * (height + padding)
        for _, images in normalized
    ]
    canvas = Image.new(
        "RGB", (canvas_width, sum(block_heights) + padding), "white")
    draw = ImageDraw.Draw(canvas)

    y_offset = 0
    for (label, images), block_height in zip(normalized, block_heights):
        draw.text((4, y_offset + padding + 4), label, fill="black")
        for index, image in enumerate(images):
            row, column = divmod(index, grid_columns)
            x = label_width + padding + column * (width + padding)
            y = y_offset + padding + row * (height + padding)
            canvas.paste(Image.fromarray(image.transpose(1, 2, 0)), (x, y))
        y_offset += block_height

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
