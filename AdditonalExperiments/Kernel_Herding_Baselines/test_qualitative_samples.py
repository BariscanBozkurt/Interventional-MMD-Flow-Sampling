"""Focused tests for deterministic qualitative sample grids."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qualitative_samples import (
    deterministic_indices,
    images_to_uint8,
    save_labeled_image_grid,
)


class QualitativeSamplesTest(unittest.TestCase):
    def test_deterministic_indices_are_reproducible_and_unique(self):
        first = deterministic_indices(100, 16, seed=17)
        second = deterministic_indices(100, 16, seed=17)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(np.unique(first)), 16)
        self.assertTrue(np.all(first[:-1] < first[1:]))

    def test_range_conversion_and_grid(self):
        row_a = images_to_uint8(
            torch.linspace(-1.0, 1.0, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8),
            "minus_one_one")
        row_b = images_to_uint8(
            torch.linspace(0.0, 1.0, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8),
            "zero_one")
        self.assertEqual(row_a.dtype, np.uint8)
        self.assertEqual(int(row_a.min()), 0)
        self.assertEqual(int(row_a.max()), 255)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.png"
            save_labeled_image_grid(
                [("continuous", row_a), ("oracle", row_b)], path,
                n_columns=2)
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertGreater(image.width, 16)
                self.assertGreater(image.height, 16)


if __name__ == "__main__":
    unittest.main()
