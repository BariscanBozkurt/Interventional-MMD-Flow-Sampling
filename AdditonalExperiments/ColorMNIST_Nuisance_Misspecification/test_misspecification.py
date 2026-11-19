"""Algebra and evaluation checks for the controlled nuisance experiment."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from colormnist_doublegen_misspecification import (  # noqa: E402
    build_target_weights,
    corrupt_importance_weights,
    uniform_w1,
)
from kernel_utils import RBF  # noqa: E402


def dense_targets(X, A, weights, folds, kernel, arm, rho, restricted,
                  fit_mode="crossfit"):
    """Independent per-observation KRR pseudo-outcome construction."""
    n = len(X)
    plugin = torch.zeros_like(weights)
    correction = torch.zeros_like(weights)
    blocks = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
              for fold in np.unique(folds)]
    if fit_mode == "nosplit":
        blocks = [(np.arange(n), np.arange(n))]
    for train, heldout in blocks:
        train = torch.as_tensor(train)
        heldout = torch.as_tensor(heldout)
        if restricted:
            train = train[X[train] < 0.5]
        K_tt = kernel(X[train, None], X[train, None])
        K_tt *= (A[train, None] == A[None, train])
        system = K_tt + rho * torch.eye(len(train), dtype=X.dtype)
        for i in heldout:
            k_x = kernel(X[train, None], X[i].reshape(1, 1)).flatten()
            beta_policy = torch.linalg.solve(system, k_x * (A[train] == arm))
            beta_logged = torch.linalg.solve(system, k_x * (A[train] == A[i]))
            plugin[train] += beta_policy / n
            correction[train] -= weights[i] * beta_logged / n
            correction[i] += weights[i] / n
    return plugin, weights / n, plugin + correction


class NuisanceExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = torch.linspace(0.025, 0.975, 20, dtype=torch.float64)
        cls.A = torch.arange(20) % 2
        cls.Z = torch.stack((cls.X, cls.X.square()), dim=1)
        cls.folds = np.tile([0, 1, 1, 0], 5)
        cls.kernel = RBF(length_scale=0.07)
        cls.arm = 1
        cls.rho = 0.3
        probability = torch.sigmoid(2.0 * (cls.X - 0.5))
        cls.weights = (cls.A == cls.arm).to(cls.X.dtype) / probability
        cls.targets, cls.metadata = build_target_weights(
            cls.X, cls.A, cls.Z, cls.weights, cls.folds, cls.kernel,
            arm=cls.arm, n_landmarks=len(cls.X), gram_ridge=cls.rho,
            seed=7, jitter=0.0,
        )

    def test_raw_corruption_changes_only_prespecified_subgroup(self):
        original = self.weights.clone()
        corrupted = corrupt_importance_weights(self.weights, self.X)
        expected = self.weights * torch.where(self.X >= 0.5, 0.25, 1.0)
        torch.testing.assert_close(corrupted, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.weights, original, rtol=0, atol=0)
        self.assertEqual(corrupted.dtype, torch.float64)
        self.assertLess(float(corrupted.sum()), float(self.weights.sum()))

    def test_preserve_mode_retains_baseline_mass_and_relative_distortion(self):
        corrupted = corrupt_importance_weights(
            self.weights, self.X, mass_mode="preserve"
        )
        torch.testing.assert_close(corrupted.sum(), self.weights.sum())
        nonzero = self.weights > 0
        multiplier = corrupted[nonzero] / self.weights[nonzero]
        red = self.X[nonzero] >= 0.5
        torch.testing.assert_close(multiplier[red].mean(),
                                   0.25 * multiplier[~red].mean())
        self.assertNotAlmostEqual(float(corrupted.sum()), 1.0)

    def test_all_landmarks_match_dense_pseudo_outcomes(self):
        """Restricted fits retain the original evaluation sample and denominator."""
        for cme, restricted in (("full", False), ("blue_only", True)):
            for weight_mode in ("baseline", "red_downweighted"):
                weights = self.weights
                if weight_mode == "red_downweighted":
                    weights = corrupt_importance_weights(weights, self.X)
                expected = dense_targets(
                    self.X, self.A, weights, self.folds, self.kernel,
                    self.arm, self.rho, restricted,
                )
                for estimator, vector in zip(("plugin", "ipw", "dr"), expected):
                    with self.subTest(cme=cme, weights=weight_mode, estimator=estimator):
                        torch.testing.assert_close(
                            self.targets[cme, weight_mode, estimator], vector,
                            rtol=1e-9, atol=1e-11,
                        )

    def test_restricted_fit_keeps_red_heldout_residuals(self):
        red = self.X >= 0.5
        for weight_mode in ("baseline", "red_downweighted"):
            weights = self.weights
            if weight_mode == "red_downweighted":
                weights = corrupt_importance_weights(weights, self.X)
            torch.testing.assert_close(
                self.targets["blue_only", weight_mode, "plugin"][red],
                torch.zeros_like(weights[red]), rtol=0, atol=0,
            )
            torch.testing.assert_close(
                self.targets["blue_only", weight_mode, "dr"][red],
                weights[red] / len(self.X), rtol=1e-12, atol=1e-12,
            )

    def test_fit_metadata_proves_split_and_landmark_restrictions(self):
        coverage = {"full": [], "blue_only": []}
        for fit in self.metadata:
            cme = fit["cme_mode"]
            train = set(fit["train_indices"])
            heldout = set(fit["heldout_indices"])
            landmarks = set(fit["landmarks"])
            self.assertFalse(train & heldout)
            self.assertEqual(landmarks, train)
            expected = set(range(len(self.X))) - heldout
            if cme == "blue_only":
                expected = {i for i in expected if self.X[i] < 0.5}
            self.assertEqual(train, expected)
            coverage[cme].extend(heldout)
        for observations in coverage.values():
            self.assertEqual(sorted(observations), list(range(len(self.X))))

    def test_nosplit_keeps_evaluation_rows_outside_cme_training_subgroup(self):
        targets, _ = build_target_weights(
            self.X, self.A, self.Z, self.weights, self.folds, self.kernel,
            arm=self.arm, fit_mode="nosplit", n_landmarks=len(self.X),
            gram_ridge=self.rho, seed=7, jitter=0.0,
        )
        for cme, restricted in (("full", False), ("blue_only", True)):
            for weight_mode in ("baseline", "red_downweighted"):
                weights = self.weights
                if weight_mode == "red_downweighted":
                    weights = corrupt_importance_weights(weights, self.X)
                expected = dense_targets(
                    self.X, self.A, weights, self.folds, self.kernel,
                    self.arm, self.rho, restricted, fit_mode="nosplit",
                )
                for estimator, vector in zip(("plugin", "ipw", "dr"), expected):
                    with self.subTest(cme=cme, weights=weight_mode, estimator=estimator):
                        torch.testing.assert_close(
                            targets[cme, weight_mode, estimator], vector,
                            rtol=1e-9, atol=1e-11,
                        )

    def test_estimator_axes_are_independent_and_ipw_stays_raw(self):
        for cme in ("full", "blue_only"):
            torch.testing.assert_close(
                self.targets[cme, "baseline", "plugin"],
                self.targets[cme, "red_downweighted", "plugin"],
            )
        for weight_mode in ("baseline", "red_downweighted"):
            torch.testing.assert_close(
                self.targets["full", weight_mode, "ipw"],
                self.targets["blue_only", weight_mode, "ipw"],
                rtol=0, atol=0,
            )
        raw = corrupt_importance_weights(self.weights, self.X) / len(self.X)
        torch.testing.assert_close(
            self.targets["full", "red_downweighted", "ipw"], raw,
            rtol=0, atol=0,
        )
        self.assertGreater(abs(float(raw.sum()) - 1), 0.1)

    def test_mass_control_is_passed_to_targets_without_final_normalization(self):
        targets, _ = build_target_weights(
            self.X, self.A, self.Z, self.weights, self.folds, self.kernel,
            arm=self.arm, n_landmarks=len(self.X), gram_ridge=self.rho,
            seed=7, jitter=0.0, corruption_mass="preserve",
        )
        preserved = corrupt_importance_weights(
            self.weights, self.X, mass_mode="preserve"
        ) / len(self.X)
        torch.testing.assert_close(
            targets["full", "red_downweighted", "ipw"], preserved
        )
        self.assertNotAlmostEqual(float(preserved.sum()), 1.0)
        for cme in ("full", "blue_only"):
            for estimator in ("plugin", "ipw", "dr"):
                torch.testing.assert_close(
                    targets[cme, "baseline", estimator],
                    self.targets[cme, "baseline", estimator],
                )

    def test_uniform_distance_includes_empirical_quantile_discretization(self):
        """Exact W1 to a continuous uniform law is positive for finite grids."""
        self.assertAlmostEqual(float(uniform_w1(torch.tensor([0.5]))), 0.25)
        self.assertAlmostEqual(float(uniform_w1(torch.tensor([0.0]))), 0.5)
        self.assertAlmostEqual(float(uniform_w1(torch.tensor([1.0]))), 0.5)
        self.assertAlmostEqual(float(uniform_w1(torch.tensor([0.25, 0.75]))), 0.125)
        self.assertAlmostEqual(float(uniform_w1(torch.tensor([0.75, 0.25]))), 0.125)


if __name__ == "__main__":
    unittest.main()
