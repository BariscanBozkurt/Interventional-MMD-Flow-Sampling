"""Independent algebra checks for the ColorMNIST nuisance severity ladder."""

import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from colormnist_misspecification_severity import (  # noqa: E402
    build_severity_targets,
    target_flow_id,
)
from kernel_utils import RBF  # noqa: E402


def dense_fullsample_targets(X, A, weights, kernel, arm, cutoff, rho):
    """Full KRR reference with explicit prediction coefficients for every row."""
    n = len(X)
    train = torch.arange(n) if cutoff == 1.0 else torch.where(X < cutoff)[0]
    K_x = kernel(X[train, None], X[:, None])
    K_logged = K_x * (A[train, None] == A[None, :])
    K_policy = K_x * (A[train, None] == arm)
    system = K_logged[:, train] + rho * torch.eye(len(train), dtype=X.dtype)
    fitted_policy = torch.linalg.solve(system, K_policy)
    fitted_logged = torch.linalg.solve(system, K_logged)
    plugin = torch.zeros_like(weights)
    plugin[train] = fitted_policy.sum(1) / n
    dr = plugin + weights / n
    dr[train] -= fitted_logged @ weights / n
    return plugin, weights / n, dr


class SeverityExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = torch.linspace(0.0125, 0.9875, 40, dtype=torch.float64)
        cls.X[-1] = 1.0
        cls.A = torch.arange(len(cls.X)) % 2
        cls.Z = torch.stack((cls.X, cls.X.square()), dim=1)
        cls.kernel = RBF(length_scale=0.04)
        cls.arm = 1
        cls.rho = 0.2
        cls.weights = (cls.A == cls.arm).double() / torch.sigmoid(
            2.0 * (cls.X - 0.5)
        )
        cls.originals = [value.clone() for value in (cls.X, cls.A, cls.Z, cls.weights)]
        cls.targets, cls.metadata, cls.settings = build_severity_targets(
            cls.X, cls.A, cls.Z, cls.weights, cls.kernel,
            arm=cls.arm, n_landmarks=len(cls.X), gram_ridge=cls.rho,
            seed=7, jitter=0.0, block_size=7,
        )

    def test_default_grid_has_seven_weight_settings_and_39_unique_flows(self):
        expected_weights = {(1.0, "baseline")} | {
            (factor, mode)
            for factor in (0.25, 0.05, 0.01)
            for mode in ("raw", "preserve")
        }
        self.assertEqual(set(self.settings), expected_weights)
        self.assertEqual(len(self.targets), 4 * 7 * 3)
        self.assertEqual(len({target_flow_id(key) for key in self.targets}), 39)
        self.assertTrue(all(key[2] == "baseline" for key in self.targets if key[1] == 1))

    def test_all_landmark_targets_match_independent_dense_krr(self):
        for cutoff in (1.0, 0.5, 0.25, 0.1):
            for (factor, mode), weights in self.settings.items():
                expected = dense_fullsample_targets(
                    self.X, self.A, weights, self.kernel, self.arm, cutoff, self.rho
                )
                for estimator, vector in zip(("plugin", "ipw", "dr"), expected):
                    with self.subTest(cutoff=cutoff, factor=factor, mode=mode,
                                      estimator=estimator):
                        torch.testing.assert_close(
                            self.targets[cutoff, factor, mode, estimator], vector,
                            rtol=2e-9, atol=2e-11,
                        )

    def test_cme_cutoff_does_not_move_weight_corruption_threshold(self):
        for factor in (0.25, 0.05, 0.01):
            expected = self.weights * torch.where(self.X >= 0.5, factor, 1.0)
            torch.testing.assert_close(self.settings[factor, "raw"], expected,
                                       rtol=0, atol=0)
            middle = (self.X >= 0.25) & (self.X < 0.5)
            torch.testing.assert_close(self.settings[factor, "raw"][middle],
                                       self.weights[middle], rtol=0, atol=0)
            torch.testing.assert_close(
                self.targets[0.25, factor, "raw", "plugin"][middle],
                torch.zeros_like(self.weights[middle]), rtol=0, atol=0,
            )

    def test_every_excluded_observation_retains_its_dr_residual(self):
        for cutoff in (0.5, 0.25, 0.1):
            excluded = self.X >= cutoff
            for (factor, mode), weights in self.settings.items():
                with self.subTest(cutoff=cutoff, factor=factor, mode=mode):
                    torch.testing.assert_close(
                        self.targets[cutoff, factor, mode, "plugin"][excluded],
                        torch.zeros_like(weights[excluded]), rtol=0, atol=0,
                    )
                    torch.testing.assert_close(
                        self.targets[cutoff, factor, mode, "dr"][excluded],
                        weights[excluded] / len(self.X), rtol=1e-12, atol=1e-12,
                    )

    def test_mass_control_preserves_baseline_mass_and_relative_distortion(self):
        torch.testing.assert_close(self.settings[1.0, "baseline"], self.weights,
                                   rtol=0, atol=0)
        nonzero = self.weights > 0
        red = self.X[nonzero] >= 0.5
        for factor in (0.25, 0.05, 0.01):
            raw = self.settings[factor, "raw"]
            preserved = self.settings[factor, "preserve"]
            torch.testing.assert_close(preserved.sum(), self.weights.sum())
            self.assertLess(float(raw.sum()), float(self.weights.sum()))
            multiplier = preserved[nonzero] / self.weights[nonzero]
            torch.testing.assert_close(multiplier[red].mean(),
                                       factor * multiplier[~red].mean())
            self.assertGreater(abs(float(preserved.sum()) - 1.0), 0.1)

    def test_preserved_dr_scales_only_the_corrupted_correction(self):
        for factor in (0.25, 0.05, 0.01):
            scale = self.weights.sum() / self.settings[factor, "raw"].sum()
            for cutoff in (1.0, 0.5, 0.25, 0.1):
                plugin = self.targets[cutoff, factor, "raw", "plugin"]
                raw_dr = self.targets[cutoff, factor, "raw", "dr"]
                torch.testing.assert_close(
                    self.targets[cutoff, factor, "preserve", "dr"],
                    plugin + scale * (raw_dr - plugin), rtol=2e-9, atol=2e-11,
                )

    def test_estimator_axis_invariances_and_input_nonmutation(self):
        for cutoff in (1.0, 0.5, 0.25, 0.1):
            plugin = self.targets[cutoff, 1.0, "baseline", "plugin"]
            for factor, mode in self.settings:
                torch.testing.assert_close(
                    self.targets[cutoff, factor, mode, "plugin"], plugin,
                    rtol=0, atol=0,
                )
                torch.testing.assert_close(
                    self.targets[cutoff, factor, mode, "ipw"],
                    self.settings[factor, mode] / len(self.X), rtol=0, atol=0,
                )
        for value, original in zip((self.X, self.A, self.Z, self.weights), self.originals):
            torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_fit_metadata_has_nested_support_and_all_landmarks(self):
        self.assertEqual(len(self.metadata), 4)
        for fit in self.metadata:
            cutoff = fit["cme_cutoff"]
            expected = set(range(len(self.X))) if cutoff == 1 else {
                i for i, x in enumerate(self.X) if x < cutoff
            }
            train = set(fit["train_indices"])
            self.assertEqual(train, expected)
            self.assertEqual(set(fit["landmarks"]), train)
            self.assertTrue(set(fit["cv_indices"]).issubset(train))
            self.assertEqual(fit["n_train"], len(train))
            for arm in (0, 1):
                count = sum(int(self.A[i]) == arm for i in train)
                self.assertGreater(count, 0)
                self.assertEqual(fit[f"n_train_arm{arm}"], count)
                self.assertEqual(fit[f"n_landmarks_arm{arm}"], count)

    def test_training_only_cv_and_landmark_subsets(self):
        _, metadata, _ = build_severity_targets(
            self.X, self.A, self.Z, self.weights, self.kernel,
            arm=self.arm, cme_cutoffs=(1.0, 0.25), weight_factors=(1.0, 0.05),
            n_landmarks=8, cv_max_n=8, seed=7, block_size=7,
        )
        for fit in metadata:
            train = set(fit["train_indices"])
            self.assertEqual(len(fit["cv_indices"]), 8)
            self.assertEqual(len(fit["landmarks"]), 8)
            self.assertTrue(set(fit["cv_indices"]).issubset(train))
            self.assertTrue(set(fit["landmarks"]).issubset(train))
            self.assertGreater(fit["gram_ridge"], 0)

    def test_batched_weights_and_block_sizes_do_not_change_existing_targets(self):
        targets, _, _ = build_severity_targets(
            self.X, self.A, self.Z, self.weights, self.kernel,
            arm=self.arm, cme_cutoffs=(1.0, 0.25), weight_factors=(1.0, 0.05),
            n_landmarks=len(self.X), gram_ridge=self.rho,
            seed=7, jitter=0.0, block_size=len(self.X),
        )
        for key, vector in targets.items():
            torch.testing.assert_close(vector, self.targets[key],
                                       rtol=2e-9, atol=2e-11)


if __name__ == "__main__":
    unittest.main()
