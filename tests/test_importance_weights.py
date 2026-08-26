"""Focused tests for continuous-action importance-weight estimation."""

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from causality_utils import (GaussianPolicy, context_shift_gaussian,  # noqa: E402
                             importance_weights, make_crossfit_fold_ids)
from dens_ratio_utils import (  # noqa: E402
    crossfit_classifier_log_ratio,
    crossfit_logging_policy_log_prob,
    estimate_classifier_importance_weights,
    estimate_logging_policy_importance_weights,
    importance_weight_diagnostics,
    importance_weights_from_log_ratio,
    importance_weights_from_logging_log_prob,
)


class ImportanceWeightEstimationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.X = rng.uniform(size=(1500, 2))
        cls.logging_policy = context_shift_gaussian(-0.2, 0.5)
        cls.target_policy = context_shift_gaussian(+0.2, 0.5)
        cls.A = cls.logging_policy.sample(cls.X, rng)
        cls.oracle = importance_weights(
            cls.target_policy, cls.logging_policy, cls.X, cls.A
        )
        cls.fold_ids = make_crossfit_fold_ids(len(cls.X), n_folds=5, seed=7)

    def test_fold_assignment_is_complete_and_balanced(self):
        counts = np.bincount(self.fold_ids)
        self.assertEqual(len(self.fold_ids), len(self.X))
        self.assertEqual(len(counts), 5)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        with self.assertRaises(ValueError):
            make_crossfit_fold_ids(2, n_folds=2, seed=0)

    def test_classifier_recovers_dsprite_style_oracle_ratio(self):
        log_ratio = crossfit_classifier_log_ratio(
            self.X,
            self.A,
            self.target_policy,
            n_target_actions=3,
            seed=11,
            fold_ids=self.fold_ids,
        )
        estimated = importance_weights_from_log_ratio(log_ratio)
        diagnostics = importance_weight_diagnostics(estimated, self.oracle)

        self.assertEqual(estimated.shape, self.oracle.shape)
        self.assertTrue(np.isfinite(estimated).all())
        self.assertTrue((estimated > 0.0).all())
        self.assertLess(diagnostics["log_rmse"], 0.15)
        self.assertGreater(diagnostics["ess"], 0.0)
        self.assertLessEqual(diagnostics["ess"], len(estimated))

    def test_classifier_is_reproducible_and_not_self_normalized(self):
        first = estimate_classifier_importance_weights(
            self.X,
            self.A,
            self.target_policy,
            n_target_actions=1,
            seed=19,
            fold_ids=self.fold_ids,
        )
        second = estimate_classifier_importance_weights(
            self.X,
            self.A,
            self.target_policy,
            n_target_actions=1,
            seed=19,
            fold_ids=self.fold_ids,
        )
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertGreater(abs(first.sum() - len(first)), 1e-6)

    def test_multiple_target_draws_keep_the_classifier_prior_balanced(self):
        estimated = estimate_classifier_importance_weights(
            self.X,
            self.A,
            self.logging_policy,
            n_target_actions=4,
            seed=23,
            fold_ids=self.fold_ids,
        )
        log_rmse = np.sqrt(np.mean(np.log(estimated) ** 2))
        self.assertLess(log_rmse, 0.15)
        self.assertLess(abs(estimated.mean() - 1.0), 0.10)

    def test_logging_density_is_reusable_for_target_policies(self):
        logging_log_prob = crossfit_logging_policy_log_prob(
            self.X,
            self.A,
            fold_ids=self.fold_ids,
        )
        estimated = importance_weights_from_logging_log_prob(
            self.target_policy, self.X, self.A, logging_log_prob
        )
        wrapper_result = estimate_logging_policy_importance_weights(
            self.X,
            self.A,
            self.target_policy,
            fold_ids=self.fold_ids,
        )
        diagnostics = importance_weight_diagnostics(estimated, self.oracle)

        np.testing.assert_allclose(estimated, wrapper_result, rtol=0.0, atol=0.0)
        self.assertLess(diagnostics["log_rmse"], 0.15)

        second_target = context_shift_gaussian(0.0, 0.5)
        second_weights = importance_weights_from_logging_log_prob(
            second_target, self.X, self.A, logging_log_prob
        )
        self.assertEqual(second_weights.shape, estimated.shape)
        self.assertTrue((second_weights > 0.0).all())

    def test_gaussian_policy_rejects_invalid_scales(self):
        with self.assertRaises(ValueError):
            GaussianPolicy(lambda X: X[:, :1], 0.0)
        with self.assertRaises(ValueError):
            GaussianPolicy(lambda X: X[:, :1], np.array([1.0, np.nan]))

    def test_diagnostics_avoid_ess_overflow(self):
        weights = np.array([np.finfo(float).max / 4.0,
                            np.finfo(float).max / 2.0])
        diagnostics = importance_weight_diagnostics(weights)
        self.assertTrue(np.isfinite(diagnostics["ess"]))
        self.assertGreaterEqual(diagnostics["ess"], 1.0)
        self.assertLessEqual(diagnostics["ess"], 2.0)


if __name__ == "__main__":
    unittest.main()
