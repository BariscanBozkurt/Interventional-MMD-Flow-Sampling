"""Focused tests for the cross-fitted doubly robust CPME weights."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from causality_utils import (context_shift_gaussian, cpme_dr_weights,  # noqa: E402
                             cpme_dr_weights_crossfit, importance_weights,
                             make_crossfit_fold_ids)
from kernel_utils import RBF  # noqa: E402


def _dsprite_style_data(n, seed):
    """Continuous 2D-action bandit with the dSprites policy family; outcome = action."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    logging_policy = context_shift_gaussian(-0.2, 0.5)
    target_policy = context_shift_gaussian(+0.2, 0.5)
    A = logging_policy.sample(X, rng)
    A_resample = target_policy.sample(X, rng)
    w = importance_weights(target_policy, logging_policy, X, A)
    return X, A, A_resample, w, target_policy, rng


class CrossfitDRWeightTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.n = 600
        X, A, A_resample, w, target_policy, rng = _dsprite_style_data(cls.n, seed=5)
        cls.X_np, cls.A_np = X, A
        kernel_x = RBF(use_length_scale_heuristic=True)
        kernel_a = RBF(use_length_scale_heuristic=True)
        X_t = torch.as_tensor(X, dtype=torch.float64)
        A_t = torch.as_tensor(A, dtype=torch.float64)
        cls.K_x = kernel_x(X_t)
        cls.K_a = kernel_a(A_t)
        cls.K_a_target = kernel_a(A_t, torch.as_tensor(A_resample, dtype=torch.float64))
        cls.w = torch.as_tensor(w, dtype=torch.float64)
        cls.A_t = A_t
        cls.target_mean = torch.as_tensor(
            target_policy.sample(np.repeat(X, 50, axis=0), rng).mean(axis=0)
        )
        cls.fold_ids = make_crossfit_fold_ids(cls.n, n_folds=3, seed=0)

    def test_matches_brute_force_per_observation_construction(self):
        """The block-algebra weights equal a literal per-fold pseudo-outcome loop."""
        lam = 0.05
        v_fast = cpme_dr_weights_crossfit(
            self.K_x, self.K_a, self.K_a_target, self.w, self.fold_ids, lam
        )

        n = self.n
        K = self.K_x * self.K_a
        K_pi = self.K_x * self.K_a_target
        v_slow = torch.zeros(n, dtype=torch.float64)
        for fold in np.unique(self.fold_ids):
            held_out = np.where(self.fold_ids == fold)[0]
            train = np.where(self.fold_ids != fold)[0]
            K_tt = K[train][:, train]
            eye = torch.eye(len(train), dtype=torch.float64)
            solve = torch.linalg.inv(K_tt + lam * eye)
            for i in held_out:
                beta_pi = solve @ K_pi[train, i]        # mu_hat_pi^(-k)(x_i) expansion
                beta_i = solve @ K[train, i]            # mu_hat^(-k)(x_i, a_i) expansion
                v_slow[train] += (beta_pi - self.w[i] * beta_i) / n
                v_slow[i] += self.w[i] / n
        self.assertLess(float((v_fast - v_slow).abs().max()), 1e-8)

    def test_full_training_block_formula_is_full_sample_dr_weights(self):
        """The fold formula with training block = all observations IS cpme_dr_weights.

        This is an algebraic identity of the formula, not a callable path: an actual
        single-fold assignment has empty training blocks and must raise instead.
        """
        lam = 0.05
        n = self.n
        K = self.K_x * self.K_a
        K_pi = self.K_x * self.K_a_target
        eye = torch.eye(n, dtype=torch.float64)
        rhs = K_pi.sum(dim=1) - K @ self.w
        v_degenerate = torch.linalg.solve(K + lam * eye, rhs) / n + self.w / n
        v_reference = cpme_dr_weights(self.K_x, self.K_a, self.K_a_target, self.w, lam)
        self.assertLess(float((v_degenerate - v_reference).abs().max()), 1e-10)

    def test_degenerate_fold_assignments_raise(self):
        """One fold (empty training block) or a fold covering n-1 rows must fail loudly."""
        with self.assertRaises(ValueError):
            cpme_dr_weights_crossfit(self.K_x, self.K_a, self.K_a_target, self.w,
                                     np.zeros(self.n, dtype=int), 0.05)
        nearly_all = np.zeros(self.n, dtype=int)
        nearly_all[0] = 1
        with self.assertRaises(ValueError):
            cpme_dr_weights_crossfit(self.K_x, self.K_a, self.K_a_target, self.w,
                                     nearly_all, 0.05)

    def test_crossfit_target_matches_interventional_action_mean(self):
        """sum_i v_i A_i estimates E_pi[A]; the logged mean is far from it."""
        v = cpme_dr_weights_crossfit(
            self.K_x, self.K_a, self.K_a_target, self.w, self.fold_ids, 0.05
        )
        estimate = (v[:, None] * self.A_t).sum(dim=0)
        logged_mean = self.A_t.mean(dim=0)
        self.assertLess(float((estimate - self.target_mean).abs().max()), 0.08)
        self.assertGreater(float((logged_mean - self.target_mean).abs().max()), 0.3)

    def test_per_fold_regularization_dict_is_accepted(self):
        lams = {int(f): lam for f, lam in zip(np.unique(self.fold_ids), (0.01, 0.05, 0.1))}
        v = cpme_dr_weights_crossfit(
            self.K_x, self.K_a, self.K_a_target, self.w, self.fold_ids, lams
        )
        self.assertEqual(v.shape, (self.n,))
        self.assertTrue(bool(torch.isfinite(v).all()))
        v_shared = cpme_dr_weights_crossfit(
            self.K_x, self.K_a, self.K_a_target, self.w, self.fold_ids, 0.05
        )
        self.assertGreater(float((v - v_shared).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
