"""Focused tests for the experiment-local continuous herding optimizer."""

import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
DEPLOY_ROOT = HERE.parents[1]
sys.path.insert(0, str(DEPLOY_ROOT / "src"))
sys.path.insert(0, str(HERE))

from causality_utils import kernel_herding
from continuous_latent_herding import (
    continuous_kernel_herding,
    herding_score_and_gradient,
    latent_quantile_bounds,
)
from kernel_utils import IMQKernel, RieszKernel


class ContinuousHerdingTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.support = torch.tensor([
            [-1.2, 0.3], [0.2, -0.8], [1.1, 0.6], [0.5, 1.4],
        ])
        self.weights = torch.tensor([0.45, -0.10, 0.35, 0.30])
        self.previous = torch.tensor([[-0.4, 0.7], [0.8, -0.1]])

    def _check_gradient(self, kernel):
        query = torch.tensor([[-0.2, 0.1], [0.7, 0.9]], requires_grad=True)
        score = (kernel(self.support, query).T @ self.weights
                 - kernel(self.previous, query).mean(dim=0))
        autograd = torch.autograd.grad(score.sum(), query)[0]
        manual_score, manual = herding_score_and_gradient(
            self.weights, kernel, self.support, query.detach(), self.previous,
            support_chunk_size=2, repulsion_chunk_size=1,
            repulsion_denominator="previous_count")
        self.assertTrue(torch.allclose(manual_score, score.detach(), atol=1e-12))
        self.assertTrue(torch.allclose(manual, autograd, atol=1e-10))

    def test_imq_gradient_matches_autograd(self):
        self._check_gradient(IMQKernel(c=1.3, beta=-0.5))

    def test_riesz_gradient_matches_autograd(self):
        self._check_gradient(RieszKernel(s=1.0, eps=1e-4))

    def test_zero_step_candidate_mode_matches_finite_herding(self):
        kernel = IMQKernel(c=1.0, beta=-0.5)
        finite = kernel_herding(
            self.weights, kernel, self.support, self.support, 4)
        continuous, diagnostics = continuous_kernel_herding(
            self.weights, kernel, self.support, 4,
            initial_candidates=self.support,
            n_restarts=1, n_steps=0, domain="unbounded",
            repulsion_denominator="previous_count")
        self.assertTrue(torch.equal(continuous, finite))
        self.assertTrue(torch.all(diagnostics["score_improvements"] >= -1e-12))

    def test_default_uses_literal_algorithm2_denominator(self):
        kernel = IMQKernel(c=1.3, beta=-0.5)
        query = torch.tensor([[0.1, 0.4]])
        score, gradient = herding_score_and_gradient(
            self.weights, kernel, self.support, query, self.previous)
        expected = (kernel(self.support, query).T @ self.weights
                    - kernel(self.previous, query).sum(dim=0) / 3.0)
        query_autograd = query.clone().requires_grad_(True)
        expected_autograd = (
            kernel(self.support, query_autograd).T @ self.weights
            - kernel(self.previous, query_autograd).sum(dim=0) / 3.0)
        expected_gradient = torch.autograd.grad(
            expected_autograd.sum(), query_autograd)[0]
        self.assertTrue(torch.allclose(score, expected, atol=1e-12))
        self.assertTrue(torch.allclose(gradient, expected_gradient, atol=1e-10))

    def test_continuous_optimizer_returns_novel_point(self):
        support = torch.tensor([[-1.0], [1.0]])
        weights = torch.tensor([0.5, 0.5])
        kernel = IMQKernel(c=2.0, beta=-0.5)
        atoms, diagnostics = continuous_kernel_herding(
            weights, kernel, support, 1,
            initial_candidates=support,
            n_restarts=1, n_steps=80, learning_rate=0.03,
            domain="unbounded")
        self.assertGreater(float(torch.cdist(atoms, support).min()), 0.2)
        self.assertGreater(float(diagnostics["score_improvements"][0]), 0.0)

    def test_multistart_is_reproducible_and_no_worse_than_anchor(self):
        kernel = IMQKernel(c=1.0, beta=-0.5)
        anchor = kernel_herding(
            self.weights, kernel, self.support, self.support, 3)

        def run():
            return continuous_kernel_herding(
                self.weights, kernel, self.support, 3,
                initial_candidates=self.support,
                anchor_points=anchor,
                n_restarts=4, n_steps=8, learning_rate=0.02,
                restart_scale=0.1, domain="support_quantile_box",
                generator=torch.Generator().manual_seed(17))

        atoms_1, diagnostics_1 = run()
        atoms_2, diagnostics_2 = run()
        self.assertTrue(torch.equal(atoms_1, atoms_2))
        self.assertTrue(torch.equal(
            diagnostics_1["selected_restarts"],
            diagnostics_2["selected_restarts"]))
        self.assertTrue(torch.all(
            diagnostics_1["score_improvements_over_anchor"] >= -1e-12))

    def test_unbounded_nonunit_riesz_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unit-mass"):
            continuous_kernel_herding(
                self.weights * 0.9,
                RieszKernel(s=1.0, eps=1e-4), self.support, 2,
                n_restarts=1, n_steps=0, domain="unbounded")

    def test_unbounded_riesz_requires_algorithm2_denominator(self):
        with self.assertRaisesRegex(ValueError, "Algorithm-2 denominator"):
            continuous_kernel_herding(
                self.weights,
                RieszKernel(s=1.0, eps=1e-4), self.support, 2,
                n_restarts=1, n_steps=0, domain="unbounded",
                repulsion_denominator="previous_count")

    def test_unbounded_unit_mass_riesz_returns_finite_atoms(self):
        atoms, diagnostics = continuous_kernel_herding(
            self.weights,
            RieszKernel(s=1.0, eps=1e-4), self.support, 3,
            n_restarts=2, n_steps=5, learning_rate=0.02,
            domain="unbounded", repulsion_denominator="algorithm2",
            generator=torch.Generator().manual_seed(19))
        self.assertTrue(torch.isfinite(atoms).all())
        self.assertTrue(torch.isfinite(diagnostics["final_scores"]).all())

    def test_zero_quantile_box_uses_padded_coordinate_range(self):
        lower, upper = latent_quantile_bounds(
            self.support, lower_quantile=0.0, upper_quantile=1.0,
            padding_fraction=0.1)
        raw_lower = self.support.amin(dim=0)
        raw_upper = self.support.amax(dim=0)
        width = raw_upper - raw_lower
        self.assertTrue(torch.equal(lower, raw_lower - 0.1 * width))
        self.assertTrue(torch.equal(upper, raw_upper + 0.1 * width))


if __name__ == "__main__":
    unittest.main()
