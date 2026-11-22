"""Independent algebra and approximation checks for cached IMQ features."""

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from kernel_utils import IMQKernel
from mmd_flow_utils.mmd_flow import mmd_gradient_flow, witness_gradient
from rff_imq import (
    cache_target_mean, exact_imq_euler_flow, exact_imq_witness_gradient,
    imq_features, rff_euler_flow, rff_witness_gradient, sample_imq_frequencies,
)


def explicit_fourier_kernel(x, y, omega, c):
    """Independent stationary cosine formula, without constructing features."""
    difference = x[:, None, :] - y[None, :, :]
    phase = torch.einsum("ijd,ld->ijl", difference, omega)
    return phase.cos().mean(2) / c


class CachedIMQRFFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(31)
        cls.initial = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        cls.targets = torch.randn(7, 3, generator=generator, dtype=torch.float64)
        cls.weights = torch.tensor([.4, -.3, .2, -.1, .05, .1, -.15],
                                   dtype=torch.float64)
        cls.c = 1.7
        cls.omega = sample_imq_frequencies(3, 128, cls.c, seed=13, dtype=torch.float64)
        cls.cached = cache_target_mean(cls.targets, cls.weights, cls.omega, cls.c)

    def test_frequency_reproducibility_prefix_and_kernel_scale(self):
        small = sample_imq_frequencies(3, 18, self.c, seed=5, dtype=torch.float64)
        large = sample_imq_frequencies(3, 134, self.c, seed=5, dtype=torch.float64)
        torch.testing.assert_close(small, large[:9], rtol=0, atol=0)
        scaled = sample_imq_frequencies(3, 18, 2 * self.c, seed=5, dtype=torch.float64)
        torch.testing.assert_close(scaled, small / 2, rtol=0, atol=0)
        other = sample_imq_frequencies(3, 18, self.c, seed=6, dtype=torch.float64)
        self.assertFalse(torch.equal(small, other))

    def test_features_have_exact_diagonal_and_stationary_cosine_kernel(self):
        x = self.initial
        features = imq_features(x, self.omega, self.c)
        kernel = explicit_fourier_kernel(x, x, self.omega, self.c)
        torch.testing.assert_close(features @ features.T, kernel, rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(features.square().sum(1),
                                   x.new_full((len(x),), 1 / self.c),
                                   rtol=2e-14, atol=2e-14)
        translated = imq_features(x + 7.3, self.omega, self.c)
        torch.testing.assert_close(translated @ translated.T, kernel,
                                   rtol=2e-13, atol=2e-13)

    def test_signed_target_cache_is_raw_and_block_invariant(self):
        expected = self.weights @ imq_features(self.targets, self.omega, self.c)
        for size in [1, 3, 20]:
            actual = cache_target_mean(self.targets, self.weights, self.omega,
                                       self.c, block_size=size)
            torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
        zero_mass = self.weights - self.weights.mean()
        actual = cache_target_mean(self.targets, zero_mass, self.omega, self.c)
        torch.testing.assert_close(actual,
                                   zero_mass @ imq_features(self.targets, self.omega, self.c),
                                   rtol=2e-14, atol=2e-14)
        self.assertGreater(float(actual.norm()), .01)

    def test_rff_gradient_matches_independent_stationary_witness_autograd(self):
        query = (self.initial[:3] + .27).clone().requires_grad_(True)
        source = explicit_fourier_kernel(query, self.initial, self.omega, self.c).mean(1)
        target = explicit_fourier_kernel(query, self.targets, self.omega, self.c) @ self.weights
        expected, = torch.autograd.grad((source - target).sum(), query)
        actual = rff_witness_gradient(self.initial, self.cached, self.omega, self.c,
                                      eval_points=query.detach())
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)

    def test_rff_gradient_has_correct_particle_scaling(self):
        x = self.initial.clone().requires_grad_(True)
        discrepancy = imq_features(x, self.omega, self.c).mean(0) - self.cached
        objective = .5 * discrepancy.square().sum()
        ordinary_gradient, = torch.autograd.grad(objective, x)
        actual = rff_witness_gradient(self.initial, self.cached, self.omega, self.c)
        torch.testing.assert_close(actual, len(x) * ordinary_gradient,
                                   rtol=2e-13, atol=2e-13)

    def test_singleton_self_force_vanishes(self):
        point = self.initial[:1]
        zero_target = self.cached.new_zeros(self.cached.shape)
        actual = rff_witness_gradient(point, zero_target, self.omega, self.c)
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=1e-16)
        expected_query = point.clone().requires_grad_(True)
        target = explicit_fourier_kernel(expected_query, self.targets,
                                         self.omega, self.c) @ self.weights
        expected, = torch.autograd.grad(-target.sum(), expected_query)
        actual = rff_witness_gradient(point, self.cached, self.omega, self.c)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)

    def test_rff_euler_matches_independent_full_objective_trajectory(self):
        reference = self.initial.clone()
        step = .3
        for _ in range(6):
            x = reference.detach().requires_grad_(True)
            pair = explicit_fourier_kernel(x, x, self.omega, self.c).mean()
            cross = (explicit_fourier_kernel(x, self.targets, self.omega, self.c)
                     @ self.weights).mean()
            grad, = torch.autograd.grad(.5 * pair - cross, x)
            reference = x.detach() - step * len(x) * grad
        initial_before = self.initial.clone()
        actual = rff_euler_flow(self.initial, self.cached, self.omega, self.c, step, 6)
        torch.testing.assert_close(actual, reference, rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(self.initial, initial_before, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)

    def test_exact_analytic_gradient_matches_existing_autograd_with_signed_weights(self):
        kernel = IMQKernel(c=self.c, beta=-.5)
        query = self.initial[:3] + .31
        expected = witness_gradient(kernel, self.initial, self.targets, query,
                                    target_weights=self.weights)
        for block_size in [None, 2, 20]:
            actual = exact_imq_witness_gradient(
                self.initial, self.targets, self.weights, self.c,
                eval_points=query, block_size=block_size)
            torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)

    def test_exact_euler_matches_existing_deployed_flow(self):
        expected, _ = mmd_gradient_flow(
            IMQKernel(c=self.c, beta=-.5), self.targets, self.initial,
            step_size=.3, n_iterations=6, target_weights=self.weights,
            log_every=6, verbose=False)
        actual = exact_imq_euler_flow(
            self.initial, self.targets, self.weights, self.c, .3, 6, block_size=3)
        torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)

    def test_exact_gradient_with_self_queries_and_duplicate_points(self):
        particles = torch.cat((self.initial, self.initial[:2]), dim=0)
        targets = torch.cat((self.targets, particles[:1]), dim=0)
        weights = torch.cat((self.weights, self.weights.new_tensor([-.17])))
        expected = witness_gradient(IMQKernel(c=self.c, beta=-.5), particles,
                                    targets, particles, target_weights=weights)
        actual = exact_imq_witness_gradient(particles, targets, weights, self.c)
        torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-13)

    def test_spectral_mixture_approximates_imq_values_and_gradients(self):
        # Fixed independent test points; this checks distribution and amplitude.
        # It is not a guarantee for an adaptively optimized particle trajectory.
        omega = sample_imq_frequencies(3, 131072, self.c, seed=5, dtype=torch.float64)
        query = self.initial[:3]
        approximate_kernel = imq_features(query, omega, self.c) @ imq_features(
            self.targets[:3], omega, self.c).T
        exact_kernel = IMQKernel(c=self.c, beta=-.5)(query, self.targets[:3])
        torch.testing.assert_close(approximate_kernel, exact_kernel, rtol=0, atol=.006)
        cached = cache_target_mean(self.targets, self.weights, omega, self.c)
        approximate_gradient = rff_witness_gradient(self.initial, cached, omega, self.c,
                                                    eval_points=query)
        exact_gradient = exact_imq_witness_gradient(
            self.initial, self.targets, self.weights, self.c, eval_points=query)
        torch.testing.assert_close(approximate_gradient, exact_gradient, rtol=0, atol=.006)

    def test_validation_and_zero_step_behavior(self):
        for features in [0, 3]:
            with self.assertRaises(ValueError):
                sample_imq_frequencies(3, features, self.c)
        with self.assertRaises(ValueError):
            sample_imq_frequencies(3, 8, 0)
        with self.assertRaises(ValueError):
            cache_target_mean(self.targets, self.weights[:-1], self.omega, self.c)
        with self.assertRaises(ValueError):
            rff_euler_flow(self.initial, self.cached[:-1], self.omega, self.c, .3, 2)
        with self.assertRaises(ValueError):
            rff_witness_gradient(self.initial.float(), self.cached, self.omega, self.c)
        with self.assertRaises(ValueError):
            exact_imq_euler_flow(self.initial, self.targets, self.weights, self.c, .3, -1)
        actual = rff_euler_flow(self.initial, self.cached, self.omega, self.c, .3, 0)
        torch.testing.assert_close(actual, self.initial, rtol=0, atol=0)
        self.assertNotEqual(actual.data_ptr(), self.initial.data_ptr())


if __name__ == "__main__":
    unittest.main()
