"""Focused tests for the analytic IMQ--Euler implementation."""

import unittest
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kernel_utils import IMQKernel
from mmd_flow_utils import witness_gradient

from imq_euler import (
    euler_flow,
    scaled_witness_gradient,
    target_self_inner_product,
)


class ImqEulerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.particles = torch.randn(9, 5)
        self.target = torch.randn(14, 5)
        self.weights = torch.randn(14)
        self.weights /= self.weights.sum()
        self.c = 1.9

    def test_analytic_gradient_matches_autograd_at_noisy_query(self):
        query = self.particles + 0.2 * torch.randn_like(self.particles)
        kernel = IMQKernel(c=self.c, beta=-0.5)
        automatic = (self.c**3) * witness_gradient(
            kernel,
            self.particles,
            self.target,
            query,
            target_weights=self.weights,
        )
        analytic = scaled_witness_gradient(
            self.particles,
            self.target,
            self.weights,
            self.c,
            eval_points=query,
        )
        relative_error = (
            (analytic - automatic).norm()
            / automatic.norm().clamp_min(1e-12)
        )
        self.assertLess(float(relative_error), 1e-5)

    def test_injected_noise_is_reproducible(self):
        target_self = target_self_inner_product(
            self.target, self.weights, self.c, chunk_size=5
        )
        kwargs = dict(
            initial_particles=self.particles,
            target=self.target,
            target_weights=self.weights,
            c=self.c,
            eta=0.1,
            n_iterations=4,
            target_self=target_self,
            noise_beta0=0.05,
            noise_alpha=0.25,
            log_every=2,
        )
        first, _ = euler_flow(seed=11, **kwargs)
        second, _ = euler_flow(seed=11, **kwargs)
        other_seed, _ = euler_flow(seed=12, **kwargs)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other_seed))

    def test_noise_is_not_added_to_state_when_step_is_zero(self):
        target_self = target_self_inner_product(
            self.target, self.weights, self.c, chunk_size=5
        )
        result, _ = euler_flow(
            self.particles,
            self.target,
            self.weights,
            c=self.c,
            eta=0.0,
            n_iterations=3,
            target_self=target_self,
            noise_beta0=0.2,
            noise_alpha=0.25,
            seed=4,
            log_every=1,
        )
        self.assertTrue(torch.equal(result, self.particles))

    def test_deterministic_small_step_reduces_objective(self):
        positive_weights = torch.rand(len(self.target))
        positive_weights /= positive_weights.sum()
        target_self = target_self_inner_product(
            self.target, positive_weights, self.c, chunk_size=5
        )
        _, info = euler_flow(
            self.particles,
            self.target,
            positive_weights,
            c=self.c,
            eta=0.02,
            n_iterations=8,
            target_self=target_self,
            log_every=4,
        )
        self.assertLess(info["mmd2_final"], info["mmd2_initial"])


if __name__ == "__main__":
    unittest.main()
