"""Algebra, replay and grouping checks for the cached population field."""

import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rff_imq import cache_target_mean, imq_features, rff_euler_flow, sample_imq_frequencies
from reference_field import (
    CUDAGraphReferenceSampler, apply_reference_flow, independent_batch_rff_flow,
    integrate_coefficient_blocks, record_reference_flow, reference_velocity,
)


class ReferenceFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(47)
        cls.initial = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        cls.targets = torch.randn(6, 3, generator=generator, dtype=torch.float64)
        cls.weights = torch.tensor([.3, -.4, .1, .2, -.15, .7], dtype=torch.float64)
        cls.c = 1.3
        cls.step = .2
        cls.omega = sample_imq_frequencies(3, 64, cls.c, seed=9, dtype=torch.float64)
        cls.target_mean = cache_target_mean(cls.targets, cls.weights, cls.omega, cls.c)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)

    def test_prescribed_velocity_matches_independent_feature_autograd(self):
        coefficient = self.target_mean - imq_features(self.initial, self.omega, self.c).mean(0)
        query = (self.initial[:3] + .31).requires_grad_(True)
        potential = (imq_features(query, self.omega, self.c) @ coefficient).sum()
        expected, = torch.autograd.grad(potential, query)
        actual = reference_velocity(query.detach(), coefficient, self.omega, self.c)
        self.assert_close(actual, expected)

    def test_record_matches_interacting_flow_and_preupdate_coefficients(self):
        expected = rff_euler_flow(self.initial, self.target_mean, self.omega,
                                  self.c, self.step, 7)
        actual, coefficients = record_reference_flow(
            self.initial, self.target_mean, self.omega, self.c, self.step, 7)
        self.assert_close(actual, expected)
        first = self.target_mean - imq_features(self.initial, self.omega, self.c).mean(0)
        self.assert_close(coefficients[0], first)
        after_one = rff_euler_flow(self.initial, self.target_mean, self.omega,
                                   self.c, self.step, 1)
        self.assert_close(coefficients[1], self.target_mean - imq_features(
            after_one, self.omega, self.c).mean(0))
        self.assertEqual(tuple(coefficients.shape), (7, 64))

    def test_guide_replay_recovers_endpoints_with_signed_nonunit_target(self):
        self.assertNotEqual(float(self.weights.sum()), 1.)
        self.assertLess(float(self.weights.min()), 0.)
        endpoints, coefficients = record_reference_flow(
            self.initial, self.target_mean, self.omega, self.c, self.step, 11)
        replay = apply_reference_flow(self.initial, coefficients, self.omega,
                                      self.c, self.step)
        self.assert_close(replay, endpoints)

    def test_batch_partition_and_permutation_do_not_change_map(self):
        _, coefficients = record_reference_flow(
            self.initial[:4], self.target_mean, self.omega, self.c, self.step, 11)
        fresh = self.initial + .16
        together = apply_reference_flow(fresh, coefficients, self.omega, self.c, self.step)
        for batch in [1, 3, 5]:
            chunks = [apply_reference_flow(fresh[start:start + batch], coefficients,
                                           self.omega, self.c, self.step)
                      for start in range(0, len(fresh), batch)]
            self.assert_close(torch.cat(chunks), together)
        permutation = torch.tensor([5, 1, 3, 7, 2, 0, 4, 6])
        permuted = apply_reference_flow(fresh[permutation], coefficients,
                                        self.omega, self.c, self.step)
        self.assert_close(permuted, together[permutation])

    def test_requests_do_not_mutate_cache_or_previous_outputs(self):
        initial_before = self.initial.clone()
        endpoints, coefficients = record_reference_flow(
            self.initial, self.target_mean, self.omega, self.c, self.step, 4)
        coefficients_before = coefficients.clone()
        first = apply_reference_flow(self.initial, coefficients, self.omega, self.c, self.step)
        first_before = first.clone()
        second = apply_reference_flow(self.initial + .4, coefficients,
                                       self.omega, self.c, self.step)
        torch.testing.assert_close(first, first_before, rtol=0, atol=0)
        torch.testing.assert_close(coefficients, coefficients_before, rtol=0, atol=0)
        torch.testing.assert_close(self.initial, initial_before, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(endpoints.requires_grad)

    def test_independent_group_updates_match_separate_interacting_flows(self):
        for batch in [1, 2, 4]:
            grouped = self.initial.reshape(-1, batch, 3)
            actual = independent_batch_rff_flow(
                grouped, self.target_mean, self.omega, self.c, self.step, 6)
            expected = torch.stack([rff_euler_flow(
                group, self.target_mean, self.omega, self.c, self.step, 6)
                for group in grouped])
            self.assert_close(actual, expected)

    def test_independent_singletons_have_no_self_repulsion(self):
        actual = independent_batch_rff_flow(self.initial[:, None, :], self.target_mean,
                                            self.omega, self.c, self.step, 1)[:, 0]
        expected = self.initial + self.step * reference_velocity(
            self.initial, self.target_mean, self.omega, self.c)
        self.assert_close(actual, expected)

    def test_coarse_blocks_use_sum_and_preserve_short_final_block(self):
        _, coefficients = record_reference_flow(
            self.initial, self.target_mean, self.omega, self.c, self.step, 7)
        blocks = integrate_coefficient_blocks(coefficients, stride=3)
        expected_blocks = torch.stack((coefficients[:3].sum(0),
                                       coefficients[3:6].sum(0), coefficients[6]))
        self.assert_close(blocks, expected_blocks)
        query = self.initial[:2] + .4
        expected = query.clone()
        for block in expected_blocks:
            expected += self.step * reference_velocity(expected, block, self.omega, self.c)
        actual = apply_reference_flow(query, coefficients, self.omega,
                                      self.c, self.step, stride=3)
        self.assert_close(actual, expected)
        self.assert_close(blocks.sum(0), coefficients.sum(0))

    def test_stride_one_is_original_sequence(self):
        _, coefficients = record_reference_flow(
            self.initial, self.target_mean, self.omega, self.c, self.step, 4)
        blocks = integrate_coefficient_blocks(coefficients, stride=1)
        self.assertIs(blocks, coefficients)
        self.assert_close(apply_reference_flow(self.initial, coefficients, self.omega,
                                               self.c, self.step, stride=1),
                          apply_reference_flow(self.initial, blocks, self.omega,
                                               self.c, self.step))

    def test_zero_steps_preserve_initial_and_empty_cache(self):
        final, coefficients = record_reference_flow(
            self.initial, self.target_mean, self.omega, self.c, self.step, 0)
        self.assertEqual(tuple(coefficients.shape), (0, 64))
        self.assert_close(final, self.initial)
        for stride in [1, 4]:
            actual = apply_reference_flow(self.initial, coefficients, self.omega,
                                          self.c, self.step, stride=stride)
            self.assert_close(actual, self.initial)
            self.assertNotEqual(actual.data_ptr(), self.initial.data_ptr())

    def test_invalid_inputs_fail_without_silent_coercion(self):
        coefficients = self.initial.new_zeros((2, 64))
        for stride in [0, -2, 1.5, True]:
            with self.assertRaises(ValueError):
                apply_reference_flow(self.initial, coefficients, self.omega,
                                     self.c, self.step, stride=stride)
        with self.assertRaises(ValueError):
            apply_reference_flow(self.initial, coefficients[:, :-1], self.omega,
                                 self.c, self.step)
        with self.assertRaises(ValueError):
            independent_batch_rff_flow(self.initial, self.target_mean, self.omega,
                                        self.c, self.step, 2)
        with self.assertRaises(ValueError):
            CUDAGraphReferenceSampler(self.initial, coefficients, self.omega,
                                      self.c, self.step)

    @unittest.skipUnless(os.environ.get("RFF_TEST_CUDA") == "1", "opt-in CUDA graph check")
    def test_cuda_graph_replays_new_inputs_and_preserves_previous_outputs(self):
        # The experiment host opts in; ordinary algebra checks need no GPU.
        initial, omega, target_mean = (x.float().cuda() for x in
                                       (self.initial, self.omega, self.target_mean))
        _, coefficients = record_reference_flow(initial, target_mean, omega,
                                                 self.c, self.step, 7)
        for stride in [1, 3]:
            for batch in [1, 4]:
                query = initial[:batch]
                sampler = CUDAGraphReferenceSampler(query, coefficients, omega,
                                                     self.c, self.step, stride=stride)
                first = sampler(query)
                first_before = first.clone()
                second = sampler(query + .4)
                expected_first = apply_reference_flow(query, coefficients, omega,
                                                      self.c, self.step, stride=stride)
                expected_second = apply_reference_flow(query + .4, coefficients, omega,
                                                       self.c, self.step, stride=stride)
                torch.testing.assert_close(first, expected_first, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(second, expected_second, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(first, first_before, rtol=0, atol=0)
                self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
