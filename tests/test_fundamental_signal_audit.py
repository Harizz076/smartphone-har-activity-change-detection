#!/usr/bin/env python3
"""Tests for the mathematical core of the raw-signal audit."""
from __future__ import annotations

import math
import unittest

import numpy as np

from analysis.fundamental_signal_audit import mean_shift_glr
from glrt.continuous_transition_glrt import scan_glrt


class FixedBoundaryGlrtTests(unittest.TestCase):
    def test_matches_gaussian_unknown_variance_likelihood_ratio(self) -> None:
        left = [0.0, 1.0]
        right = [2.0, 4.0]
        overall = sum(left + right) / 4
        sse0 = sum((value - overall) ** 2 for value in left + right)
        sse1 = sum((value - 0.5) ** 2 for value in left)
        sse1 += sum((value - 3.0) ** 2 for value in right)
        expected = 4 * math.log(sse0 / sse1)
        score, effect = mean_shift_glr(left, right)
        self.assertAlmostEqual(score, expected, places=12)
        self.assertGreater(effect, 0.0)

    def test_no_mean_change_has_zero_score(self) -> None:
        score, effect = mean_shift_glr([0.0, 2.0], [0.0, 2.0])
        self.assertAlmostEqual(score, 0.0, places=12)
        self.assertAlmostEqual(effect, 0.0, places=12)

class ContinuousScanTests(unittest.TestCase):
    def test_common_variance_scan_localizes_mean_change(self) -> None:
        rng = np.random.default_rng(7)
        signal = np.concatenate((rng.normal(0, 0.2, (100, 2)),
                                 rng.normal(2, 0.2, (100, 2))))
        result = scan_glrt(signal, horizon=200, minimum_segment=20,
                           hop=1, candidate_step=1, variance_model="common")
        self.assertEqual(len(result.scores), 1)
        self.assertLessEqual(abs(int(result.change_samples[0]) - 100), 2)
        self.assertGreater(float(result.scores[0]), 100.0)

    def test_separate_variance_scan_detects_variance_change(self) -> None:
        rng = np.random.default_rng(11)
        signal = np.concatenate((rng.normal(0, 0.1, (100, 1)),
                                 rng.normal(0, 1.0, (100, 1))))
        common = scan_glrt(signal, horizon=200, minimum_segment=20,
                           variance_model="common")
        separate = scan_glrt(signal, horizon=200, minimum_segment=20,
                             variance_model="separate")
        self.assertGreater(float(separate.scores[0]), float(common.scores[0]))
        self.assertLessEqual(abs(int(separate.change_samples[0]) - 100), 5)

    def test_constant_signal_has_zero_score(self) -> None:
        result = scan_glrt(np.zeros((80, 3)), horizon=60, minimum_segment=10,
                           hop=10, candidate_step=2)
        np.testing.assert_allclose(result.scores, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
