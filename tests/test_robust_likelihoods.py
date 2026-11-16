#!/usr/bin/env python3
"""Mathematical tests for Student-t and Laplace likelihood models."""
from __future__ import annotations

import math
import unittest

from glrt.robust_likelihoods import (
    LaplaceClassModel,
    StudentTClassModel,
    fit_laplace_models,
    fit_student_models,
    predict_robust,
)


class DensityTests(unittest.TestCase):
    def test_univariate_student_t_density_at_center(self) -> None:
        model = StudentTClassModel("a", 5, (0.0,), ((1.0,),), 0.0, 5.0, 1, True)
        expected = math.lgamma(3.0) - math.lgamma(2.5) - 0.5 * math.log(5 * math.pi)
        self.assertAlmostEqual(model.log_likelihood((0.0,)), expected, places=12)

    def test_independent_laplace_density(self) -> None:
        model = LaplaceClassModel("a", 5, (0.0, 1.0), (2.0, 0.5))
        expected = -math.log(4.0) - 0.5 - math.log(1.0) - 2.0
        self.assertAlmostEqual(model.log_likelihood((1.0, 2.0)), expected, places=12)


class RobustClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = ("a", "b")
        self.vectors = [(-2.2, -1.8), (-1.8, -2.2), (-2.0, -2.1), (-1.9, -1.8),
                        (2.2, 1.8), (1.8, 2.2), (2.0, 2.1), (1.9, 1.8)]
        self.actual = ["a"] * 4 + ["b"] * 4

    def test_student_t_classifies_separated_centers(self) -> None:
        standardizer, models = fit_student_models(
            self.vectors, self.actual, self.labels, 5.0, 0.2, 1e-6, max_iterations=15
        )
        self.assertEqual(predict_robust((-2.0, -2.0), standardizer, models).label, "a")
        self.assertEqual(predict_robust((2.0, 2.0), standardizer, models).label, "b")
        self.assertTrue(all(model.em_iterations <= 15 for model in models))

    def test_laplace_classifies_separated_centers(self) -> None:
        standardizer, models = fit_laplace_models(
            self.vectors, self.actual, self.labels, 0.2, 1e-6
        )
        self.assertEqual(predict_robust((-2.0, -2.0), standardizer, models).label, "a")
        self.assertEqual(predict_robust((2.0, 2.0), standardizer, models).label, "b")


if __name__ == "__main__":
    unittest.main()
