#!/usr/bin/env python3
"""Deterministic unit tests for the six-class GLRT mathematical core."""
from __future__ import annotations

import math
import unittest

from glrt.six_class_glrt import (
    GaussianClassModel,
    Standardizer,
    cholesky,
    extract_features,
    fit_models,
    forward_solve,
    predict,
)


class LinearAlgebraTests(unittest.TestCase):
    def test_cholesky_reconstructs_known_matrix(self) -> None:
        matrix = ((4.0, 2.0), (2.0, 3.0))
        lower = cholesky(matrix)
        reconstructed = [[sum(lower[row][k] * lower[col][k] for k in range(2))
                          for col in range(2)] for row in range(2)]
        for actual_row, expected_row in zip(reconstructed, matrix):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=12)

    def test_forward_solve(self) -> None:
        lower = ((2.0, 0.0), (1.0, 3.0))
        solution = forward_solve(lower, (4.0, 11.0))
        self.assertAlmostEqual(solution[0], 2.0)
        self.assertAlmostEqual(solution[1], 3.0)

    def test_one_dimensional_log_likelihood(self) -> None:
        model = GaussianClassModel("class", 10, (0.0,), ((2.0,),), math.log(4.0))
        expected = -0.5 * (math.log(2 * math.pi) + math.log(4.0) + 1.0)
        self.assertAlmostEqual(model.log_likelihood((2.0,)), expected, places=12)


class FeatureTests(unittest.TestCase):
    def test_features_are_finite_and_detect_frequency(self) -> None:
        sample_rate, frequency, count = 20.0, 1.0, 200
        samples = tuple((math.sin(2 * math.pi * frequency * index / sample_rate), 0.5, 9.81)
                        for index in range(count))
        vector = extract_features(samples, sample_rate)
        self.assertEqual(len(vector), 15)
        self.assertTrue(all(math.isfinite(value) for value in vector))
        self.assertAlmostEqual(vector[12], frequency, places=12)


class ClassifierTests(unittest.TestCase):
    def test_margin_is_twice_winner_runner_up_log_ratio(self) -> None:
        standardizer = Standardizer((0.0,), (1.0,))
        models = (
            GaussianClassModel("a", 10, (0.0,), ((1.0,),), 0.0),
            GaussianClassModel("b", 10, (2.0,), ((1.0,),), 0.0),
        )
        result = predict((0.5,), standardizer, models)
        self.assertEqual(result.label, "a")
        self.assertAlmostEqual(result.glr_margin, 2.0, places=12)

    def test_six_separated_classes_predict_their_centers(self) -> None:
        labels = tuple(f"class_{index}" for index in range(6))
        vectors = []
        actual = []
        for index, label in enumerate(labels):
            for offset in (-0.15, -0.05, 0.05, 0.15):
                vectors.append((index * 3.0 + offset, index * -2.0 + offset * offset))
                actual.append(label)
        standardizer, models = fit_models(vectors, actual, labels, shrinkage=0.2, ridge=1e-6)
        for index, label in enumerate(labels):
            result = predict((index * 3.0, index * -2.0), standardizer, models)
            self.assertEqual(result.label, label)
            self.assertGreaterEqual(result.glr_margin, 0.0)

    def test_all_covariance_modes_classify_separated_classes(self) -> None:
        labels = ("a", "b")
        vectors = [(-2.1, -1.9), (-1.9, -2.1), (-2.0, -2.2),
                   (2.1, 1.9), (1.9, 2.1), (2.0, 2.2)]
        actual = ["a", "a", "a", "b", "b", "b"]
        for mode in ("class_full", "class_diagonal", "shared_full", "shared_diagonal"):
            with self.subTest(mode=mode):
                standardizer, models = fit_models(vectors, actual, labels, 0.2, 1e-6, mode)
                self.assertEqual(predict((-2.0, -2.0), standardizer, models).label, "a")
                self.assertEqual(predict((2.0, 2.0), standardizer, models).label, "b")

    def test_fit_rejects_missing_class(self) -> None:
        with self.assertRaises(ValueError):
            fit_models([(0.0,), (1.0,)], ["a", "a"], ("a", "b"), 0.1, 1e-6)


if __name__ == "__main__":
    unittest.main()
