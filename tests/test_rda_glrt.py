import unittest

from glrt.rda_glrt import fit_rda_models
from glrt.six_class_glrt import fit_models, predict


class RdaGlrtTests(unittest.TestCase):
    def setUp(self):
        self.vectors = [
            (-2.0, -1.0), (-1.7, -1.2), (-2.2, -0.8),
            (2.0, 1.0), (1.7, 1.2), (2.2, 0.8),
        ]
        self.labels = ["left"] * 3 + ["right"] * 3
        self.classes = ("left", "right")

    def assert_models_close(self, left, right):
        for model_left, model_right in zip(left, right):
            self.assertEqual(model_left.label, model_right.label)
            self.assertAlmostEqual(model_left.log_determinant,
                                   model_right.log_determinant, places=10)
            for row_left, row_right in zip(model_left.cholesky,
                                           model_right.cholesky):
                for value_left, value_right in zip(row_left, row_right):
                    self.assertAlmostEqual(value_left, value_right, places=10)

    def test_lambda_zero_matches_class_specific_gaussian(self):
        standard, models = fit_models(
            self.vectors, self.labels, self.classes, 0.0, 1e-6,
            "class_full",
        )
        rda_standard, rda_models = fit_rda_models(
            self.vectors, self.labels, self.classes, 0.0, 0.0, 1e-6,
        )
        self.assertEqual(standard, rda_standard)
        self.assert_models_close(models, rda_models)

    def test_lambda_one_matches_shared_gaussian(self):
        standard, models = fit_models(
            self.vectors, self.labels, self.classes, 0.0, 1e-6,
            "shared_full",
        )
        rda_standard, rda_models = fit_rda_models(
            self.vectors, self.labels, self.classes, 1.0, 0.0, 1e-6,
        )
        self.assertEqual(standard, rda_standard)
        self.assert_models_close(models, rda_models)

    def test_rda_classifies_separated_centers(self):
        standard, models = fit_rda_models(
            self.vectors, self.labels, self.classes, 0.5, 0.25, 1e-6,
        )
        self.assertEqual(predict((-2.0, -1.0), standard, models).label,
                         "left")
        self.assertEqual(predict((2.0, 1.0), standard, models).label,
                         "right")

    def test_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            fit_rda_models(self.vectors, self.labels, self.classes,
                           -0.1, 0.0)
        with self.assertRaises(ValueError):
            fit_rda_models(self.vectors, self.labels, self.classes,
                           0.0, 1.1)


if __name__ == "__main__":
    unittest.main()
