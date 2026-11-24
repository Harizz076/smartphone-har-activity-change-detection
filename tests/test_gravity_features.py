import math
import unittest

from glrt.gravity_features import GRAVITY_FEATURE_NAMES, extract_gravity_features
from glrt.six_class_glrt import FEATURE_NAMES


class GravityFeatureTests(unittest.TestCase):
    def test_features_are_finite(self):
        samples = tuple((0.1 * math.sin(index / 3),
                         0.2 * math.cos(index / 4), 1.0)
                        for index in range(128))
        features = extract_gravity_features(samples, 50.0)
        self.assertEqual(len(features), len(FEATURE_NAMES) + len(GRAVITY_FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in features))

    def test_rotation_about_gravity_preserves_aligned_features(self):
        samples = tuple((math.sin(index / 5), math.cos(index / 7), 9.81)
                        for index in range(100))
        rotated = tuple((-y, x, z) for x, y, z in samples)
        original = extract_gravity_features(samples, 20.0)[len(FEATURE_NAMES):]
        transformed = extract_gravity_features(rotated, 20.0)[len(FEATURE_NAMES):]
        for left, right in zip(original, transformed):
            self.assertAlmostEqual(left, right, places=10)


if __name__ == "__main__":
    unittest.main()
