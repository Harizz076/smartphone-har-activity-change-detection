#!/usr/bin/env python3
"""Tests for subject-disjoint model-selection mechanics."""
from __future__ import annotations

import unittest

from analysis.optimize_six_class_glrt import group_folds


class SubjectFoldTests(unittest.TestCase):
    def test_folds_are_deterministic_disjoint_and_nonempty(self) -> None:
        groups = tuple(group for group in range(1, 10) for _ in range(6))
        labels = tuple(label for _ in range(1, 10) for label in ("a", "a", "b", "b", "c", "c"))
        first, sizes = group_folds(groups, labels, ("a", "b", "c"), folds=3)
        second, second_sizes = group_folds(groups, labels, ("a", "b", "c"), folds=3)
        self.assertEqual(first, second)
        self.assertEqual(sizes, second_sizes)
        self.assertEqual(set(first), set(groups))
        self.assertEqual(set(first.values()), {0, 1, 2})
        self.assertEqual(sum(sizes), len(groups))
        self.assertEqual(sizes, [18, 18, 18])
        for fold in range(3):
            fold_labels = {label for group, label in zip(groups, labels) if first[group] == fold}
            self.assertEqual(fold_labels, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
