import importlib.util
from pathlib import Path
import unittest

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT / 'experiment/compare.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_exact_transcript_groups_ignore_case_and_punctuation(self):
        self.assertEqual(comparison.normalize('Oh, wonderful!'), comparison.normalize('oh wonderful'))

    def test_size_matching_preserves_class_counts_and_subset(self):
        labels = np.tile([0, 1], 50)
        baseline = np.array([0]*15 + [1]*20)
        candidates = np.arange(10, 100)
        selected = comparison.matched_subset(candidates, labels, baseline, 42)
        self.assertEqual(len(selected), len(baseline))
        self.assertEqual(len(set(selected)), len(selected))
        self.assertTrue(set(selected).issubset(candidates))
        np.testing.assert_array_equal(np.bincount(labels[selected]), np.bincount(baseline))

    def test_identical_groups_never_cross_folds(self):
        groups = np.repeat(np.arange(30), 2)
        labels = np.repeat(np.tile([0, 1], 15), 2)
        for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(groups, labels, groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))


if __name__ == '__main__':
    unittest.main()
