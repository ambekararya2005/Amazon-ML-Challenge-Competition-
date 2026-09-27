"""Unit tests for the v3 group helpers. Run: python -m unittest -v"""
import unittest

import numpy as np

from src import features_v3 as f3


class TestGroupHelpers(unittest.TestCase):
    """Within-group ranks and top-2."""

    def test_group_rank(self):
        """Rank 1 = best score within each key; ties broken by row order."""
        key = np.array([1, 1, 2, 1, 2])
        score = np.array([0.5, 0.9, 0.1, 0.5, 0.3])
        np.testing.assert_array_equal(f3.group_rank(key, score), [2, 1, 2, 3, 1])

    def test_group_top2(self):
        """Group max and second max broadcast to rows; singletons get -inf."""
        mx, sec = f3.group_top2(np.array([1, 1, 2]), np.array([0.2, 0.7, 0.4]))
        np.testing.assert_allclose(mx, [0.7, 0.7, 0.4])
        self.assertEqual(sec[0], 0.2)
        self.assertTrue(np.isneginf(sec[2]))


if __name__ == "__main__":
    unittest.main()
