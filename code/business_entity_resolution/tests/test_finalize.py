"""Unit tests for the stage-1 reduction and baseline scorer helpers. Run: python -m unittest -v"""
import unittest

import numpy as np
import pandas as pd

from src import finalize as fz
from src.metric import macro_f05


class TestFastMetric(unittest.TestCase):
    """macro_f05_fast must equal the official metric."""

    def test_matches_official_metric(self):
        """Random predictions over entities incl. singletons give the same score as metric.macro_f05."""
        rng = np.random.default_rng(42)
        n_ent = 300
        n_true = rng.integers(0, 5, n_ent)
        true_map = {e: [f"T{e}_{j}" for j in range(n_true[e])] for e in range(n_ent)}
        ent, is_true, pred_map = [], [], {}
        for e in range(n_ent):
            k = rng.integers(0, 4)
            picks = [f"T{e}_{j}" if (j < n_true[e] and rng.random() < 0.6) else f"F{e}_{j}" for j in range(k)]
            pred_map[e] = picks
            ent += [e] * len(picks)
            is_true += [p.startswith("T") for p in picks]
        fast = fz.macro_f05_fast(np.asarray(ent, dtype=np.int64), np.asarray(is_true, dtype=float), n_true)
        self.assertAlmostEqual(fast, macro_f05(pred_map, true_map), places=12)


class TestSelection(unittest.TestCase):
    """Top-k reduction and one-to-one selection."""

    def test_top_k_and_best_per_query(self):
        """top_k keeps the k best by cheap score per query; best_per_query keeps one row per query."""
        df = pd.DataFrame({"query_id": np.array([1, 1, 1, 2, 2], dtype=np.int64),
                           "s1_id": np.array([10, 11, 12, 10, 13], dtype=np.int32),
                           "scoreA": np.array([0, 2, 0, 1, 0], dtype=np.float32),
                           "scoreC": np.array([0.9, 0.1, 0.5, 0.0, 0.3], dtype=np.float32)})
        red = fz.top_k(df, w_a=0.5, k=2)
        self.assertEqual(sorted(zip(red["query_id"], red["s1_id"])), [(1, 10), (1, 11), (2, 10), (2, 13)])
        red = red.sort_values(["query_id", "s1_id"]).reset_index(drop=True)
        mask = fz.best_per_query(red["query_id"].to_numpy(), np.array([0.2, 0.9, 0.5, 0.5]), red["s1_id"].to_numpy())
        self.assertEqual(mask.tolist(), [False, True, True, False])   # tie on query 2 -> lower s1_id

    def test_num_match_and_grid(self):
        """num_match is shared / min size (0 when empty); the weight grid covers the simplex."""
        a = np.array(["570 013", "", "12", "7 8 9"], dtype=object)
        b = np.array(["013", "5", "13", "9 8"], dtype=object)
        np.testing.assert_allclose(fz.num_match(a, b), [1.0, 0.0, 0.0, 1.0])
        grid = fz.weight_grid(0.1)
        self.assertEqual(len(grid), 55)
        self.assertTrue(all(abs(sum(w) - 1) < 1e-9 and w[0] > 0 for w in grid))


if __name__ == "__main__":
    unittest.main()
