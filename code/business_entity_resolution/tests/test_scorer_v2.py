"""Unit tests for rule scorer v2 selection and metrics. Run: python -m unittest -v"""
import unittest

import numpy as np
import pandas as pd

from src import scorer_v2 as sv
from src.metric import macro_f05


class TestScorerV2(unittest.TestCase):
    """Veto, one-to-one, t_keep and the fast metric."""

    def setUp(self):
        """Two S1 (10, 11); queries 1-3 true for 10, query 4 a decoy with a conflicting number."""
        self.df = pd.DataFrame({
            "query_id": [1, 2, 3, 4, 4], "s1_id": [10, 10, 10, 10, 11],
            "name_tsort": [1.0, 0.9, 0.95, 1.0, 0.5], "addr_tsort": [1.0, 0.9, np.nan, 0.95, 0.4],
            "num_compatible": [1, 1, 0, 0, 0], "num_conflict": [0, 0, 0, 1, 0],
            "extra_tokens_q": [0, 0, 0, 1, 0], "extra_tokens_s1": [0, 0, 0, 0, 0], "num_agree_major": [1, 1, np.nan, 0, np.nan],
            "label": [1, 1, 1, 0, 0]})
        self.p = {"w_name": 0.5, "w_addr": 0.4, "w_num": 0.1, "w_extra": 0.2, "veto": True}

    def test_veto_and_selection(self):
        """The conflicting decoy is vetoed; query 4 falls back to S1 11 but fails the threshold."""
        s = sv.score_v2(self.df, self.p)
        self.assertEqual(s[3], -1.0)
        q, s1 = self.df["query_id"].to_numpy(), self.df["s1_id"].to_numpy()
        from src.finalize import best_per_query
        keep = sv.select(q, s1, s, best_per_query(q, s, s1), 0.5, 0.5)
        self.assertEqual(keep.tolist(), [True, True, True, False, False])
        keep = sv.select(q, s1, s, best_per_query(q, s, s1), 0.4, 2.0)   # t_keep above every score -> all empty
        self.assertFalse(keep.any())

    def test_metrics_match_official(self):
        """metrics() macro F0.5 equals metric.macro_f05 (entities incl. a singleton)."""
        m = sv.metrics(np.array([0, 0, 1]), np.array([1.0, 0.0, 0.0]), np.array([2, 0]))
        ref = macro_f05({"a": ["t1", "f1"], "b": ["f2"]}, {"a": ["t1", "t2"], "b": []})
        self.assertAlmostEqual(m["macro_f05"], ref)
        self.assertEqual(m["f05_singletons"], 0.0)


if __name__ == "__main__":
    unittest.main()
