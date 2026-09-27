"""Unit tests for blocking v4 helpers. Run: python -m unittest -v"""
import logging
import unittest

import numpy as np
import pandas as pd

from src import blocking_v4 as b4


class TestDictionary(unittest.TestCase):
    """Cross-script token alignment and mining."""

    def test_align(self):
        """Names align leftovers by position; address components by best ratio with equal token count."""
        self.assertEqual(b4.align_names("snraij enterprises", "sunrise enterprises"), [("snraij", "sunrise")])
        self.assertEqual(b4.align_names("a b c", "a x"), [])
        self.assertEqual(b4.align_addresses("12 x, mdhy prdes", "12 x, madhya pradesh"),
                         [("mdhy", "madhya"), ("prdes", "pradesh")])

    def test_mine_thresholds(self):
        """A pair needs support >= 3 and precision >= 0.6; only one side may be non-Latin."""
        n = 4
        q = {"business_name": ["\u0938 x"] * n + ["plain"], "business_address": [""] * (n + 1),
             "name_core": ["phuds raj"] * n + ["phuds raj"], "addr_clean": [""] * (n + 1)}
        s = {"business_name": ["foods raj"] * (n + 1), "business_address": [""] * (n + 1),
             "name_core": ["foods raj"] * (n + 1), "addr_clean": [""] * (n + 1)}
        d = b4.mine_dictionary(q, s, logging.getLogger("t"))
        self.assertEqual(d, {"phuds": "foods"})
        self.assertEqual(b4.map_text("raj phuds, dilli", d), "raj foods, dilli")


class TestTopK(unittest.TestCase):
    """Adaptive top-k keeps 8 when the 1st-5th margin is small; pass D is always kept."""

    def test_adaptive(self):
        """Query 1 has a flat score list (top-8), query 2 a steep one (top-5) plus a pass-D pair."""
        rows = []
        for i in range(10):
            rows.append((1, i, 0.0, 0.50 - 0.001 * i, 0.0))
            rows.append((2, 100 + i, 0.0, 0.9 - 0.1 * i, 0.3 if i == 9 else 0.0))
        u = pd.DataFrame(rows, columns=["query_id", "s1_id", "scoreA", "scoreC", "scoreD"]).astype(
            {"scoreA": np.float32, "scoreC": np.float32, "scoreD": np.float32})
        out = b4.adaptive_top_k(u, 0.05)
        self.assertEqual(int((out["query_id"] == 1).sum()), 8)
        self.assertEqual(int((out["query_id"] == 2).sum()), 6)
        self.assertEqual(int((b4.adaptive_top_k(u, 0.0)["query_id"] == 1).sum()), 5)


if __name__ == "__main__":
    unittest.main()
