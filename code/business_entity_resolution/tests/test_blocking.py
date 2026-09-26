"""Unit tests for the blocking passes on tiny synthetic data. Run: python -m unittest -v"""
import unittest

import numpy as np

from src import blocking as bl


class TestPassA(unittest.TestCase):
    """Address-number key blocking."""

    def setUp(self):
        """Build a small S1 index: two Robyn Road records, one Main Street, plus a big generic block."""
        self.s1_nk = ["007", "007", "129", "005"] + ["001"] * 60
        self.s1_tok = ["robyn road springdale ar", "robyn road fayetteville ar", "main street ottawa oh",
                       "elm street little rock ar"] + ["common road street town zz"] * 60
        self.s1_ids = np.arange(len(self.s1_nk), dtype=np.int32) + 100
        self.df = bl.token_df(self.s1_tok)
        self.keys = bl.build_s1_keys(self.s1_nk, self.s1_tok, self.s1_ids, self.df)

    def test_rarest_tokens_and_query_skips_unseen(self):
        """Keys use the 2 rarest S1 tokens; query tokens unseen in S1 are skipped."""
        keys, owners = bl.record_keys(["007"], ["robyn road springdale ar unseenword"], self.df, query=True)
        self.assertEqual(sorted(keys), ["007|robyn", "007|springdale"])
        self.assertEqual(owners.tolist(), [0, 0])

    def test_large_blocks_dropped(self):
        """A key shared by more than A_MAX_BLOCK S1 records is dropped."""
        big = bl.hash_keys(["001|common", "001|street"])
        self.assertFalse(self.keys["key"].isin(big).any())

    def test_ranking_by_shared_keys(self):
        """The S1 record sharing more keys ranks first; the number-mismatched record is absent."""
        res = bl.pass_a_chunk(np.array([7]), ["007"], ["robyn road springdale ar"], self.keys, self.df)
        self.assertEqual(res["s1_id"].tolist()[0], 100)
        self.assertEqual(res["rank"].tolist()[0], 1)
        self.assertGreater(res["score"].iloc[0], res["score"].iloc[-1] - 1e-9)
        self.assertNotIn(102, res["s1_id"].tolist())
        self.assertEqual(list(res.columns), ["query_id", "s1_id", "pass", "score", "rank"])

    def test_no_numbers_no_candidates(self):
        """A query without num_keys yields no pass-A candidates."""
        res = bl.pass_a_chunk(np.array([7]), [""], ["robyn road"], self.keys, self.df)
        self.assertEqual(len(res), 0)


class TestPassC(unittest.TestCase):
    """Char TF-IDF top-k search."""

    def test_true_match_ranks_first(self):
        """A noisy copy of an S1 record finds that record at rank 1 with ids mapped back."""
        s1 = ["springdalecity springdale city 2007 robyn road, springdale, ar",
              "cardiologysafecare cardiology safe care 617 firehouse road, floyd county, va",
              "brownwarfield brown warfield 390 plainfield road, griswold, ct"] * 2
        vec, b = bl.fit_c_index(s1)
        s1_ids = np.array([10, 11, 12, 13, 14, 15])
        res = bl.pass_c_chunk(vec, b, np.array([500, 501]),
                              ["springdalecity springdale city 007 robyn road, springdale, ar",
                               "cardiologysafecare cardiology 5afe care 617 firehouse road, willis, va"], s1_ids)
        top = res[res["rank"] == 1].set_index("query_id")["s1_id"]
        self.assertIn(top[500], (10, 13))
        self.assertIn(top[501], (11, 14))
        self.assertLessEqual(res.groupby("query_id").size().max(), bl.C_TOP_N)
        self.assertTrue((res["score"] > 0).all())


class TestUnion(unittest.TestCase):
    """Outer join of the two passes."""

    def test_union(self):
        """Pairs from either pass appear once with flags and per-pass scores."""
        a = bl._cand_frame([1, 1], [5, 6], [2.0, 1.0], [1, 2], "A")
        c = bl._cand_frame([1, 2], [6, 9], [0.8, 0.4], [1, 1], "C")
        u = bl.union_frame(a, c)
        self.assertEqual(list(u.columns), ["query_id", "s1_id", "scoreA", "scoreC", "in_A", "in_C"])
        self.assertEqual(len(u), 3)
        row = u[(u.query_id == 1) & (u.s1_id == 6)].iloc[0]
        self.assertTrue(row.in_A and row.in_C)
        self.assertTrue(np.isnan(u[(u.query_id == 2)].iloc[0].scoreA))

    def test_pair_keys_roundtrip(self):
        """Pair keys decode back to (query_id, s1_id)."""
        k = bl.pair_keys(np.array([3 * bl.QUERY_ID_MULT + 5_285_602]), np.array([2_206_820]))
        self.assertEqual(int(k[0] >> bl.PAIR_KEY_SHIFT), 3 * bl.QUERY_ID_MULT + 5_285_602)
        self.assertEqual(int(k[0] & ((1 << bl.PAIR_KEY_SHIFT) - 1)), 2_206_820)


if __name__ == "__main__":
    unittest.main()
