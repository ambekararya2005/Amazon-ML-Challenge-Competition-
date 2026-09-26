"""Unit tests for the decoy-aware features. Run: python -m unittest -v"""
import unittest

import numpy as np
import pandas as pd

from src import decoy_features as dfe


class TestNumbers(unittest.TestCase):
    """Corruption-aware main-number relations."""

    def rel(self, a, b):
        """Return (equal, compatible, conflict) for two raw numbers."""
        return dfe.num_relation(dfe.main_number(a), dfe.main_number(b))

    def test_spec_cases(self):
        """2007~007 and 2007~02007 compatible; 2417/2412, 01317/1304, 37/32 conflict; missing = neither."""
        self.assertEqual(self.rel("2007", "007"), (False, True, False))
        self.assertEqual(self.rel("2007", "02007"), (True, True, False))
        self.assertEqual(self.rel("2417", "2412"), (False, False, True))
        self.assertEqual(self.rel("01317", "1304"), (False, False, True))
        self.assertEqual(self.rel("37", "32"), (False, False, True))
        self.assertEqual(self.rel("", "32"), (False, False, False))
        self.assertEqual(self.rel("", ""), (False, False, False))

    def test_main_number_is_first_and_zero_stripped(self):
        """The main number is the first number of the field, leading zeros removed."""
        self.assertEqual(dfe.main_number("0045 12"), "45")
        self.assertEqual(dfe.main_number("000"), "0")
        self.assertEqual(dfe.main_number(""), "")

    def test_number_features_arrays(self):
        """number_features returns aligned arrays with edit distance and log difference."""
        f = dfe.number_features(["2417 5", "37", "", "12 99"], ["2412", "37", "8", "7 99"])
        np.testing.assert_array_equal(f["num_conflict"], [1, 0, 0, 1])
        np.testing.assert_array_equal(f["num_equal"], [0, 1, 0, 0])
        np.testing.assert_array_equal(f["num_missing"], [0, 0, 1, 0])
        np.testing.assert_array_equal(f["any_shared_number"], [0, 1, 0, 1])
        self.assertEqual(f["num_edit"][0], 1)
        self.assertTrue(np.isnan(f["num_edit"][2]))


class TestNames(unittest.TestCase):
    """Extra-token detection."""

    def test_extra_tokens_fuzzy(self):
        """Typos still match (ratio >= 85); decoy words are extra on one side only."""
        f = dfe.extra_token_features(["cerballiance amicale holding", "chorale auto", "kj corporation education"],
                                     ["cerballiance amciale", "chorale auto international", "kj education corporation"])
        np.testing.assert_array_equal(f["extra_tokens_q"], [1, 0, 0])
        np.testing.assert_array_equal(f["extra_tokens_s1"], [0, 1, 0])


class TestPairAndContext(unittest.TestCase):
    """End-to-end feature table on a tiny candidate set."""

    def test_pair_and_context(self):
        """A decoy with a conflicting number disagrees with the S1's majority claimants."""
        qt = {"name_core": np.array(["trusted avalanche care", "trusted avalanche care", "trusted avalanche care central"], dtype=object),
              "name_compact": np.array(["trustedavalanchecare", "trustedavalanchecare", "trustedavalanchecarecentral"], dtype=object),
              "addr_clean": np.array(["1304 kane st, la crosse", "1304 kane street, la crosse", "1317 kane st, la crosse"], dtype=object),
              "numbers": np.array(["1304", "1304", "1317"], dtype=object)}
        st = {k: np.array([v] * 3, dtype=object) for k, v in
              {"name_core": "trusted avalanche care", "name_compact": "trustedavalanchecare",
               "addr_clean": "1304 kane street, city of la crosse", "numbers": "1304"}.items()}
        f = dfe.pair_features(qt, st)
        np.testing.assert_array_equal(f["num_conflict"], [0, 0, 1])
        np.testing.assert_array_equal(f["extra_tokens_q"], [0, 0, 1])
        self.assertGreater(f["name_tset"][2], f["name_tsort"][2])   # token_set hides the extra word
        df = pd.DataFrame({"query_id": [1, 2, 3], "s1_id": [7, 7, 7], **f})
        main_q = np.array([dfe.main_number(x) for x in qt["numbers"]], dtype=object)
        df = dfe.context_features(df, main_q)
        self.assertEqual(df["s1_claims"].tolist(), [3.0, 3.0, 3.0])
        self.assertEqual(df["num_agree_major"].tolist()[2], 0.0)
        self.assertEqual(df["num_agree_major"].tolist()[0], 1.0)
        self.assertEqual(df["extra_vs_min"].tolist(), [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
