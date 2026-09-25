"""Unit tests for src/metric.py. Run from code/business_entity_resolution/: python -m unittest -v"""
import unittest

from src.metric import f05_breakdown, f05_entity, macro_f05


class TestF05Entity(unittest.TestCase):
    """Per-entity F0.5 against the problem statement and the singleton rules."""

    def test_problem_statement_example(self):
        """README example: 3 predicted, 2 true, both found -> 0.714."""
        score = f05_entity(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
        self.assertEqual(round(score, 3), 0.714)

    def test_singleton_empty_is_one(self):
        """A true singleton with an empty prediction scores 1.0."""
        self.assertEqual(f05_entity([], []), 1.0)

    def test_singleton_any_prediction_is_zero(self):
        """A true singleton with any prediction scores 0.0."""
        self.assertEqual(f05_entity(["S2-1"], []), 0.0)
        self.assertEqual(f05_entity(["S2-1", "S3-2"], []), 0.0)

    def test_non_singleton_empty_is_zero(self):
        """An entity with true matches and an empty prediction scores 0.0."""
        self.assertEqual(f05_entity([], ["S2-1", "S3-2"]), 0.0)

    def test_plan_table_values(self):
        """Values from the plan's n=4 table: 3 correct 0.938, 4 correct + 1 wrong 0.833."""
        true = ["a", "b", "c", "d"]
        self.assertAlmostEqual(f05_entity(["a", "b", "c"], true), 0.9375)
        self.assertAlmostEqual(f05_entity(["a", "b", "c", "d", "x"], true), 0.8333, places=4)

    def test_duplicates_in_prediction_are_ignored(self):
        """Predicted ids are treated as a set."""
        self.assertEqual(f05_entity(["a", "a"], ["a"]), 1.0)


class TestMacroF05(unittest.TestCase):
    """Macro averaging and missing predictions."""

    def test_missing_prediction_counts_as_empty(self):
        """Entities absent from pred_map are scored as empty predictions."""
        true_map = {"S1-1": ["S2-1"], "S1-2": []}
        self.assertEqual(macro_f05({}, true_map), 0.5)  # 0.0 + 1.0

    def test_macro_average(self):
        """Macro F0.5 is the plain mean over true_map; extra pred_map keys are ignored."""
        true_map = {"S1-1": ["S2-00047", "S3-00812"], "S1-2": [], "S1-3": ["S2-9"]}
        pred_map = {"S1-1": ["S2-00047", "S2-00193", "S3-00812"], "S1-2": ["S2-5"],
                    "S1-3": ["S2-9"], "S1-99": ["S2-7"]}
        expected = (1.25 * 2 / (3 + 0.5) + 0.0 + 1.0) / 3
        self.assertAlmostEqual(macro_f05(pred_map, true_map), expected)


class TestBreakdown(unittest.TestCase):
    """Per-country breakdown with an open set of country labels."""

    def test_breakdown(self):
        """Overall and per-country numbers, precision/recall and empty-prediction rates."""
        true_map = {"a": ["S2-1"], "b": [], "c": ["S2-2", "S3-3"], "d": []}
        pred_map = {"a": ["S2-1"], "b": ["S2-9"], "c": ["S2-2"]}
        country = {"a": "US", "b": "US", "c": "France", "d": "France"}
        out = f05_breakdown(pred_map, true_map, country)
        self.assertAlmostEqual(out["overall"]["macro_f05"], macro_f05(pred_map, true_map))
        self.assertEqual(set(out["by_country"]), {"US", "France"})
        self.assertAlmostEqual(out["by_country"]["US"]["macro_f05"], 0.5)
        self.assertAlmostEqual(out["by_country"]["France"]["macro_f05"], (1.25 / 1.5 + 1.0) / 2)
        self.assertAlmostEqual(out["overall"]["mean_recall"], (1 + 1 + 0.5 + 1) / 4)
        self.assertAlmostEqual(out["overall"]["mean_precision"], (1 + 0 + 1 + 1) / 4)
        self.assertAlmostEqual(out["overall"]["empty_pred_rate_singletons"], 0.5)
        self.assertAlmostEqual(out["overall"]["empty_pred_rate_non_singletons"], 0.0)

    def test_missing_country_raises(self):
        """Every evaluated entity must have a country label."""
        with self.assertRaises(ValueError):
            f05_breakdown({}, {"a": []}, {})


if __name__ == "__main__":
    unittest.main()
