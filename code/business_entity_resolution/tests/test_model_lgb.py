"""Unit tests for the model-stage helpers (isotonic, target encoding). Run: python -m unittest -v"""
import unittest

import numpy as np
import pyarrow as pa

from src import model_lgb as ml


class TestIsotonic(unittest.TestCase):
    """PAV calibration."""

    def test_pav_monotone(self):
        """Violations are pooled with weights."""
        np.testing.assert_allclose(ml.pav(np.array([0.1, 0.5, 0.3, 0.9]), np.array([1, 1, 3, 1.0])),
                                   [0.1, 0.35, 0.35, 0.9])

    def test_fit_apply(self):
        """Calibration of a miscalibrated score is monotone and close to the true rate."""
        rng = np.random.default_rng(42)
        p = rng.uniform(size=50_000)
        y = (rng.uniform(size=p.size) < p ** 2).astype(float)
        cal = ml.fit_isotonic(p, y, n_bins=100)
        q = ml.apply_isotonic(cal, np.array([0.2, 0.5, 0.9]))
        self.assertTrue(np.all(np.diff(cal["y"]) >= 0))
        np.testing.assert_allclose(q, [0.04, 0.25, 0.81], atol=0.05)


class TestTargetEncoding(unittest.TestCase):
    """Extra-word target encoding."""

    def test_fit_apply(self):
        """A decoy-only word scores high, unseen words fall back to the prior and are counted."""
        xs = ["holdings", "holdings", "", "center", "center group"]
        label = np.array([0, 0, 1, 1, 0])
        v = ml.te_fit(xs, label, np.arange(5))
        f = ml.te_apply(pa.array(["holdings", "", "zzz center"]), v, v.attrs["prior"], "q", 3)
        self.assertGreater(f["te_q_max"][0], f["te_q_mean"][2] - 1)          # defined
        self.assertTrue(np.isnan(f["te_q_max"][1]))
        np.testing.assert_array_equal(f["te_q_unseen"], [0, 0, 1])
        score = dict(zip(v["token"], v["score"]))
        self.assertGreater(score["holdings"], score["center"])


if __name__ == "__main__":
    unittest.main()
