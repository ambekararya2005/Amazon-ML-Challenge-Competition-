"""Unit tests for the set decoders and the error budget. Run: python -m unittest -v"""
import itertools
import math
import unittest

import numpy as np

from src import decoder as dec


def brute_expected(p, k, h=None, lam=0.0):
    """Return E[F0.5] of predicting the top-k of sorted p by enumeration (Poisson missed count up to 4)."""
    q = np.minimum(np.asarray(p) / h, 1.0) if h is not None else np.asarray(p)
    pm = np.array([math.exp(-lam) * lam ** m / math.factorial(m) for m in range(dec.M_MAX + 1)])
    pm /= pm.sum()
    tot = 0.0
    for z in itertools.product([0, 1], repeat=len(q)):
        pz = np.prod([qi if zi else 1 - qi for qi, zi in zip(q, z)])
        for m, pmm in enumerate(pm):
            tp, n = sum(z[:k]), sum(z) + m
            f = (1.0 if n == 0 else 0.0) if k == 0 else (0.0 if tp == 0 else 1.25 * tp / (k + 0.25 * n))
            tot += pz * pmm * f
    return tot if h is None else (1 - h) * (k == 0) + h * tot


class TestExpectedF05(unittest.TestCase):
    """Exact expected F0.5 against enumeration."""

    def test_matches_brute_force(self):
        """Every prefix, with and without the has-match mixture and missed matches."""
        rng = np.random.default_rng(42)
        p = np.sort(rng.uniform(0.05, 0.95, size=5))[::-1]
        P = np.zeros((1, dec.K_MAX))
        P[0, :5] = p
        for h, miss in ((None, 0.0), (0.8, 0.0), (None, 0.1), (0.9, 0.05)):
            hh = None if h is None else np.array([h])
            E = dec.expected_f05_table(P, hh, miss)
            q = p if h is None else np.minimum(p / h, 1)
            lam = miss / (1 - miss) * q.sum()
            for k in range(6):
                self.assertAlmostEqual(E[0, k], brute_expected(p, k, h, lam), places=9)

    def test_decode_choices(self):
        """Confident pairs are kept, a lone weak candidate gives the empty set."""
        ent = np.array([0, 0, 0, 1, 2])
        p = np.array([0.95, 0.9, 0.05, 0.2, 0.97])
        keep = dec.expected_f05_decode(ent, p, 3)
        np.testing.assert_array_equal(keep, [True, True, False, False, True])

    def test_threshold_empty_rule(self):
        """An S1 keeps its pairs only if its best p reaches t_keep."""
        ent = np.array([0, 0, 1])
        p = np.array([0.7, 0.55, 0.6])
        np.testing.assert_array_equal(dec.threshold_decode(ent, p, 0.5, 0.65), [True, True, False])


class TestBudget(unittest.TestCase):
    """Error budget sums to 1 - F0.5."""

    def test_budget(self):
        """Singleton hit, a decoy FP, a blocking miss and a scoring miss."""
        # entities: 0 singleton predicted non-empty; 1 has 2 true (1 found+predicted, 1 missed by blocking) + decoy FP;
        #           2 has 1 true found but not predicted
        ent = np.array([0, 1, 1, 2])
        pred = np.array([True, True, True, False])
        label = np.array([0, 1, 0, 1])
        q_true = np.array([-1, 1, -1, 2])
        n_true = np.array([0, 2, 1])
        b = dec.error_budget(ent, pred, label, q_true, n_true)
        self.assertAlmostEqual(b["check_sum"], b["total_lost"], places=5)
        self.assertAlmostEqual(b["singleton_nonempty"], 1 / 3, places=5)
        self.assertGreater(b["fp_decoy"], 0)
        self.assertEqual(b["fp_other"], 0)
        self.assertAlmostEqual(b["fn_scoring"], 1 / 3, places=5)
        m = dec.fold_metrics(ent, pred, label, n_true)
        self.assertAlmostEqual(1 - m["macro_f05"], b["total_lost"], places=5)


if __name__ == "__main__":
    unittest.main()
