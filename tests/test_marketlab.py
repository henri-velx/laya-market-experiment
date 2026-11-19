"""Unit tests for the pieces that are easy to get silently wrong.

    python -m unittest discover -s tests
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from marketlab import config, metrics, splits
from marketlab.features import rsi, triple_barrier


def _bars(closes):
    n = len(closes)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": [c * 1.001 for c in closes],
            "Low": [c * 0.999 for c in closes],
            "Close": list(closes),
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class RsiTest(unittest.TestCase):
    def test_uninterrupted_uptrend_is_100(self):
        out = rsi(pd.Series(np.arange(1.0, 41.0)))
        self.assertEqual(float(out.iloc[-1]), 100.0)

    def test_uninterrupted_downtrend_is_0(self):
        out = rsi(pd.Series(np.arange(40.0, 0.0, -1.0)))
        self.assertAlmostEqual(float(out.iloc[-1]), 0.0)

    def test_flat_series_is_neutral(self):
        out = rsi(pd.Series(np.full(40, 10.0)))
        self.assertEqual(float(out.iloc[-1]), 50.0)

    def test_warmup_is_nan(self):
        out = rsi(pd.Series(np.arange(1.0, 41.0)))
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertFalse(np.isnan(out.iloc[config.VOL_WINDOW]))


class TripleBarrierTest(unittest.TestCase):
    def test_rise_labels_buy(self):
        closes = [100 * 1.01**i for i in range(15)]
        label, ret, _ = triple_barrier(_bars(closes), np.full(15, 0.002))
        self.assertEqual(label[0], "buy")
        self.assertGreater(ret[0], 0)

    def test_fall_labels_sell(self):
        closes = [100 * 0.99**i for i in range(15)]
        label, ret, _ = triple_barrier(_bars(closes), np.full(15, 0.002))
        self.assertEqual(label[0], "sell")
        self.assertLess(ret[0], 0)

    def test_flat_labels_hold(self):
        label, ret, _ = triple_barrier(_bars([100.0] * 15), np.full(15, 0.02))
        self.assertEqual(label[0], "hold")
        self.assertAlmostEqual(ret[0], 0.0)

    def test_rows_without_a_full_horizon_are_unlabelled(self):
        label, _, _ = triple_barrier(_bars([100.0] * 15), np.full(15, 0.02))
        self.assertIsNone(label[-1])

    def test_gap_above_the_barrier_fills_at_the_open(self):
        closes = [100.0] * 15
        opens, highs, lows = list(closes), [100.1] * 15, [99.9] * 15
        opens[2], highs[2], lows[2] = 107.0, 108.0, 95.0  # opens over the top, then trades down
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1_000_000] * 15},
            index=pd.bdate_range("2020-01-01", periods=15),
        )
        label, ret, _ = triple_barrier(df, np.full(15, 0.002))
        self.assertEqual(label[0], "buy")
        self.assertAlmostEqual(ret[0], 0.07)  # 107 / 100 - 1, not the barrier price

    def test_broken_open_is_not_a_decision_day(self):
        closes = [100.0] * 15
        opens = list(closes)
        opens[5] = 0.0
        df = pd.DataFrame(
            {"Open": opens, "High": [100.1] * 15, "Low": [99.9] * 15, "Close": closes, "Volume": [1_000_000] * 15},
            index=pd.bdate_range("2020-01-01", periods=15),
        )
        label, _, _ = triple_barrier(df, np.full(15, 0.002))
        self.assertIsNone(label[4])


class SplitTest(unittest.TestCase):
    def test_blocks_and_purging(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2021-06-01", "2021-12-20", "2021-12-20", "2022-06-01", "2023-12-28", "2024-03-01"]
                ),
                "t1": pd.to_datetime(
                    ["2021-06-15", "2022-01-05", "2021-12-30", "2022-07-01", "2024-01-10", "2024-03-15"]
                ),
            }
        )
        out = splits.assign(df)
        self.assertEqual(list(out["split"]), ["train", "purged", "train", "val", "purged", "test"])


class MetricsTest(unittest.TestCase):
    def test_buy_signals_and_cost(self):
        y = np.array(["buy", "buy", "hold", "sell", "sell"])
        pred = np.array(["buy", "sell", "hold", "sell", "buy"])
        fwd = np.array([0.02, -0.01, 0.0, -0.02, 0.01])
        row = metrics.evaluate("x", y, pred, fwd)
        self.assertEqual(row["n_buy"], 2)
        self.assertAlmostEqual(row["ret_buy"], 0.015)
        self.assertAlmostEqual(row["net"], 0.015 - config.COST)
        self.assertFalse(np.isnan(row["se"]))

    def test_no_buy_signal_is_nan_not_a_crash(self):
        y = np.array(["hold", "sell"])
        pred = np.array(["hold", "sell"])
        row = metrics.evaluate("x", y, pred, np.array([0.0, -0.01]))
        self.assertEqual(row["n_buy"], 0)
        self.assertTrue(np.isnan(row["edge"]))


if __name__ == "__main__":
    unittest.main()
