"""Turn a day of numbers into a short English description.

Language models read text, not feature vectors, so the state is rendered as
words. Moves are measured in units of the ticker's own volatility (a z-score),
so a quiet ETF and a wild single stock are described on the same scale.

This is the part of the study worth stealing for other projects: same
information, different rendering, and the rendering is what the model sees.
"""
from __future__ import annotations

import numpy as np


def _move(z: float) -> str:
    if z <= -2:
        return "sharp drop"
    if z <= -1:
        return "drop"
    if z <= -0.25:
        return "slightly down"
    if z < 0.25:
        return "flat"
    if z < 1:
        return "slightly up"
    if z < 2:
        return "rise"
    return "sharp rise"


def _pick(x: float, cuts, names) -> str:
    for cut, name in zip(cuts, names):
        if x < cut:
            return name
    return names[-1]


def build(r) -> dict:
    vol = max(float(r["vol_20d"]), 1e-6)
    ann_vol = vol * np.sqrt(252) * 100
    above50, above200 = r["vs_sma50"] > 0, r["vs_sma200"] > 0
    trend = {
        (True, True): "above 50d and 200d average",
        (True, False): "above 50d, below 200d average",
        (False, True): "below 50d, above 200d average",
        (False, False): "below 50d and 200d average",
    }[(above50, above200)]
    return {
        "trend": trend,
        "move_1d": _move(r["ret_1d"] / vol),
        "move_5d": _move(r["ret_5d"] / (vol * np.sqrt(5))),
        "move_20d": _move(r["ret_20d"] / (vol * np.sqrt(20))),
        "volatility": _pick(ann_vol, [12, 25, 40], ["low", "normal", "high", "very high"]),
        "vs_52w_high": _pick(r["dist_high_252d"] * 100, [-25, -10, -2],
                             ["far below high", "below high", "near high", "at high"]),
        "vs_20d_high": _pick(r["dist_high_20d"] * 100, [-8, -3, -0.5],
                             ["far below high", "below high", "near high", "at high"]),
        "rsi": _pick(r["rsi_14"], [30, 45, 55, 70],
                     ["oversold", "weak", "neutral", "strong", "overbought"]),
        "volume": _pick(r["vol_ratio"], [0.7, 1.3, 2.0], ["low", "normal", "high", "very high"]),
    }
