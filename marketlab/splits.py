"""Chronological train / validation / test split, with purging.

A row on day t is only usable for training if its label has resolved before the
next block begins; otherwise a training label would overlap the validation
period. Rows that fail that check are marked "purged" and never used.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def assign(df: pd.DataFrame) -> pd.DataFrame:
    d, t1 = df["date"], df["t1"]
    train_end = pd.Timestamp(config.TRAIN_END)
    val_start, val_end = pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END)
    test_start = pd.Timestamp(config.TEST_START)

    split = np.full(len(df), "purged", dtype=object)
    split[(d <= train_end) & (t1 < val_start)] = "train"
    split[(d >= val_start) & (d <= val_end) & (t1 < test_start)] = "val"
    split[d >= test_start] = "test"

    out = df.copy()
    out["split"] = split
    return out


def build(labels: pd.DataFrame) -> pd.DataFrame:
    """Split, write data/built/dataset.csv and results/summary.json."""
    df = labels.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = assign(df)
    config.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.BUILD_DIR / "dataset.csv", index=False)

    print("Rows per split:")
    print(df["split"].value_counts().to_string())
    print("\nClass share per split:")
    print(df.groupby("split")["label"].value_counts(normalize=True).unstack().round(3).to_string())

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "tickers": sorted(df["ticker"].unique().tolist()),
        "rows": int(len(df)),
        "rows_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "class_share": {k: round(float(v), 4) for k, v in df["label"].value_counts(normalize=True).items()},
        "params": {
            "horizon_days": config.HORIZON,
            "barrier_k": config.K,
            "vol_window": config.VOL_WINDOW,
            "cost_round_trip": config.COST,
            "train_end": config.TRAIN_END,
            "val_start": config.VAL_START,
            "val_end": config.VAL_END,
            "test_start": config.TEST_START,
        },
    }
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return df
