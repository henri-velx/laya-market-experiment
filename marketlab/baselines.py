"""Baselines on the validation block.

Training data ends in 2021, validation is 2022-2023. Four simple baselines come
first, then two learned models on the same features. If a model cannot beat
"always buy" or "always hold", it is not saying anything about the market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, metrics


def make_models() -> dict:
    return {
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=0
        ),
    }


def run(dataset: pd.DataFrame) -> list[dict]:
    train, val = dataset[dataset.split == "train"], dataset[dataset.split == "val"]
    if train.empty or val.empty:
        raise SystemExit("Train or validation split is empty. Check the dates in config.py.")

    yv, fv = val["label"].to_numpy(), val["fwd_ret"].to_numpy()
    priors = train["label"].value_counts(normalize=True).reindex(config.LABELS).fillna(0)
    rng = np.random.default_rng(0)

    rule = np.where(
        (val.vs_sma50 > 0) & (val.vs_sma200 > 0) & (val.ret_20d > 0), "buy",
        np.where((val.vs_sma50 < 0) & (val.ret_20d < 0), "sell", "hold"),
    )

    rows = [
        metrics.evaluate("always buy (market drift)", yv, np.full(len(val), "buy"), fv),
        metrics.evaluate("always most common class", yv, np.full(len(val), priors.idxmax()), fv),
        metrics.evaluate("random by class prior", yv, rng.choice(config.LABELS, size=len(val), p=priors.to_numpy()), fv),
        metrics.evaluate("rule: SMA + momentum", yv, rule, fv),
    ]

    for name, model in make_models().items():
        model.fit(train[config.FEATURES], train["label"])
        rows.append(metrics.evaluate(name, yv, model.predict(val[config.FEATURES]), fv))

    print("Validation block (train: up to 2021, validation: 2022-2023):\n")
    metrics.print_table(rows)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "baselines_val.csv", index=False)
    return rows
