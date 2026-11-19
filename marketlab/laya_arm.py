"""Optional: score the validation block with Laya, zero-shot.

Needs the `laya` package and a CUDA GPU (the checkpoint is ~0.8 GB and the
English weights are downloaded on first use). Nothing else in this repo needs it.

The question deliberately avoids yes/no words and negations; the options are
read literally by the model.
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import config, metrics
from .state import build as build_state

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Given this market situation, what is the most likely outcome?",
        "criteria": {
            "buy": f"price rises clearly within {config.HORIZON} trading days",
            "hold": f"price stays in a range for {config.HORIZON} trading days",
            "sell": f"price falls clearly within {config.HORIZON} trading days",
        },
    }
}


def run(batch_size: int = 64) -> pd.DataFrame:
    try:
        import laya
    except ImportError as exc:  # keep the rest of the repo usable
        raise SystemExit(
            "The Laya arm needs the `laya` package and a CUDA GPU.\n"
            "Install it with `pip install laya`, or run the other stages instead."
        ) from exc

    dataset = pd.read_csv(config.BUILD_DIR / "dataset.csv", parse_dates=["date", "t1"])
    val = dataset[dataset.split == "val"].reset_index(drop=True)
    records = val.to_dict("records")
    states = [build_state(r) for r in records]
    print(f"{len(states)} validation rows")

    with warnings.catch_warnings():
        # The shipped checkpoint logs a known temperature notice; it does not move answers.
        warnings.filterwarnings("ignore", message=".*invalid temperatures.*", category=RuntimeWarning)
        agent = laya.load("convaiinnovations/laya")
        results = agent.predict_batch(states, QUESTIONS, batch_size=batch_size)

    rows = []
    for rec, res in zip(records, results):
        answer = res["answers"]["action"]
        probs = answer["probabilities"]
        rows.append({
            "ticker": rec["ticker"], "date": rec["date"].date().isoformat(),
            "label": rec["label"], "fwd_ret": rec["fwd_ret"], "pred": answer["choice"],
            "p_buy": round(probs["buy"], 4), "p_hold": round(probs["hold"], 4),
            "p_sell": round(probs["sell"], 4),
        })

    out = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.RESULTS_DIR / "laya_val.csv", index=False)

    print("\nPredicted classes:")
    print(out["pred"].value_counts(normalize=True).round(3).to_string())

    comparison = config.RESULTS_DIR / "baselines_val.csv"
    if not comparison.exists():
        print("\nNo baselines_val.csv yet: run `python run.py baselines` for the comparison table.")
        return out

    table = pd.read_csv(comparison).to_dict("records")
    table.append(metrics.evaluate("Laya, zero-shot", out["label"], out["pred"], out["fwd_ret"]))
    print("\nComparison on the validation block:\n")
    metrics.print_table(table)
    return out
