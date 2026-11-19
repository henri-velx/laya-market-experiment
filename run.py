#!/usr/bin/env python3
"""Command line entrypoint.

    python run.py all          # download, label, baselines, walk-forward, report
    python run.py report       # just rebuild report.html from saved results
    python run.py laya         # optional, needs a GPU

Every stage can also be run on its own; see `python run.py --help`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

STAGES = ("download", "label", "baselines", "walkforward", "report", "laya", "all")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="marketlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--data-dir", help="where raw and built data live (default: ./data)")
    p.add_argument("--results-dir", help="where result files live (default: ./results)")
    p.add_argument("--tickers", nargs="+", help="override the universe, e.g. --tickers SPY QQQ")
    p.add_argument("--start", help="first date to download, e.g. 2015-01-01")
    p.add_argument("--end", help="last date to download (default: config.END, which pins the results)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.data_dir:
        os.environ["MARKETLAB_DATA"] = str(Path(args.data_dir).resolve())
    if args.results_dir:
        os.environ["MARKETLAB_RESULTS"] = str(Path(args.results_dir).resolve())

    # Imported after the environment is set, because config reads it at import time.
    from marketlab import baselines, data, features, report, splits, walkforward

    if args.stage in ("download", "all"):
        data.download(tickers=args.tickers, start=args.start, end=args.end)
    if args.stage in ("label", "all"):
        labels = features.build(tickers=args.tickers)
        dataset = splits.build(labels)
        if args.stage == "label":
            return 0
    else:
        dataset = None

    if args.stage in ("baselines", "all"):
        if dataset is None:
            import pandas as pd
            from marketlab import config
            dataset = pd.read_csv(config.BUILD_DIR / "dataset.csv", parse_dates=["date", "t1"])
        baselines.run(dataset)

    if args.stage in ("walkforward", "all"):
        if dataset is None:
            import pandas as pd
            from marketlab import config
            dataset = pd.read_csv(config.BUILD_DIR / "dataset.csv", parse_dates=["date", "t1"])
        walkforward.run(dataset)

    if args.stage == "laya":
        from marketlab import laya_arm
        laya_arm.run()

    if args.stage in ("report", "all"):
        path = report.build()
        print(f"\nWrote {path}")
        print("Open it in a browser, or enable GitHub Pages to publish it.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
