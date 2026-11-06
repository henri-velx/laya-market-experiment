"""A small, reproducible study of intraday-to-daily market signals.

Stages live in their own modules and are driven by ``run.py``:

    data        download daily bars
    features    build features and triple-barrier labels
    splits      chronological train/val/test split with purging
    baselines   simple and learned baselines on the validation block
    walkforward one-year-ahead test, repeated for every year
    report      render report.html from the saved results
    laya        optional: score the validation block with Laya, zero-shot
"""
