"""
Train the probabilistic model and save it to pickle/model.pkl.

This runs OFFLINE (not part of run.sh). The contest pipeline does NOT retrain --
it only loads the pickle we commit. We:
  1. build features from data/
  2. backtest on past months (honest accuracy check)
  3. train final models on ALL usable rows
  4. save one artifact (the 3 quantile models + the feature schema)

  python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from data_loader import load_unified_data
from features import build_panel, training_frame
from model import (
    QUANTILES, RANDOM_SEED,
    build_design_matrix, train_quantile_models, predict_quantiles,
)


def backtest(frame: pd.DataFrame, n_test_periods: int = 4):
    """Train on early months, test on the most recent ones, report honesty metrics."""
    periods = sorted(frame["period"].unique())
    if len(periods) <= n_test_periods + 3:
        print("  [backtest] not enough history for a split; skipping.")
        return
    cutoff = periods[-n_test_periods]
    train_df = frame[frame["period"] < cutoff]
    test_df = frame[frame["period"] >= cutoff]

    X_tr, feat = build_design_matrix(train_df)
    X_te, _ = build_design_matrix(test_df)
    models = train_quantile_models(X_tr, train_df["target_revenue"])
    preds = predict_quantiles(models, X_te, feat)

    # attach predictions back to the test rows so we can aggregate them
    ev = test_df[["period", "channel", "target_revenue"]].reset_index(drop=True).copy()
    ev["p10"], ev["p50"], ev["p90"] = preds["p10"].values, preds["p50"].values, preds["p90"].values

    def _metrics(g):
        actual = g["target_revenue"].to_numpy()
        p50 = g["p50"].to_numpy()
        inside = ((actual >= g["p10"].to_numpy()) & (actual <= g["p90"].to_numpy())).mean()
        mae = np.mean(np.abs(actual - p50))
        denom = np.where(actual == 0, np.nan, actual)
        mape = np.nanmean(np.abs((actual - p50) / denom)) * 100
        return mae, mape, inside * 100

    # 1) finest grain (per channel x type cell) -- noisiest view
    mae, mape, cov = _metrics(ev)
    print(f"  [backtest] test rows: {len(test_df)} (last {n_test_periods} months)")
    print(f"  [per-cell ] MAPE {mape:5.1f}% | coverage {cov:3.0f}% | MAE ${mae:,.0f}")

    # 2) aggregate to TOTAL revenue per month (what the business / contest scores)
    total = ev.groupby("period").agg(
        target_revenue=("target_revenue", "sum"),
        p10=("p10", "sum"), p50=("p50", "sum"), p90=("p90", "sum"),
    ).reset_index()
    mae, mape, cov = _metrics(total)
    print(f"  [TOTAL    ] MAPE {mape:5.1f}% | coverage {cov:3.0f}% | MAE ${mae:,.0f}")

    # 3) per-channel monthly totals
    by_ch = ev.groupby(["period", "channel"]).agg(
        target_revenue=("target_revenue", "sum"),
        p10=("p10", "sum"), p50=("p50", "sum"), p90=("p90", "sum"),
    ).reset_index()
    mae, mape, cov = _metrics(by_ch)
    print(f"  [by-channel] MAPE {mape:5.1f}% | coverage {cov:3.0f}% | MAE ${mae:,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Train + save the quantile model.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-out", default="./pickle/model.pkl")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("Loading + building features...")
    unified = load_unified_data(args.data_dir)
    panel = build_panel(unified, group_keys=("channel", "campaign_type"))
    frame = training_frame(panel)
    print(f"Trainable rows: {len(frame)}")

    print("Backtesting...")
    backtest(frame)

    print("Training final models on all rows...")
    X, feature_names = build_design_matrix(frame)
    models = train_quantile_models(X, frame["target_revenue"])

    artifact = {
        "quantile_models": models,
        "feature_names": feature_names,
        "quantiles": QUANTILES,
        "log_target": True,
        "group_keys": ["channel", "campaign_type"],
        "trained_rows": len(frame),
        "seed": RANDOM_SEED,
    }
    joblib.dump(artifact, args.model_out)
    print(f"Saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
