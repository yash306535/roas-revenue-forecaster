"""
Entry script for prediction (called by run.sh).

  python src/predict.py --features features.parquet \
      --model ./pickle/model.pkl --output ./output/predictions.csv

Loads the committed model, rolls the forecast forward 30/60/90 days, derives
ROAS, rolls up to channel + total, and writes predictions.csv.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from forecast import run_forecast, summarize


def main():
    parser = argparse.ArgumentParser(description="Produce probabilistic forecasts.")
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--output", default="./output/predictions.csv")
    args = parser.parse_args()

    panel = pd.read_parquet(args.features)
    artifact = joblib.load(args.model)

    forecast_long = run_forecast(artifact, panel)
    predictions = summarize(forecast_long)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)

    print(f"Wrote {len(predictions)} forecast rows -> {out_path}")
    total_90 = predictions[(predictions.level == "total") & (predictions.horizon_days == 90)]
    if not total_90.empty:
        r = total_90.iloc[0]
        print("90-day blended outlook:")
        print(f"  revenue: ${r.revenue_p10:,.0f}  ..  ${r.revenue_p50:,.0f}  ..  ${r.revenue_p90:,.0f}")
        print(f"  ROAS:    {r.roas_p10:.2f}  ..  {r.roas_p50:.2f}  ..  {r.roas_p90:.2f}")


if __name__ == "__main__":
    main()
