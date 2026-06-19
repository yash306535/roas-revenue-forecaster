"""
Entry script for feature generation (called by run.sh).

  python src/generate_features.py --data-dir ./data --out features.parquet

Reads whatever CSVs are in --data-dir, unifies them, builds the monthly feature
panel, and writes it to a parquet file for the predict step to consume.
"""

import argparse

from data_loader import load_unified_data
from features import build_panel


def main():
    parser = argparse.ArgumentParser(description="Generate the monthly feature panel.")
    parser.add_argument("--data-dir", default="./data", help="folder with input CSVs")
    parser.add_argument("--out", default="features.parquet", help="output parquet path")
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    unified = load_unified_data(args.data_dir)
    print(f"Unified rows: {len(unified)}")

    panel = build_panel(unified, group_keys=("channel", "campaign_type"))
    print(f"Feature panel rows: {len(panel)}  (channel x campaign_type x month)")

    panel.to_parquet(args.out, index=False)
    print(f"Wrote features -> {args.out}")


if __name__ == "__main__":
    main()
