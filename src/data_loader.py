"""
Data loading + unification.

This module holds the three "adapters" (one per platform). An adapter is a
small function whose only job is to take one messy CSV and convert it into our
canonical schema (see config.py) -- just like a charger converts AC->DC into
the form your device expects.

Flow:
    load_unified_data(data_dir)
        -> finds every CSV in data_dir
        -> detects which platform each file is (by its columns)
        -> runs the matching adapter
        -> stacks (concats) them into one tidy long table
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import CANONICAL_COLUMNS, NUMERIC_COLUMNS, normalize_campaign_type

MICROS_PER_DOLLAR = 1_000_000


# ---------------------------------------------------------------------------
# Small helper: every adapter ends by calling this so all three return the
# exact same columns in the exact same order, with clean types.
# ---------------------------------------------------------------------------
def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    # add any canonical column that this platform simply doesn't have
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[CANONICAL_COLUMNS].copy()

    # real datetime, not text
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # force numeric columns to be numbers; junk -> NaN
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ids/names/types as clean strings
    df["campaign_id"] = df["campaign_id"].astype(str)
    df["campaign_name"] = df["campaign_name"].astype(str)
    df["campaign_type"] = df["campaign_type"].apply(normalize_campaign_type)

    return df


# ---------------------------------------------------------------------------
# Adapter 1: Google Ads
#   - spend is in "micros" (millionths of a dollar) -> divide by 1,000,000
#   - revenue lives in metrics_conversions_value
# ---------------------------------------------------------------------------
def adapt_google(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = df["segments_date"]
    out["channel"] = "google"
    out["campaign_id"] = df["campaign_id"]
    out["campaign_name"] = df["campaign_name"]
    out["campaign_type"] = df["campaign_advertising_channel_type"]
    out["spend"] = df["metrics_cost_micros"] / MICROS_PER_DOLLAR
    out["revenue"] = df["metrics_conversions_value"]
    out["clicks"] = df["metrics_clicks"]
    out["impressions"] = df["metrics_impressions"]
    out["conversions"] = df["metrics_conversions"]
    return _finalize(out)


# ---------------------------------------------------------------------------
# Adapter 2: Bing / Microsoft Ads
#   - already in normal dollars, just rename
# ---------------------------------------------------------------------------
def adapt_bing(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = df["TimePeriod"]
    out["channel"] = "bing"
    out["campaign_id"] = df["CampaignId"]
    out["campaign_name"] = df["CampaignName"]
    out["campaign_type"] = df["CampaignType"]
    out["spend"] = df["Spend"]
    out["revenue"] = df["Revenue"]
    out["clicks"] = df["Clicks"]
    out["impressions"] = df["Impressions"]
    out["conversions"] = df["Conversions"]
    return _finalize(out)


# ---------------------------------------------------------------------------
# Adapter 3: Meta (Facebook / Instagram)
#   - NO revenue column and NO campaign type in the data.
#   - We keep spend/activity and leave revenue + type as "not available".
# ---------------------------------------------------------------------------
def adapt_meta(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = df["date_start"]
    out["channel"] = "meta"
    out["campaign_id"] = df["campaign_id"]
    out["campaign_name"] = df["campaign_name"]
    out["campaign_type"] = "unknown"          # Meta data has no type
    out["spend"] = df["spend"]
    out["revenue"] = np.nan                    # Meta data has no revenue
    out["clicks"] = df["clicks"]
    out["impressions"] = df["impressions"]
    out["conversions"] = df["conversion"]      # note: singular in source
    return _finalize(out)


# ---------------------------------------------------------------------------
# Detect which platform a file is, purely from its columns.
# This is more robust than trusting filenames -- the test harness could rename
# files, but the schema (column names) stays the same.
# ---------------------------------------------------------------------------
def detect_adapter(columns):
    cols = set(columns)
    if "metrics_cost_micros" in cols:
        return adapt_google
    if {"CampaignId", "TimePeriod"}.issubset(cols):
        return adapt_bing
    if {"date_start", "cpm"}.issubset(cols):
        return adapt_meta
    return None


def load_unified_data(data_dir: str = "./data") -> pd.DataFrame:
    """Read every CSV in data_dir, adapt each, and stack into one table."""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for csv_file in csv_files:
        raw = pd.read_csv(csv_file)
        # drop the leading unnamed index column if the file has one
        if str(raw.columns[0]).startswith("Unnamed"):
            raw = raw.drop(columns=raw.columns[0])

        adapter = detect_adapter(raw.columns)
        if adapter is None:
            print(f"  [skip] {csv_file.name}: unrecognized schema")
            continue

        adapted = adapter(raw)
        frames.append(adapted)
        print(f"  [ok]   {csv_file.name}: {len(adapted):>6} rows -> {adapted['channel'].iloc[0]}")

    if not frames:
        raise ValueError("No recognizable platform files were loaded.")

    unified = pd.concat(frames, ignore_index=True)
    unified = unified.sort_values(["date", "channel", "campaign_id"]).reset_index(drop=True)
    return unified


if __name__ == "__main__":
    # quick manual check when run directly
    df = load_unified_data("./data")
    print("\nUnified shape:", df.shape)
    print("\nRows per channel:")
    print(df["channel"].value_counts())
    print("\nSample:")
    print(df.head())
