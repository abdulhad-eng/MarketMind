"""
fetch_data.py
Downloads OHLCV price data for all 10 tickers using yfinance.
Computes 3-day forward return ground truth labels (UP/DOWN/NEUTRAL).
Saves prices.csv, labels.csv, and test_labels.csv to the data/ folder.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
           "META", "JPM",  "XOM",  "WMT",  "GOOGL"]

TICKER_NAMES = {
    "AAPL": "Apple",     "MSFT": "Microsoft", "NVDA": "Nvidia",
    "TSLA": "Tesla",     "AMZN": "Amazon",     "META": "Meta",
    "JPM":  "JPMorgan",  "XOM":  "ExxonMobil", "WMT":  "Walmart",
    "GOOGL":"Google"
}

TRAIN_START = "2025-07-01"
TRAIN_END   = "2026-02-28"
TEST_START  = "2026-03-01"
TEST_END    = "2026-04-10"

DATA_DIR = Path("data")


def fetch_prices():
    DATA_DIR.mkdir(exist_ok=True)
    all_frames = []
    for ticker in TICKERS:
        print(f"  Fetching {ticker} ({TICKER_NAMES[ticker]})...")
        try:
            df = yf.Ticker(ticker).history(start=TRAIN_START, end=TEST_END)
            if df.empty:
                print(f"  WARNING: No data for {ticker}")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).date
            df.index.name = "Date"
            df = df.reset_index()
            df["Ticker"] = ticker
            all_frames.append(df)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    prices = pd.concat(all_frames, ignore_index=True)
    prices.to_csv(DATA_DIR / "prices.csv", index=False)
    print(f"\nSaved {len(prices)} rows to data/prices.csv")
    return prices


def compute_labels(prices):
    all_labels = []
    for ticker in TICKERS:
        df = prices[prices["Ticker"] == ticker].copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.sort_values("Date").reset_index(drop=True)
        for i in range(len(df) - 3):
            price_now   = df.loc[i,     "Close"]
            price_later = df.loc[i + 3, "Close"]
            pct_change  = (price_later - price_now) / price_now * 100
            label = "UP" if pct_change > 1.0 else "DOWN" if pct_change < -1.0 else "NEUTRAL"
            all_labels.append({
                "Ticker":     ticker,
                "Date":       df.loc[i, "Date"],
                "Close":      round(price_now,   2),
                "Close_D3":   round(price_later, 2),
                "Pct_Change": round(pct_change,  4),
                "Label":      label
            })
    labels_df = pd.DataFrame(all_labels)
    labels_df.to_csv(DATA_DIR / "labels.csv", index=False)
    print(f"Saved {len(labels_df)} labeled rows to data/labels.csv")
    return labels_df


def get_test_labels(labels_df):
    labels_df = labels_df.copy()
    labels_df["Date"] = pd.to_datetime(labels_df["Date"]).dt.date
    test_start = pd.to_datetime(TEST_START).date()
    test_end   = pd.to_datetime(TEST_END).date()
    mask    = (labels_df["Date"] >= test_start) & (labels_df["Date"] <= test_end)
    test_df = labels_df[mask].copy()
    test_df.to_csv(DATA_DIR / "test_labels.csv", index=False)
    print(f"\nTest window: {len(test_df)} predictions across {test_df['Ticker'].nunique()} stocks")
    print(f"Label distribution:\n{test_df['Label'].value_counts().to_string()}")
    return test_df


if __name__ == "__main__":
    prices = fetch_prices()
    labels = compute_labels(prices)
    test   = get_test_labels(labels)
    print(test.head(10).to_string(index=False))
