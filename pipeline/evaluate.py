"""
evaluate.py
Accuracy metrics and ablation table builder for MarketMind.
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")


def compute_accuracy(df, config_col):
    if df.empty or config_col not in df.columns:
        return 0.0
    return round((df[config_col] == df["Label"]).sum() / len(df) * 100, 2)


def build_ablation_table(df):
    configs = [
        ("Baseline",              "Pred_Baseline"),
        ("+ RAG + Sentiment",     "Pred_RAG"),
        ("+ Decomposition + CoVe","Pred_Tools"),
    ]
    rows = []
    for name, col in configs:
        if col not in df.columns:
            continue
        acc     = compute_accuracy(df, col)
        correct = (df[col] == df["Label"]).sum()
        rows.append({
            "Configuration": name,
            "Correct":       int(correct),
            "Total":         len(df),
            "Accuracy":      f"{acc}%"
        })
    return pd.DataFrame(rows)


def build_per_stock_table(df):
    configs = [
        ("Baseline", "Pred_Baseline"),
        ("RAG",      "Pred_RAG"),
        ("Full Chain","Pred_Tools"),
    ]
    rows = []
    for ticker in sorted(df["Ticker"].unique()):
        subset = df[df["Ticker"] == ticker]
        row    = {"Ticker": ticker, "Predictions": len(subset)}
        for name, col in configs:
            if col in df.columns:
                row[name] = f"{compute_accuracy(subset, col)}%"
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nTest predictions: {len(df)}")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    print("\n--- ABLATION TABLE ---")
    print(build_ablation_table(df).to_string(index=False))
    print("\n--- PER STOCK ---")
    print(build_per_stock_table(df).to_string(index=False))
    print("\n--- PER LABEL ACCURACY (Full Chain) ---")
    if "Pred_Tools" in df.columns:
        for label in ["UP", "DOWN", "NEUTRAL"]:
            sub = df[df["Label"] == label]
            if not sub.empty:
                acc = (sub["Pred_Tools"] == sub["Label"]).mean() * 100
                print(f"  {label}: {acc:.1f}%")


if __name__ == "__main__":
    try:
        df = pd.read_csv(RESULTS_DIR / "predictions.csv")
        print_summary(df)
    except FileNotFoundError:
        print("No predictions.csv found. Run python run_pipeline.py first.")
