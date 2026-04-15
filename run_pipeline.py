"""
run_pipeline.py
MarketMind historical ablation study runner.

Runs three prediction configurations on the held-out test window:
  Config 1: Baseline             — zero-shot, ticker name only
  Config 2: RAG + Sentiment      — FAISS retrieval + LLM sentiment
  Config 3: Decomposition + CoVe — full 4-step chain + verification

Usage:
  python run_pipeline.py              # full run (fetches all fresh data)
  python run_pipeline.py --skip-fetch # reuse cached data CSVs
"""

import sys
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

from pipeline.fetch_data import fetch_prices, compute_labels, get_test_labels
from pipeline.fetch_news import fetch_all_news, get_news_for_date
from pipeline.rag        import build_rag_from_df
from pipeline.sentiment  import analyze_sentiment
from pipeline.decompose  import run_full_chain
from pipeline.verify     import verify_prediction
from pipeline.evaluate   import build_ablation_table, build_per_stock_table, print_summary

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")


def baseline_predict(ticker: str) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock analyst. Answer with only UP, DOWN, or NEUTRAL."},
                {"role": "user",   "content":
                 f"Based on your knowledge of {ticker} as a publicly traded company, "
                 f"predict whether the stock will go UP, DOWN, or NEUTRAL over the "
                 f"next 3 trading days. Respond with ONLY one word."}
            ],
            temperature=0
        )
        v = resp.choices[0].message.content.strip().upper()
        return v if v in ("UP", "DOWN", "NEUTRAL") else "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def rag_predict(ticker: str, sentiment: str, headlines: list) -> str:
    if headlines:
        ctx = (f"News sentiment: {sentiment}\n"
               f"Headlines:\n" + "\n".join(f"  - {h}" for h in headlines[:3]))
    else:
        ctx = "No recent news available."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock analyst. Answer with only UP, DOWN, or NEUTRAL."},
                {"role": "user",   "content":
                 f"Predict {ticker} direction over next 3 trading days.\n\n"
                 f"{ctx}\n\nRespond with ONLY one word: UP, DOWN, or NEUTRAL."}
            ],
            temperature=0
        )
        v = resp.choices[0].message.content.strip().upper()
        return v if v in ("UP", "DOWN", "NEUTRAL") else "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def run(skip_fetch: bool = False):
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("  MARKETMIND — HISTORICAL ABLATION STUDY")
    print("="*60)

    # Step 1: Price data
    print("\n[1/5] Price data and ground truth labels")
    if not skip_fetch or not (DATA_DIR / "labels.csv").exists():
        prices = fetch_prices()
        labels = compute_labels(prices)
    else:
        print("  Using cached data/labels.csv")
        labels = pd.read_csv(DATA_DIR / "labels.csv")
    test_labels = get_test_labels(labels)

    # Step 2: News
    print("\n[2/5] News headlines")
    if not skip_fetch or not (DATA_DIR / "news.csv").exists():
        news_df = fetch_all_news()
    else:
        print("  Using cached data/news.csv")
        news_df = pd.read_csv(DATA_DIR / "news.csv")
    print(f"  Articles available: {len(news_df)}")

    # Step 3: Build FAISS index
    print("\n[3/5] Building FAISS RAG index")
    rag = build_rag_from_df(news_df)

    # Step 4: Predictions
    total = len(test_labels)
    print(f"\n[4/5] Running 3 configurations x {total} test records")
    print("  Config 1: Baseline  Config 2: RAG+Sentiment  Config 3: Decomposition+CoVe\n")

    predictions = []
    for i, (_, row) in enumerate(test_labels.iterrows()):
        ticker = row["Ticker"]
        date   = row["Date"]
        actual = row["Label"]

        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} done...")

        query     = f"What is the latest news affecting {ticker} stock price today?"
        headlines = rag.retrieve(ticker, query, k=5)
        if not headlines:
            headlines = get_news_for_date(ticker, str(date), news_df)

        # Config 1: Baseline
        p1 = baseline_predict(ticker)

        # Config 2: RAG + Sentiment
        try:
            sentiment, _ = analyze_sentiment(ticker, headlines) if headlines else ("NEUTRAL", "")
        except Exception:
            sentiment = "NEUTRAL"
        p2 = rag_predict(ticker, sentiment, headlines)

        # Config 3: Decomposition + CoVe
        try:
            chain    = run_full_chain(ticker, headlines)
            verified = verify_prediction(chain)
            p3       = verified["verified_prediction"]
        except Exception as e:
            print(f"  Chain error {ticker} {date}: {e}")
            p3 = "NEUTRAL"

        predictions.append({
            "Ticker":           ticker,
            "Date":             date,
            "Label":            actual,
            "Close":            row.get("Close"),
            "Pct_Change":       row.get("Pct_Change"),
            "Headlines":        " | ".join(headlines[:3]) if headlines else "",
            "Sentiment":        sentiment,
            "Pred_Baseline":    p1,
            "Pred_RAG":         p2,
            "Pred_Tools":       p3,
            "Correct_Baseline": p1 == actual,
            "Correct_RAG":      p2 == actual,
            "Correct_Tools":    p3 == actual
        })
        time.sleep(0.2)

    # Step 5: Evaluate
    print("\n[5/5] Saving and evaluating")
    preds_df = pd.DataFrame(predictions)
    preds_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    ablation  = build_ablation_table(preds_df)
    per_stock = build_per_stock_table(preds_df)
    ablation.to_csv(RESULTS_DIR  / "ablation.csv",  index=False)
    per_stock.to_csv(RESULTS_DIR / "per_stock.csv", index=False)

    print_summary(preds_df)

    print("\n" + "="*60)
    print("  Done! Run: streamlit run app.py")
    print("="*60)
    return preds_df


if __name__ == "__main__":
    run(skip_fetch="--skip-fetch" in sys.argv)
