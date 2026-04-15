"""
live_bot.py
MarketMind Live Prediction Bot.

Makes one prediction per stock every 60 minutes during NYSE market hours.
Validates each prediction after 60 minutes using actual price movement.
Updates results/live_predictions.csv continuously for the dashboard.

US Market Hours (ET): 9:30 AM to 4:00 PM, Monday to Friday
Pakistan (PKT = UTC+5):  7:00 PM to 1:30 AM

Usage:
  python live_bot.py --simulate             # demo mode, works any time
  python live_bot.py --simulate --cycles 5  # run 5 simulated cycles
  python live_bot.py --once                 # one real cycle and exit
  python live_bot.py                        # full live mode (market hours)
"""

import sys
import time
import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

try:
    import pytz
    ET = pytz.timezone("America/New_York")
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

from dotenv import load_dotenv
load_dotenv()

from pipeline.fetch_news import fetch_all_news, get_news_for_date
from pipeline.rag        import build_rag_from_df
from pipeline.decompose  import run_full_chain
from pipeline.verify     import verify_prediction

TICKERS     = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
               "META", "JPM",  "XOM",  "WMT",  "GOOGL"]
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
LIVE_FILE   = RESULTS_DIR / "live_predictions.csv"
THRESHOLD   = 0.3

LIVE_COLUMNS = [
    "cycle_id", "timestamp", "ticker",
    "prediction", "verified_prediction", "confidence", "reasoning",
    "news_summary", "tech_analysis", "argument",
    "verification_note", "was_revised",
    "price_at_prediction", "price_at_validation",
    "actual_change_pct", "actual_label", "correct"
]


def is_market_open() -> bool:
    if not HAS_PYTZ:
        now       = datetime.utcnow()
        minutes   = now.hour * 60 + now.minute
        open_utc  = 13 * 60 + 30
        close_utc = 20 * 60
        return now.weekday() < 5 and open_utc <= minutes <= close_utc
    now      = datetime.now(ET)
    open_dt  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_dt = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return now.weekday() < 5 and open_dt <= now <= close_dt


def get_current_price(ticker: str) -> float:
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 0.0


def label_change(pct: float) -> str:
    if pct >  THRESHOLD: return "UP"
    if pct < -THRESHOLD: return "DOWN"
    return "NEUTRAL"


def load_live() -> pd.DataFrame:
    if LIVE_FILE.exists():
        try:
            return pd.read_csv(LIVE_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=LIVE_COLUMNS)


def save_live(df: pd.DataFrame):
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(LIVE_FILE, index=False)


def run_cycle(news_df, rag, cycle_id) -> pd.DataFrame:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  PREDICTION CYCLE {cycle_id}  |  {ts}")
    print(f"{'='*60}")

    rows = []
    for ticker in TICKERS:
        price_now = get_current_price(ticker)
        query     = f"What is the latest financial news affecting {ticker} stock price today?"
        headlines = rag.retrieve(ticker, query, k=5)
        if not headlines:
            headlines = get_news_for_date(ticker, datetime.now().strftime("%Y-%m-%d"), news_df)

        chain    = run_full_chain(ticker, headlines)
        verified = verify_prediction(chain)
        pred     = verified["verified_prediction"]
        conf     = verified["confidence"]
        revised  = verified["was_revised"]

        marker = " [CoVe revised]" if revised else ""
        print(f"  [{ticker}] {pred} ({conf}){marker}")

        rows.append({
            "cycle_id":            cycle_id,
            "timestamp":           ts,
            "ticker":              ticker,
            "prediction":          verified["prediction"],
            "verified_prediction": pred,
            "confidence":          conf,
            "reasoning":           verified["reasoning"],
            "news_summary":        verified["news_summary"],
            "tech_analysis":       verified["tech_analysis"],
            "argument":            verified["argument"],
            "verification_note":   verified["verification_note"],
            "was_revised":         revised,
            "price_at_prediction": price_now,
            "price_at_validation": None,
            "actual_change_pct":   None,
            "actual_label":        None,
            "correct":             None
        })
    return pd.DataFrame(rows)


def score_cycle(df: pd.DataFrame, cycle_id) -> pd.DataFrame:
    mask = df["cycle_id"] == cycle_id
    if not mask.any():
        return df
    print(f"\n  Scoring cycle {cycle_id}...")
    for idx, row in df[mask].iterrows():
        price_then = row["price_at_prediction"]
        price_now  = get_current_price(row["ticker"])
        if price_then and price_then > 0 and price_now > 0:
            pct = (price_now - price_then) / price_then * 100
            df.loc[idx, "price_at_validation"] = price_now
            df.loc[idx, "actual_change_pct"]   = round(pct, 3)
            df.loc[idx, "actual_label"]         = label_change(pct)
            df.loc[idx, "correct"] = (row["verified_prediction"] == label_change(pct))
    scored = df[mask].dropna(subset=["correct"])
    if not scored.empty:
        acc = scored["correct"].mean() * 100
        print(f"  Cycle {cycle_id} accuracy: {acc:.1f}% "
              f"({int(scored['correct'].sum())}/{len(scored)})")
    return df


def run_simulation(n_cycles: int = 3):
    print("\n" + "="*60)
    print("  MARKETMIND LIVE BOT — SIMULATION MODE")
    print(f"  Running {n_cycles} simulated cycles")
    print("="*60)

    RESULTS_DIR.mkdir(exist_ok=True)

    if (DATA_DIR / "news.csv").exists():
        news_df = pd.read_csv(DATA_DIR / "news.csv")
        print(f"\nLoaded {len(news_df)} cached news articles")
    else:
        print("\nFetching news...")
        news_df = fetch_all_news()

    print("\nBuilding FAISS RAG index...")
    rag = build_rag_from_df(news_df)

    labels_path = DATA_DIR / "test_labels.csv"
    if not labels_path.exists():
        print("ERROR: No test_labels.csv found. Run python run_pipeline.py first.")
        return

    labels_df  = pd.read_csv(labels_path)
    labels_df["Date"] = pd.to_datetime(labels_df["Date"]).dt.date
    test_dates = sorted(labels_df["Date"].unique())[:n_cycles]

    all_preds = load_live()
    if not all_preds.empty:
        all_preds = all_preds[
            ~all_preds["cycle_id"].astype(str).str.startswith("SIM")]

    overall_correct = 0
    overall_total   = 0

    for i, date in enumerate(test_dates, 1):
        cycle_id = f"SIM{i}"
        print(f"\n{'='*60}")
        print(f"  SIMULATED CYCLE {cycle_id}  |  {date}")
        print(f"{'='*60}")

        rows = []
        for ticker in TICKERS:
            label_row    = labels_df[(labels_df["Ticker"] == ticker) & (labels_df["Date"] == date)]
            actual_label = label_row["Label"].values[0]  if not label_row.empty else "NEUTRAL"
            price_now    = float(label_row["Close"].values[0]) if not label_row.empty else 0.0

            query     = f"What is the latest financial news affecting {ticker} stock today?"
            headlines = rag.retrieve(ticker, query, k=5)

            chain    = run_full_chain(ticker, headlines)
            verified = verify_prediction(chain)
            pred     = verified["verified_prediction"]
            correct  = (pred == actual_label)
            overall_correct += int(correct)
            overall_total   += 1

            tick = "✓" if correct else "✗"
            marker = " [CoVe revised]" if verified["was_revised"] else ""
            print(f"  [{ticker}] {pred} vs {actual_label} {tick}{marker}")

            rows.append({
                "cycle_id":            cycle_id,
                "timestamp":           str(date),
                "ticker":              ticker,
                "prediction":          verified["prediction"],
                "verified_prediction": pred,
                "confidence":          verified["confidence"],
                "reasoning":           verified["reasoning"],
                "news_summary":        verified["news_summary"],
                "tech_analysis":       verified["tech_analysis"],
                "argument":            verified["argument"],
                "verification_note":   verified["verification_note"],
                "was_revised":         verified["was_revised"],
                "price_at_prediction": price_now,
                "price_at_validation": None,
                "actual_change_pct":   label_row["Pct_Change"].values[0] if not label_row.empty else None,
                "actual_label":        actual_label,
                "correct":             correct
            })

        cycle_df  = pd.DataFrame(rows)
        cycle_acc = cycle_df["correct"].mean() * 100
        print(f"\n  Cycle {cycle_id} accuracy: {cycle_acc:.1f}%")

        all_preds = pd.concat([all_preds, cycle_df], ignore_index=True)
        save_live(all_preds)

    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    if overall_total > 0:
        print(f"  Overall accuracy: {overall_correct/overall_total*100:.1f}% "
              f"({overall_correct}/{overall_total})")
    print(f"  Run: streamlit run app.py")
    print("="*60)


def run_live():
    print("\n" + "="*60)
    print("  MARKETMIND LIVE BOT")
    print("  Predictions every 60 minutes during NYSE market hours")
    print("  Tip: use --simulate for demos outside market hours")
    print("="*60)

    RESULTS_DIR.mkdir(exist_ok=True)
    news_df       = fetch_all_news()
    rag           = build_rag_from_df(news_df)
    all_preds     = load_live()
    cycle_id      = (all_preds["cycle_id"].nunique() + 1) if not all_preds.empty else 1
    prev_cycle_id = None

    while True:
        if not is_market_open():
            print("\n  Market closed. Checking again in 30 minutes.")
            print("  Tip: use --simulate for demos outside market hours.")
            time.sleep(1800)
            continue

        if prev_cycle_id is not None:
            all_preds = score_cycle(all_preds, prev_cycle_id)
            save_live(all_preds)

        new_rows      = run_cycle(news_df, rag, cycle_id)
        all_preds     = pd.concat([all_preds, new_rows], ignore_index=True)
        save_live(all_preds)
        prev_cycle_id = cycle_id
        cycle_id     += 1

        print(f"\n  Saved. Next cycle in 60 minutes.")
        time.sleep(3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketMind Live Bot")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate using historical data (works any time)")
    parser.add_argument("--once",     action="store_true",
                        help="Run one live prediction cycle and exit")
    parser.add_argument("--cycles",   type=int, default=3,
                        help="Number of simulation cycles (default 3)")
    args = parser.parse_args()

    if args.simulate:
        run_simulation(n_cycles=args.cycles)
    elif args.once:
        news_df   = fetch_all_news()
        rag       = build_rag_from_df(news_df)
        all_preds = load_live()
        cycle_id  = (all_preds["cycle_id"].nunique() + 1) if not all_preds.empty else 1
        new_rows  = run_cycle(news_df, rag, cycle_id)
        all_preds = pd.concat([all_preds, new_rows], ignore_index=True)
        save_live(all_preds)
        print(f"\nDone. Run: streamlit run app.py")
    else:
        run_live()
