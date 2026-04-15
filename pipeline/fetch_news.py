"""
fetch_news.py
Fetches news headlines using yfinance built-in news as the primary source.
yfinance news is completely free, requires no API key, has no date
restrictions, and always returns recent headlines.

NewsAPI is used as a bonus if a key is available — but without date
parameters so it works on the free tier.
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DATA_DIR     = Path("data")

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
           "META", "JPM",  "XOM",  "WMT",  "GOOGL"]

TICKER_NAMES = {
    "AAPL": "Apple",     "MSFT": "Microsoft", "NVDA": "Nvidia",
    "TSLA": "Tesla",     "AMZN": "Amazon",     "META": "Meta",
    "JPM":  "JPMorgan",  "XOM":  "ExxonMobil", "WMT":  "Walmart",
    "GOOGL":"Google"
}


def fetch_yfinance_news(ticker):
    """Primary source — always works, no API key needed."""
    results = []
    try:
        articles = yf.Ticker(ticker).news or []
        for article in articles:
            pub_ts = article.get("providerPublishTime", 0)
            pub_date = (datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d")
                        if pub_ts else datetime.now().strftime("%Y-%m-%d"))
            title = article.get("title", "") or ""
            if not title or title.lower() == "[removed]":
                continue
            results.append({
                "Ticker":      ticker,
                "Date":        pub_date,
                "Title":       title.strip(),
                "Description": "",
                "Source":      article.get("publisher", "")
            })
    except Exception as e:
        print(f"  yfinance news error for {ticker}: {e}")
    return results


def fetch_newsapi_headlines(ticker):
    """Bonus source — no date filter so free tier works."""
    if not NEWS_API_KEY:
        return []
    try:
        params = {
            "q":        f"{TICKER_NAMES[ticker]} stock",
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 20,
            "apiKey":   NEWS_API_KEY
        }
        response = requests.get("https://newsapi.org/v2/everything",
                                params=params, timeout=10)
        if response.status_code != 200:
            return []
        results = []
        for article in response.json().get("articles", []):
            title = article.get("title", "") or ""
            if not title or title.lower() == "[removed]":
                continue
            results.append({
                "Ticker":      ticker,
                "Date":        article.get("publishedAt", "")[:10],
                "Title":       title.strip(),
                "Description": (article.get("description") or "").strip(),
                "Source":      (article.get("source") or {}).get("name", "")
            })
        return results
    except Exception:
        return []


def fetch_all_news(from_date=None, to_date=None):
    """
    Fetch news for all 10 tickers.
    from_date and to_date are accepted for API compatibility but ignored —
    yfinance always returns recent news which is used as context.
    """
    DATA_DIR.mkdir(exist_ok=True)
    all_articles = []

    print("Fetching news for all tickers (yfinance + NewsAPI)...\n")

    for ticker in TICKERS:
        print(f"  {ticker}...", end="", flush=True)
        yf_articles = fetch_yfinance_news(ticker)
        na_articles = fetch_newsapi_headlines(ticker)
        time.sleep(0.5)
        combined = yf_articles + na_articles
        all_articles.extend(combined)
        print(f" {len(combined)} articles "
              f"({len(yf_articles)} yfinance, {len(na_articles)} NewsAPI)")

    if not all_articles:
        print("\nWARNING: No articles fetched.")
        news_df = pd.DataFrame(columns=["Ticker","Date","Title","Description","Source"])
    else:
        news_df = pd.DataFrame(all_articles).drop_duplicates(subset=["Ticker","Title"])

    news_df.to_csv(DATA_DIR / "news.csv", index=False)
    print(f"\nSaved {len(news_df)} articles to data/news.csv")
    return news_df


def get_news_for_date(ticker, date, news_df, days_back=14):
    """
    Return up to 10 headlines for a ticker near the given date.
    Falls back to all available headlines for the ticker if none
    match the date window.
    """
    if news_df is None or news_df.empty:
        return []
    try:
        date_obj  = pd.to_datetime(date).date()
        start_obj = date_obj - timedelta(days=days_back)
        df = news_df[news_df["Ticker"] == ticker].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

        windowed  = df[(df["Date"] >= start_obj) & (df["Date"] <= date_obj)]
        headlines = [r["Title"] for _, r in windowed.iterrows()
                     if r.get("Title") and str(r["Title"]) != "nan"]
        if headlines:
            return headlines[:10]

        # Fallback: any available headlines for this ticker
        all_t = df.dropna(subset=["Title"])
        return [r["Title"] for _, r in all_t.iterrows()
                if r.get("Title") and str(r["Title"]) != "nan"][:10]
    except Exception:
        return []


if __name__ == "__main__":
    news = fetch_all_news()
    if not news.empty:
        print("\nSample articles:")
        print(news[["Ticker","Date","Title"]].head(10).to_string(index=False))
