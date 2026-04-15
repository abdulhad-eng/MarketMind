"""
sentiment.py
LLM-based financial news sentiment analysis.
Returns POSITIVE, NEGATIVE, or NEUTRAL for a set of headlines.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def analyze_sentiment(ticker, headlines):
    if not headlines:
        return "NEUTRAL", "No news available."

    hl_text = "\n".join(f"- {h}" for h in headlines[:5])
    prompt  = f"""You are a financial news sentiment analyst.

Analyze the following recent news headlines about {ticker} stock.
Classify the overall sentiment as exactly one of: POSITIVE, NEGATIVE, or NEUTRAL

Guidelines:
- POSITIVE: strong earnings, revenue growth, new products, upgrades, partnerships
- NEGATIVE: losses, lawsuits, leadership problems, downgrades, market share loss
- NEUTRAL:  mixed signals, routine updates, no clear directional implication

Headlines:
{hl_text}

Respond with ONLY one word: POSITIVE, NEGATIVE, or NEUTRAL"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Respond with only POSITIVE, NEGATIVE, or NEUTRAL."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0
        )
        sentiment = response.choices[0].message.content.strip().upper()
        if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            sentiment = "NEUTRAL"
    except Exception as e:
        print(f"  Sentiment API error: {e}")
        sentiment = "NEUTRAL"

    return sentiment, hl_text


if __name__ == "__main__":
    s, _ = analyze_sentiment("AAPL", [
        "Apple beats Q2 earnings estimates by 8%",
        "iPhone sales surge in emerging markets"
    ])
    print(f"Sentiment: {s}")
