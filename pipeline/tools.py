"""
tools.py
Real OpenAI function calling for MarketMind.

The model is given two tool schemas and autonomously decides
which tools to call and with what parameters during Step 2
of the decomposition chain. We execute the real yfinance-backed
functions and return results to the model.
"""

import json
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": (
                "Fetches real-time technical indicators for a stock ticker: "
                "RSI (14-day), 5-day price momentum, and 20-day moving average position. "
                "Call this when you need to understand the recent price trend."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol e.g. AAPL, TSLA, MSFT"
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Days to look back for calculations. Default 20.",
                        "default": 20
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_intraday_change",
            "description": (
                "Fetches the percentage price change for a stock over the last N hours. "
                "Call this to understand very recent intraday momentum."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Number of hours to look back (1 to 6). Default 1.",
                        "default": 1
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]


def get_technical_indicators(ticker: str, lookback_days: int = 20) -> dict:
    try:
        end   = datetime.now()
        start = end - timedelta(days=lookback_days + 10)
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d")
        )
        if hist.empty or len(hist) < 5:
            return {"error": f"Insufficient data for {ticker}"}

        closes      = hist["Close"].values
        momentum_5d = float((closes[-1] - closes[-5]) / closes[-5] * 100)
        ma20        = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))

        if len(closes) >= 15:
            deltas   = np.diff(closes[-15:])
            gains    = deltas[deltas > 0]
            losses   = -deltas[deltas < 0]
            avg_gain = float(np.mean(gains))  if len(gains)  > 0 else 0.001
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.001
            rsi      = round(100 - (100 / (1 + avg_gain / avg_loss)), 1)
        else:
            rsi = 50.0

        return {
            "ticker":          ticker,
            "current_price":   round(float(closes[-1]), 2),
            "momentum_5d_pct": round(momentum_5d, 2),
            "trend":           ("UPTREND" if momentum_5d > 1
                                else "DOWNTREND" if momentum_5d < -1
                                else "SIDEWAYS"),
            "above_20d_ma":    bool(closes[-1] > ma20),
            "rsi_14":          rsi,
            "rsi_signal":      ("OVERBOUGHT" if rsi > 70
                                else "OVERSOLD" if rsi < 30
                                else "NEUTRAL")
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_intraday_change(ticker: str, hours: int = 1) -> dict:
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1h")
        if hist.empty or len(hist) < 2:
            return {"error": f"No intraday data for {ticker}"}

        periods    = min(hours, len(hist) - 1)
        price_now  = float(hist["Close"].iloc[-1])
        price_prev = float(hist["Close"].iloc[-1 - periods])
        change_pct = (price_now - price_prev) / price_prev * 100

        return {
            "ticker":        ticker,
            "current_price": round(price_now,  2),
            "change_pct":    round(change_pct, 3),
            "direction":     ("UP"   if change_pct >  0.3
                              else "DOWN" if change_pct < -0.3
                              else "FLAT"),
            "hours_back":    periods
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def execute_tool_call(name: str, args: dict) -> dict:
    if name == "get_technical_indicators":
        return get_technical_indicators(
            ticker        = args["ticker"],
            lookback_days = args.get("lookback_days", 20)
        )
    elif name == "get_intraday_change":
        return get_intraday_change(
            ticker = args["ticker"],
            hours  = args.get("hours", 1)
        )
    return {"error": f"Unknown tool: {name}"}


def run_with_tools(messages: list, max_rounds: int = 3) -> tuple:
    """
    Run a chat completion with automatic tool use.
    The model decides if and when to call tools.
    Handles the multi-turn loop automatically.
    Returns (final_text: str, tool_results: dict).
    """
    tool_results = {}
    msgs         = list(messages)

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0
        )
        choice = response.choices[0]

        if choice.finish_reason != "tool_calls":
            return choice.message.content or "", tool_results

        msgs.append(choice.message)
        for tc in choice.message.tool_calls:
            name   = tc.function.name
            args   = json.loads(tc.function.arguments)
            result = execute_tool_call(name, args)
            tool_results[name] = result
            msgs.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result)
            })

    # Safety fallback after max rounds
    final = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0
    )
    return final.choices[0].message.content or "", tool_results
