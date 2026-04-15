"""
decompose.py
4-Step Task Decomposition chain for MarketMind.

Each step is a separate LLM call. Output of each step feeds the next.
Step 2 uses real OpenAI function calling — the model autonomously
decides to call get_technical_indicators() and/or get_intraday_change().

Step 1: Summarize news into plain English sentiment
Step 2: Analyze technicals via function calling
Step 3: Build explicit bull and bear case from both signals
Step 4: Make final prediction with confidence and one-line reasoning
"""

from openai import OpenAI
from dotenv import load_dotenv
from pipeline.tools import run_with_tools

load_dotenv()
client = OpenAI()


def _call(system: str, user: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=0
    )
    return r.choices[0].message.content.strip()


def step1_summarize_news(ticker: str, headlines: list) -> str:
    if not headlines:
        return f"No recent news is available for {ticker}."
    hl_text = "\n".join(f"- {h}" for h in headlines[:6])
    return _call(
        system="You are a financial news analyst. Be concise and factual.",
        user=(
            f"Summarize the following headlines about {ticker} in 1-2 sentences.\n"
            f"Focus only on what is relevant to short-term price movement.\n\n"
            f"Headlines:\n{hl_text}\n\nSummary:"
        )
    )


def step2_analyze_technicals(ticker: str, news_summary: str) -> tuple:
    """
    The model autonomously calls tools to get live price data.
    We do not pre-compute or force tool calls — the model decides.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical analyst. Use the available tools to fetch "
                "live data for the stock. Then write a 2-3 sentence technical analysis."
            )
        },
        {
            "role": "user",
            "content": (
                f"Analyze the technical picture for {ticker}.\n\n"
                f"News context: {news_summary}\n\n"
                f"Use tools to get current technical data, then explain: "
                f"do the technicals support or contradict the news sentiment?"
            )
        }
    ]
    return run_with_tools(messages)


def step3_build_argument(ticker: str, news_summary: str, tech_analysis: str) -> str:
    return _call(
        system="You are a balanced equity analyst. Consider both sides objectively.",
        user=(
            f"For {ticker}:\n\n"
            f"News: {news_summary}\n\n"
            f"Technical analysis: {tech_analysis}\n\n"
            f"Write exactly:\n\n"
            f"BULL CASE (2 concrete reasons price could go UP):\n1.\n2.\n\n"
            f"BEAR CASE (2 concrete reasons price could go DOWN):\n1.\n2."
        )
    )


def step4_make_prediction(ticker, news_summary, tech_analysis, argument) -> dict:
    raw = _call(
        system="You are making a final stock prediction. Be decisive. Use only the evidence provided.",
        user=(
            f"Final prediction for {ticker} over the next 60 minutes:\n\n"
            f"News:         {news_summary}\n"
            f"Technicals:   {tech_analysis}\n"
            f"Bull vs Bear: {argument}\n\n"
            f"Respond in exactly this format:\n"
            f"PREDICTION: UP or DOWN or NEUTRAL\n"
            f"CONFIDENCE: HIGH or MEDIUM or LOW\n"
            f"REASONING: one sentence explaining the decisive factor"
        )
    )

    result = {"prediction": "NEUTRAL", "confidence": "LOW", "reasoning": raw}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("PREDICTION:"):
            v = line.replace("PREDICTION:", "").strip().upper()
            if v in ("UP", "DOWN", "NEUTRAL"):
                result["prediction"] = v
        elif line.startswith("CONFIDENCE:"):
            v = line.replace("CONFIDENCE:", "").strip().upper()
            if v in ("HIGH", "MEDIUM", "LOW"):
                result["confidence"] = v
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    return result


def run_full_chain(ticker: str, headlines: list) -> dict:
    """
    Run all four decomposition steps for one ticker.
    Stores every intermediate output for display in the dashboard.
    """
    news_summary                = step1_summarize_news(ticker, headlines)
    tech_analysis, tool_results = step2_analyze_technicals(ticker, news_summary)
    argument                    = step3_build_argument(ticker, news_summary, tech_analysis)
    prediction                  = step4_make_prediction(ticker, news_summary, tech_analysis, argument)

    return {
        "ticker":         ticker,
        "headlines_used": headlines,
        "news_summary":   news_summary,
        "tech_analysis":  tech_analysis,
        "tool_results":   tool_results,
        "argument":       argument,
        "prediction":     prediction["prediction"],
        "confidence":     prediction["confidence"],
        "reasoning":      prediction["reasoning"]
    }
