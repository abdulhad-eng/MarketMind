"""
verify.py
Chain-of-Verification (CoVe) for MarketMind.

After the 4-step decomposition chain produces a prediction, this step
re-reads the original evidence and checks the reasoning for inconsistencies.
If it finds one, the prediction is revised.

Reference: Dhuliawala et al. 2023 —
"Chain-of-Verification Reduces Hallucination in Large Language Models"
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def verify_prediction(chain_output: dict) -> dict:
    """
    Given the full chain output from run_full_chain(), run one verification pass.

    The verifier checks:
    1. Does the news summary accurately reflect the original headlines?
    2. Is the technical analysis consistent with what the tools returned?
    3. Is the final prediction logically justified by the combined evidence?

    Returns the chain_output dict with three added fields:
        verified_prediction: str  (may differ from original if revised)
        verification_note:   str
        was_revised:         bool
    """
    ticker        = chain_output["ticker"]
    news_summary  = chain_output["news_summary"]
    tech_analysis = chain_output["tech_analysis"]
    argument      = chain_output["argument"]
    prediction    = chain_output["prediction"]
    confidence    = chain_output["confidence"]
    reasoning     = chain_output["reasoning"]
    headlines     = chain_output.get("headlines_used", [])
    tool_results  = chain_output.get("tool_results", {})

    hl_text   = "\n".join(f"  - {h}" for h in headlines[:5]) if headlines else "  (none)"
    tool_text = json.dumps(tool_results, indent=2) if tool_results else "  (no tool data)"

    prompt = (
        f"You are verifying a prediction for {ticker}.\n\n"
        f"ORIGINAL PREDICTION: {prediction} (confidence: {confidence})\n"
        f"REASONING: {reasoning}\n\n"
        f"ORIGINAL HEADLINES:\n{hl_text}\n\n"
        f"NEWS SUMMARY PRODUCED: {news_summary}\n\n"
        f"TOOL DATA RETURNED:\n{tool_text}\n\n"
        f"TECHNICAL ANALYSIS PRODUCED: {tech_analysis}\n\n"
        f"BULL/BEAR ARGUMENT: {argument}\n\n"
        f"YOUR TASKS:\n"
        f"1. Does the news summary accurately reflect the headlines above?\n"
        f"2. Is the technical analysis consistent with the tool data above?\n"
        f"3. Is the prediction ({prediction}) logically justified by the combined evidence?\n\n"
        f"If the prediction is well-supported:\n"
        f"VERIFIED: {prediction}\n"
        f"NOTE: [brief confirmation]\n\n"
        f"If you find a material inconsistency:\n"
        f"REVISED: [UP or DOWN or NEUTRAL]\n"
        f"NOTE: [what you found and why you revised]"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous fact-checker for financial predictions. "
                        "Only revise if you find a genuine inconsistency in the evidence."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        content = f"Verification error: {e}"

    verified_prediction = prediction
    note                = content
    was_revised         = False

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("VERIFIED:"):
            verified_prediction = prediction
            was_revised         = False
        elif line.startswith("REVISED:"):
            v = line.replace("REVISED:", "").strip().upper()
            if v in ("UP", "DOWN", "NEUTRAL"):
                verified_prediction = v
                was_revised         = (v != prediction)
        elif line.startswith("NOTE:"):
            note = line.replace("NOTE:", "").strip()

    result                        = chain_output.copy()
    result["verified_prediction"] = verified_prediction
    result["verification_note"]   = note
    result["was_revised"]         = was_revised
    return result
