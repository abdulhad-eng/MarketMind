# MarketMind — Multi-Stage LLM Pipeline for Stock Price Direction Prediction

**Course:** 17-630 Prompt Engineering, Carnegie Mellon University, Spring 2026  
**Student:** Hadi Qureshi 
**Model:** gpt-4o-mini  


---

## What This Project Does

MarketMind predicts whether a stock will go **UP**, **DOWN**, or **NEUTRAL** over a 3-day horizon (historical mode) or 60-minute horizon (live mode). It combines four prompt engineering techniques into a sequential chain and was evaluated on 10 large-cap US stocks: AAPL, MSFT, NVDA, TSLA, AMZN, META, JPM, XOM, WMT, GOOGL.

The key finding: **zero-shot and RAG approaches produced 0 DOWN predictions out of 250.** The full pipeline produced 49 DOWN predictions with 40.8% accuracy. Function calling — which gave the model numerical evidence (RSI, momentum) independent of news sentiment — was the only technique that broke this directional blind spot.

---

## Architecture Overview

```
NEWS (yfinance + NewsAPI)
    │
    ▼
FAISS EMBED (all-MiniLM-L6-v2) → Vector Index (one per ticker)
    │
    ▼ Retrieve top-5 by cosine similarity
    │
┌───┴────────────────────────────────────────────────────────┐
│                  4-STEP DECOMPOSITION CHAIN                │
│                                                            │
│  Step 1: Summarize headlines → plain-English price signal  │
│      │ (output: news_summary)                              │
│      ▼                                                     │
│  Step 2: Technical analysis via FUNCTION CALLING           │
│      │ Model calls get_technical_indicators()              │
│      │ and/or get_intraday_change() autonomously           │
│      │ (output: tech_analysis + raw tool_results)          │
│      ▼                                                     │
│  Step 3: Build bull case (2 reasons) + bear case (2)       │
│      │ (input: news_summary + tech_analysis)               │
│      │ (output: argument)                                  │
│      ▼                                                     │
│  Step 4: Final prediction — UP/DOWN/NEUTRAL + confidence   │
│      │ (input: news_summary + tech_analysis + argument)    │
│                                                            │
└───┬────────────────────────────────────────────────────────┘
    │
    ▼
CoVe VERIFICATION
    │ Re-reads ALL intermediate outputs + raw tool data
    │ Checks: news summary ↔ headlines, tech analysis ↔ tool data,
    │         prediction ↔ combined evidence
    │ If inconsistency found → REVISED prediction
    │
    ▼
FINAL OUTPUT: UP / DOWN / NEUTRAL + HIGH / MEDIUM / LOW confidence
```

**Data flow detail:**
- Step 1 → Step 2: `news_summary` becomes the "News context" in Step 2's prompt
- Step 2 → Step 3: `tech_analysis` (text) flows forward; raw `tool_results` (JSON) is stored for CoVe only
- Step 3 → Step 4: `argument` flows forward alongside `news_summary` and `tech_analysis`
- CoVe receives everything: original headlines, news_summary, tool_results, tech_analysis, argument, and the prediction

---

## File Structure

```
marketmind_final/
├── pipeline/
│   ├── __init__.py
│   ├── fetch_data.py      # yfinance OHLCV prices + ground truth label computation
│   ├── fetch_news.py      # News from yfinance (primary) + NewsAPI (supplementary)
│   ├── rag.py             # FAISS index build + semantic retrieval
│   ├── sentiment.py       # LLM sentiment classification (Config 2: RAG + Sentiment)
│   ├── tools.py           # OpenAI function calling + yfinance tool implementations
│   ├── decompose.py       # 4-step decomposition chain (the core pipeline)
│   ├── verify.py          # Chain-of-Verification (CoVe)
│   └── evaluate.py        # Accuracy metrics + ablation table builder
├── data/                  # Cached CSVs: prices, labels, test_labels, news
├── results/               # Output CSVs: predictions, ablation, per_stock, live
├── app.py                 # Streamlit dashboard (5 tabs)
├── live_bot.py            # 60-minute live prediction bot
├── run_pipeline.py        # Historical ablation study runner
├── requirements.txt
├── .env.template          # OPENAI_API_KEY and NEWS_API_KEY placeholders
└── README.md              # This file
```

---

## Code Walkthrough

### `pipeline/rag.py` — FAISS Semantic Retrieval

**What it does:** Embeds news headlines using `all-MiniLM-L6-v2` from `sentence-transformers` and stores them in a FAISS vector index (one index per ticker). At prediction time, retrieves the 5 most semantically relevant headlines using cosine similarity.

**Why this matters:** In Iteration 1, I used date-based filtering (just grab headlines from the same day). That's not real RAG — it's just a date filter. Genuine RAG as defined by Lewis et al. (2020) uses semantic similarity to find relevant context regardless of when it was published. This change was necessary to meet the course requirement for retrieval-augmented generation.

**Design choice:** The embedding model is loaded once and cached in memory across all predictions. This avoids redundant model loading and keeps the pipeline fast (~$0 cost since embeddings are local, not API calls).

**Reflection:** Even with genuine semantic retrieval, RAG alone amplified the positive news bias rather than helping predict DOWN. This was surprising — I expected more context to help, but financial news structurally skews positive, and the model picked up on that skew.

### `pipeline/decompose.py` — 4-Step Decomposition Chain

**What it does:** Runs four sequential LLM calls for each prediction. Each step has a distinct role and its output feeds the next step.

**Step 1 (`step1_summarize_news`):** Takes raw headlines and produces a 1-2 sentence plain-English summary focused on price-relevant information. This compresses noisy, repetitive headlines into a clean signal.

**Step 2 (`step2_analyze_technicals`):** This is the critical innovation. The model is given two tool schemas via OpenAI's function calling API and autonomously decides when and how to call them. The tools are:
- `get_technical_indicators(ticker, lookback_days=20)` → RSI-14, 5-day momentum %, 20-day MA position, trend label, current price
- `get_intraday_change(ticker, hours=1)` → hourly price change % and direction

The model is **never forced** to call these tools — it decides based on context. This is real function calling in the style of Schick et al. (2023), not pre-computed data pasted into a prompt.

**Step 3 (`step3_build_argument`):** Takes the news summary and technical analysis and constructs an explicit bull case (2 reasons for UP) and bear case (2 reasons for DOWN). This forces the model to consider both sides before committing to a direction.

**Step 4 (`step4_make_prediction`):** Takes all three prior outputs and makes a final prediction in a structured format: PREDICTION (UP/DOWN/NEUTRAL), CONFIDENCE (HIGH/MEDIUM/LOW), and a one-line REASONING.

**Design choice — temperature 0:** All calls use `temperature=0` for deterministic, reproducible predictions. Stock prediction is not a creative task; we want the model's most likely assessment.

**Design choice — structured output parsing:** Step 4's response is parsed line-by-line looking for `PREDICTION:`, `CONFIDENCE:`, and `REASONING:` prefixes. If parsing fails, it defaults to NEUTRAL/LOW — a safe fallback.

**Reflection:** The 4-step chain is what made DOWN predictions possible. Without Step 2 (function calling), the model only had text to work with, and financial text skews positive. The numerical evidence from RSI and momentum gave the model something concrete to anchor a bearish prediction on.

### `pipeline/tools.py` — Function Calling Implementation

**What it does:** Implements the OpenAI function calling loop. Defines tool schemas, sends them to the API, executes tool calls against live yfinance data, and returns results to the model for synthesis.

**How it works:**
1. Tool schemas are defined as JSON following OpenAI's function calling spec
2. The model's response is checked for `tool_calls`
3. If tools are called, the corresponding Python function executes a live yfinance query
4. Tool results are appended to the conversation as `tool` role messages
5. The model is called again to synthesize the results into a technical analysis

**Why real function calling matters:** In Iteration 1, I pre-computed RSI and momentum externally and pasted the numbers into the prompt. That's not function calling — the model has no agency over what data to request. With real function calling, the model decides: "I need to check the technical indicators for this stock" and calls the tool itself. This is the distinction between tool augmentation (Schick et al., 2023) and static context injection.

### `pipeline/verify.py` — Chain-of-Verification (CoVe)

**What it does:** After the 4-step chain produces a prediction, CoVe runs a separate verification prompt that re-reads all evidence and checks for three types of inconsistencies:
1. Does the news summary accurately reflect the original headlines?
2. Is the technical analysis consistent with what the tools actually returned?
3. Is the final prediction logically justified by the combined evidence?

If a material inconsistency is found, the prediction is revised.

**Why end-of-chain, not mid-chain:** In Iteration 2, I tried a "confirmation gate" between Steps 1 and 2. It rejected valid signals and collapsed recall. This aligns with Laban et al. (2024) — challenging an LLM's output mid-process can degrade performance. CoVe at the end works better because it has the full picture.

**Real example (WMT, April 14, 2026):** Step 4 predicted UP based on Jim Cramer's bullish endorsement. But the raw tool data showed RSI 34.3, momentum -1.74%, DOWNTREND. CoVe caught this contradiction — bullish news but bearish technicals — and revised the prediction to DOWN. The actual result was DOWN. This is CoVe working exactly as designed.

**Reflection:** CoVe is effective but fires rarely — only 2-3 revisions per 250 predictions. It works as a safety net, not a primary accuracy driver. The WMT example proves its value, but it can't catch everything — the TSLA failure (sensational headline overwhelming mixed technicals) passed CoVe because the reasoning was internally consistent even though the conclusion was wrong.

### `pipeline/sentiment.py` — RAG Sentiment (Config 2)

**What it does:** Used in Config 2 (RAG + Sentiment) only. Takes the FAISS-retrieved headlines and classifies overall sentiment as POSITIVE, NEGATIVE, or NEUTRAL. This sentiment label feeds a single enriched prediction prompt.

**Why it's separate from Config 3:** Config 3 (Decomposition + CoVe) doesn't use this file at all. Config 3 uses `decompose.py` which has its own Step 1 (news summarization) that produces a richer output than a single sentiment label.

### `pipeline/fetch_data.py` — Price Data & Labels

**What it does:** Fetches OHLCV data from yfinance and computes ground truth labels. A stock is labeled UP if 3-day forward return exceeds +1.0%, DOWN if below -1.0%, NEUTRAL otherwise.

**Label distribution in test set:** DOWN: 101 (40%), UP: 90 (36%), NEUTRAL: 59 (24%). This is important context — DOWN is actually the most common label, which makes the fact that baseline and RAG predicted 0 DOWNs even more striking.

### `pipeline/fetch_news.py` — News Collection

**What it does:** Fetches headlines from yfinance (primary) and NewsAPI (supplementary). NewsAPI doesn't support date-range filtering, so all headlines are fetched and then embedded for semantic retrieval.

### `run_pipeline.py` — Historical Ablation Runner

**What it does:** Runs all three configurations (Baseline, RAG+Sentiment, Decomposition+CoVe) sequentially across the test set. Supports `--skip-fetch` to reuse cached price/news data.

### `live_bot.py` — Live Trading Bot

**What it does:** Runs predictions every 60 minutes during NYSE market hours (9:30 AM - 4:00 PM ET). Each cycle predicts all 10 stocks, waits 60 minutes, then scores against actual price movement.

**Simulation mode:** `--simulate` flag uses historical data so the bot can be tested any time, not just during market hours.

### `app.py` — Streamlit Dashboard

**What it does:** Bloomberg terminal-style dashboard with 5 tabs:
1. **Overview** — Pipeline architecture, ablation chart, per-label accuracy, label distribution
2. **Live Bot** — Accuracy per cycle (color-coded), expandable prediction cards with full reasoning
3. **Stock Explorer** — Candlestick chart with prediction markers overlaid
4. **All Predictions** — Filterable table by stock, label, configuration
5. **Accuracy** — Ablation bar chart, per-stock grouped bars

**Cycle sort fix:** Cycle IDs mix integers (5-14) and strings (SIM1-SIM3). A `sort_key` function prevents alphabetical sorting.

---

## Three Configurations (Ablation Study)

| Config | Description | Accuracy | DOWN Predictions |
|--------|-------------|----------|-----------------|
| 1. Baseline | Zero-shot: single prompt, ticker name only | 25.2% | 0 |
| 2. RAG + Sentiment | FAISS retrieves 5 headlines → sentiment label → enriched prompt | 33.2% | 0 |
| 3. Decomp + CoVe | 4-step chain with function calling + CoVe verification | 36.8% | 49 (40.8% acc.) |

---

## Iteration History

### Iteration 1 — Flat Single-Prompt Pipeline (REBUILT)
All three configs used a single prediction prompt. Technical indicators were pre-computed and pasted in. FAISS used date-based filtering. This was not genuine multi-step orchestration and was rebuilt from scratch.

### Iteration 2 — Confirmation Gate (REMOVED)
Added a gate prompt between Steps 1 and 2 to validate signal quality. The gate rejected valid signals on certain dates, collapsing recall. This aligns with the FlipFlop finding (Laban et al., 2024) — challenging an LLM's initial output can degrade performance. Removed entirely.

### Iteration 3 — Full Decomposition + Real RAG (FINAL)
Genuine FAISS semantic retrieval, real function calling, 4-step decomposition chain, CoVe verification. This is the deployed version.

**Model comparison:**
- gpt-4o-mini: **36.8%** (best)
- gpt-4o: 33.2% (worse — larger model over-reasons on noisy short-horizon data)

---

## Key Findings

1. **Function calling is the critical innovation.** It was the only technique that gave the model numerical evidence to generate DOWN predictions independent of news tone.

2. **Financial news structurally skews positive.** RAG amplified UP bias (150 UP predictions vs 43 baseline) rather than helping predict DOWN. Adding context without structure can hurt.

3. **Smaller models win on noisy tasks.** gpt-4o-mini outperformed gpt-4o (36.8% vs 33.2%).

4. **NEUTRAL is nearly unpredictable.** 13.6% accuracy (8/59). The prompt structure forces directional commitment.

5. **Per-stock results vary dramatically.** TSLA (+36pp) and GOOGL (+40pp) benefited most. XOM (+0pp) saw no improvement — oil prices drive XOM, not company news.

---

## Setup & Running

### Prerequisites
- Python 3.10+
- OpenAI API key
- NewsAPI key (optional, yfinance news works alone)

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy environment template and add your keys
cp .env.template .env
# Edit .env: add OPENAI_API_KEY and NEWS_API_KEY
```

### Running

```bash
# Historical ablation (fetches fresh data)
python run_pipeline.py

# Historical ablation (reuse cached CSVs)
python run_pipeline.py --skip-fetch

# Live bot simulation (works any time)
python live_bot.py --simulate --cycles 3

# Live bot real-time (NYSE hours: 9:30 AM - 4:00 PM ET)
python live_bot.py

# Dashboard
streamlit run app.py
```

---

## References

- Brown et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.
- Dhuliawala et al. (2024). "Chain-of-Verification Reduces Hallucination in LLMs." Findings of ACL 2024.
- Fama, E. (1970). "Efficient Capital Markets." Journal of Finance, 25(2).
- Gao et al. (2024). "Retrieval-Augmented Generation for LLMs: A Survey." arXiv:2312.10997.
- Khot et al. (2023). "Decomposed Prompting: A Modular Approach." ICLR 2023.
- Laban et al. (2024). "Are You Sure? The FlipFlop Experiment." arXiv:2311.08596.
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
- Schick et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." NeurIPS 2023.
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs." NeurIPS 2022.
