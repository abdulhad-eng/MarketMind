"""
app.py
MarketMind Dashboard — Bloomberg Terminal aesthetic.
Dark background, green/red price colors, monospace data typography.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="MarketMind",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")

TICKERS = ["AAPL","MSFT","NVDA","TSLA","AMZN",
           "META","JPM", "XOM", "WMT", "GOOGL"]

TICKER_NAMES = {
    "AAPL":"Apple",    "MSFT":"Microsoft","NVDA":"Nvidia",
    "TSLA":"Tesla",    "AMZN":"Amazon",   "META":"Meta",
    "JPM":"JPMorgan",  "XOM":"ExxonMobil","WMT":"Walmart","GOOGL":"Google"
}

SECTOR = {
    "AAPL":"TECH","MSFT":"TECH","NVDA":"SEMI","TSLA":"AUTO",
    "AMZN":"TECH","META":"TECH","JPM":"FIN","XOM":"ENRG",
    "WMT":"RTIL","GOOGL":"TECH"
}

UP_COLOR      = "#00FF88"
DOWN_COLOR    = "#FF3355"
NEUTRAL_COLOR = "#FFB800"
BG_COLOR      = "#0A0E1A"
CARD_COLOR    = "#0F1628"
BORDER_COLOR  = "#1E2D4A"
TEXT_PRIMARY  = "#E8EDF5"
TEXT_DIM      = "#4A6080"
ACCENT_BLUE   = "#1E88E5"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Barlow+Condensed:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    background-color: {BG_COLOR} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

[data-testid="stSidebar"] {{
    background: #080C18 !important;
    border-right: 1px solid {BORDER_COLOR} !important;
}}
[data-testid="stSidebar"] * {{
    color: {TEXT_PRIMARY} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BORDER_COLOR} !important;
    gap: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {TEXT_DIM} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 1.4rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase !important;
}}
.stTabs [aria-selected="true"] {{
    color: {UP_COLOR} !important;
    border-bottom: 2px solid {UP_COLOR} !important;
    background: rgba(0,255,136,0.04) !important;
}}

[data-testid="stMetric"] {{
    background: {CARD_COLOR} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_DIM} !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    background: {CARD_COLOR} !important;
    border: 1px solid {BORDER_COLOR} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}}

[data-testid="stExpander"] {{
    background: {CARD_COLOR} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 4px !important;
    margin-bottom: 4px !important;
}}
[data-testid="stExpander"] summary {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: {TEXT_PRIMARY} !important;
    padding: 0.6rem 1rem !important;
}}

.stButton > button {{
    background: transparent !important;
    border: 1px solid {BORDER_COLOR} !important;
    color: {TEXT_DIM} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 3px !important;
}}
.stButton > button:hover {{
    border-color: {UP_COLOR} !important;
    color: {UP_COLOR} !important;
}}

hr {{ border-color: {BORDER_COLOR} !important; margin: 0.8rem 0 !important; }}

.mm-logo {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {TEXT_PRIMARY};
    line-height: 1;
}}
.mm-logo span {{ color: {UP_COLOR}; }}
.mm-tagline {{
    font-size: 0.55rem;
    letter-spacing: 0.22em;
    color: {TEXT_DIM};
    text-transform: uppercase;
    margin-top: 0.25rem;
}}
.section-label {{
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    color: {TEXT_DIM};
    text-transform: uppercase;
    border-left: 2px solid {UP_COLOR};
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.7rem;
}}
.chain-box {{
    background: #070B15;
    border: 1px solid {BORDER_COLOR};
    border-left: 3px solid {ACCENT_BLUE};
    border-radius: 3px;
    padding: 0.75rem 1rem;
    margin: 0.35rem 0;
    font-size: 0.73rem;
    line-height: 1.65;
    color: #8BA0BC;
}}
.chain-step {{
    font-size: 0.54rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {ACCENT_BLUE};
    margin-bottom: 0.35rem;
    font-weight: 600;
}}
.chain-box.cove {{
    border-left-color: {NEUTRAL_COLOR};
}}
.chain-step.cove {{ color: {NEUTRAL_COLOR}; }}
.accuracy-badge {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}}
.accuracy-badge.good {{ color: {UP_COLOR}; }}
.accuracy-badge.mid  {{ color: {NEUTRAL_COLOR}; }}
.accuracy-badge.bad  {{ color: {DOWN_COLOR}; }}
.pred-result-correct {{ color: {UP_COLOR}; font-weight: 700; }}
.pred-result-wrong   {{ color: {DOWN_COLOR}; font-weight: 700; }}
.pred-result-pending {{ color: {TEXT_DIM}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#070B15",
    font=dict(family="JetBrains Mono", color=TEXT_PRIMARY),
    xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR,
               tickfont=dict(size=9, color=TEXT_DIM)),
    yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR,
               tickfont=dict(size=9, color=TEXT_DIM)),
    margin=dict(t=30, b=20, l=10, r=10),
    legend=dict(font=dict(size=10, family="JetBrains Mono"),
                bgcolor="rgba(0,0,0,0)", bordercolor=BORDER_COLOR)
)


@st.cache_data(ttl=60)
def load_data():
    d = {}
    for key, path in [
        ("predictions", RESULTS_DIR / "predictions.csv"),
        ("ablation",    RESULTS_DIR / "ablation.csv"),
        ("per_stock",   RESULTS_DIR / "per_stock.csv"),
        ("live",        RESULTS_DIR / "live_predictions.csv"),
        ("news",        DATA_DIR    / "news.csv"),
    ]:
        try:
            df = pd.read_csv(path)
            if "Date"      in df.columns: df["Date"]      = pd.to_datetime(df["Date"],      errors="coerce")
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            d[key] = df
        except FileNotFoundError:
            d[key] = None
    return d


def direction_color(v):
    v = str(v).upper()
    if v == "UP":      return UP_COLOR
    if v == "DOWN":    return DOWN_COLOR
    return NEUTRAL_COLOR

def accuracy_class(v):
    if v >= 45: return "good"
    if v >= 28: return "mid"
    return "bad"


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1rem 0 0.8rem;">
      <div class="mm-logo">MARKET<span>MIND</span></div>
      <div class="mm-tagline">AI Signal Engine · CMU 17-630</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">Pipeline Steps</div>', unsafe_allow_html=True)
    for num, label, color in [
        ("01", "FAISS Semantic RAG",           ACCENT_BLUE),
        ("02", "News Summarization",            TEXT_PRIMARY),
        ("03", "Function Calling (Technicals)", NEUTRAL_COLOR),
        ("04", "Bull / Bear Argument",          TEXT_PRIMARY),
        ("05", "Final Prediction",              TEXT_PRIMARY),
        ("06", "CoVe Verification",             UP_COLOR),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;
                    padding:0.3rem 0;border-bottom:1px solid {BORDER_COLOR};">
          <span style="color:{TEXT_DIM};font-size:0.58rem;min-width:18px;">{num}</span>
          <span style="color:{color};font-size:0.7rem;">{label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="section-label" style="margin-top:1.2rem;">System Config</div>', unsafe_allow_html=True)
    for k, v in [("MODEL","gpt-4o-mini"),("UNIVERSE","10 large-cap"),
                  ("LIVE HORIZON","60 min / cycle"),("HIST HORIZON","3 trading days"),
                  ("LABEL THRESHOLD","±1.0%")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.22rem 0;
                    font-size:0.62rem;border-bottom:1px solid {BORDER_COLOR};">
          <span style="color:{TEXT_DIM};">{k}</span>
          <span style="color:{TEXT_PRIMARY};font-weight:600;">{v}</span>
        </div>""", unsafe_allow_html=True)

    data = load_data()
    if data["live"] is not None:
        scored = data["live"].dropna(subset=["correct"])
        if not scored.empty:
            acc = scored["correct"].mean() * 100
            cls = accuracy_class(acc)
            st.markdown(f"""
            <div style="margin-top:1.2rem;text-align:center;
                        background:{CARD_COLOR};border:1px solid {BORDER_COLOR};
                        border-radius:4px;padding:1rem 0.8rem;">
              <div style="font-size:0.55rem;letter-spacing:0.18em;
                          color:{TEXT_DIM};text-transform:uppercase;margin-bottom:0.4rem;">
                Live Accuracy
              </div>
              <div class="accuracy-badge {cls}">{acc:.1f}%</div>
              <div style="font-size:0.6rem;color:{TEXT_DIM};margin-top:0.3rem;">
                {int(scored['correct'].sum())} / {len(scored)} scored
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⟳  REFRESH"):
        st.cache_data.clear()
        st.rerun()

data = load_data()

# ── HEADER ─────────────────────────────────────────────────────────────────────
col_logo, col_time = st.columns([3, 1])
with col_logo:
    st.markdown(f"""
    <div style="padding:1rem 0 0.4rem;border-bottom:1px solid {BORDER_COLOR};margin-bottom:1rem;">
      <div class="mm-logo" style="font-size:2rem;">
        MARKET<span>MIND</span>
        <span style="font-size:0.65rem;color:{TEXT_DIM};font-weight:400;
                     margin-left:1.2rem;letter-spacing:0.16em;
                     font-family:'JetBrains Mono',monospace;">
          RAG · DECOMPOSITION · FUNCTION CALLING · COVE
        </span>
      </div>
    </div>""", unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style="padding:1rem 0 0.4rem;text-align:right;
                border-bottom:1px solid {BORDER_COLOR};margin-bottom:1rem;">
      <div style="color:{UP_COLOR};font-size:0.65rem;font-weight:700;
                  letter-spacing:0.1em;">● SYSTEM LIVE</div>
      <div style="color:{TEXT_DIM};font-size:0.6rem;margin-top:0.2rem;">
        {datetime.now().strftime('%H:%M  %b %d %Y')}
      </div>
    </div>""", unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs([
    "OVERVIEW", "LIVE BOT", "STOCK EXPLORER", "ALL PREDICTIONS", "ACCURACY"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    if data["predictions"] is not None:
        preds = data["predictions"]
        base  = preds["Correct_Baseline"].mean() * 100
        rag   = preds["Correct_RAG"].mean()      * 100
        full  = preds["Correct_Tools"].mean()    * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BASELINE",          f"{base:.1f}%")
        c2.metric("+ RAG + SENTIMENT", f"{rag:.1f}%",  delta=f"{rag-base:+.1f}pp vs baseline")
        c3.metric("+ DECOMP + COVE",   f"{full:.1f}%", delta=f"{full-base:+.1f}pp vs baseline")
        c4.metric("TEST PREDICTIONS",  f"{len(preds)}")

    st.markdown(f'<div class="section-label">Architecture</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#070B15;border:1px solid {BORDER_COLOR};
                border-radius:4px;padding:1.2rem 1.5rem;
                font-size:0.72rem;line-height:2.1;color:#4A6888;
                font-family:'JetBrains Mono',monospace;">
<span style="color:{UP_COLOR};font-weight:600;">NEWS</span> (yfinance + NewsAPI)
  →  <span style="color:{ACCENT_BLUE};">FAISS EMBED</span>  →  Semantic Retrieval  (k=5 most relevant)
        │
        ├─  <span style="color:{TEXT_PRIMARY};font-weight:600;">STEP 1</span>  Summarize news into plain-English signal
        │
        ├─  <span style="color:{TEXT_PRIMARY};font-weight:600;">STEP 2</span>  Technical analysis  <span style="color:{NEUTRAL_COLOR};">[function calling]</span>
        │         <span style="color:{NEUTRAL_COLOR};">get_technical_indicators()</span>  ·  RSI · momentum · MA
        │         <span style="color:{NEUTRAL_COLOR};">get_intraday_change()</span>       ·  1-hour delta
        │
        ├─  <span style="color:{TEXT_PRIMARY};font-weight:600;">STEP 3</span>  Bull case  vs  Bear case
        │
        ├─  <span style="color:{TEXT_PRIMARY};font-weight:600;">STEP 4</span>  Final prediction  +  confidence
        │
        └─  <span style="color:{UP_COLOR};font-weight:700;">CoVe</span>  Verify reasoning  →  Revise if inconsistent
                    │
                    ▼
            <span style="color:{UP_COLOR};font-weight:700;">▲ UP</span>   <span style="color:{DOWN_COLOR};font-weight:700;">▼ DOWN</span>   <span style="color:{NEUTRAL_COLOR};font-weight:700;">● NEUTRAL</span>
    </div>""", unsafe_allow_html=True)

    if data["predictions"] is not None:
        preds = data["predictions"]

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(f'<div class="section-label">Ablation Study</div>', unsafe_allow_html=True)
            cfgs   = ["Baseline", "+ RAG + Sentiment", "+ Decomp + CoVe"]
            accs   = [base, rag, full]
            colors = [ACCENT_BLUE, NEUTRAL_COLOR, UP_COLOR]

            fig = go.Figure()
            for cfg, acc, col in zip(cfgs, accs, colors):
                fig.add_trace(go.Bar(
                    x=[cfg], y=[acc],
                    marker_color=col, marker_line_width=0, width=0.45,
                    text=[f"{acc:.1f}%"],
                    textposition="outside",
                    textfont=dict(size=15, color=col, family="Barlow Condensed"),
                    name=cfg
                ))
            fig.update_layout(
                showlegend=False, height=260,
                yaxis=dict(range=[0, 50], ticksuffix="%",
                           **PLOTLY_LAYOUT["yaxis"]),
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"]),
                bargap=0.5,
                **{k:v for k,v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis","yaxis","legend")}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(f'<div class="section-label">Per-Label Accuracy (Full Chain)</div>',
                        unsafe_allow_html=True)
            for label, col in [("UP", UP_COLOR), ("DOWN", DOWN_COLOR), ("NEUTRAL", NEUTRAL_COLOR)]:
                sub = preds[preds["Label"] == label]
                if sub.empty: continue
                acc = sub["Correct_Tools"].mean() * 100
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.7rem;margin-bottom:0.25rem;">
                    <span style="color:{col};font-weight:700;letter-spacing:0.05em;">{label}</span>
                    <span style="color:{TEXT_PRIMARY};">{acc:.1f}%
                      <span style="color:{TEXT_DIM};font-size:0.6rem;">
                        ({int(sub['Correct_Tools'].sum())}/{len(sub)})
                      </span>
                    </span>
                  </div>
                  <div style="background:{BORDER_COLOR};height:5px;border-radius:3px;">
                    <div style="background:{col};width:{int(acc)}%;height:5px;
                                border-radius:3px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="section-label" style="margin-top:1rem;">Label Distribution</div>',
                        unsafe_allow_html=True)
            lc = preds["Label"].value_counts()
            total = len(preds)
            for label, col in [("DOWN", DOWN_COLOR), ("UP", UP_COLOR), ("NEUTRAL", NEUTRAL_COLOR)]:
                n   = lc.get(label, 0)
                pct = n / total * 100
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            font-size:0.65rem;padding:0.2rem 0;
                            border-bottom:1px solid {BORDER_COLOR};">
                  <span style="color:{col};">{label}</span>
                  <span style="color:{TEXT_PRIMARY};">{n}
                    <span style="color:{TEXT_DIM};">({pct:.0f}%)</span>
                  </span>
                </div>""", unsafe_allow_html=True)

    if data["live"] is not None:
        scored = data["live"].dropna(subset=["correct"])
        if not scored.empty:
            st.divider()
            st.markdown(f'<div class="section-label">Live Bot Summary</div>', unsafe_allow_html=True)
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("LIVE ACCURACY",      f"{scored['correct'].mean()*100:.1f}%")
            lc2.metric("PREDICTIONS SCORED",  len(scored))
            lc3.metric("COVE REVISIONS",
                       int(data["live"]["was_revised"].sum())
                       if "was_revised" in data["live"].columns else 0)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE BOT
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    if data["live"] is None or data["live"].empty:
        st.markdown(f"""
        <div style="text-align:center;padding:4rem 2rem;
                    border:1px solid {BORDER_COLOR};border-radius:4px;
                    background:{CARD_COLOR};margin-top:1rem;">
          <div style="font-size:0.7rem;letter-spacing:0.18em;
                      color:{TEXT_DIM};text-transform:uppercase;">
            No live data yet
          </div>
          <div style="font-size:0.65rem;color:#1E3050;margin-top:0.8rem;
                      font-family:'JetBrains Mono',monospace;">
            python live_bot.py
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        ldf    = data["live"].copy()
        scored = ldf.dropna(subset=["correct"])

        c1, c2, c3, c4 = st.columns(4)
        acc = scored["correct"].mean()*100 if not scored.empty else 0
        c1.metric("LIVE ACCURACY",      f"{acc:.1f}%")
        c2.metric("PREDICTIONS SCORED",  len(scored))
        c3.metric("TOTAL PREDICTIONS",   len(ldf))
        c4.metric("COVE REVISIONS",
                  int(ldf["was_revised"].sum()) if "was_revised" in ldf.columns else 0)

        if not scored.empty and "cycle_id" in scored.columns:
            st.markdown(f'<div class="section-label">Accuracy Per Cycle</div>',
                        unsafe_allow_html=True)
            ca = (scored.groupby("cycle_id")["correct"].mean()*100).reset_index()
            ca.columns = ["Cycle","Accuracy"]
            def sort_key(v):
                s = str(v)
                if s.startswith("SIM"):
                     return (1, int(s.replace("SIM","")))
                try:
                     return (0, int(s))
                except:
                    return (0, 0)
            ca = ca.iloc[sorted(range(len(ca)), key=lambda i: sort_key(ca["Cycle"].iloc[i]))]
            ca["Cycle"] = ca["Cycle"].astype(str)

            fig = go.Figure(go.Bar(
                x=ca["Cycle"].astype(str), y=ca["Accuracy"],
                marker_color=[UP_COLOR if a>=40 else NEUTRAL_COLOR if a>=25 else DOWN_COLOR
                              for a in ca["Accuracy"]],
                marker_line_width=0, width=0.5,
                text=[f"{a:.0f}%" for a in ca["Accuracy"]],
                textposition="outside",
                textfont=dict(size=12, family="Barlow Condensed")
            ))
            fig.update_layout(
                height=200, showlegend=False,
                yaxis=dict(range=[0,100], ticksuffix="%", **PLOTLY_LAYOUT["yaxis"]),
                xaxis=dict(**PLOTLY_LAYOUT["xaxis"]),
                **{k:v for k,v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis","yaxis","legend")}
            )
            st.plotly_chart(fig, use_container_width=True)

        latest_cycle = ldf["cycle_id"].iloc[-1]
        latest       = ldf[ldf["cycle_id"] == latest_cycle]
        st.markdown(f'<div class="section-label">Latest Cycle — {latest_cycle}</div>',
                    unsafe_allow_html=True)

        for _, row in latest.iterrows():
            pred    = str(row.get("verified_prediction","—")).upper()
            actual  = str(row.get("actual_label","")).upper()
            correct = row.get("correct", None)
            conf    = str(row.get("confidence","—")).upper()
            revised = bool(row.get("was_revised", False))
            dc      = direction_color(pred)
            is_pending = (correct is None or str(correct) == "nan"
                          or actual in ("NONE","NAN",""))

            title = (f"{row['ticker']} ({SECTOR.get(row['ticker'],'')})  "
                     f"—  {pred}  ·  {conf}"
                     f"  {'⟲ CoVe revised' if revised else ''}")

            with st.expander(title, expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div style="font-size:0.72rem;line-height:2;">
                      <div><span style="color:{TEXT_DIM};">PREDICTION  </span>
                        <span style="color:{dc};font-weight:700;font-size:1rem;"> {pred}</span>
                      </div>
                      <div><span style="color:{TEXT_DIM};">CONFIDENCE  </span>
                        <span style="color:{'#00FF88' if conf=='HIGH' else '#FFB800' if conf=='MEDIUM' else TEXT_DIM};">
                          {conf}
                        </span>
                      </div>
                      <div><span style="color:{TEXT_DIM};">COVE REVISED  </span>
                        <span style="color:{'#FFB800' if revised else TEXT_DIM};">
                          {'YES' if revised else 'NO'}
                        </span>
                      </div>
                      <div><span style="color:{TEXT_DIM};">ACTUAL  </span>
                        <span style="color:{direction_color(actual) if not is_pending else TEXT_DIM};">
                          {'PENDING ⏳' if is_pending else actual}
                        </span>
                      </div>
                      {"" if is_pending else
                       f'<div><span style="color:{TEXT_DIM};">RESULT  </span>' +
                       (f'<span class="pred-result-correct">✓ CORRECT</span></div>'
                        if correct else
                        f'<span class="pred-result-wrong">✗ WRONG</span></div>')}
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div style="font-size:0.7rem;color:#8BA0BC;line-height:1.7;">
                      <div style="color:{TEXT_DIM};font-size:0.56rem;
                                  letter-spacing:0.14em;text-transform:uppercase;
                                  margin-bottom:0.3rem;">REASONING</div>
                      {row.get('reasoning','—')}
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="chain-box">
                  <div class="chain-step">STEP 1 — NEWS SUMMARY</div>
                  {row.get('news_summary','—')}
                </div>
                <div class="chain-box">
                  <div class="chain-step">STEP 2 — TECHNICAL ANALYSIS (FUNCTION CALLING)</div>
                  {row.get('tech_analysis','—')}
                </div>
                <div class="chain-box">
                  <div class="chain-step">STEP 3 — BULL / BEAR ARGUMENT</div>
                  {row.get('argument','—')}
                </div>
                """, unsafe_allow_html=True)

                if revised:
                    st.markdown(f"""
                    <div class="chain-box cove">
                      <div class="chain-step cove">COVE — REVISION MADE</div>
                      {row.get('verification_note','—')}
                    </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="section-label">Prediction History</div>', unsafe_allow_html=True)
        disp = ldf[["cycle_id","ticker","verified_prediction","confidence",
                    "actual_label","correct","was_revised"]].copy()
        disp["correct"] = disp["correct"].map({True:"✓",False:"✗",None:"⏳"}).fillna("⏳")
        disp.columns = ["CYCLE","TICKER","PREDICTION","CONF","ACTUAL","RESULT","REVISED"]
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — STOCK EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        st.markdown(f'<div class="section-label">Stock</div>', unsafe_allow_html=True)
        sel = st.selectbox("", TICKERS,
                           format_func=lambda x: f"{x}  {TICKER_NAMES[x]}",
                           label_visibility="collapsed")
        cfg = st.radio("", ["Baseline","RAG + Sentiment","Decomp + CoVe"],
                       label_visibility="collapsed")

        if data["predictions"] is not None:
            sub = data["predictions"][data["predictions"]["Ticker"] == sel]
            if not sub.empty:
                st.markdown(f'<div class="section-label">Accuracy</div>', unsafe_allow_html=True)
                for name, col in [("BASELINE","Correct_Baseline"),
                                   ("RAG","Correct_RAG"),
                                   ("FULL CHAIN","Correct_Tools")]:
                    if col in sub.columns:
                        a = sub[col].mean() * 100
                        c = UP_COLOR if a >= 40 else NEUTRAL_COLOR if a >= 25 else DOWN_COLOR
                        st.markdown(f"""
                        <div style="margin-bottom:0.5rem;">
                          <div style="display:flex;justify-content:space-between;font-size:0.64rem;">
                            <span style="color:{TEXT_DIM};">{name}</span>
                            <span style="color:{c};font-weight:700;">{a:.1f}%</span>
                          </div>
                          <div style="background:{BORDER_COLOR};height:3px;border-radius:2px;margin-top:0.15rem;">
                            <div style="background:{c};width:{int(a)}%;height:3px;border-radius:2px;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

    cmap = {
        "Baseline":        ("Pred_Baseline", "Correct_Baseline"),
        "RAG + Sentiment": ("Pred_RAG",      "Correct_RAG"),
        "Decomp + CoVe":   ("Pred_Tools",    "Correct_Tools")
    }
    pcol, ccol = cmap[cfg]

    with cc2:
        try:
            hist = yf.Ticker(sel).history(start="2026-03-01", end="2026-04-14")
            fig  = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"],   close=hist["Close"],
                name="PRICE",
                increasing=dict(line=dict(color=UP_COLOR,   width=1), fillcolor=UP_COLOR),
                decreasing=dict(line=dict(color=DOWN_COLOR, width=1), fillcolor=DOWN_COLOR)
            ))

            if data["predictions"] is not None:
                sub = data["predictions"][data["predictions"]["Ticker"] == sel]
                for _, row in sub.iterrows():
                    pred    = str(row.get(pcol,"NEUTRAL")).upper()
                    correct = row.get(ccol, False)
                    sym = ("triangle-up" if pred=="UP" else
                           "triangle-down" if pred=="DOWN" else "circle")
                    fig.add_trace(go.Scatter(
                        x=[row["Date"]], y=[row.get("Close")],
                        mode="markers",
                        marker=dict(symbol=sym, size=10,
                                    color=direction_color(pred),
                                    line=dict(color=UP_COLOR if correct else DOWN_COLOR, width=2)),
                        showlegend=False,
                        hovertemplate=(f"<b>%{{x}}</b><br>"
                                       f"Pred: {pred}<br>"
                                       f"Actual: {row.get('Label','')}<br>"
                                       f"{'✓' if correct else '✗'}<extra></extra>")
                    ))

            fig.update_layout(
                title=dict(
                    text=f"{sel}  ·  {TICKER_NAMES[sel]}  ·  MAR 2026",
                    font=dict(family="JetBrains Mono", size=11, color=TEXT_DIM)
                ),
                height=420, xaxis_rangeslider_visible=False,
                xaxis=dict(type="date", **PLOTLY_LAYOUT["xaxis"]),
                yaxis=dict(tickprefix="$", **PLOTLY_LAYOUT["yaxis"]),
                **{k:v for k,v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis","yaxis","legend")}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    if data["predictions"] is not None:
        sub = data["predictions"][data["predictions"]["Ticker"]==sel].copy()
        if "Date" in sub.columns:
            sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
        sub["RESULT"] = sub[ccol].map({True:"✓ CORRECT", False:"✗ WRONG"})
        cols = ["Date","Label",pcol,"RESULT","Pct_Change"]
        cols = [c for c in cols if c in sub.columns]
        st.dataframe(sub[cols].rename(columns={pcol:"PREDICTED","Label":"ACTUAL",
                                                "Date":"DATE","Pct_Change":"3D CHG%"}),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALL PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    if data["predictions"] is None:
        st.warning("No predictions. Run python run_pipeline.py first.")
    else:
        preds = data["predictions"].copy()
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            tf = st.multiselect("STOCKS", TICKERS, default=TICKERS, label_visibility="collapsed")
        with fc2:
            lf = st.multiselect("LABELS", ["UP","DOWN","NEUTRAL"],
                                default=["UP","DOWN","NEUTRAL"], label_visibility="collapsed")
        with fc3:
            cs = st.selectbox("CONFIG", ["Baseline","RAG + Sentiment","Decomp + CoVe"],
                              label_visibility="collapsed")

        cm = {
            "Baseline":        ("Pred_Baseline", "Correct_Baseline"),
            "RAG + Sentiment": ("Pred_RAG",      "Correct_RAG"),
            "Decomp + CoVe":   ("Pred_Tools",    "Correct_Tools")
        }
        pc, cc = cm[cs]
        filt = preds[preds["Ticker"].isin(tf) & preds["Label"].isin(lf)].copy()
        if "Date" in filt.columns:
            filt["Date"] = filt["Date"].dt.strftime("%Y-%m-%d")
        filt["RESULT"] = filt[cc].map({True:"✓ CORRECT", False:"✗ WRONG"})
        show = ["Date","Ticker","Label",pc,"RESULT","Pct_Change"]
        show = [c for c in show if c in filt.columns]
        st.dataframe(filt[show].rename(columns={pc:"PREDICTED","Ticker":"TICKER",
                                                 "Label":"ACTUAL","Date":"DATE",
                                                 "Pct_Change":"3D CHG%"}),
                     use_container_width=True, hide_index=True)
        if len(filt):
            acc = filt[cc].mean() * 100
            cls = accuracy_class(acc)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1.2rem;margin-top:0.8rem;
                        padding:0.8rem 1.2rem;background:{CARD_COLOR};
                        border:1px solid {BORDER_COLOR};border-radius:4px;">
              <div class="accuracy-badge {cls}">{acc:.1f}%</div>
              <div style="font-size:0.68rem;color:{TEXT_DIM};">
                {cs} on {len(filt)} filtered predictions
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ACCURACY
# ════════════════════════════════════════════════════════════════════════════════
with t5:
    if data["predictions"] is None:
        st.warning("No predictions. Run python run_pipeline.py first.")
    else:
        preds = data["predictions"]

        st.markdown(f'<div class="section-label">Ablation Study — Each Technique\'s Contribution</div>',
                    unsafe_allow_html=True)
        cfgs  = ["Baseline","+ RAG + Sentiment","+ Decomp + CoVe"]
        accs  = [preds["Correct_Baseline"].mean()*100,
                 preds["Correct_RAG"].mean()     *100,
                 preds["Correct_Tools"].mean()   *100]

        fig = go.Figure()
        for cfg, acc, col in zip(cfgs, accs, [ACCENT_BLUE, NEUTRAL_COLOR, UP_COLOR]):
            fig.add_trace(go.Bar(
                x=[cfg], y=[acc],
                marker_color=col, marker_line_width=0, width=0.4,
                text=[f"{acc:.1f}%"],
                textposition="outside",
                textfont=dict(size=16, color=col, family="Barlow Condensed"),
                name=cfg
            ))
        fig.update_layout(
            showlegend=False, height=290,
            yaxis=dict(range=[0,50], ticksuffix="%", **PLOTLY_LAYOUT["yaxis"]),
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"]),
            bargap=0.5,
            **{k:v for k,v in PLOTLY_LAYOUT.items()
               if k not in ("xaxis","yaxis","legend")}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="section-label">Per-Stock Accuracy</div>', unsafe_allow_html=True)
        srows = []
        for t in TICKERS:
            s = preds[preds["Ticker"] == t]
            if s.empty: continue
            srows.append({
                "Ticker":     t,
                "Baseline":   s["Correct_Baseline"].mean()*100,
                "RAG":        s["Correct_RAG"].mean()     *100,
                "Full Chain": s["Correct_Tools"].mean()   *100
            })
        sdf = pd.DataFrame(srows)

        fig2 = go.Figure()
        for col, color, name in zip(
            ["Baseline","RAG","Full Chain"],
            [ACCENT_BLUE, NEUTRAL_COLOR, UP_COLOR],
            ["Baseline","+ RAG","+ Decomp + CoVe"]
        ):
            fig2.add_trace(go.Bar(
                name=name, x=sdf["Ticker"], y=sdf[col],
                marker_color=color, marker_line_width=0
            ))
        fig2.update_layout(
            barmode="group",
            yaxis=dict(range=[0,70], ticksuffix="%", **PLOTLY_LAYOUT["yaxis"]),
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"]),
            height=380,
            **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")}
        )
        st.plotly_chart(fig2, use_container_width=True)

        if data["per_stock"] is not None:
            st.markdown(f'<div class="section-label">Summary Table</div>', unsafe_allow_html=True)
            st.dataframe(data["per_stock"], use_container_width=True, hide_index=True)