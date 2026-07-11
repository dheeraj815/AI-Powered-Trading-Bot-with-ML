import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import ta
import warnings
import json
import math
import time
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QUANT.AI — Algorithmic Trading Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL THEME — PROFESSIONAL DARK TERMINAL
# ═══════════════════════════════════════════════════════════════
THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&family=Syne:wght@400..800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Backgrounds */
    --void:         #020408;
    --bg0:          #060a12;
    --bg1:          #0a0f1e;
    --bg2:          #0e1428;
    --bg3:          #131a30;
    --bg4:          #192038;
    --bgHover:      #1f2842;

    /* Borders */
    --bdr:          rgba(255,255,255,0.055);
    --bdrG:         rgba(0,255,148,0.22);
    --bdrR:         rgba(255,50,80,0.22);
    --bdrB:         rgba(56,139,253,0.22);

    /* Colors */
    --green:        #00ff94;
    --greenDim:     #00c970;
    --greenGlow:    rgba(0,255,148,0.1);
    --greenGlow2:   rgba(0,255,148,0.04);
    --red:          #ff3250;
    --redDim:       #cc2840;
    --redGlow:      rgba(255,50,80,0.1);
    --amber:        #ffb347;
    --amberGlow:    rgba(255,179,71,0.1);
    --blue:         #388bfd;
    --blueGlow:     rgba(56,139,253,0.1);
    --cyan:         #00c8e0;
    --purple:       #b06eff;
    --pink:         #ff6eb4;

    /* Text */
    --t1:           #dde4f0;
    --t2:           #7e8fa8;
    --t3:           #3d4e65;
    --t4:           #1e2a38;

    /* Fonts */
    --mono:         'JetBrains Mono', monospace;
    --display:      'Syne', sans-serif;
    --body:         'Inter', sans-serif;

    /* Sizes */
    --r-sm:         4px;
    --r-md:         8px;
    --r-lg:         12px;
    --r-xl:         16px;
}

/* ── RESET ── */
* { box-sizing: border-box; }
html, body, .stApp { background: var(--void) !important; }
.main .block-container {
    padding: 0.75rem 1.25rem 2rem !important;
    max-width: 1680px !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg0) !important;
    border-right: 1px solid var(--bdrG) !important;
    width: 290px !important;
}
section[data-testid="stSidebar"] > div { padding: 1rem 0.9rem !important; }

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4 { font-family: var(--display) !important; letter-spacing: -0.02em; }
h1 { font-size: 1.55rem !important; font-weight: 800 !important; color: var(--t1) !important; }
h2 { font-size: 0.8rem !important; font-weight: 700 !important; color: var(--t2) !important;
     text-transform: uppercase; letter-spacing: 0.12em; }
p, span, div, li { color: var(--t1); font-family: var(--body) !important; }
.stMarkdown, .stMarkdown p { font-size: 0.9rem; }
label { font-family: var(--mono) !important; font-size: 0.68rem !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important;
        color: var(--t3) !important; font-weight: 600 !important; }

/* ── METRICS ── */
div[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: var(--r-lg) !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.2s;
}
div[data-testid="stMetric"]:hover { border-color: var(--bdrG) !important; }
div[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: var(--green) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--t3) !important;
}
div[data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 0.8rem !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--bdrG) !important;
    color: var(--green) !important;
    border-radius: var(--r-md) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 0.9rem !important;
    transition: all 0.15s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: var(--greenGlow) !important;
    border-color: var(--green) !important;
    box-shadow: 0 0 14px var(--greenGlow), 0 0 2px var(--green) !important;
}
.stButton > button[kind="primary"] {
    background: var(--green) !important;
    color: var(--void) !important;
    border-color: var(--green) !important;
    font-weight: 800 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--greenDim) !important;
    box-shadow: 0 0 24px rgba(0,255,148,0.35) !important;
}

/* ── INPUTS ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: var(--r-md) !important;
    color: var(--t1) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px var(--greenGlow) !important;
    outline: none !important;
}
.stSelectbox [data-baseweb="select"] > div { background: var(--bg2) !important; border-color: var(--bdr) !important; }
.stSelectbox [data-baseweb="popover"] > div { background: var(--bg3) !important; border: 1px solid var(--bdrG) !important; }
.stSelectbox [data-baseweb="option"] { background: var(--bg3) !important; }
.stSelectbox [data-baseweb="option"]:hover { background: var(--bg4) !important; }

/* ── SLIDERS ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--green) !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color: var(--green) !important; font-family: var(--mono) !important; font-size: 0.7rem !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--bdr) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--t3) !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    padding: 0.75rem 1.25rem !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--t1) !important; background: var(--bg2) !important; }
.stTabs [aria-selected="true"] {
    border-bottom-color: var(--green) !important;
    color: var(--green) !important;
    background: var(--greenGlow2) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border: 1px solid var(--bdr) !important; border-radius: var(--r-lg) !important; overflow: hidden !important; }
.dvn-scroller { background: var(--bg2) !important; }

/* ── EXPANDERS ── */
[data-testid="stExpander"] details {
    background: var(--bg2) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: var(--r-lg) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--t2) !important;
    font-weight: 700 !important;
}

/* ── ALERTS ── */
.stSuccess  { background: rgba(0,255,148,0.07) !important; border-left: 3px solid var(--green) !important; border-radius: var(--r-md) !important; }
.stError    { background: rgba(255,50,80,0.07) !important; border-left: 3px solid var(--red) !important; border-radius: var(--r-md) !important; }
.stWarning  { background: rgba(255,179,71,0.07) !important; border-left: 3px solid var(--amber) !important; border-radius: var(--r-md) !important; }
.stInfo     { background: rgba(56,139,253,0.07) !important; border-left: 3px solid var(--blue) !important; border-radius: var(--r-md) !important; }

/* ── PROGRESS ── */
.stProgress > div > div > div > div { background: var(--green) !important; border-radius: 2px !important; }
.stProgress > div > div { background: var(--bg3) !important; border-radius: 2px !important; height: 3px !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--t3); }

/* ── CUSTOM COMPONENTS ── */
.sec-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--t3);
    border-bottom: 1px solid var(--bdr);
    padding-bottom: 0.4rem;
    margin-bottom: 0.75rem;
}

.kpi-box {
    background: var(--bg2);
    border: 1px solid var(--bdr);
    border-radius: var(--r-lg);
    padding: 0.85rem 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.kpi-box:hover { border-color: var(--bdrG); box-shadow: 0 0 20px var(--greenGlow2); }
.kpi-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--t3);
    margin-bottom: 0.3rem;
}
.kpi-val {
    font-family: var(--mono);
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--t1);
}
.kpi-pos { color: var(--green) !important; }
.kpi-neg { color: var(--red) !important; }
.kpi-neu { color: var(--amber) !important; }

.signal-card {
    border-radius: var(--r-xl);
    padding: 1.5rem 1rem;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
    font-family: var(--mono);
    margin-bottom: 0.65rem;
}
.signal-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.06;
    background: radial-gradient(circle at 50% 0%, currentColor 0%, transparent 65%);
}
.sig-buy  { background: rgba(0,255,148,0.04); border-color: rgba(0,255,148,0.4); color: var(--green); }
.sig-sell { background: rgba(255,50,80,0.04); border-color: rgba(255,50,80,0.4); color: var(--red); }
.sig-hold { background: rgba(255,179,71,0.04); border-color: rgba(255,179,71,0.4); color: var(--amber); }

.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.pill-active {
    background: rgba(0,255,148,0.1);
    border: 1px solid rgba(0,255,148,0.4);
    color: var(--green);
}
.pill-active::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: blink 1.4s infinite;
}
.pill-inactive {
    background: rgba(255,50,80,0.1);
    border: 1px solid rgba(255,50,80,0.35);
    color: var(--red);
}
.pill-inactive::before { content: ''; width: 6px; height: 6px; background: var(--red); border-radius: 50%; }

@keyframes blink {
    0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(0,255,148,0.5); }
    50%      { opacity:0.6; box-shadow:0 0 0 4px rgba(0,255,148,0); }
}

.trade-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.8rem;
    background: var(--bg2);
    border: 1px solid var(--bdr);
    border-radius: var(--r-md);
    margin-bottom: 3px;
    font-family: var(--mono);
    font-size: 0.73rem;
    transition: border-color 0.15s;
}
.trade-row:hover { border-color: var(--bdrG); }

.ind-bar { background: var(--bg0); border-radius: 2px; height: 5px; overflow: hidden; margin-top: 0.3rem; }
.ind-fill { height: 100%; border-radius: 2px; }

.order-box-buy  { background: rgba(0,255,148,0.04); border: 1px solid rgba(0,255,148,0.2); border-radius: var(--r-lg); padding: 0.85rem; margin-bottom: 0.55rem; }
.order-box-sell { background: rgba(255,50,80,0.04); border: 1px solid rgba(255,50,80,0.2); border-radius: var(--r-lg); padding: 0.85rem; margin-bottom: 0.55rem; }
.order-box-hold { background: rgba(255,179,71,0.04); border: 1px solid rgba(255,179,71,0.2); border-radius: var(--r-lg); padding: 0.85rem; margin-bottom: 0.55rem; text-align: center; }

.ensemble-card {
    background: var(--bg2);
    border: 1px solid var(--bdr);
    border-radius: var(--r-lg);
    padding: 1rem;
    border-left: 3px solid;
}

.news-card {
    background: var(--bg2);
    border: 1px solid var(--bdr);
    border-radius: var(--r-lg);
    padding: 0.85rem 1rem;
    margin-bottom: 4px;
    transition: border-color 0.15s;
}
.news-card:hover { border-color: var(--bdrB); }

.scan-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.9rem;
    background: var(--bg2);
    border: 1px solid var(--bdr);
    border-radius: var(--r-md);
    margin-bottom: 3px;
    font-family: var(--mono);
    font-size: 0.73rem;
}

.heatmap-cell {
    border-radius: var(--r-sm);
    padding: 0.5rem;
    text-align: center;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════
_defaults = {
    "portfolio_value": 100_000.0,
    "cash":            100_000.0,
    "positions":       {},      # {symbol: shares}
    "avg_costs":       {},      # {symbol: avg_cost}
    "trade_history":   [],
    "bot_active":      False,
    "initial_capital": 100_000.0,
    "trade_count":     0,
    "win_count":       0,
    "loss_count":      0,
    "max_drawdown":    0.0,
    "peak_value":      100_000.0,
    "watchlist":       ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY"],
    "alerts":          [],
    "last_scan_time":  None,
    "portfolio_history": [100_000.0],
    "portfolio_dates":   [datetime.now().strftime("%Y-%m-%d")],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# DATA FETCHING (CACHED)
# ═══════════════════════════════════════════════════════════════


@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol: str, period: str = "1y"):
    """Fetch OHLCV history for a symbol. Returns None on failure."""
    try:
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=True,
            prepost=True,
        )
        if df is None or df.empty:
            return None
        df = df.dropna()
        if len(df) < 10:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Yahoo Finance Error: {e}")
        return None


@st.cache_data(ttl=180, show_spinner=False)
def fetch_info(symbol: str) -> dict:
    defaults = {"name": symbol, "sector": "N/A", "industry": "N/A",
                "price": 0.0, "change": 0.0, "volume": 0, "mktcap": 0,
                "hi52": 0.0, "lo52": 0.0, "pe": 0.0, "div": 0.0,
                "beta": 1.0, "eps": 0.0, "fwd_pe": 0.0, "peg": 0.0,
                "ps": 0.0, "pb": 0.0, "roe": 0.0, "profit_margin": 0.0,
                "revenue": 0, "gross_margin": 0.0, "debt_equity": 0.0,
                "current_ratio": 0.0, "analyst_target": 0.0, "rec": "N/A"}
    try:
        info = yf.Ticker(symbol.upper()).info
        return {
            "name":           info.get("longName", symbol),
            "sector":         info.get("sector", "N/A"),
            "industry":       info.get("industry", "N/A"),
            "price":          info.get("currentPrice", info.get("regularMarketPrice", 0.0)) or 0.0,
            "change":         info.get("regularMarketChangePercent", 0.0) or 0.0,
            "volume":         info.get("volume", 0) or 0,
            "mktcap":         info.get("marketCap", 0) or 0,
            "hi52":           info.get("fiftyTwoWeekHigh", 0.0) or 0.0,
            "lo52":           info.get("fiftyTwoWeekLow", 0.0) or 0.0,
            "pe":             info.get("trailingPE", 0.0) or 0.0,
            "fwd_pe":         info.get("forwardPE", 0.0) or 0.0,
            "peg":            info.get("pegRatio", 0.0) or 0.0,
            "div":            info.get("dividendYield", 0.0) or 0.0,
            "beta":           info.get("beta", 1.0) or 1.0,
            "eps":            info.get("trailingEps", 0.0) or 0.0,
            "ps":             info.get("priceToSalesTrailing12Months", 0.0) or 0.0,
            "pb":             info.get("priceToBook", 0.0) or 0.0,
            "roe":            info.get("returnOnEquity", 0.0) or 0.0,
            "profit_margin":  info.get("profitMargins", 0.0) or 0.0,
            "revenue":        info.get("totalRevenue", 0) or 0,
            "gross_margin":   info.get("grossMargins", 0.0) or 0.0,
            "debt_equity":    info.get("debtToEquity", 0.0) or 0.0,
            "current_ratio":  info.get("currentRatio", 0.0) or 0.0,
            "analyst_target": info.get("targetMeanPrice", 0.0) or 0.0,
            "rec":            info.get("recommendationKey", "N/A") or "N/A",
        }
    except Exception:
        return defaults


@st.cache_data(ttl=300, show_spinner=False)
def fetch_multi(symbols: list, period: str = "3mo") -> dict:
    result = {}
    for sym in symbols:
        d = fetch_data(sym, period)
        if d is not None:
            result[sym] = d
    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_chain(symbol: str) -> tuple:
    try:
        tkr = yf.Ticker(symbol.upper())
        exps = tkr.options
        if not exps:
            return None, None
        chain = tkr.option_chain(exps[0])
        return chain.calls, chain.puts
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def fmt_large(n: float) -> str:
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    if n >= 1e3:
        return f"${n/1e3:.1f}K"
    return f"${n:,.0f}"


def sharpe(returns: pd.Series, rfr: float = 0.04) -> float:
    excess = returns - rfr / 252
    std = excess.std()
    return float(np.sqrt(252) * excess.mean() / std) if std > 0 else 0.0


def sortino(returns: pd.Series, rfr: float = 0.04) -> float:
    excess = returns - rfr / 252
    neg = excess[excess < 0]
    ds = neg.std()
    return float(np.sqrt(252) * excess.mean() / ds) if ds > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min() * 100)


def calmar(total_ret_pct: float, max_dd_pct: float) -> float:
    return total_ret_pct / abs(max_dd_pct) if max_dd_pct != 0 else 0.0


def update_portfolio():
    total = st.session_state.cash
    for sym, qty in st.session_state.positions.items():
        if qty > 0:
            d = fetch_data(sym, "5d")
            if d is not None and not d.empty:
                total += qty * float(d["Close"].iloc[-1])
    st.session_state.portfolio_value = total
    if total > st.session_state.peak_value:
        st.session_state.peak_value = total
    dd = (st.session_state.peak_value - total) / \
        st.session_state.peak_value * 100
    if dd > st.session_state.max_drawdown:
        st.session_state.max_drawdown = dd


def execute_buy(symbol: str, shares: int, price: float):
    cost = shares * price
    if st.session_state.cash < cost:
        return False, "Insufficient cash"
    st.session_state.cash -= cost
    prev_qty = st.session_state.positions.get(symbol, 0)
    prev_cost = st.session_state.avg_costs.get(symbol, 0.0)
    new_qty = prev_qty + shares
    new_avg = (prev_qty * prev_cost + shares * price) / new_qty
    st.session_state.positions[symbol] = new_qty
    st.session_state.avg_costs[symbol] = new_avg
    st.session_state.trade_history.append({
        "time": datetime.now(), "symbol": symbol,
        "action": "BUY", "shares": shares, "price": price,
        "total": cost, "pnl": None,
    })
    st.session_state.trade_count += 1
    return True, f"Bought {shares} × {symbol} @ ${price:.2f}"


def execute_sell(symbol: str, shares: int, price: float):
    held = st.session_state.positions.get(symbol, 0)
    if held < shares:
        return False, f"Only {held} shares held"
    value = shares * price
    avg_cost = st.session_state.avg_costs.get(symbol, price)
    pnl = (price - avg_cost) * shares
    st.session_state.cash += value
    new_qty = held - shares
    st.session_state.positions[symbol] = new_qty
    if new_qty == 0:
        st.session_state.avg_costs.pop(symbol, None)
    else:
        st.session_state.avg_costs[symbol] = avg_cost  # avg cost unchanged
    if pnl > 0:
        st.session_state.win_count += 1
    else:
        st.session_state.loss_count += 1
    st.session_state.trade_history.append({
        "time": datetime.now(), "symbol": symbol,
        "action": "SELL", "shares": shares, "price": price,
        "total": value, "pnl": pnl,
    })
    st.session_state.trade_count += 1
    return True, f"Sold {shares} × {symbol} @ ${price:.2f}  P&L: ${pnl:+,.2f}"


def kpi_box(label: str, value: str, cls: str = "") -> str:
    return f"""
    <div class="kpi-box">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val {cls}">{value}</div>
    </div>"""


def model_breakdown_html(individual: dict) -> str:
    """Build clean, valid HTML rows for the per-model probability breakdown."""
    rows = []
    for name, val in individual.items():
        rows.append(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:var(--mono);font-size:0.65rem;margin-bottom:2px;">'
            f'<span style="color:var(--t3);">{name.upper()}</span>'
            f'<span style="color:var(--t2);">{val*100:.1f}%</span>'
            f'</div>'
        )
    return "".join(rows)


# ═══════════════════════════════════════════════════════════════
# ML ENGINE  — QuantMLEngine v3
# ═══════════════════════════════════════════════════════════════
class QuantMLEngine:
    """
    Ensemble of Random Forest + Gradient Boosting + Extra Trees + Logistic Regression.
    80+ engineered features. TimeSeriesSplit cross-validation. Full backtest engine.
    """
    WEIGHTS = {"rf": 0.40, "gb": 0.30, "et": 0.20, "lr": 0.10}

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=400, max_depth=16,
                                         min_samples_split=5, min_samples_leaf=2,
                                         class_weight="balanced", n_jobs=-1, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=250, learning_rate=0.04,
                                             max_depth=8, subsample=0.8,
                                             min_samples_leaf=2, random_state=42)
        self.et = ExtraTreesClassifier(n_estimators=300, max_depth=14,
                                       min_samples_split=5, class_weight="balanced",
                                       n_jobs=-1, random_state=42)
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     C=0.5, solver="lbfgs", random_state=42)
        self.scaler = RobustScaler()
        self.trained = False
        self.features = []
        self.val_scores = {}        # {model: [fold_scores]}
        self.feat_imp = {}        # {feature: importance}
        self.cv_accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    # ── FEATURE ENGINEERING ───────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        c = d["Close"]
        h = d["High"]
        lo = d["Low"]
        vol = d["Volume"]

        # ── Price & Returns
        d["ret_1"] = c.pct_change(1)
        d["ret_2"] = c.pct_change(2)
        d["ret_5"] = c.pct_change(5)
        d["ret_10"] = c.pct_change(10)
        d["ret_20"] = c.pct_change(20)
        d["log_ret"] = np.log(c / c.shift(1))
        d["ret_accel"] = d["ret_1"] - d["ret_1"].shift(1)

        # ── Price ratios
        d["hl_ratio"] = h / lo
        d["co_ratio"] = c / d["Open"]
        d["hc_ratio"] = (h - c) / (h - lo + 1e-9)
        d["lc_ratio"] = (c - lo) / (h - lo + 1e-9)
        d["body_size"] = abs(c - d["Open"]) / (h - lo + 1e-9)

        # ── Volatility regimes
        for w in [5, 10, 20, 60]:
            d[f"vol_{w}"] = d["ret_1"].rolling(w).std()
            d[f"vol_rank_{w}"] = d[f"vol_{w}"].rank(pct=True)
        d["vol_ratio_5_20"] = d["vol_5"] / (d["vol_20"] + 1e-9)
        d["vol_ratio_10_60"] = d["vol_10"] / (d["vol_60"] + 1e-9)
        d["vol_z_20"] = (d["vol_5"] - d["vol_20"]) / (d["vol_20"].std() + 1e-9)

        # ── Volume
        d["vol_chg"] = vol.pct_change()
        d["vol_ma5"] = vol.rolling(5).mean()
        d["vol_ma20"] = vol.rolling(20).mean()
        d["vol_ratio2"] = vol / (d["vol_ma20"] + 1e-9)
        d["vol_z"] = (vol - d["vol_ma20"]) / (vol.rolling(20).std() + 1e-9)
        d["vpt"] = (c - c.shift(1)) * vol
        d["dollar_vol"] = c * vol

        # ── RSI family
        for w in [7, 14, 21]:
            d[f"rsi_{w}"] = ta.momentum.RSIIndicator(c, window=w).rsi()
        d["rsi_slope"] = d["rsi_14"] - d["rsi_14"].shift(5)
        d["rsi_div"] = (d["rsi_14"] - 50) / 50
        d["rsi_7_14"] = d["rsi_7"] - d["rsi_14"]

        # ── MACD
        macd_obj = ta.trend.MACD(
            c, window_slow=26, window_fast=12, window_sign=9)
        d["macd"] = macd_obj.macd()
        d["macd_sig"] = macd_obj.macd_signal()
        d["macd_hist"] = macd_obj.macd_diff()
        d["macd_slope"] = d["macd"] - d["macd"].shift(3)
        d["macd_hist_slope"] = d["macd_hist"] - d["macd_hist"].shift(1)
        d["macd_norm"] = d["macd"] / (c + 1e-9)

        # ── Bollinger Bands
        bb = ta.volatility.BollingerBands(c)
        d["bb_h"] = bb.bollinger_hband()
        d["bb_l"] = bb.bollinger_lband()
        d["bb_m"] = bb.bollinger_mavg()
        d["bb_w"] = bb.bollinger_wband()
        d["bb_pos"] = (c - d["bb_l"]) / (d["bb_h"] - d["bb_l"] + 1e-9)
        d["bb_bw"] = (d["bb_h"] - d["bb_l"]) / (d["bb_m"] + 1e-9)
        d["bb_pos_sq"] = d["bb_pos"] ** 2

        # ── Stochastic
        stoch = ta.momentum.StochasticOscillator(h, lo, c)
        d["stoch_k"] = stoch.stoch()
        d["stoch_d"] = stoch.stoch_signal()
        d["stoch_diff"] = d["stoch_k"] - d["stoch_d"]
        d["stoch_k_slope"] = d["stoch_k"] - d["stoch_k"].shift(3)

        # ── ADX / DMI
        adx_obj = ta.trend.ADXIndicator(h, lo, c)
        d["adx"] = adx_obj.adx()
        d["dmi_p"] = adx_obj.adx_pos()
        d["dmi_n"] = adx_obj.adx_neg()
        d["dmi_diff"] = d["dmi_p"] - d["dmi_n"]
        d["adx_slope"] = d["adx"] - d["adx"].shift(5)

        # ── CCI / WPR
        d["cci"] = ta.trend.CCIIndicator(h, lo, c).cci()
        d["wpr"] = ta.momentum.WilliamsRIndicator(h, lo, c).williams_r()
        d["cci_norm"] = d["cci"] / 200.0
        d["wpr_norm"] = (d["wpr"] + 100) / 100.0

        # ── ATR
        atr_obj = ta.volatility.AverageTrueRange(h, lo, c)
        d["atr"] = atr_obj.average_true_range()
        d["atr_pct"] = d["atr"] / (c + 1e-9)
        d["atr_z"] = (d["atr"] - d["atr"].rolling(20).mean()) / \
            (d["atr"].rolling(20).std() + 1e-9)

        # ── OBV / MFI
        obv = ta.volume.OnBalanceVolumeIndicator(c, vol)
        d["obv"] = obv.on_balance_volume()
        d["obv_slope"] = d["obv"] - d["obv"].shift(5)
        d["obv_ema"] = d["obv"].ewm(span=20).mean()
        d["obv_ratio"] = d["obv"] / (d["obv_ema"] + 1e-9)
        d["mfi"] = ta.volume.MFIIndicator(h, lo, c, vol).money_flow_index()
        d["mfi_slope"] = d["mfi"] - d["mfi"].shift(5)

        # ── Moving averages
        for w in [5, 10, 20, 50, 100, 200]:
            d[f"sma_{w}"] = c.rolling(w).mean()
            d[f"ema_{w}"] = c.ewm(span=w).mean()
            d[f"pr_sma_{w}"] = c / (d[f"sma_{w}"] + 1e-9)
        d["sma_5_20"] = (d["sma_5"] - d["sma_20"]) / (d["sma_20"] + 1e-9)
        d["ema_5_20"] = (d["ema_5"] - d["ema_20"]) / (d["ema_20"] + 1e-9)
        d["ema_20_50"] = (d["ema_20"] - d["ema_50"]) / (d["ema_50"] + 1e-9)
        d["golden_x"] = (d["sma_50"] - d["sma_200"]) / (d["sma_200"] + 1e-9)
        d["ma_align"] = ((d["ema_5"] > d["ema_20"]) & (
            d["ema_20"] > d["ema_50"])).astype(int)

        # ── Lag features
        for lag in [1, 2, 3, 5, 10]:
            d[f"ret_lag_{lag}"] = d["ret_1"].shift(lag)
            d[f"vol_lag_{lag}"] = vol.shift(lag)
        for lag in [1, 2, 3]:
            d[f"rsi_lag_{lag}"] = d["rsi_14"].shift(lag)
            d[f"macd_lag_{lag}"] = d["macd"].shift(lag)

        # ── Trend strength
        d["trend_str"] = abs(c - d["sma_20"]) / (d["atr"] + 1e-9)
        d["trend_sign"] = np.sign(c - d["sma_20"])

        # ── VWAP approximation
        d["vwap_approx"] = (d["dollar_vol"].rolling(20).sum() /
                            (vol.rolling(20).sum() + 1e-9))
        d["pr_vwap"] = c / (d["vwap_approx"] + 1e-9)

        return d.replace([np.inf, -np.inf], np.nan).dropna()

    # ── TRAIN ────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        df = self.engineer_features(df)
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()

        _skip = {"target", "Open", "High", "Low", "Close", "Volume",
                 "Dividends", "Stock Splits", "Capital Gains"}
        self.features = [c for c in df.columns if c not in _skip]

        X = df[self.features].fillna(0)
        y = df["target"]

        tscv = TimeSeriesSplit(n_splits=5)
        fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

        for tr_idx, te_idx in tscv.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            Xtr_s = self.scaler.fit_transform(Xtr)
            Xte_s = self.scaler.transform(Xte)
            self.rf.fit(Xtr_s, ytr)
            yp = self.rf.predict(Xte_s)
            fold_acc.append(accuracy_score(yte, yp))
            fold_prec.append(precision_score(yte, yp, zero_division=0))
            fold_rec.append(recall_score(yte, yp, zero_division=0))
            fold_f1.append(f1_score(yte, yp, zero_division=0))

        # Final fit on all data
        Xs = self.scaler.fit_transform(X)
        self.rf.fit(Xs, y)
        self.gb.fit(Xs, y)
        self.et.fit(Xs, y)
        self.lr.fit(Xs, y)

        self.feat_imp = dict(zip(self.features, self.rf.feature_importances_))
        self.trained = True
        self.cv_accuracy = float(np.mean(fold_acc))
        self.precision = float(np.mean(fold_prec))
        self.recall = float(np.mean(fold_rec))
        self.f1 = float(np.mean(fold_f1))

        return {
            "accuracy":  self.cv_accuracy,
            "precision": self.precision,
            "recall":    self.recall,
            "f1":        self.f1,
            "n_features": len(self.features),
            "n_samples":  len(df),
        }

    # ── PREDICT ──────────────────────────────────────────────
    def predict(self, df: pd.DataFrame):
        if not self.trained:
            return None
        df = self.engineer_features(df)
        X = df[self.features].iloc[-1:].fillna(0)
        Xs = self.scaler.transform(X)

        probs = {}
        for name, model in [("rf", self.rf), ("gb", self.gb),
                            ("et", self.et), ("lr", self.lr)]:
            probs[name] = model.predict_proba(Xs)[0]

        ens = sum(self.WEIGHTS[n] * probs[n] for n in self.WEIGHTS)
        buy_p = float(ens[1])
        sell_p = float(ens[0])
        conf = float(max(ens))
        strength = float(abs(buy_p - sell_p))

        if buy_p > 0.72:
            signal = "STRONG BUY"
        elif buy_p > 0.62:
            signal = "BUY"
        elif sell_p > 0.72:
            signal = "STRONG SELL"
        elif sell_p > 0.62:
            signal = "SELL"
        else:
            signal = "HOLD"

        individual = {n: float(probs[n][1]) for n in probs}
        return {
            "signal":     signal,
            "confidence": conf,
            "strength":   strength,
            "buy_prob":   buy_p,
            "sell_prob":  sell_p,
            "individual": individual,
        }

    # ── BULK PROBABILITIES ────────────────────────────────────
    def bulk_proba(self, df: pd.DataFrame):
        if not self.trained:
            return None
        df = self.engineer_features(df)
        X = df[self.features].fillna(0)
        Xs = self.scaler.transform(X)
        rf_p = self.rf.predict_proba(Xs)[:, 1]
        gb_p = self.gb.predict_proba(Xs)[:, 1]
        et_p = self.et.predict_proba(Xs)[:, 1]
        lr_p = self.lr.predict_proba(Xs)[:, 1]
        return (self.WEIGHTS["rf"]*rf_p + self.WEIGHTS["gb"]*gb_p +
                self.WEIGHTS["et"]*et_p + self.WEIGHTS["lr"]*lr_p)

    # ── BACKTEST ─────────────────────────────────────────────
    def backtest(self, df: pd.DataFrame, initial: float = 100_000.0,
                 buy_thr: float = 0.65, sell_thr: float = 0.35,
                 stop_loss_pct: float = 5.0, take_profit_pct: float = 20.0,
                 commission: float = 0.001) -> dict:
        df = self.engineer_features(df)
        X = df[self.features].fillna(0)
        Xs = self.scaler.transform(X)
        rf_p = self.rf.predict_proba(Xs)[:, 1]
        gb_p = self.gb.predict_proba(Xs)[:, 1]
        et_p = self.et.predict_proba(Xs)[:, 1]
        lr_p = self.lr.predict_proba(Xs)[:, 1]
        sigs = (self.WEIGHTS["rf"]*rf_p + self.WEIGHTS["gb"]*gb_p +
                self.WEIGHTS["et"]*et_p + self.WEIGHTS["lr"]*lr_p)

        cap = initial
        pos = 0
        entry = 0.0
        trades = []
        equity = [initial]
        dates = [df.index[0]]
        peak = initial
        max_dd = 0.0

        for i in range(len(df)):
            price = float(df["Close"].iloc[i])
            sig = sigs[i]
            date = df.index[i]

            # Stop loss / take profit
            if pos > 0:
                pct_chg = (price - entry) / entry * 100
                if pct_chg <= -stop_loss_pct or pct_chg >= take_profit_pct:
                    proceeds = pos * price * (1 - commission)
                    cap += proceeds
                    trades.append({"type": "SELL", "price": price,
                                   "shares": pos, "date": date,
                                   "pnl": (price - entry)*pos,
                                   "reason": "SL/TP"})
                    pos = 0

            # ML signal
            if sig >= buy_thr and pos == 0:
                shares = int(cap // (price * (1 + commission)))
                if shares > 0:
                    cost = shares * price * (1 + commission)
                    cap -= cost
                    entry = price
                    pos = shares
                    trades.append({"type": "BUY", "price": price,
                                   "shares": shares, "date": date, "pnl": None})
            elif sig <= sell_thr and pos > 0:
                proceeds = pos * price * (1 - commission)
                cap += proceeds
                trades.append({"type": "SELL", "price": price,
                               "shares": pos, "date": date,
                               "pnl": (price - entry)*pos, "reason": "Signal"})
                pos = 0

            val = cap + pos * price
            equity.append(val)
            dates.append(date)
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Close at end
        if pos > 0:
            final_price = float(df["Close"].iloc[-1])
            cap += pos * final_price * (1 - commission)
            trades.append({"type": "SELL", "price": final_price,
                           "shares": pos, "date": df.index[-1],
                           "pnl": (final_price - entry)*pos, "reason": "EOD"})

        final_val = cap
        sells = [t for t in trades if t["type"] ==
                 "SELL" and t.get("pnl") is not None]
        wins = [t for t in sells if t["pnl"] > 0]
        losses = [t for t in sells if t["pnl"] <= 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        profit_factor = (sum(t["pnl"] for t in wins) /
                         abs(sum(t["pnl"] for t in losses)) if losses else float("inf"))

        eq_s = pd.Series(equity)
        rets_s = eq_s.pct_change().dropna()
        sharpe_r = sharpe(rets_s)
        sortino_r = sortino(rets_s)
        total_ret = (final_val - initial) / initial * 100
        calmar_r = calmar(total_ret, max_dd)

        # BH benchmark
        bh_start = float(df["Close"].iloc[0])
        bh_end = float(df["Close"].iloc[-1])
        bh_ret = (bh_end - bh_start) / bh_start * 100
        alpha = total_ret - bh_ret

        return {
            "final_value":    final_val,
            "total_return":   total_ret,
            "bh_return":      bh_ret,
            "alpha":          alpha,
            "num_trades":     len(trades),
            "num_buys":       len([t for t in trades if t["type"] == "BUY"]),
            "num_sells":      len(sells),
            "win_rate":       win_rate,
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "profit_factor":  profit_factor,
            "max_drawdown":   max_dd,
            "sharpe":         sharpe_r,
            "sortino":        sortino_r,
            "calmar":         calmar_r,
            "equity_curve":   equity,
            "equity_dates":   dates,
            "returns":        rets_s,
            "trades":         trades,
        }


# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(6,10,18,0)",
    plot_bgcolor="rgba(6,10,18,0)",
    margin=dict(l=0, r=0, t=4, b=0),
    font=dict(family="JetBrains Mono", size=10, color="#7e8fa8"),
    hoverlabel=dict(bgcolor="#0a0f1e", font_family="JetBrains Mono",
                    font_size=11, bordercolor="#0a0f1e"),
)
GRID = dict(showgrid=True, gridcolor="rgba(255,255,255,0.035)", zeroline=False)


def candle_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    for span in [20, 50]:
        df[f"ema_{span}"] = df["Close"].ewm(span=span).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    macd_obj = ta.trend.MACD(df["Close"])
    df["macd"] = macd_obj.macd()
    df["macd_sig"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.17, 0.14, 0.14],
        vertical_spacing=0.018,
    )

    # ── Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing=dict(line=dict(color="#00ff94", width=1),
                        fillcolor="rgba(0,255,148,0.3)"),
        decreasing=dict(line=dict(color="#ff3250", width=1),
                        fillcolor="rgba(255,50,80,0.3)"),
    ), row=1, col=1)

    for s, color, w in [("ema_20", "#388bfd", 1.5), ("ema_50", "#ffb347", 1.5), ("sma_200", "#b06eff", 2)]:
        fig.add_trace(go.Scatter(x=df.index, y=df[s], line=dict(color=color, width=w),
                                 showlegend=False, opacity=0.9), row=1, col=1)

    # ── Volume
    vol_colors = ["rgba(0,255,148,0.45)" if c >= o else "rgba(255,50,80,0.45)"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume"].rolling(20).mean(),
                             line=dict(color="#ffb347", width=1, dash="dot"),
                             showlegend=False), row=2, col=1)

    # ── RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"],
                             line=dict(color="#00c8e0", width=1.5),
                             showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot",
                  line_color="rgba(255,50,80,0.5)",  line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot",
                  line_color="rgba(0,255,148,0.5)", line_width=1, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,50,80,0.03)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,148,0.03)",
                  line_width=0, row=3, col=1)

    # ── MACD
    hist_colors = ["rgba(0,255,148,0.55)" if v >= 0 else "rgba(255,50,80,0.55)"
                   for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
                         marker_color=hist_colors, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
                             line=dict(color="#388bfd", width=1.2), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_sig"],
                             line=dict(color="#ff6eb4", width=1.2), showlegend=False), row=4, col=1)

    fig.update_layout(**PLOTLY_BASE, height=620, xaxis_rangeslider_visible=False,
                      hovermode="x unified")
    for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
        fig.update_layout(**{ax: GRID})
    fig.update_yaxes(tickprefix="$", row=1, col=1, **GRID)
    fig.update_yaxes(row=2, col=1, showgrid=False)
    fig.update_yaxes(range=[0, 100], tickvals=[
                     30, 50, 70], row=3, col=1, **GRID)
    fig.update_yaxes(row=4, col=1, **GRID)

    fig.add_annotation(x=0.005, y=0.99, xref="paper", yref="paper",
                       text="<b style='color:#388bfd'>━</b> EMA20  "
                       "<b style='color:#ffb347'>━</b> EMA50  "
                       "<b style='color:#b06eff'>━</b> SMA200",
                       showarrow=False, font=dict(size=10, color="#7e8fa8"),
                       bgcolor="rgba(6,10,18,0.7)", borderpad=5, align="left")
    return fig


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
update_portfolio()

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:800;
            color:#dde4f0;letter-spacing:-0.03em;">QUANT<span style="color:#00ff94;">.AI</span></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#1e2a38;
            letter-spacing:0.2em;text-transform:uppercase;margin-top:2px;">Terminal v3.0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Instrument</div>',
                unsafe_allow_html=True)
    symbol = st.text_input("", "AAPL", key="sym_input",
                           placeholder="e.g. AAPL, TSLA, NVDA").upper().strip()
    info = fetch_info(symbol)

    chg_color = "#00ff94" if info["change"] >= 0 else "#ff3250"
    chg_sign = "+" if info["change"] >= 0 else ""
    st.markdown(f"""
    <div style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r-lg);
        padding:0.8rem;margin-bottom:1rem;">
        <div style="font-family:var(--mono);font-size:0.65rem;color:var(--t3);letter-spacing:0.05em;">
            {info['sector']} · {info['industry'][:22] if info['industry'] != 'N/A' else ''}</div>
        <div style="font-family:var(--mono);font-size:0.88rem;font-weight:700;color:var(--t1);
            margin:0.25rem 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            {info['name'][:30]}</div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.3rem;">
            <span style="font-family:var(--mono);font-size:1.1rem;font-weight:700;color:var(--t1);">
                ${info['price']:.2f}</span>
            <span style="font-family:var(--mono);font-size:0.78rem;font-weight:700;color:{chg_color};">
                {chg_sign}{info['change']:.2f}%</span>
        </div>
        <div style="font-family:var(--mono);font-size:0.62rem;color:var(--t3);margin-top:0.3rem;">
            Vol: {info['volume']:,}  ·  {fmt_large(info['mktcap'])}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Order Setup</div>',
                unsafe_allow_html=True)
    investment = st.number_input("Investment Amount ($)", 100,
                                 int(max(st.session_state.cash, 100)),
                                 min(5000, int(max(st.session_state.cash, 100))), 500)
    c1, c2 = st.columns(2)
    with c1:
        stop_loss = st.number_input("Stop Loss %", 1, 25, 5)
    with c2:
        take_profit = st.number_input("Take Profit %", 2, 100, 15)

    risk_level = st.select_slider("Risk Profile",
                                  ["Conservative", "Moderate", "Aggressive"], "Moderate")

    st.markdown('<div class="sec-label" style="margin-top:0.9rem;">Engine Controls</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶  START", use_container_width=True, type="primary"):
            st.session_state.bot_active = True
            st.toast("🟢 Bot activated", icon="⚡")
    with c2:
        if st.button("⏸  PAUSE", use_container_width=True):
            st.session_state.bot_active = False
            st.toast("⏸ Bot paused")

    if st.button("↺  RESET PORTFOLIO", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown('<div class="sec-label" style="margin-top:0.9rem;">Performance</div>',
                unsafe_allow_html=True)
    roi = (st.session_state.portfolio_value - st.session_state.initial_capital
           ) / st.session_state.initial_capital * 100
    win_rate = (st.session_state.win_count /
                (st.session_state.win_count + st.session_state.loss_count) * 100
                if (st.session_state.win_count + st.session_state.loss_count) > 0 else 0)

    st.metric("Total ROI",     f"{roi:+.2f}%",
              delta=f"${st.session_state.portfolio_value - st.session_state.initial_capital:+,.0f}")
    st.metric("Win Rate",      f"{win_rate:.1f}%")
    st.metric("Max Drawdown",  f"{st.session_state.max_drawdown:.2f}%")
    st.metric("Total Trades",  str(st.session_state.trade_count))

    # Watchlist quick-add
    st.markdown('<div class="sec-label" style="margin-top:0.9rem;">Watchlist</div>',
                unsafe_allow_html=True)
    wl_add = st.text_input("Add symbol", "", placeholder="e.g. NFLX")
    if wl_add:
        wl_sym = wl_add.upper().strip()
        if wl_sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(wl_sym)

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
pnl = st.session_state.portfolio_value - st.session_state.initial_capital
pnl_pct = pnl / st.session_state.initial_capital * 100
status = ('<span class="pill pill-active">BOT ACTIVE</span>'
          if st.session_state.bot_active else
          '<span class="pill pill-inactive">BOT OFFLINE</span>')

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
    margin-bottom:1.1rem;padding-bottom:0.75rem;border-bottom:1px solid var(--bdr);">
    <div style="display:flex;align-items:center;gap:1rem;">
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;
                color:var(--t1);letter-spacing:-0.04em;line-height:1.1;">
                QUANT<span style="color:#00ff94;">.AI</span>
                <span style="font-size:0.62rem;color:var(--t3);font-weight:600;
                    letter-spacing:0.14em;margin-left:0.8rem;font-family:'JetBrains Mono',monospace;">
                    ALGORITHMIC TRADING TERMINAL  v3.0</span>
            </div>
        </div>
        {status}
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--t3);">
        {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
    </div>
</div>""", unsafe_allow_html=True)

# Top KPI row
kc = st.columns(5)
with kc[0]:
    st.markdown(kpi_box("Portfolio Value",
                f"${st.session_state.portfolio_value:,.2f}", "kpi-pos"), unsafe_allow_html=True)
with kc[1]:
    st.markdown(kpi_box("Cash Available",
                f"${st.session_state.cash:,.2f}"), unsafe_allow_html=True)
with kc[2]:
    st.markdown(kpi_box("Unrealized P&L",
                f"${pnl:+,.2f}", "kpi-pos" if pnl >= 0 else "kpi-neg"), unsafe_allow_html=True)
with kc[3]:
    st.markdown(kpi_box("Total Return",    f"{pnl_pct:+.2f}%",
                "kpi-pos" if pnl_pct >= 0 else "kpi-neg"), unsafe_allow_html=True)
with kc[4]:
    open_pos = sum(1 for v in st.session_state.positions.values() if v > 0)
    st.markdown(kpi_box("Open Positions", str(open_pos)),
                unsafe_allow_html=True)

st.markdown("<div style='margin:0.85rem 0;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_live, tab_ml, tab_port, tab_bt, tab_scan, tab_options, tab_news = st.tabs([
    "⚡  LIVE TRADING",
    "🧠  ML ENGINE",
    "💼  PORTFOLIO",
    "📊  BACKTEST",
    "🔍  MARKET SCAN",
    "📈  OPTIONS",
    "📰  NEWS & DATA",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIVE TRADING
# ═══════════════════════════════════════════════════════════════
with tab_live:
    df = fetch_data(symbol, "1y")
    if df is None or df.empty:
        st.error(f"Could not fetch data for **{symbol}**. Verify the ticker.")
    else:
        col_chart, col_sig = st.columns([3.1, 1])

        with col_chart:
            st.markdown(f'<div class="sec-label">{symbol} — Price Action  ·  EMA20/50  ·  SMA200  ·  Volume  ·  RSI  ·  MACD</div>',
                        unsafe_allow_html=True)
            fig_c = candle_chart(df)
            st.plotly_chart(fig_c, use_container_width=True,
                            config={"displayModeBar": False})

        with col_sig:
            st.markdown(
                '<div class="sec-label">AI Signal Engine</div>', unsafe_allow_html=True)

            with st.spinner("Training ensemble..."):
                model = QuantMLEngine()
                metrics = model.train(df)
                pred = model.predict(df)

            cur_price = float(df["Close"].iloc[-1])
            prev_price = float(df["Close"].iloc[-2])
            chg_pct = (cur_price - prev_price) / prev_price * 100
            chg_col = "#00ff94" if chg_pct >= 0 else "#ff3250"

            # Price display
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r-lg);
                padding:0.85rem;margin-bottom:0.65rem;">
                <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                    letter-spacing:0.12em;text-transform:uppercase;">{symbol} · LAST</div>
                <div style="font-family:var(--mono);font-size:1.75rem;font-weight:700;
                    color:var(--t1);margin:0.2rem 0;letter-spacing:-0.04em;">${cur_price:.2f}</div>
                <div style="font-family:var(--mono);font-size:0.8rem;color:{chg_col};font-weight:700;">
                    {'+' if chg_pct>=0 else ''}{chg_pct:.2f}% today</div>
                <div style="display:flex;justify-content:space-between;margin-top:0.4rem;">
                    <span style="font-family:var(--mono);font-size:0.6rem;color:var(--t3);">
                        CV ACC <span style="color:#00ff94;">{metrics['accuracy']*100:.1f}%</span></span>
                    <span style="font-family:var(--mono);font-size:0.6rem;color:var(--t3);">
                        F1 <span style="color:#388bfd;">{metrics['f1']*100:.1f}%</span></span>
                    <span style="font-family:var(--mono);font-size:0.6rem;color:var(--t3);">
                        {metrics['n_features']} FTR</span>
                </div>
            </div>""", unsafe_allow_html=True)

            if pred:
                sig_map = {
                    "STRONG BUY":  ("sig-buy",  "▲▲", "#00ff94", "🚀"),
                    "BUY":         ("sig-buy",  "▲",  "#00ff94", "📈"),
                    "HOLD":        ("sig-hold", "◆",  "#ffb347", "⏸"),
                    "SELL":        ("sig-sell", "▼",  "#ff3250", "📉"),
                    "STRONG SELL": ("sig-sell", "▼▼", "#ff3250", "⚠️"),
                }
                css, icon, sig_col, emoji = sig_map[pred["signal"]]

                st.markdown(f"""
                <div class="signal-card {css}">
                    <div style="font-size:2.2rem;margin-bottom:0.25rem;">{emoji}</div>
                    <div style="font-size:1.3rem;font-weight:800;letter-spacing:0.05em;
                        color:{sig_col};">{pred['signal']}</div>
                    <div style="font-size:0.65rem;color:rgba(255,255,255,0.4);
                        letter-spacing:0.12em;margin-top:0.35rem;">
                        CONFIDENCE · {pred['confidence']*100:.1f}%</div>
                    <div style="background:rgba(0,0,0,0.35);border-radius:2px;
                        height:3px;margin-top:0.5rem;overflow:hidden;">
                        <div style="width:{pred['confidence']*100:.0f}%;height:100%;
                            background:{sig_col};border-radius:2px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Probability bars
                st.markdown(f"""
                <div style="background:var(--bg2);border:1px solid var(--bdr);
                    border-radius:var(--r-lg);padding:0.8rem;margin-bottom:0.65rem;">
                    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                        letter-spacing:0.12em;margin-bottom:0.55rem;">PROBABILITY</div>
                    <div style="margin-bottom:0.5rem;">
                        <div style="display:flex;justify-content:space-between;
                            font-family:var(--mono);font-size:0.7rem;">
                            <span style="color:#00ff94;">BUY</span>
                            <span style="color:#00ff94;font-weight:700;">{pred['buy_prob']*100:.1f}%</span>
                        </div>
                        <div class="ind-bar"><div class="ind-fill"
                            style="width:{pred['buy_prob']*100:.0f}%;background:#00ff94;"></div></div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;
                            font-family:var(--mono);font-size:0.7rem;">
                            <span style="color:#ff3250;">SELL</span>
                            <span style="color:#ff3250;font-weight:700;">{pred['sell_prob']*100:.1f}%</span>
                        </div>
                        <div class="ind-bar"><div class="ind-fill"
                            style="width:{pred['sell_prob']*100:.0f}%;background:#ff3250;"></div></div>
                    </div>
                    <div style="margin-top:0.55rem;border-top:1px solid var(--bdr);padding-top:0.45rem;">
                        <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                            letter-spacing:0.1em;margin-bottom:0.35rem;">MODEL BREAKDOWN</div>
                        {model_breakdown_html(pred['individual'])}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Technical indicators
                rsi_v = ta.momentum.RSIIndicator(df["Close"]).rsi().iloc[-1]
                macd_v = ta.trend.MACD(df["Close"]).macd().iloc[-1]
                bb_obj = ta.volatility.BollingerBands(df["Close"])
                bb_pos = ((float(df["Close"].iloc[-1]) - float(bb_obj.bollinger_lband().iloc[-1])) /
                          (float(bb_obj.bollinger_hband().iloc[-1]) - float(bb_obj.bollinger_lband().iloc[-1]) + 1e-9) * 100)
                adx_v = ta.trend.ADXIndicator(
                    df["High"], df["Low"], df["Close"]).adx().iloc[-1]
                mfi_v = ta.volume.MFIIndicator(
                    df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index().iloc[-1]
                stk_v = ta.momentum.StochasticOscillator(
                    df["High"], df["Low"], df["Close"]).stoch().iloc[-1]
                atr_v = ta.volatility.AverageTrueRange(
                    df["High"], df["Low"], df["Close"]).average_true_range().iloc[-1]
                cci_v = ta.trend.CCIIndicator(
                    df["High"], df["Low"], df["Close"]).cci().iloc[-1]

                def rsi_st(r):
                    if r > 70:
                        return "#ff3250", "OB"
                    if r < 30:
                        return "#00ff94", "OS"
                    return "#ffb347", "N"

                inds = [
                    ("RSI 14",  f"{rsi_v:.1f}",  *rsi_st(rsi_v)),
                    ("MACD",    f"{macd_v:.4f}",  "+", "#00ff94") if macd_v > 0 else (
                        "MACD", f"{macd_v:.4f}", "−", "#ff3250"),
                    ("BB POS",  f"{bb_pos:.0f}%", "HI" if bb_pos > 60 else (
                        "LO" if bb_pos < 40 else "MID"), "#ffb347"),
                    ("ADX",     f"{adx_v:.1f}",   "TRD" if adx_v >
                     25 else "RNG", "#388bfd" if adx_v > 25 else "#7e8fa8"),
                    ("MFI",     f"{mfi_v:.1f}",   "OB" if mfi_v > 80 else (
                        "OS" if mfi_v < 20 else "N"), "#b06eff"),
                    ("STOCH",   f"{stk_v:.1f}",   "OB" if stk_v > 80 else (
                        "OS" if stk_v < 20 else "N"), "#00c8e0"),
                    ("ATR%",    f"{atr_v/cur_price*100:.2f}%",
                     "VOL", "#ff6eb4"),
                    ("CCI",     f"{cci_v:.0f}",   "OB" if cci_v > 100 else (
                        "OS" if cci_v < -100 else "N"), "#ffb347"),
                ]
                st.markdown('<div class="sec-label">Indicators</div>',
                            unsafe_allow_html=True)
                for nm, vl, lb, cl in inds:
                    st.markdown(f"""
                    <div class="trade-row" style="padding:0.4rem 0.7rem;margin-bottom:2px;">
                        <span style="color:var(--t2);font-size:0.68rem;">{nm}</span>
                        <div style="display:flex;align-items:center;gap:0.45rem;">
                            <span style="color:var(--t3);font-size:0.6rem;border:1px solid var(--t4);
                                border-radius:2px;padding:0px 3px;">{lb}</span>
                            <span style="color:{cl};font-weight:700;font-size:0.75rem;">{vl}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Order execution
                st.markdown('<div class="sec-label" style="margin-top:0.6rem;">Order</div>',
                            unsafe_allow_html=True)
                conf = pred["confidence"]

                if "BUY" in pred["signal"] and conf >= 0.58:
                    shares = max(1, int(investment / cur_price))
                    cost = shares * cur_price
                    target_p = cur_price * (1 + take_profit/100)
                    sl_p = cur_price * (1 - stop_loss/100)
                    st.markdown(f"""
                    <div class="order-box-buy">
                        <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                            letter-spacing:0.1em;">SUGGESTED BUY</div>
                        <div style="font-family:var(--mono);font-size:0.82rem;color:var(--t1);margin-top:0.3rem;">
                            {shares} shares @ ${cur_price:.2f}</div>
                        <div style="font-family:var(--mono);font-size:0.82rem;color:#00ff94;">
                            Cost: ${cost:,.2f}</div>
                        <div style="font-family:var(--mono);font-size:0.65rem;color:var(--t3);margin-top:0.3rem;">
                            TP: ${target_p:.2f}  ·  SL: ${sl_p:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("▶  EXECUTE BUY", use_container_width=True, type="primary"):
                        ok, msg = execute_buy(symbol, shares, cur_price)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                elif "SELL" in pred["signal"] and conf >= 0.58:
                    held = st.session_state.positions.get(symbol, 0)
                    if held > 0:
                        val = held * cur_price
                        avg_cost = st.session_state.avg_costs.get(
                            symbol, cur_price)
                        pnl_est = (cur_price - avg_cost) * held
                        st.markdown(f"""
                        <div class="order-box-sell">
                            <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                                letter-spacing:0.1em;">SUGGESTED SELL</div>
                            <div style="font-family:var(--mono);font-size:0.82rem;color:var(--t1);margin-top:0.3rem;">
                                {held} shares @ ${cur_price:.2f}</div>
                            <div style="font-family:var(--mono);font-size:0.82rem;
                                color:{'#00ff94' if pnl_est>=0 else '#ff3250'};">
                                P&L: ${pnl_est:+,.2f}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button("▶  EXECUTE SELL", use_container_width=True, type="primary"):
                            ok, msg = execute_sell(symbol, held, cur_price)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        st.markdown("""<div class="order-box-sell">
                            <div style="font-family:var(--mono);font-size:0.75rem;color:#7e8fa8;text-align:center;">
                                No position to close</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="order-box-hold">
                        <div style="font-family:var(--mono);font-size:0.78rem;color:#ffb347;font-weight:700;">
                            ◆ HOLD — No action</div></div>""", unsafe_allow_html=True)

                # Manual trading
                with st.expander("✏️  Manual Order", expanded=False):
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        m_shares = st.number_input("Shares", 1, 10000, 1)
                    with mc2:
                        m_price = st.number_input(
                            "Price", 0.01, 999999.0, float(cur_price))
                    mc3, mc4 = st.columns(2)
                    with mc3:
                        if st.button("BUY", use_container_width=True, type="primary"):
                            ok, msg = execute_buy(symbol, m_shares, m_price)
                            (st.success if ok else st.error)(msg)
                            if ok:
                                st.rerun()
                    with mc4:
                        if st.button("SELL", use_container_width=True):
                            ok, msg = execute_sell(symbol, m_shares, m_price)
                            (st.success if ok else st.error)(msg)
                            if ok:
                                st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 2 — ML ENGINE
# ═══════════════════════════════════════════════════════════════
with tab_ml:
    df2 = fetch_data(symbol, "1y")
    if df2 is None or df2.empty:
        st.error("No data available.")
    else:
        with st.spinner("Training full ensemble..."):
            model2 = QuantMLEngine()
            m2 = model2.train(df2)

        st.markdown('<div class="sec-label">Model Performance — 5-Fold TimeSeriesSplit CV</div>',
                    unsafe_allow_html=True)
        mc = st.columns(6)
        ml_kpis = [
            ("CV Accuracy",   f"{m2['accuracy']*100:.2f}%",  "kpi-pos"),
            ("Precision",     f"{m2['precision']*100:.2f}%", ""),
            ("Recall",        f"{m2['recall']*100:.2f}%",    ""),
            ("F1 Score",      f"{m2['f1']*100:.2f}%",        ""),
            ("Features",      str(m2['n_features']),          ""),
            ("Training Bars", str(m2['n_samples']),           ""),
        ]
        for col, (lbl, val, cls) in zip(mc, ml_kpis):
            with col:
                st.markdown(kpi_box(lbl, val, cls), unsafe_allow_html=True)

        st.markdown("<div style='margin:0.85rem 0;'></div>",
                    unsafe_allow_html=True)
        col_feat, col_sig2 = st.columns(2)

        with col_feat:
            st.markdown('<div class="sec-label">Feature Importance — Top 30</div>',
                        unsafe_allow_html=True)
            fi_df = (pd.DataFrame(list(model2.feat_imp.items()),
                                  columns=["Feature", "Importance"])
                     .sort_values("Importance", ascending=False).head(30))

            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "rgba(56,139,253,0.4)"], [
                        1, "rgba(0,255,148,1)"]],
                    line=dict(width=0),
                ),
            ))
            fig_fi.update_layout(**PLOTLY_BASE, height=580,
                                 xaxis={**GRID},
                                 yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_fi, use_container_width=True,
                            config={"displayModeBar": False})

        with col_sig2:
            st.markdown('<div class="sec-label">Signal Probability vs Price</div>',
                        unsafe_allow_html=True)
            df_eng = model2.engineer_features(df2.copy())
            proba = model2.bulk_proba(df2)
            if proba is not None:
                fig_sp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.6, 0.4], vertical_spacing=0.04)
                fig_sp.add_trace(go.Scatter(x=df_eng.index, y=df_eng["Close"],
                                            line=dict(
                                                color="rgba(221,228,240,0.7)", width=1.5),
                                            showlegend=False), row=1, col=1)
                # buy/sell dots
                buy_idx = df_eng.index[proba >= 0.65]
                sell_idx = df_eng.index[proba <= 0.35]
                if len(buy_idx):
                    fig_sp.add_trace(go.Scatter(
                        x=buy_idx, y=df_eng.loc[buy_idx, "Close"],
                        mode="markers", name="BUY signal",
                        marker=dict(color="#00ff94", size=5, symbol="triangle-up")), row=1, col=1)
                if len(sell_idx):
                    fig_sp.add_trace(go.Scatter(
                        x=sell_idx, y=df_eng.loc[sell_idx, "Close"],
                        mode="markers", name="SELL signal",
                        marker=dict(color="#ff3250", size=5, symbol="triangle-down")), row=1, col=1)

                fig_sp.add_trace(go.Scatter(
                    x=df_eng.index, y=proba,
                    mode="lines", line=dict(color="rgba(0,255,148,0.55)", width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,255,148,0.05)",
                    showlegend=False), row=2, col=1)
                for y_val, col_ in [(0.65, "rgba(0,255,148,0.5)"), (0.35, "rgba(255,50,80,0.5)")]:
                    fig_sp.add_hline(y=y_val, line_dash="dot", line_color=col_,
                                     line_width=1, row=2, col=1)

                fig_sp.update_layout(**PLOTLY_BASE, height=580,
                                     xaxis={**GRID}, xaxis2={**GRID},
                                     yaxis={**GRID, "tickprefix": "$"},
                                     yaxis2={**GRID, "range": [0, 1]},
                                     legend=dict(font=dict(size=9, color="#7e8fa8"),
                                                 orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_sp, use_container_width=True,
                                config={"displayModeBar": False})

        # Ensemble architecture
        st.markdown('<div class="sec-label" style="margin-top:0.5rem;">Ensemble Architecture</div>',
                    unsafe_allow_html=True)
        ea1, ea2, ea3, ea4 = st.columns(4)
        ens_cards = [
            ("Random Forest",    "40%",
             "400 trees · depth 16 · balanced",       "#00ff94"),
            ("Gradient Boost",   "30%",
             "250 trees · lr=0.04 · subsample 0.8",   "#388bfd"),
            ("Extra Trees",      "20%",
             "300 trees · depth 14 · balanced",        "#b06eff"),
            ("Logistic Reg.",    "10%",
             "L2 · C=0.5 · max_iter 2000",            "#ffb347"),
        ]
        for col, (name, wt, desc, col_) in zip([ea1, ea2, ea3, ea4], ens_cards):
            with col:
                st.markdown(f"""
                <div class="ensemble-card" style="border-left-color:{col_};">
                    <div style="font-family:var(--mono);font-size:0.62rem;color:var(--t3);
                        text-transform:uppercase;letter-spacing:0.1em;">{name}</div>
                    <div style="font-family:var(--mono);font-size:1.6rem;font-weight:700;
                        color:{col_};margin:0.25rem 0;">{wt}</div>
                    <div style="font-size:0.72rem;color:var(--t2);line-height:1.4;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        # Feature categories
        st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Feature Category Breakdown</div>',
                    unsafe_allow_html=True)
        cats = {
            "Price/Returns":    [f for f in model2.features if "ret" in f or "log" in f or "ratio" in f or "size" in f],
            "Moving Averages":  [f for f in model2.features if "sma" in f or "ema" in f or "pr_" in f or "golden" in f or "align" in f],
            "Momentum":         [f for f in model2.features if any(x in f for x in ["rsi", "macd", "stoch", "cci", "wpr", "mom"])],
            "Volatility":       [f for f in model2.features if any(x in f for x in ["vol_", "bb_", "atr"])],
            "Volume":           [f for f in model2.features if any(x in f for x in ["obv", "mfi", "vpt", "dollar", "vol_ma", "vol_z", "vol_r"])],
            "Trend":            [f for f in model2.features if any(x in f for x in ["adx", "dmi", "trend", "vwap"])],
        }
        fc = st.columns(6)
        for col, (cat, flist) in zip(fc, cats.items()):
            with col:
                avg_imp = np.mean([model2.feat_imp.get(f, 0)
                                  for f in flist]) if flist else 0
                st.markdown(f"""
                <div style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r-lg);
                    padding:0.75rem;text-align:center;">
                    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                        text-transform:uppercase;letter-spacing:0.1em;">{cat}</div>
                    <div style="font-family:var(--mono);font-size:1.3rem;font-weight:700;
                        color:#00ff94;margin:0.2rem 0;">{len(flist)}</div>
                    <div style="font-family:var(--mono);font-size:0.6rem;color:var(--t3);">
                        avg imp {avg_imp*100:.2f}%</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════
with tab_port:
    invested = st.session_state.portfolio_value - st.session_state.cash
    alloc_pct = invested / st.session_state.portfolio_value * \
        100 if st.session_state.portfolio_value else 0
    total_pnl = st.session_state.portfolio_value - st.session_state.initial_capital
    total_ret = total_pnl / st.session_state.initial_capital * 100
    win_r2 = (st.session_state.win_count /
              (st.session_state.win_count + st.session_state.loss_count) * 100
              if (st.session_state.win_count + st.session_state.loss_count) > 0 else 0)

    st.markdown('<div class="sec-label">Portfolio Overview</div>',
                unsafe_allow_html=True)
    pc = st.columns(6)
    pm = [
        ("Portfolio Value",
         f"${st.session_state.portfolio_value:,.2f}", "kpi-pos"),
        ("Invested",          f"${invested:,.2f}",                         ""),
        ("Cash",              f"${st.session_state.cash:,.2f}",            ""),
        ("Total P&L",         f"${total_pnl:+,.2f}",
         "kpi-pos" if total_pnl >= 0 else "kpi-neg"),
        ("Total Return",      f"{total_ret:+.2f}%",
         "kpi-pos" if total_ret >= 0 else "kpi-neg"),
        ("Win Rate",          f"{win_r2:.1f}%",
         "kpi-pos" if win_r2 > 50 else ""),
    ]
    for col, (lbl, val, cls) in zip(pc, pm):
        with col:
            st.markdown(kpi_box(lbl, val, cls), unsafe_allow_html=True)

    st.markdown("<div style='margin:0.85rem 0;'></div>",
                unsafe_allow_html=True)
    col_p, col_h = st.columns([1.35, 1])

    with col_p:
        st.markdown('<div class="sec-label">Active Positions</div>',
                    unsafe_allow_html=True)
        active = {s: q for s, q in st.session_state.positions.items() if q > 0}
        if active:
            rows = []
            for sym, qty in active.items():
                d_ = fetch_data(sym, "5d")
                price_ = float(d_["Close"].iloc[-1]) if d_ is not None else 0.0
                val_ = qty * price_
                avg_c = st.session_state.avg_costs.get(sym, price_)
                pos_pnl = (price_ - avg_c) * qty
                pos_pnl_pct = (price_ - avg_c) / avg_c * 100 if avg_c else 0
                rows.append({
                    "Symbol": sym, "Qty": qty,
                    "Avg Cost": f"${avg_c:.2f}",
                    "Last":     f"${price_:.2f}",
                    "Mkt Val":  f"${val_:,.2f}",
                    "P&L $":    f"${pos_pnl:+,.2f}",
                    "P&L %":    f"{pos_pnl_pct:+.2f}%",
                    "Weight":   f"{val_/st.session_state.portfolio_value*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

            # Allocation pie
            labels = [r["Symbol"] for r in rows]
            values = [float(r["Mkt Val"].replace(
                "$", "").replace(",", "")) for r in rows]
            # Add cash
            labels.append("CASH")
            values.append(st.session_state.cash)

            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.6,
                marker=dict(colors=["#00ff94", "#388bfd", "#b06eff", "#ffb347",
                                    "#ff3250", "#00c8e0", "#ff6eb4", "#3d4e65"]),
                textfont=dict(family="JetBrains Mono", size=10),
            ))
            fig_pie.update_layout(**PLOTLY_BASE, height=280, showlegend=True,
                                  legend=dict(font=dict(size=9, color="#7e8fa8"),
                                              orientation="h", yanchor="bottom", y=-0.15))
            fig_pie.add_annotation(text="ALLOC", x=0.5, y=0.5, font_size=11,
                                   font_color="#7e8fa8", font_family="JetBrains Mono",
                                   showarrow=False)
            st.plotly_chart(fig_pie, use_container_width=True,
                            config={"displayModeBar": False})

            # Quick close buttons
            st.markdown('<div class="sec-label">Quick Close</div>',
                        unsafe_allow_html=True)
            qcols = st.columns(min(len(active), 4))
            for i, (sym, qty) in enumerate(list(active.items())[:4]):
                with qcols[i]:
                    if st.button(f"SELL {sym}", use_container_width=True):
                        d_ = fetch_data(sym, "5d")
                        if d_ is not None:
                            ok, msg = execute_sell(
                                sym, qty, float(d_["Close"].iloc[-1]))
                            (st.success if ok else st.error)(msg)
                            if ok:
                                st.rerun()
        else:
            st.info("No active positions. Execute trades from the Live Trading tab.")

    with col_h:
        st.markdown('<div class="sec-label">Trade History</div>',
                    unsafe_allow_html=True)
        if st.session_state.trade_history:
            for t in reversed(st.session_state.trade_history[-20:]):
                is_buy = t["action"] == "BUY"
                a_col = "#00ff94" if is_buy else "#ff3250"
                a_bg = "rgba(0,255,148,0.04)" if is_buy else "rgba(255,50,80,0.04)"
                a_bdr = "rgba(0,255,148,0.18)" if is_buy else "rgba(255,50,80,0.18)"
                ts = (t["time"].strftime("%m/%d %H:%M")
                      if isinstance(t["time"], datetime) else str(t["time"])[:16])
                pnl_str = f"<span style='color:{'#00ff94' if (t.get('pnl') or 0)>=0 else '#ff3250'};font-size:0.65rem;'>P&L ${t['pnl']:+,.0f}</span>" if t.get(
                    "pnl") is not None else ""
                st.markdown(f"""
                <div style="background:{a_bg};border:1px solid {a_bdr};border-radius:var(--r-md);
                    padding:0.5rem 0.8rem;margin-bottom:3px;
                    display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-family:var(--mono);font-size:0.7rem;font-weight:800;
                            color:{a_col};">{t['action']}</span>
                        <span style="font-family:var(--mono);font-size:0.7rem;color:var(--t2);
                            margin-left:0.4rem;">{t['symbol']}</span>
                        <div style="font-family:var(--mono);font-size:0.62rem;color:var(--t3);">
                            {ts}  {pnl_str}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:var(--mono);font-size:0.75rem;font-weight:700;
                            color:var(--t1);">${t['price']:.2f}</div>
                        <div style="font-family:var(--mono);font-size:0.65rem;color:var(--t2);">
                            {t['shares']} sh · ${t['total']:,.0f}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No trades yet.")

        # Stats
        if st.session_state.trade_count > 0:
            st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Trade Stats</div>',
                        unsafe_allow_html=True)
            won = st.session_state.win_count
            lost = st.session_state.loss_count
            ttl = won + lost
            pnl_trades = [t.get("pnl", 0) or 0 for t in st.session_state.trade_history if t.get(
                "pnl") is not None]
            avg_w = np.mean([p for p in pnl_trades if p > 0]) if any(
                p > 0 for p in pnl_trades) else 0
            avg_l = np.mean([p for p in pnl_trades if p <= 0]) if any(
                p <= 0 for p in pnl_trades) else 0
            pf = abs(sum(p for p in pnl_trades if p > 0) / sum(p for p in pnl_trades if p < 0)
                     ) if any(p < 0 for p in pnl_trades) else float("inf")
            stats_d = [
                ("Wins",           str(won), "#00ff94"),
                ("Losses",         str(lost), "#ff3250"),
                ("Avg Win $",      f"${avg_w:,.0f}", "#00ff94"),
                ("Avg Loss $",     f"${avg_l:,.0f}", "#ff3250"),
                ("Profit Factor",  f"{min(pf,99):.2f}", "#ffb347"),
                ("Max Drawdown",
                 f"{st.session_state.max_drawdown:.2f}%", "#ff3250"),
            ]
            for lbl, val, col_ in stats_d:
                st.markdown(f"""<div class="trade-row" style="margin-bottom:2px;">
                    <span style="font-size:0.68rem;color:var(--t2);">{lbl}</span>
                    <span style="font-family:var(--mono);font-weight:700;font-size:0.75rem;color:{col_};">{val}</span>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════
with tab_bt:
    df_bt = fetch_data(symbol, "2y")
    if df_bt is None or df_bt.empty:
        st.error("No data for backtest.")
    else:
        st.markdown(
            '<div class="sec-label">Backtest Configuration</div>', unsafe_allow_html=True)
        bc = st.columns(5)
        with bc[0]:
            bt_cap = st.number_input(
                "Capital ($)",    10_000, 10_000_000, 100_000, 10_000)
        with bc[1]:
            bt_buy = st.slider("Buy Threshold",   0.55, 0.90, 0.65, 0.01)
        with bc[2]:
            bt_sell = st.slider("Sell Threshold",  0.10, 0.45, 0.35, 0.01)
        with bc[3]:
            bt_sl = st.slider("Stop Loss %",     1, 25, 5)
        with bc[4]:
            bt_tp = st.slider("Take Profit %",   2, 100, 20)

        bt_comm = 0.001  # 0.1% commission

        if st.button("▶  RUN BACKTEST", type="primary"):
            with st.spinner("Running historical simulation with full risk management..."):
                btm = QuantMLEngine()
                btm.train(df_bt)
                res = btm.backtest(df_bt, bt_cap, bt_buy,
                                   bt_sell, bt_sl, bt_tp, bt_comm)

            st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Results</div>',
                        unsafe_allow_html=True)
            bc2 = st.columns(8)
            bt_kpis = [
                ("Final Value",     f"${res['final_value']:,.0f}",
                 "kpi-pos" if res["total_return"] >= 0 else "kpi-neg"),
                ("Total Return",    f"{res['total_return']:+.2f}%",
                 "kpi-pos" if res["total_return"] >= 0 else "kpi-neg"),
                ("vs BH",           f"{res['alpha']:+.2f}%",
                 "kpi-pos" if res["alpha"] >= 0 else "kpi-neg"),
                ("Sharpe",          f"{res['sharpe']:.3f}",
                 "kpi-pos" if res["sharpe"] >= 1 else ""),
                ("Sortino",         f"{res['sortino']:.3f}",
                 "kpi-pos" if res["sortino"] >= 1 else ""),
                ("Max DD",
                 f"{res['max_drawdown']:.2f}%",   "kpi-neg"),
                ("Win Rate",        f"{res['win_rate']:.1f}%",
                 "kpi-pos" if res["win_rate"] >= 50 else ""),
                ("# Trades",        str(res['num_trades']),          ""),
            ]
            for col, (lbl, val, cls) in zip(bc2, bt_kpis):
                with col:
                    st.markdown(kpi_box(lbl, val, cls), unsafe_allow_html=True)

            st.markdown("<div style='margin:0.75rem 0;'></div>",
                        unsafe_allow_html=True)
            c_left, c_right = st.columns([2.5, 1])

            with c_left:
                # Equity + drawdown
                fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       row_heights=[0.7, 0.3], vertical_spacing=0.03)
                eq_s = pd.Series(res["equity_curve"])
                eq_c = "#00ff94" if res["total_return"] >= 0 else "#ff3250"
                fig_eq.add_trace(go.Scatter(
                    x=res["equity_dates"] if len(res["equity_dates"]) == len(
                        eq_s) else list(range(len(eq_s))),
                    y=eq_s, mode="lines", name="Strategy",
                    line=dict(color=eq_c, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,255,148,0.06)" if res["total_return"] >= 0 else "rgba(255,50,80,0.06)",
                ), row=1, col=1)

                # BH benchmark
                bh_curve = [bt_cap * (float(df_bt["Close"].iloc[i]) / float(df_bt["Close"].iloc[0]))
                            for i in range(min(len(df_bt), len(eq_s)))]
                fig_eq.add_trace(go.Scatter(
                    x=list(df_bt.index[:len(bh_curve)]), y=bh_curve,
                    mode="lines", name="Buy & Hold",
                    line=dict(color="rgba(56,139,253,0.65)", width=1.5, dash="dot")), row=1, col=1)

                # Drawdown
                roll_max = eq_s.cummax()
                drawdown_ = (eq_s - roll_max) / roll_max * 100
                fig_eq.add_trace(go.Scatter(
                    y=drawdown_, mode="lines", name="Drawdown",
                    line=dict(color="rgba(255,50,80,0.7)", width=1),
                    fill="tozeroy", fillcolor="rgba(255,50,80,0.08)",
                    showlegend=False), row=2, col=1)

                # Trade markers
                buy_trades = [t for t in res["trades"] if t["type"] == "BUY"]
                sell_trades = [t for t in res["trades"] if t["type"]
                               == "SELL" and t.get("pnl") is not None]
                if buy_trades:
                    fig_eq.add_trace(go.Scatter(
                        x=[t["date"] for t in buy_trades],
                        y=[bt_cap * (float(df_bt.loc[t["date"], "Close"]) / float(df_bt["Close"].iloc[0]))
                           if t["date"] in df_bt.index else None for t in buy_trades],
                        mode="markers", name="Buy",
                        marker=dict(color="#00ff94", size=7, symbol="triangle-up")), row=1, col=1)
                if sell_trades:
                    fig_eq.add_trace(go.Scatter(
                        x=[t["date"] for t in sell_trades],
                        y=[bt_cap * (float(df_bt.loc[t["date"], "Close"]) / float(df_bt["Close"].iloc[0]))
                           if t["date"] in df_bt.index else None for t in sell_trades],
                        mode="markers", name="Sell",
                        marker=dict(color="#ff3250", size=7, symbol="triangle-down")), row=1, col=1)

                fig_eq.update_layout(**PLOTLY_BASE, height=480,
                                     legend=dict(font=dict(size=9, color="#7e8fa8"),
                                                 orientation="h", yanchor="bottom", y=1.02),
                                     xaxis={**GRID}, xaxis2={**GRID},
                                     yaxis={**GRID, "tickprefix": "$"},
                                     yaxis2={**GRID, "ticksuffix": "%"})
                st.plotly_chart(fig_eq, use_container_width=True,
                                config={"displayModeBar": False})

                # Returns distribution
                st.markdown('<div class="sec-label">Return Distribution</div>',
                            unsafe_allow_html=True)
                rets_arr = res["returns"] * 100
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=rets_arr, nbinsx=60,
                    marker=dict(color="rgba(0,255,148,0.5)",
                                line=dict(color="rgba(0,255,148,0.8)", width=0.5))
                ))
                fig_dist.add_vline(
                    x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                fig_dist.add_vline(x=float(rets_arr.mean()), line_color="#ffb347",
                                   line_dash="dot", line_width=1.5,
                                   annotation_text=f"μ={rets_arr.mean():.3f}%",
                                   annotation_font_color="#ffb347", annotation_font_size=9)
                fig_dist.update_layout(**PLOTLY_BASE, height=200,
                                       xaxis={**GRID, "ticksuffix": "%"},
                                       yaxis={"showgrid": False}, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True,
                                config={"displayModeBar": False})

            with c_right:
                st.markdown('<div class="sec-label">Detailed Stats</div>',
                            unsafe_allow_html=True)
                detail_stats = [
                    ("Total Return",     f"{res['total_return']:+.2f}%",
                     "#00ff94" if res["total_return"] >= 0 else "#ff3250"),
                    ("Buy & Hold",
                     f"{res['bh_return']:+.2f}%",     "#388bfd"),
                    ("Alpha",            f"{res['alpha']:+.2f}%",
                     "#00ff94" if res["alpha"] >= 0 else "#ff3250"),
                    ("Sharpe Ratio",     f"{res['sharpe']:.3f}",
                     "#00ff94" if res["sharpe"] >= 1 else "#ffb347"),
                    ("Sortino Ratio",    f"{res['sortino']:.3f}",
                     "#00ff94" if res["sortino"] >= 1 else "#ffb347"),
                    ("Calmar Ratio",     f"{res['calmar']:.3f}",
                     "#00ff94" if res["calmar"] >= 0.5 else "#ffb347"),
                    ("Max Drawdown",
                     f"{res['max_drawdown']:.2f}%",   "#ff3250"),
                    ("Win Rate",         f"{res['win_rate']:.1f}%",
                     "#00ff94" if res["win_rate"] >= 50 else "#ff3250"),
                    ("Avg Win",
                     f"${res['avg_win']:,.0f}",        "#00ff94"),
                    ("Avg Loss",
                     f"${res['avg_loss']:,.0f}",       "#ff3250"),
                    ("Profit Factor",    f"{min(res['profit_factor'],99):.2f}",
                     "#00ff94" if res["profit_factor"] >= 1 else "#ff3250"),
                    ("Num Buys",         str(
                        res["num_buys"]),             "#7e8fa8"),
                    ("Num Sells",        str(
                        res["num_sells"]),            "#7e8fa8"),
                    ("Commission",
                     f"{bt_comm*100:.1f}%",           "#7e8fa8"),
                ]
                for lbl, val, col_ in detail_stats:
                    st.markdown(f"""<div class="trade-row" style="margin-bottom:2px;">
                        <span style="font-size:0.65rem;color:var(--t2);">{lbl}</span>
                        <span style="font-family:var(--mono);font-weight:700;font-size:0.73rem;color:{col_};">{val}</span>
                    </div>""", unsafe_allow_html=True)

                # Monthly returns
                st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Monthly Returns</div>',
                            unsafe_allow_html=True)
                eq_df = pd.Series(res["equity_curve"],
                                  index=res["equity_dates"] if len(res["equity_dates"]) == len(res["equity_curve"]) else range(len(res["equity_curve"])))
                try:
                    eq_df.index = pd.to_datetime(eq_df.index)
                    monthly = eq_df.resample(
                        "ME").last().pct_change().dropna() * 100
                    for period, ret in monthly.tail(12).items():
                        col_ = "#00ff94" if ret >= 0 else "#ff3250"
                        st.markdown(f"""<div class="trade-row" style="margin-bottom:2px;">
                            <span style="font-size:0.65rem;color:var(--t2);">{period.strftime('%b %Y')}</span>
                            <span style="font-family:var(--mono);font-weight:700;font-size:0.73rem;color:{col_};">{ret:+.2f}%</span>
                        </div>""", unsafe_allow_html=True)
                except Exception:
                    st.caption("Monthly breakdown unavailable")
        else:
            st.markdown("""
            <div style="background:var(--bg2);border:1px dashed rgba(0,255,148,0.2);
                border-radius:var(--r-xl);padding:2.5rem;text-align:center;margin-top:0.5rem;">
                <div style="font-family:var(--mono);font-size:0.85rem;color:var(--t3);">
                    Configure parameters above and click
                    <strong style="color:#00ff94;">▶ RUN BACKTEST</strong>
                </div>
                <div style="font-family:var(--mono);font-size:0.68rem;color:var(--t4);margin-top:0.4rem;">
                    Features: commission modeling · stop-loss · take-profit · buy&hold comparison · full risk metrics
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — MARKET SCAN
# ═══════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown('<div class="sec-label">Market Scanner — Watchlist</div>',
                unsafe_allow_html=True)

    scan_col, scan_opt = st.columns([3, 1])
    with scan_opt:
        scan_period = st.selectbox(
            "Scan Period", ["3mo", "6mo", "1y"], index=1)
        if st.button("🔄  SCAN NOW", type="primary", use_container_width=True):
            st.session_state.last_scan_time = datetime.now()

    with st.spinner("Scanning watchlist..."):
        scan_data = fetch_multi(st.session_state.watchlist, scan_period)

    scan_results = []
    for sym in st.session_state.watchlist:
        if sym not in scan_data:
            continue
        d_ = scan_data[sym]
        if len(d_) < 50:
            continue
        try:
            c_ = d_["Close"]
            ret_1d = (c_.iloc[-1] - c_.iloc[-2]) / c_.iloc[-2] * 100
            ret_1w = (c_.iloc[-1] - c_.iloc[-6]) / \
                c_.iloc[-6] * 100 if len(c_) >= 6 else 0
            ret_1m = (c_.iloc[-1] - c_.iloc[-22]) / \
                c_.iloc[-22] * 100 if len(c_) >= 22 else 0
            rsi_ = float(ta.momentum.RSIIndicator(c_).rsi().iloc[-1])
            macd_obj2 = ta.trend.MACD(c_)
            macd_hist_ = float(macd_obj2.macd_diff().iloc[-1])
            adx_ = float(ta.trend.ADXIndicator(
                d_["High"], d_["Low"], c_).adx().iloc[-1])
            vol_ = float(d_["Volume"].iloc[-1] /
                         d_["Volume"].rolling(20).mean().iloc[-1])
            ema20_ = float(c_.ewm(span=20).mean().iloc[-1])
            ema50_ = float(c_.ewm(span=50).mean().iloc[-1])
            price_ = float(c_.iloc[-1])
            score = 0
            if ret_1d > 0:
                score += 1
            if ret_1w > 0:
                score += 1
            if 40 < rsi_ < 65:
                score += 1
            if macd_hist_ > 0:
                score += 1
            if adx_ > 25:
                score += 1
            if ema20_ > ema50_:
                score += 1
            if vol_ > 1.2:
                score += 1
            scan_results.append({
                "sym": sym, "price": price_, "ret_1d": ret_1d,
                "ret_1w": ret_1w, "ret_1m": ret_1m, "rsi": rsi_,
                "macd_hist": macd_hist_, "adx": adx_, "vol_ratio": vol_,
                "trend": "↑" if ema20_ > ema50_ else "↓",
                "score": score,
            })
        except Exception:
            continue

    scan_results.sort(key=lambda x: x["score"], reverse=True)

    # Scan table
    st.markdown('<div class="sec-label">Ranked Results (by Score)</div>',
                unsafe_allow_html=True)
    hdr = st.columns([1.2, 1, 1, 1, 1, 1, 1, 1, 1])
    for col, lbl in zip(hdr, ["Symbol", "Price", "1D", "1W", "1M", "RSI", "ADX", "Vol×", "Score"]):
        col.markdown(f"<div style='font-family:var(--mono);font-size:0.6rem;font-weight:700;"
                     f"color:var(--t3);text-transform:uppercase;letter-spacing:0.1em;'>{lbl}</div>",
                     unsafe_allow_html=True)

    for r in scan_results:
        cols = st.columns([1.2, 1, 1, 1, 1, 1, 1, 1, 1])
        c1d = "#00ff94" if r["ret_1d"] >= 0 else "#ff3250"
        c1w = "#00ff94" if r["ret_1w"] >= 0 else "#ff3250"
        c1m = "#00ff94" if r["ret_1m"] >= 0 else "#ff3250"
        cr = "#ff3250" if r["rsi"] > 70 else (
            "#00ff94" if r["rsi"] < 30 else "#ffb347")
        cd = "#00ff94" if r["adx"] > 25 else "#7e8fa8"
        cv = "#00ff94" if r["vol_ratio"] > 1.5 else "#7e8fa8"
        sc_color = "#00ff94" if r["score"] >= 5 else "#ffb347" if r["score"] >= 3 else "#ff3250"

        vals = [
            (r["sym"],           "#dde4f0"),
            (f"${r['price']:.2f}", "#7e8fa8"),
            (f"{r['ret_1d']:+.2f}%", c1d),
            (f"{r['ret_1w']:+.2f}%", c1w),
            (f"{r['ret_1m']:+.2f}%", c1m),
            (f"{r['rsi']:.0f}", cr),
            (f"{r['adx']:.0f} {r['trend']}", cd),
            (f"{r['vol_ratio']:.2f}×", cv),
            (f"{'█'*r['score']}{'░'*(7-r['score'])} {r['score']}/7", sc_color),
        ]
        for col, (val, c_) in zip(cols, vals):
            col.markdown(f"<div style='font-family:var(--mono);font-size:0.73rem;"
                         f"font-weight:700;color:{c_};padding:0.3rem 0;'>{val}</div>",
                         unsafe_allow_html=True)

    if not scan_results:
        st.info("Add symbols to watchlist in sidebar, then scan.")

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    # Heatmap
    st.markdown('<div class="sec-label">1-Day Return Heatmap</div>',
                unsafe_allow_html=True)
    if scan_results:
        max_abs = max(abs(r["ret_1d"]) for r in scan_results) or 1
        hcols = st.columns(min(len(scan_results), 8))
        for i, r in enumerate(scan_results[:8]):
            with hcols[i]:
                intensity = abs(r["ret_1d"]) / max_abs
                if r["ret_1d"] >= 0:
                    bg = f"rgba(0,255,148,{0.08 + 0.35*intensity:.2f})"
                    bc = "rgba(0,255,148,0.4)"
                    tc = "#00ff94"
                else:
                    bg = f"rgba(255,50,80,{0.08 + 0.35*intensity:.2f})"
                    bc = "rgba(255,50,80,0.4)"
                    tc = "#ff3250"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bc};border-radius:var(--r-lg);
                    padding:0.9rem 0.5rem;text-align:center;margin-bottom:0.5rem;">
                    <div style="font-family:var(--mono);font-size:0.7rem;font-weight:800;
                        color:var(--t1);">{r['sym']}</div>
                    <div style="font-family:var(--mono);font-size:1.1rem;font-weight:700;
                        color:{tc};margin:0.15rem 0;">{r['ret_1d']:+.2f}%</div>
                    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);">
                        RSI {r['rsi']:.0f}</div>
                </div>""", unsafe_allow_html=True)

    # Correlation matrix
    if len(scan_data) >= 3:
        st.markdown('<div class="sec-label" style="margin-top:0.5rem;">Correlation Matrix (Closing Returns)</div>',
                    unsafe_allow_html=True)
        ret_df = pd.DataFrame()
        for sym, d_ in list(scan_data.items())[:10]:
            ret_df[sym] = d_["Close"].pct_change()
        ret_df = ret_df.dropna()
        corr = ret_df.corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#ff3250"], [0.5, "#0a0f1e"], [1, "#00ff94"]],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=9),
            colorbar=dict(tickfont=dict(family="JetBrains Mono", size=8)),
        ))
        fig_corr.update_layout(**PLOTLY_BASE, height=360,
                               xaxis=dict(tickfont=dict(size=9)),
                               yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_corr, use_container_width=True,
                        config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════
# TAB 6 — OPTIONS
# ═══════════════════════════════════════════════════════════════
with tab_options:
    st.markdown('<div class="sec-label">Options Chain</div>',
                unsafe_allow_html=True)
    calls, puts = fetch_options_chain(symbol)

    if calls is not None and puts is not None and not calls.empty:
        cur_p = fetch_info(symbol)["price"]
        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--bdrG);border-radius:var(--r-lg);
            padding:0.75rem 1rem;margin-bottom:0.75rem;font-family:var(--mono);font-size:0.75rem;
            color:var(--t2);">
            Underlying <span style="color:var(--t1);font-weight:700;">{symbol}</span>
            ·  Last <span style="color:#00ff94;font-weight:700;">${cur_p:.2f}</span>
            ·  Showing nearest expiry
        </div>""", unsafe_allow_html=True)

        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown('<div class="sec-label">Calls</div>',
                        unsafe_allow_html=True)
            calls_show = calls[["strike", "lastPrice", "bid", "ask", "volume", "openInterest",
                                "impliedVolatility", "inTheMoney"]].copy()
            calls_show["IV%"] = (
                calls_show["impliedVolatility"] * 100).round(1)
            calls_show = calls_show.rename(columns={
                "lastPrice": "Last", "openInterest": "OI", "inTheMoney": "ITM"
            }).drop(columns=["impliedVolatility"])
            # highlight ITM

            def highlight_itm(row):
                if row["ITM"]:
                    return ["background-color:rgba(0,255,148,0.06)"]*len(row)
                return [""]*len(row)
            st.dataframe(calls_show.style.apply(highlight_itm, axis=1),
                         use_container_width=True, hide_index=True)

        with oc2:
            st.markdown('<div class="sec-label">Puts</div>',
                        unsafe_allow_html=True)
            puts_show = puts[["strike", "lastPrice", "bid", "ask", "volume", "openInterest",
                              "impliedVolatility", "inTheMoney"]].copy()
            puts_show["IV%"] = (puts_show["impliedVolatility"] * 100).round(1)
            puts_show = puts_show.rename(columns={
                "lastPrice": "Last", "openInterest": "OI", "inTheMoney": "ITM"
            }).drop(columns=["impliedVolatility"])
            st.dataframe(puts_show, use_container_width=True, hide_index=True)

        # IV smile
        st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Implied Volatility Smile</div>',
                    unsafe_allow_html=True)
        fig_iv = go.Figure()
        calls_iv = calls.dropna(subset=["impliedVolatility"])
        puts_iv = puts.dropna(subset=["impliedVolatility"])
        fig_iv.add_trace(go.Scatter(
            x=calls_iv["strike"], y=calls_iv["impliedVolatility"]*100,
            mode="lines+markers", name="Calls IV",
            line=dict(color="#00ff94", width=2),
            marker=dict(size=5)))
        fig_iv.add_trace(go.Scatter(
            x=puts_iv["strike"], y=puts_iv["impliedVolatility"]*100,
            mode="lines+markers", name="Puts IV",
            line=dict(color="#ff3250", width=2),
            marker=dict(size=5)))
        fig_iv.add_vline(x=cur_p, line_dash="dot",
                         line_color="rgba(255,179,71,0.7)", line_width=1.5,
                         annotation_text=f"SPOT ${cur_p:.2f}",
                         annotation_font_color="#ffb347", annotation_font_size=9)
        fig_iv.update_layout(**PLOTLY_BASE, height=320,
                             xaxis={**GRID, "tickprefix": "$"},
                             yaxis={**GRID, "ticksuffix": "%", "title": "IV %"},
                             legend=dict(font=dict(size=9, color="#7e8fa8")))
        st.plotly_chart(fig_iv, use_container_width=True,
                        config={"displayModeBar": False})

        # Open Interest profile
        oc3, oc4 = st.columns(2)
        with oc3:
            st.markdown('<div class="sec-label">Open Interest — Calls vs Puts</div>',
                        unsafe_allow_html=True)
            strikes = sorted(
                set(calls["strike"]).intersection(puts["strike"]))[:30]
            c_oi = calls.set_index("strike")[
                "openInterest"].reindex(strikes).fillna(0)
            p_oi = puts.set_index("strike")[
                "openInterest"].reindex(strikes).fillna(0)
            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(name="Calls", x=strikes, y=c_oi,
                                    marker_color="rgba(0,255,148,0.55)"))
            fig_oi.add_trace(go.Bar(name="Puts",  x=strikes, y=p_oi,
                                    marker_color="rgba(255,50,80,0.55)"))
            fig_oi.add_vline(x=cur_p, line_dash="dot",
                             line_color="rgba(255,179,71,0.6)", line_width=1)
            fig_oi.update_layout(**PLOTLY_BASE, height=300, barmode="group",
                                 xaxis={**GRID, "tickprefix": "$"},
                                 yaxis={**GRID},
                                 legend=dict(font=dict(size=9, color="#7e8fa8")))
            st.plotly_chart(fig_oi, use_container_width=True,
                            config={"displayModeBar": False})

        with oc4:
            st.markdown('<div class="sec-label">Put/Call Ratio by Strike</div>',
                        unsafe_allow_html=True)
            pc_ratio = (p_oi / (c_oi + 1e-9)).clip(0, 5)
            pc_colors = ["#ff3250" if v > 1 else "#00ff94" for v in pc_ratio]
            fig_pcr = go.Figure(go.Bar(
                x=strikes, y=pc_ratio,
                marker_color=pc_colors,
            ))
            fig_pcr.add_hline(y=1, line_dash="dot",
                              line_color="rgba(255,255,255,0.25)", line_width=1)
            fig_pcr.update_layout(**PLOTLY_BASE, height=300,
                                  xaxis={**GRID, "tickprefix": "$"},
                                  yaxis={**GRID})
            st.plotly_chart(fig_pcr, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.info(f"No options data available for {symbol}. "
                "Options data is available for US equities with active options markets.")

# ═══════════════════════════════════════════════════════════════
# TAB 7 — NEWS & DATA
# ═══════════════════════════════════════════════════════════════
with tab_news:
    info7 = fetch_info(symbol)
    df7 = fetch_data(symbol, "1y")

    nc1, nc2 = st.columns([1.6, 1])

    with nc1:
        st.markdown('<div class="sec-label">Fundamentals</div>',
                    unsafe_allow_html=True)
        f_data = [
            ("Market Cap",       fmt_large(
                info7["mktcap"]) if info7["mktcap"] else "N/A"),
            ("Revenue",          fmt_large(
                info7["revenue"]) if info7["revenue"] else "N/A"),
            ("Trailing P/E",
             f"{info7['pe']:.2f}" if info7["pe"] else "N/A"),
            ("Forward P/E",
             f"{info7['fwd_pe']:.2f}" if info7["fwd_pe"] else "N/A"),
            ("PEG Ratio",
             f"{info7['peg']:.2f}" if info7["peg"] else "N/A"),
            ("Price/Sales",
             f"{info7['ps']:.2f}" if info7["ps"] else "N/A"),
            ("Price/Book",
             f"{info7['pb']:.2f}" if info7["pb"] else "N/A"),
            ("EPS (TTM)",
             f"${info7['eps']:.2f}" if info7["eps"] else "N/A"),
            ("ROE",
             f"{info7['roe']*100:.2f}%" if info7["roe"] else "N/A"),
            ("Profit Margin",
             f"{info7['profit_margin']*100:.2f}%" if info7["profit_margin"] else "N/A"),
            ("Gross Margin",
             f"{info7['gross_margin']*100:.2f}%" if info7["gross_margin"] else "N/A"),
            ("Debt/Equity",
             f"{info7['debt_equity']:.2f}" if info7["debt_equity"] else "N/A"),
            ("Current Ratio",
             f"{info7['current_ratio']:.2f}" if info7["current_ratio"] else "N/A"),
            ("Beta",
             f"{info7['beta']:.2f}" if info7["beta"] else "N/A"),
            ("Dividend Yield",
             f"{info7['div']*100:.2f}%" if info7["div"] else "N/A"),
            ("52W High",
             f"${info7['hi52']:.2f}" if info7["hi52"] else "N/A"),
            ("52W Low",
             f"${info7['lo52']:.2f}" if info7["lo52"] else "N/A"),
            ("Analyst Target",
             f"${info7['analyst_target']:.2f}" if info7["analyst_target"] else "N/A"),
            ("Recommendation",   info7["rec"].upper()),
            ("Volume",
             f"{info7['volume']:,}" if info7["volume"] else "N/A"),
        ]
        fc_cols = st.columns(4)
        for i, (lbl, val) in enumerate(f_data):
            color = "#7e8fa8"
            if "%" in val and val not in ("N/A",):
                try:
                    num = float(val.replace("%", "").replace(
                        "$", "").replace(",", ""))
                    color = "#00ff94" if num > 0 else "#ff3250"
                except Exception:
                    pass
            with fc_cols[i % 4]:
                st.markdown(f"""
                <div style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r-md);
                    padding:0.65rem 0.75rem;margin-bottom:4px;">
                    <div style="font-family:var(--mono);font-size:0.56rem;color:var(--t3);
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;">{lbl}</div>
                    <div style="font-family:var(--mono);font-size:0.88rem;font-weight:700;color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)

        if df7 is not None and not df7.empty:
            st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Statistical Summary</div>',
                        unsafe_allow_html=True)
            rets7 = df7["Close"].pct_change().dropna()
            vol_ann = rets7.std() * np.sqrt(252) * 100
            ret_ann = rets7.mean() * 252 * 100
            sh_r = sharpe(rets7)
            so_r = sortino(rets7)
            skew = float(rets7.skew())
            kurt = float(rets7.kurt())
            best = float(rets7.max()) * 100
            worst = float(rets7.min()) * 100
            vwap = float((df7["Close"] * df7["Volume"]
                          ).sum() / df7["Volume"].sum())
            vs200 = (float(df7["Close"].iloc[-1]) /
                     float(df7["Close"].rolling(200).mean().iloc[-1]) - 1) * 100

            stat_cols = st.columns(3)
            stats_list = [
                ("Annualized Return",    f"{ret_ann:+.2f}%",
                 "#00ff94" if ret_ann >= 0 else "#ff3250"),
                ("Annualized Volatility", f"{vol_ann:.2f}%",    "#ffb347"),
                ("Sharpe Ratio",         f"{sh_r:.3f}",
                 "#00ff94" if sh_r >= 1 else "#ffb347"),
                ("Sortino Ratio",        f"{so_r:.3f}",
                 "#00ff94" if so_r >= 1 else "#ffb347"),
                ("Skewness",             f"{skew:.3f}",
                 "#00ff94" if skew > 0 else "#ff3250"),
                ("Excess Kurtosis",      f"{kurt:.3f}",         "#7e8fa8"),
                ("Best Day",             f"{best:+.2f}%",       "#00ff94"),
                ("Worst Day",            f"{worst:+.2f}%",      "#ff3250"),
                ("VWAP (1Y)",            f"${vwap:.2f}",        "#7e8fa8"),
                ("vs SMA200",            f"{vs200:+.1f}%",
                 "#00ff94" if vs200 >= 0 else "#ff3250"),
                ("Avg Daily Volume",
                 f"{df7['Volume'].mean():,.0f}", "#7e8fa8"),
                ("Avg Dollar Volume",    fmt_large(
                    float((df7["Close"]*df7["Volume"]).mean())), "#7e8fa8"),
            ]
            for i, (lbl, val, col_) in enumerate(stats_list):
                with stat_cols[i % 3]:
                    st.markdown(f"""<div class="trade-row" style="margin-bottom:2px;">
                        <span style="font-size:0.65rem;color:var(--t2);">{lbl}</span>
                        <span style="font-family:var(--mono);font-weight:700;font-size:0.73rem;color:{col_};">{val}</span>
                    </div>""", unsafe_allow_html=True)

            # Rolling metrics chart
            st.markdown('<div class="sec-label" style="margin-top:0.75rem;">Rolling 30-Day Sharpe & Vol</div>',
                        unsafe_allow_html=True)
            roll_sh = rets7.rolling(30).apply(
                lambda x: sharpe(pd.Series(x)), raw=False)
            roll_vol = rets7.rolling(30).std() * np.sqrt(252) * 100

            fig_roll = make_subplots(rows=1, cols=2)
            fig_roll.add_trace(go.Scatter(x=df7.index[-len(roll_sh):], y=roll_sh,
                                          line=dict(
                                              color="#00ff94", width=1.5),
                                          showlegend=False, name="Sharpe"), row=1, col=1)
            fig_roll.add_hline(y=1, line_dash="dot", line_color="rgba(0,255,148,0.4)",
                               line_width=1, row=1, col=1)
            fig_roll.add_trace(go.Scatter(x=df7.index[-len(roll_vol):], y=roll_vol,
                                          line=dict(
                                              color="#ffb347", width=1.5),
                                          showlegend=False, name="Vol"), row=1, col=2)
            fig_roll.update_layout(**PLOTLY_BASE, height=220,
                                   xaxis={**GRID}, xaxis2={**GRID},
                                   yaxis={**GRID, "title": "Sharpe"},
                                   yaxis2={**GRID, "title": "Annualized Vol %"})
            st.plotly_chart(fig_roll, use_container_width=True,
                            config={"displayModeBar": False})

    with nc2:
        st.markdown('<div class="sec-label">52-Week Price Channel</div>',
                    unsafe_allow_html=True)
        if df7 is not None and not df7.empty:
            df7["rh20"] = df7["High"].rolling(20).max()
            df7["rl20"] = df7["Low"].rolling(20).min()
            fig_ch = go.Figure()
            fig_ch.add_trace(go.Scatter(x=df7.index, y=df7["rh20"], fill=None,
                                        mode="lines", line=dict(color="rgba(0,255,148,0.15)", width=0),
                                        showlegend=False))
            fig_ch.add_trace(go.Scatter(x=df7.index, y=df7["rl20"],
                                        fill="tonexty", fillcolor="rgba(0,255,148,0.03)",
                                        mode="lines", line=dict(color="rgba(0,255,148,0.15)", width=0),
                                        showlegend=False))
            fig_ch.add_trace(go.Scatter(x=df7.index, y=df7["Close"],
                                        mode="lines", line=dict(color="#dde4f0", width=1.5),
                                        showlegend=False))
            fig_ch.update_layout(**PLOTLY_BASE, height=240,
                                 xaxis={**GRID}, yaxis={**GRID, "tickprefix": "$"})
            st.plotly_chart(fig_ch, use_container_width=True,
                            config={"displayModeBar": False})

        # 52W range bar
        if info7["hi52"] and info7["lo52"] and info7["price"]:
            rng = info7["hi52"] - info7["lo52"]
            pos = (info7["price"] - info7["lo52"]) / \
                rng * 100 if rng > 0 else 50
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--r-lg);
                padding:0.85rem;margin-bottom:0.65rem;">
                <div style="font-family:var(--mono);font-size:0.58rem;color:var(--t3);
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">52W Position</div>
                <div style="display:flex;justify-content:space-between;
                    font-family:var(--mono);font-size:0.65rem;margin-bottom:0.3rem;">
                    <span style="color:#ff3250;">${info7['lo52']:.2f}</span>
                    <span style="color:#00ff94;font-weight:700;">${info7['price']:.2f}
                        ({pos:.0f}%)</span>
                    <span style="color:#00ff94;">${info7['hi52']:.2f}</span>
                </div>
                <div style="background:var(--bg0);border-radius:3px;height:8px;overflow:hidden;
                    position:relative;">
                    <div style="width:{pos:.0f}%;height:100%;
                        background:linear-gradient(90deg,#ff3250,#ffb347,#00ff94);
                        border-radius:3px;"></div>
                    <div style="position:absolute;top:0;left:{pos:.0f}%;
                        height:100%;width:2px;background:#fff;opacity:0.8;
                        transform:translateX(-1px);"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Peer comparison
        st.markdown('<div class="sec-label">Index Correlation (3mo)</div>',
                    unsafe_allow_html=True)
        peers_d = fetch_multi(["SPY", "QQQ", "IWM", "DIA", "VIX"], "3mo")
        if df7 is not None:
            sym_r = df7["Close"].pct_change().dropna()
            for peer, d_p in peers_d.items():
                p_r = d_p["Close"].pct_change().dropna()
                al = pd.concat([sym_r, p_r], axis=1).dropna()
                if len(al) < 10:
                    continue
                corr_ = float(al.iloc[:, 0].corr(al.iloc[:, 1]))
                corr_c = "#00ff94" if corr_ > 0.6 else "#ffb347" if corr_ > 0.3 else "#ff3250"
                st.markdown(f"""
                <div class="trade-row" style="margin-bottom:2px;">
                    <span style="font-size:0.68rem;color:var(--t2);">vs {peer}</span>
                    <div>
                        <span style="font-family:var(--mono);font-weight:700;
                            font-size:0.75rem;color:{corr_c};">{corr_:.3f}</span>
                        <div style="background:var(--bg0);border-radius:2px;height:3px;
                            width:80px;margin-top:2px;overflow:hidden;">
                            <div style="width:{abs(corr_)*100:.0f}%;height:100%;
                                background:{corr_c};border-radius:2px;"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Volume profile
        st.markdown('<div class="sec-label" style="margin-top:0.65rem;">Volume Profile (60d)</div>',
                    unsafe_allow_html=True)
        if df7 is not None and len(df7) >= 20:
            df7["vma20"] = df7["Volume"].rolling(20).mean()
            v_colors = ["rgba(0,255,148,0.5)" if c >= o else "rgba(255,50,80,0.5)"
                        for c, o in zip(df7["Close"].tail(60), df7["Open"].tail(60))]
            fig_vp = go.Figure()
            fig_vp.add_trace(go.Bar(x=df7.index[-60:], y=df7["Volume"].tail(60),
                                    marker_color=v_colors, showlegend=False))
            fig_vp.add_trace(go.Scatter(x=df7.index[-60:], y=df7["vma20"].tail(60),
                                        line=dict(color="#ffb347",
                                                  width=1.5, dash="dot"),
                                        showlegend=False))
            fig_vp.update_layout(**PLOTLY_BASE, height=200,
                                 xaxis=dict(showgrid=False),
                                 yaxis={**GRID})
            st.plotly_chart(fig_vp, use_container_width=True,
                            config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:1.75rem;padding:0.85rem 1.25rem;background:var(--bg1);
    border:1px solid var(--bdr);border-radius:var(--r-lg);
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--t3);">
        QUANT.AI v3.0  ·  Ensemble ML: RF + GB + ET + LR  ·  80+ Features  ·
        TimeSeriesSplit CV  ·  Options Chain  ·  Market Scanner  ·  Full Risk Mgmt
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--t4);">
        ⚠ Educational purposes only  ·  Not financial advice  ·  Past performance ≠ future results
    </div>
</div>""", unsafe_allow_html=True)
