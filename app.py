
from __future__ import annotations

import io
import json
import math
import re
import sqlite3
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except Exception:
    FPDF_OK = False

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantPilot AI — Autonomous Quant Trading Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&family=Space+Grotesk:wght@400..700&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
  --void:#050810; --bg0:#080c16; --bg1:#0c1120; --bg2:#111a2e; --bg3:#16203a; --bg4:#1b2745;
  --bdr:rgba(255,255,255,0.06); --bdrG:rgba(0,230,150,0.25); --bdrR:rgba(255,70,90,0.25); --bdrB:rgba(70,140,255,0.25);
  --green:#00e696; --greenDim:#00b876; --red:#ff4a5a; --redDim:#d63a48; --amber:#ffb545; --blue:#4d8cff; --cyan:#22d3ee; --purple:#a06bff;
  --t1:#e6ecf6; --t2:#8b98b4; --t3:#4a5670; --t4:#232f47;
  --mono:'JetBrains Mono',monospace; --display:'Space Grotesk',sans-serif; --body:'Inter',sans-serif;
  --r-sm:4px; --r-md:10px; --r-lg:14px; --r-xl:18px;
}
*{box-sizing:border-box;}
html,body,.stApp{background:var(--void)!important; color:var(--t1);}
.main .block-container{padding:0.9rem 1.4rem 2.4rem!important; max-width:1720px!important;}
#MainMenu, footer, header, .stDeployButton{visibility:hidden; display:none;}
section[data-testid="stSidebar"]{background:var(--bg0)!important; border-right:1px solid var(--bdr);}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
h1,h2,h3,h4{font-family:var(--display); color:var(--t1); letter-spacing:-0.02em;}
p,span,div,label{font-family:var(--body);}
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"]{
  background:var(--bg2)!important; color:var(--t1)!important; border:1px solid var(--bdr)!important; border-radius:var(--r-md)!important;
  font-family:var(--mono)!important;
}
.stButton>button{
  border-radius:var(--r-md)!important; border:1px solid var(--bdr)!important; font-family:var(--mono)!important;
  font-weight:600!important; letter-spacing:0.04em; background:var(--bg3)!important; color:var(--t1)!important;
}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--green),var(--greenDim))!important; color:#03110b!important; border:none!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px; border-bottom:1px solid var(--bdr);}
.stTabs [data-baseweb="tab"]{
  background:var(--bg1); border-radius:var(--r-md) var(--r-md) 0 0; color:var(--t2); font-family:var(--mono);
  font-size:0.78rem; font-weight:600; padding:0.55rem 1rem;
}
.stTabs [aria-selected="true"]{background:var(--bg3)!important; color:var(--green)!important;}
.card{background:linear-gradient(180deg,var(--bg2),var(--bg1)); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:1rem 1.1rem;}
.kpi{background:var(--bg2); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:0.75rem 0.95rem;}
.kpi .lab{font-family:var(--mono); font-size:0.62rem; color:var(--t3); text-transform:uppercase; letter-spacing:0.1em;}
.kpi .val{font-family:var(--mono); font-size:1.25rem; font-weight:700; color:var(--t1); margin-top:0.15rem;}
.kpi .pos{color:var(--green)!important;} .kpi .neg{color:var(--red)!important;}
.pill{font-family:var(--mono); font-size:0.62rem; font-weight:700; padding:3px 10px; border-radius:20px; letter-spacing:0.08em;}
.pill-ok{background:rgba(0,230,150,0.12); color:var(--green); border:1px solid var(--bdrG);}
.pill-warn{background:rgba(255,181,69,0.12); color:var(--amber); border:1px solid rgba(255,181,69,0.3);}
.pill-off{background:rgba(139,152,180,0.1); color:var(--t2); border:1px solid var(--bdr);}
.agent-row{display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0.6rem; border-radius:var(--r-md); margin-bottom:3px; font-family:var(--mono); font-size:0.74rem;}
.agent-done{background:rgba(0,230,150,0.06); color:var(--green);}
.agent-run{background:rgba(77,140,255,0.08); color:var(--blue);}
.agent-pend{background:transparent; color:var(--t3);}
.sec-label{font-family:var(--mono); font-size:0.66rem; color:var(--t3); text-transform:uppercase; letter-spacing:0.14em; margin:0.7rem 0 0.35rem 0;}
hr{border-color:var(--bdr)!important;}
.streamlit-expanderHeader{font-family:var(--mono)!important; font-size:0.8rem!important;}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE — trade journal / paper-trading persistence / agent logs
# ═══════════════════════════════════════════════════════════════════════════
DB_PATH = Path.home() / ".quantpilot_ai.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS paper_trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT,
        qty REAL, price REAL, value REAL, cash_after REAL, reason TEXT, session_id TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS agent_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, agent TEXT, message TEXT, level TEXT, session_id TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS strategy_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, idea TEXT, symbol TEXT, strategy_type TEXT,
        params TEXT, net_profit REAL, sharpe REAL, win_rate REAL, max_dd REAL, session_id TEXT)""")
    conn.commit()
    conn.close()


init_db()

# ═══════════════════════════════════════════════════════════════════════════
# ASSET RESOLUTION — maps natural language to tickers / asset classes
# ═══════════════════════════════════════════════════════════════════════════
ASSET_ALIASES = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD", "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "ripple": "XRP-USD", "xrp": "XRP-USD", "litecoin": "LTC-USD", "cardano": "ADA-USD",
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "meta": "META", "facebook": "META", "nvidia": "NVDA", "netflix": "NFLX",
    "amd": "AMD", "intel": "INTC", "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS", "nifty": "^NSEI", "nifty50": "^NSEI", "banknifty": "^NSEBANK",
    "sensex": "^BSESN", "s&p500": "^GSPC", "sp500": "^GSPC", "spx": "^GSPC", "dowjones": "^DJI",
    "dow": "^DJI", "nasdaq": "^IXIC", "gold": "GC=F", "silver": "SI=F", "crudeoil": "CL=F",
    "oil": "CL=F", "naturalgas": "NG=F", "eurusd": "EURUSD=X", "gbpusd": "GBPUSD=X",
    "usdjpy": "USDJPY=X", "usdinr": "USDINR=X", "audusd": "AUDUSD=X", "spy": "SPY", "qqq": "QQQ",
}

CRYPTO_SET = {"BTC-USD", "ETH-USD", "SOL-USD",
              "DOGE-USD", "XRP-USD", "LTC-USD", "ADA-USD"}


def classify_asset(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("-USD") or s in CRYPTO_SET:
        return "Cryptocurrency"
    if s.endswith("=X"):
        return "Forex"
    if s.startswith("^"):
        return "Index"
    if s.endswith("=F"):
        return "Commodity"
    if s in {"SPY", "QQQ", "VOO", "IVV", "DIA", "GLD", "SLV", "ARKK"}:
        return "ETF"
    return "Stock"


def resolve_symbol(idea: str, override: str = "") -> str:
    if override.strip():
        return override.strip().upper()
    low = idea.lower()
    # multi-word aliases first (longest match)
    for name in sorted(ASSET_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            return ASSET_ALIASES[name]
    # bare uppercase ticker-looking token e.g. "AAPL", "TSLA"
    tokens = re.findall(r"\b[A-Z]{2,6}\b", idea)
    ignore = {"EMA", "SMA", "RSI", "MACD", "ATR", "BB", "AI", "SL", "TP"}
    for t in tokens:
        if t not in ignore:
            return t
    return "AAPL"

# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyPlan:
    idea: str
    symbol: str
    asset_class: str
    strategy_type: str
    timeframe: str          # "Intraday" | "Swing"
    interval: str
    period: str
    fast: int = 20
    slow: int = 50
    rsi_period: int = 14
    rsi_low: int = 30
    rsi_high: int = 70
    use_rsi_confirm: bool = False
    use_macd_confirm: bool = False
    bb_period: int = 20
    bb_std: float = 2.0
    breakout_lookback: int = 20
    direction: str = "Long Only"


@dataclass
class RiskParams:
    initial_capital: float = 100_000.0
    risk_per_trade_pct: float = 1.0
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 6.0
    trailing_stop_pct: float = 0.0
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 20.0
    commission_bps: float = 5.0
    slippage_bps: float = 3.0
    max_portfolio_exposure_pct: float = 100.0


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1 — STRATEGY PLANNER  (natural language -> StrategyPlan)
# ═══════════════════════════════════════════════════════════════════════════
class StrategyPlannerAgent:
    """Parses a single natural-language trading idea into a structured plan."""

    name = "Strategy Planner Agent"

    def run(self, idea: str, symbol_override: str = "") -> StrategyPlan:
        low = idea.lower()
        symbol = resolve_symbol(idea, symbol_override)
        asset_class = classify_asset(symbol)

        intraday = any(k in low for k in [
                       "intraday", "scalp", "5 min", "15 min", "minute", "day trad"])
        timeframe = "Intraday" if intraday else "Swing"
        interval = "15m" if intraday else "1d"
        period = "60d" if intraday else "2y"

        nums = [int(n) for n in re.findall(r"\d+", idea)]

        if "macd" in low:
            strategy_type = "MACD Crossover"
        elif "breakout" in low or "range" in low:
            strategy_type = "Breakout"
        elif "bollinger" in low:
            strategy_type = "Bollinger Mean Reversion"
        elif "mean reversion" in low or "reversion" in low:
            strategy_type = "RSI Mean Reversion"
        elif "momentum" in low:
            strategy_type = "Momentum"
        elif "ema" in low or "crossover" in low or "moving average" in low or "sma" in low:
            strategy_type = "SMA Crossover" if (
                "sma" in low and "ema" not in low) else "EMA Crossover"
        else:
            strategy_type = "EMA Crossover"

        fast, slow = 20, 50
        pair = re.findall(r"(\d{1,3})\s*[/\-]\s*(\d{1,3})", idea)
        if pair:
            fast, slow = int(pair[0][0]), int(pair[0][1])
        elif len(nums) >= 2 and strategy_type in ("EMA Crossover", "SMA Crossover"):
            small = sorted(n for n in nums if 2 <= n <= 400)
            if len(small) >= 2:
                fast, slow = small[0], small[-1]
        if fast >= slow:
            fast, slow = 20, 50

        use_rsi_confirm = "rsi" in low and strategy_type not in (
            "RSI Mean Reversion",)
        use_macd_confirm = "macd" in low and strategy_type != "MACD Crossover"

        rsi_low, rsi_high = 30, 70
        rsi_thresh = re.findall(r"rsi[^\d]{0,15}(\d{1,2})", low)
        if len(rsi_thresh) >= 2:
            rsi_low, rsi_high = int(rsi_thresh[0]), int(rsi_thresh[1])

        breakout_lookback = 20
        bo = re.findall(
            r"(\d{1,3})\s*[- ]?(?:day|bar|period)?\s*breakout", low)
        if bo:
            breakout_lookback = int(bo[0])
        elif strategy_type == "Breakout" and nums:
            cand = [n for n in nums if 5 <= n <= 250]
            if cand:
                breakout_lookback = cand[0]

        direction = "Long/Short" if any(
            k in low for k in ["short", "long/short", "both direction"]) else "Long Only"

        return StrategyPlan(
            idea=idea, symbol=symbol, asset_class=asset_class, strategy_type=strategy_type,
            timeframe=timeframe, interval=interval, period=period, fast=fast, slow=slow,
            use_rsi_confirm=use_rsi_confirm, use_macd_confirm=use_macd_confirm,
            rsi_low=rsi_low, rsi_high=rsi_high, breakout_lookback=breakout_lookback, direction=direction,
        )

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2 — MARKET DATA  +  AGENT 3 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════


class MarketDataAgent:
    name = "Market Data Agent"

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _download(symbol: str, period: str, interval: str) -> pd.DataFrame:
        last_err = None
        for attempt in range(3):
            try:
                df = yf.download(symbol, period=period, interval=interval,
                                 auto_adjust=True, progress=False, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                last_err = e
                time.sleep(0.4 * (attempt + 1))
        if last_err:
            raise RuntimeError(f"Failed to fetch {symbol}: {last_err}")
        return pd.DataFrame()

    def run(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        try:
            df = self._download(symbol, period, interval)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            # graceful fallback to daily data if intraday unavailable for this symbol
            try:
                df = self._download(symbol, "2y", "1d")
            except Exception:
                df = pd.DataFrame()
        return df


class DataCleaningAgent:
    name = "Data Cleaning Agent"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        keep_cols = [c for c in ["Open", "High", "Low",
                                 "Close", "Volume"] if c in df.columns]
        df = df[keep_cols]
        df = df[(df["Close"] > 0) & (df["Open"] > 0)]
        df = df.ffill().dropna()
        return df


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4 — TECHNICAL INDICATORS  (native pandas/numpy implementations)
# ═══════════════════════════════════════════════════════════════════════════
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(s: pd.Series, fast=12, slow=26, signal=9):
    macd_line = _ema(s, fast) - _ema(s, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = _sma(s, n)
    std = s.rolling(n, min_periods=max(2, n // 2)).std()
    return mid + k * std, mid, mid - k * std


class TechnicalIndicatorAgent:
    name = "Technical Indicator Agent"

    def run(self, df: pd.DataFrame, plan: StrategyPlan) -> pd.DataFrame:
        d = df.copy()
        c = d["Close"]
        d["EMA_FAST"] = _ema(c, plan.fast)
        d["EMA_SLOW"] = _ema(c, plan.slow)
        d["SMA_FAST"] = _sma(c, plan.fast)
        d["SMA_SLOW"] = _sma(c, plan.slow)
        d["RSI"] = _rsi(c, plan.rsi_period)
        d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = _macd(c)
        d["ATR"] = _atr(d, 14)
        d["BB_UPPER"], d["BB_MID"], d["BB_LOWER"] = _bollinger(
            c, plan.bb_period, plan.bb_std)
        d["ROC"] = c.pct_change(10) * 100
        d["EMA200"] = _ema(c, min(200, max(20, len(d) // 2)))
        d["TREND"] = np.where(c > d["EMA200"], "Uptrend", "Downtrend")

        # breakout levels / support & resistance (rolling pivot extremes)
        look = plan.breakout_lookback
        d["ROLL_HIGH"] = d["High"].rolling(look).max()
        d["ROLL_LOW"] = d["Low"].rolling(look).min()
        d["RESISTANCE"] = d["High"].rolling(look, center=True).max()
        d["SUPPORT"] = d["Low"].rolling(look, center=True).min()

        # simple candlestick pattern flags
        body = (d["Close"] - d["Open"]).abs()
        rng = (d["High"] - d["Low"]).replace(0, np.nan)
        prev_open, prev_close = d["Open"].shift(), d["Close"].shift()
        d["PAT_DOJI"] = (body / rng < 0.1)
        d["PAT_BULL_ENGULF"] = (d["Close"] > d["Open"]) & (prev_close < prev_open) & \
            (d["Close"] >= prev_open) & (d["Open"] <= prev_close)
        d["PAT_BEAR_ENGULF"] = (d["Close"] < d["Open"]) & (prev_close > prev_open) & \
            (d["Open"] >= prev_close) & (d["Close"] <= prev_open)

        # higher-timeframe (weekly) trend for multi-timeframe confirmation
        try:
            weekly = d["Close"].resample("W").last().dropna()
            weekly_ema = _ema(weekly, 10)
            htf = (weekly > weekly_ema).reindex(d.index, method="ffill")
            d["HTF_UPTREND"] = htf.fillna(False)
        except Exception:
            d["HTF_UPTREND"] = True

        return d.dropna(subset=["Close"])

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 5 — STRATEGY BUILDER  (entry/exit signal generation)
# ═══════════════════════════════════════════════════════════════════════════


class StrategyBuilderAgent:
    name = "Strategy Builder Agent"

    def run(self, df: pd.DataFrame, plan: StrategyPlan) -> pd.DataFrame:
        d = df.copy()
        long_entry = pd.Series(False, index=d.index)
        long_exit = pd.Series(False, index=d.index)

        st_type = plan.strategy_type
        if st_type == "EMA Crossover":
            cross_up = (d["EMA_FAST"] > d["EMA_SLOW"]) & (
                d["EMA_FAST"].shift() <= d["EMA_SLOW"].shift())
            cross_dn = (d["EMA_FAST"] < d["EMA_SLOW"]) & (
                d["EMA_FAST"].shift() >= d["EMA_SLOW"].shift())
            long_entry, long_exit = cross_up, cross_dn
        elif st_type == "SMA Crossover":
            cross_up = (d["SMA_FAST"] > d["SMA_SLOW"]) & (
                d["SMA_FAST"].shift() <= d["SMA_SLOW"].shift())
            cross_dn = (d["SMA_FAST"] < d["SMA_SLOW"]) & (
                d["SMA_FAST"].shift() >= d["SMA_SLOW"].shift())
            long_entry, long_exit = cross_up, cross_dn
        elif st_type == "MACD Crossover":
            cross_up = (d["MACD"] > d["MACD_SIGNAL"]) & (
                d["MACD"].shift() <= d["MACD_SIGNAL"].shift())
            cross_dn = (d["MACD"] < d["MACD_SIGNAL"]) & (
                d["MACD"].shift() >= d["MACD_SIGNAL"].shift())
            long_entry, long_exit = cross_up, cross_dn
        elif st_type == "Breakout":
            long_entry = d["Close"] > d["ROLL_HIGH"].shift()
            long_exit = d["Close"] < d["ROLL_LOW"].shift()
        elif st_type == "Bollinger Mean Reversion":
            long_entry = d["Close"] < d["BB_LOWER"]
            long_exit = d["Close"] > d["BB_MID"]
        elif st_type == "RSI Mean Reversion":
            long_entry = d["RSI"] < plan.rsi_low
            long_exit = d["RSI"] > plan.rsi_high
        elif st_type == "Momentum":
            long_entry = d["ROC"] > 0
            long_exit = d["ROC"] < 0
        else:
            cross_up = (d["EMA_FAST"] > d["EMA_SLOW"]) & (
                d["EMA_FAST"].shift() <= d["EMA_SLOW"].shift())
            cross_dn = (d["EMA_FAST"] < d["EMA_SLOW"]) & (
                d["EMA_FAST"].shift() >= d["EMA_SLOW"].shift())
            long_entry, long_exit = cross_up, cross_dn

        if plan.use_rsi_confirm:
            long_entry = long_entry & (d["RSI"] > 40) & (d["RSI"] < 70)
        if plan.use_macd_confirm:
            long_entry = long_entry & (d["MACD"] > d["MACD_SIGNAL"])

        d["LONG_ENTRY"] = long_entry.fillna(False)
        d["LONG_EXIT"] = long_exit.fillna(False)
        d["SIGNAL"] = 0
        d.loc[d["LONG_ENTRY"], "SIGNAL"] = 1
        d.loc[d["LONG_EXIT"], "SIGNAL"] = -1
        return d


# ═══════════════════════════════════════════════════════════════════════════
# AI CONFIDENCE SCORER  (lightweight ML confirmation, optional)
# ═══════════════════════════════════════════════════════════════════════════
class MLConfidenceAgent:
    name = "Market Intelligence Agent"

    def run(self, df: pd.DataFrame) -> float:
        if not SKLEARN_OK or len(df) < 120:
            return 50.0
        d = df.copy()
        feats = ["RSI", "MACD_HIST", "ROC"]
        d["FUT"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
        d = d.dropna(subset=feats + ["FUT"])
        if len(d) < 60:
            return 50.0
        X, y = d[feats].values, d["FUT"].values
        split = int(len(d) * 0.8)
        try:
            model = RandomForestClassifier(
                n_estimators=150, max_depth=5, random_state=42)
            model.fit(X[:split], y[:split])
            proba = model.predict_proba(X[-1:].reshape(1, -1))[0]
            up_idx = list(model.classes_).index(
                1) if 1 in model.classes_ else 0
            return float(proba[up_idx] * 100)
        except Exception:
            return 50.0

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 6 — RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════


class RiskManagementAgent:
    name = "Risk Management Agent"

    def position_size(self, capital: float, price: float, risk: RiskParams) -> float:
        if price <= 0:
            return 0.0
        risk_amount = capital * (risk.risk_per_trade_pct / 100.0)
        stop_distance = price * (risk.stop_loss_pct / 100.0)
        if stop_distance <= 0:
            return 0.0
        qty = risk_amount / stop_distance
        max_qty_by_exposure = (
            capital * (risk.max_portfolio_exposure_pct / 100.0)) / price
        return max(0.0, min(qty, max_qty_by_exposure))

    def evaluate(self, trades: pd.DataFrame, equity: pd.Series, risk: RiskParams) -> dict:
        report = {"breaches": [], "status": "OK"}
        if equity is None or equity.empty:
            return report
        dd = (equity / equity.cummax() - 1.0) * 100
        worst_dd = dd.min()
        if worst_dd <= -risk.max_drawdown_pct:
            report["breaches"].append(
                f"Max drawdown limit breached: {worst_dd:.2f}% vs limit -{risk.max_drawdown_pct:.1f}%")
        if not trades.empty and "pnl" in trades.columns:
            daily = trades.copy()
            daily["date"] = pd.to_datetime(daily["exit_time"]).dt.date
            worst_day = daily.groupby("date")["pnl"].sum().min()
            worst_day_pct = (worst_day / risk.initial_capital) * \
                100 if risk.initial_capital else 0
            if worst_day_pct <= -risk.max_daily_loss_pct:
                report["breaches"].append(
                    f"Max daily loss limit breached: {worst_day_pct:.2f}% on worst day")
        report["status"] = "BREACH" if report["breaches"] else "OK"
        report["worst_drawdown_pct"] = float(worst_dd)
        return report


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 7 — BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class BacktestingAgent:
    name = "Backtesting Agent"

    def run(self, df: pd.DataFrame, plan: StrategyPlan, risk: RiskParams) -> dict:
        d = df.dropna(subset=["SIGNAL"]).copy()
        cash = risk.initial_capital
        position_qty = 0.0
        entry_price = 0.0
        trail_peak = 0.0
        trades = []
        equity_curve = []
        risk_agent = RiskManagementAgent()
        fee = (risk.commission_bps + risk.slippage_bps) / 10_000.0

        for ts, row in d.iterrows():
            price = float(row["Close"])
            equity_now = cash + position_qty * price
            equity_curve.append((ts, equity_now))

            if position_qty > 0:
                trail_peak = max(trail_peak, price)
                stop_price = entry_price * (1 - risk.stop_loss_pct / 100.0)
                target_price = entry_price * (1 + risk.take_profit_pct / 100.0)
                trailing_price = trail_peak * \
                    (1 - risk.trailing_stop_pct /
                     100.0) if risk.trailing_stop_pct > 0 else -1

                exit_reason = None
                if price <= stop_price:
                    exit_reason = "Stop Loss"
                elif price >= target_price:
                    exit_reason = "Take Profit"
                elif risk.trailing_stop_pct > 0 and price <= trailing_price:
                    exit_reason = "Trailing Stop"
                elif bool(row.get("LONG_EXIT", False)):
                    exit_reason = "Signal Exit"

                if exit_reason:
                    proceeds = position_qty * price * (1 - fee)
                    pnl = proceeds - (position_qty * entry_price)
                    cash += proceeds
                    trades.append({
                        "entry_time": entry_ts, "exit_time": ts, "entry_price": entry_price,
                        "exit_price": price, "qty": position_qty, "pnl": pnl,
                        "return_pct": (price / entry_price - 1) * 100, "reason": exit_reason,
                    })
                    position_qty = 0.0
                    entry_price = 0.0

            elif bool(row.get("LONG_ENTRY", False)) and cash > 0:
                qty = risk_agent.position_size(cash, price, risk)
                cost = qty * price * (1 + fee)
                if qty > 0 and cost <= cash:
                    position_qty = qty
                    entry_price = price
                    entry_ts = ts
                    trail_peak = price
                    cash -= cost

        final_price = float(d["Close"].iloc[-1]) if len(d) else 0.0
        if position_qty > 0:
            proceeds = position_qty * final_price * (1 - fee)
            pnl = proceeds - (position_qty * entry_price)
            cash += proceeds
            trades.append({
                "entry_time": entry_ts, "exit_time": d.index[-1], "entry_price": entry_price,
                "exit_price": final_price, "qty": position_qty, "pnl": pnl,
                "return_pct": (final_price / entry_price - 1) * 100, "reason": "End of Period",
            })
            position_qty = 0.0

        equity_series = pd.Series({t: v for t, v in equity_curve})
        trades_df = pd.DataFrame(trades)
        return {"trades": trades_df, "equity": equity_series, "final_cash": cash}


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 8 — PERFORMANCE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
def sharpe_ratio(returns: pd.Series, rfr: float = 0.04, periods: int = 252) -> float:
    if returns is None or len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - rfr / periods
    return float(np.sqrt(periods) * excess.mean() / returns.std())


def sortino_ratio(returns: pd.Series, rfr: float = 0.04, periods: int = 252) -> float:
    if returns is None or len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if downside.std() == 0 or len(downside) == 0:
        return 0.0
    excess = returns - rfr / periods
    return float(np.sqrt(periods) * excess.mean() / downside.std())


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    dd = (equity / equity.cummax() - 1.0) * 100
    return float(dd.min())


def calmar_ratio(total_return_pct: float, mdd_pct: float) -> float:
    if mdd_pct == 0:
        return 0.0
    return float(total_return_pct / abs(mdd_pct))


class PerformanceAnalyticsAgent:
    name = "Performance Analytics Agent"

    def run(self, trades: pd.DataFrame, equity: pd.Series, initial_capital: float) -> dict:
        m = {
            "net_profit": 0.0, "net_profit_pct": 0.0, "win_rate": 0.0, "loss_rate": 0.0,
            "profit_factor": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd": 0.0,
            "avg_trade": 0.0, "expectancy": 0.0, "risk_reward": 0.0, "total_trades": 0,
            "wins": 0, "losses": 0,
        }
        if equity is not None and not equity.empty:
            m["max_dd"] = max_drawdown_pct(equity)
            daily_returns = equity.pct_change().dropna()
            m["sharpe"] = sharpe_ratio(daily_returns)
            m["sortino"] = sortino_ratio(daily_returns)
            net_profit_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            m["net_profit_pct"] = float(net_profit_pct)
            m["calmar"] = calmar_ratio(net_profit_pct, m["max_dd"])

        if trades is not None and not trades.empty:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]
            m["total_trades"] = len(trades)
            m["wins"], m["losses"] = len(wins), len(losses)
            m["win_rate"] = len(wins) / len(trades) * 100
            m["loss_rate"] = len(losses) / len(trades) * 100
            gross_win = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())
            m["profit_factor"] = float(
                gross_win / gross_loss) if gross_loss > 0 else float(gross_win > 0) * 99.0
            m["net_profit"] = float(trades["pnl"].sum())
            m["avg_trade"] = float(trades["pnl"].mean())
            avg_win = wins["pnl"].mean() if len(wins) else 0
            avg_loss = abs(losses["pnl"].mean()) if len(losses) else 0
            m["risk_reward"] = float(
                avg_win / avg_loss) if avg_loss > 0 else 0.0
            m["expectancy"] = float(
                (m["win_rate"] / 100 * avg_win) - (m["loss_rate"] / 100 * avg_loss))

        monthly = pd.DataFrame()
        if equity is not None and not equity.empty:
            monthly_ret = equity.resample(
                "ME").last().pct_change().dropna() * 100
            monthly = monthly_ret.to_frame("Return %")
            monthly.index = monthly.index.strftime("%Y-%m")

        return {"metrics": m, "monthly": monthly}

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 9 — PARAMETER OPTIMIZATION  (grid search + walk-forward split)
# ═══════════════════════════════════════════════════════════════════════════


class OptimizationAgent:
    name = "Optimization Agent"

    def run(self, raw_df: pd.DataFrame, plan: StrategyPlan, risk: RiskParams,
            indicator_agent: TechnicalIndicatorAgent, builder_agent: StrategyBuilderAgent,
            backtest_agent: BacktestingAgent, perf_agent: PerformanceAnalyticsAgent) -> pd.DataFrame:

        split = int(len(raw_df) * 0.7)
        train_df, test_df = raw_df.iloc[:split], raw_df.iloc[split:]

        if plan.strategy_type in ("EMA Crossover", "SMA Crossover"):
            grid = [(f, s) for f in (10, 20, 30, 50)
                    for s in (50, 100, 150, 200) if f < s]
            param_names = ("fast", "slow")
        elif plan.strategy_type == "RSI Mean Reversion":
            grid = [(lo, hi) for lo in (20, 25, 30, 35)
                    for hi in (65, 70, 75, 80)]
            param_names = ("rsi_low", "rsi_high")
        elif plan.strategy_type == "Breakout":
            grid = [(n,) for n in (10, 20, 30, 40, 55)]
            param_names = ("breakout_lookback",)
        else:
            grid = [(plan.fast, plan.slow)]
            param_names = ("fast", "slow")

        rows = []
        for combo in grid:
            test_plan = StrategyPlan(
                **{**asdict(plan), **dict(zip(param_names, combo))})
            try:
                ind_train = indicator_agent.run(train_df, test_plan)
                sig_train = builder_agent.run(ind_train, test_plan)
                res_train = backtest_agent.run(sig_train, test_plan, risk)
                perf_train = perf_agent.run(
                    res_train["trades"], res_train["equity"], risk.initial_capital)["metrics"]

                ind_test = indicator_agent.run(test_df, test_plan)
                sig_test = builder_agent.run(ind_test, test_plan)
                res_test = backtest_agent.run(sig_test, test_plan, risk)
                perf_test = perf_agent.run(
                    res_test["trades"], res_test["equity"], risk.initial_capital)["metrics"]

                row = {name: val for name, val in zip(param_names, combo)}
                row.update({
                    "train_sharpe": round(perf_train["sharpe"], 3),
                    "train_return_%": round(perf_train["net_profit_pct"], 2),
                    "test_sharpe": round(perf_test["sharpe"], 3),
                    "test_return_%": round(perf_test["net_profit_pct"], 2),
                    "test_win_rate_%": round(perf_test["win_rate"], 1),
                    "test_max_dd_%": round(perf_test["max_dd"], 2),
                })
                rows.append(row)
            except Exception:
                continue

        leaderboard = pd.DataFrame(rows)
        if not leaderboard.empty:
            leaderboard = leaderboard.sort_values(
                "test_sharpe", ascending=False).reset_index(drop=True)
        return leaderboard


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 10 — PORTFOLIO MANAGER
# ═══════════════════════════════════════════════════════════════════════════
class PortfolioManagerAgent:
    name = "Portfolio Manager Agent"

    def run(self, equity: pd.Series, allocation_pct: float, total_capital: float) -> dict:
        if equity is None or equity.empty:
            return {"portfolio_equity": pd.Series(dtype=float), "cash_reserve": total_capital}
        allocated_capital = total_capital * (allocation_pct / 100.0)
        cash_reserve = total_capital - allocated_capital
        scale = allocated_capital / equity.iloc[0]
        strategy_equity = equity * scale
        portfolio_equity = strategy_equity + cash_reserve
        return {
            "portfolio_equity": portfolio_equity,
            "cash_reserve": cash_reserve,
            "allocated_capital": allocated_capital,
        }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 11 — PAPER TRADING  (simulated broker, persisted trade journal)
# ═══════════════════════════════════════════════════════════════════════════
class PaperTradingAgent:
    name = "Paper Trading Agent"

    def log_trade(self, session_id: str, symbol: str, side: str, qty: float,
                  price: float, cash_after: float, reason: str):
        conn = get_conn()
        conn.execute(
            "INSERT INTO paper_trades(ts,symbol,side,qty,price,value,cash_after,reason,session_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), symbol, side, qty,
             price, qty * price, cash_after, reason, session_id),
        )
        conn.commit()
        conn.close()

    def history(self, session_id: str) -> pd.DataFrame:
        conn = get_conn()
        df = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE session_id=? ORDER BY id DESC", conn, params=(session_id,)
        )
        conn.close()
        return df

    def place_order(self, side: str, symbol: str, qty: float, price: float,
                    session_id: str, reason: str = "Manual"):
        book = st.session_state.paper_broker
        if side == "BUY":
            cost = qty * price
            if cost > book["cash"]:
                return False, "Insufficient paper cash for this order."
            book["cash"] -= cost
            book["positions"][symbol] = book["positions"].get(
                symbol, 0.0) + qty
        else:
            held = book["positions"].get(symbol, 0.0)
            if qty > held:
                return False, "Cannot sell more than current paper position."
            book["cash"] += qty * price
            book["positions"][symbol] = held - qty
        self.log_trade(session_id, symbol, side, qty,
                       price, book["cash"], reason)
        return True, "Order filled (paper trading)."


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION SUPERVISOR — validates every order before it is allowed to fill
# ═══════════════════════════════════════════════════════════════════════════
class ExecutionSupervisorAgent:
    name = "Execution Supervisor Agent"

    def validate(self, side: str, qty: float, price: float, cash: float,
                 positions: dict, symbol: str, risk: RiskParams) -> tuple[bool, str]:
        if qty <= 0 or price <= 0:
            return False, "Invalid order size or price."
        if side == "BUY":
            exposure_after = (sum(positions.values()) * price + qty * price)
            if qty * price > cash:
                return False, "Order rejected — insufficient buying power."
            if risk.max_portfolio_exposure_pct < 100 and exposure_after > cash * (risk.max_portfolio_exposure_pct / 100):
                return False, "Order rejected — exceeds max portfolio exposure limit."
        else:
            if qty > positions.get(symbol, 0.0):
                return False, "Order rejected — no sufficient open position to sell."
        return True, "Order passed risk validation."

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 12 — REPORTING  (CSV always available, PDF when fpdf2 is installed)
# ═══════════════════════════════════════════════════════════════════════════


class ReportingAgent:
    name = "Reporting Agent"

    def trades_csv(self, trades: pd.DataFrame) -> bytes:
        return trades.to_csv(index=False).encode("utf-8")

    def equity_csv(self, equity: pd.Series) -> bytes:
        return equity.rename("equity").to_csv().encode("utf-8")

    def pdf_report(self, plan: StrategyPlan, risk: RiskParams, metrics: dict) -> Optional[bytes]:
        if not FPDF_OK:
            return None
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "QuantPilot AI - Strategy Performance Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Strategy Overview", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for label, val in [
            ("Trading Idea", plan.idea), ("Symbol",
                                          plan.symbol), ("Asset Class", plan.asset_class),
            ("Strategy Type", plan.strategy_type), ("Timeframe", plan.timeframe),
            ("Fast/Slow", f"{plan.fast}/{plan.slow}"),
        ]:
            pdf.multi_cell(0, 6, f"{label}: {val}")
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Performance Metrics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for label, key, suffix in [
            ("Net Profit", "net_profit", ""), ("Net Return", "net_profit_pct", "%"),
            ("Win Rate", "win_rate", "%"), ("Loss Rate", "loss_rate", "%"),
            ("Profit Factor", "profit_factor", ""), ("Sharpe Ratio", "sharpe", ""),
            ("Sortino Ratio", "sortino", ""), ("Calmar Ratio", "calmar", ""),
            ("Max Drawdown", "max_dd", "%"), ("Average Trade", "avg_trade", ""),
            ("Expectancy", "expectancy", ""), ("Risk/Reward", "risk_reward", ""),
            ("Total Trades", "total_trades", ""),
        ]:
            val = metrics.get(key, 0)
            pdf.cell(0, 6, f"{label}: {val:,.2f}{suffix}" if isinstance(
                val, float) else f"{label}: {val}{suffix}", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Risk Configuration", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for label, val in [
            ("Initial Capital", f"${risk.initial_capital:,.2f}"), (
                "Risk per Trade", f"{risk.risk_per_trade_pct}%"),
            ("Stop Loss", f"{risk.stop_loss_pct}%"), ("Take Profit",
                                                      f"{risk.take_profit_pct}%"),
            ("Trailing Stop", f"{risk.trailing_stop_pct}%"), (
                "Max Daily Loss", f"{risk.max_daily_loss_pct}%"),
            ("Max Drawdown Limit", f"{risk.max_drawdown_pct}%"),
        ]:
            pdf.cell(0, 6, f"{label}: {val}", ln=True)
        out = pdf.output()
        if isinstance(out, str):
            return out.encode("latin-1", errors="replace")
        return bytes(out)


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATION ENGINE — runs every agent in sequence, logs progress
# ═══════════════════════════════════════════════════════════════════════════
class OrchestrationEngine:
    """Centralized orchestrator coordinating all autonomous agents."""

    def __init__(self):
        self.planner = StrategyPlannerAgent()
        self.data_agent = MarketDataAgent()
        self.cleaner = DataCleaningAgent()
        self.indicators = TechnicalIndicatorAgent()
        self.builder = StrategyBuilderAgent()
        self.ml_intel = MLConfidenceAgent()
        self.backtester = BacktestingAgent()
        self.risk_agent = RiskManagementAgent()
        self.optimizer = OptimizationAgent()
        self.portfolio_agent = PortfolioManagerAgent()
        self.paper_agent = PaperTradingAgent()
        self.perf_agent = PerformanceAnalyticsAgent()
        self.reporter = ReportingAgent()
        self.log = []

    def _log(self, agent: str, message: str, level: str = "info"):
        entry = {"ts": datetime.now().strftime("%H:%M:%S"), "agent": agent,
                 "message": message, "level": level}
        self.log.append(entry)

    def run_pipeline(self, idea: str, symbol_override: str, risk: RiskParams,
                     allocation_pct: float, run_optimization: bool) -> dict:
        result = {"success": False}

        self._log(self.planner.name, "Parsing natural-language trading idea…")
        plan = self.planner.run(idea, symbol_override)
        self._log(self.planner.name,
                  f"Plan ready → {plan.symbol} ({plan.asset_class}) · {plan.strategy_type}")

        self._log(self.data_agent.name,
                  f"Downloading {plan.period} of {plan.interval} data for {plan.symbol}…")
        raw = self.data_agent.run(plan.symbol, plan.period, plan.interval)
        if raw is None or raw.empty or len(raw) < 60:
            self._log(self.data_agent.name,
                      "Insufficient market data returned.", "error")
            result["error"] = f"Could not retrieve enough historical data for '{plan.symbol}'. Try another symbol or idea."
            result["log"] = self.log
            return result

        self._log("Data Cleaning Agent",
                  f"Validating & cleaning {len(raw)} bars…")
        clean = self.cleaner.run(raw)

        self._log(self.indicators.name,
                  "Generating indicators, patterns, support/resistance…")
        with_ind = self.indicators.run(clean, plan)

        self._log(self.builder.name,
                  f"Building {plan.strategy_type} entry/exit logic…")
        with_sig = self.builder.run(with_ind, plan)

        self._log(self.ml_intel.name,
                  "Scoring short-term directional confidence…")
        confidence = self.ml_intel.run(with_sig)

        self._log(self.backtester.name,
                  "Running historical backtest simulation…")
        bt = self.backtester.run(with_sig, plan, risk)

        self._log(self.risk_agent.name,
                  "Evaluating drawdown & daily-loss risk limits…")
        risk_report = self.risk_agent.evaluate(
            bt["trades"], bt["equity"], risk)

        self._log(self.perf_agent.name, "Computing performance analytics…")
        perf = self.perf_agent.run(
            bt["trades"], bt["equity"], risk.initial_capital)

        optimization = pd.DataFrame()
        if run_optimization:
            self._log(self.optimizer.name,
                      "Optimizing parameters via grid search + walk-forward validation…")
            optimization = self.optimizer.run(clean, plan, risk, self.indicators, self.builder,
                                              self.backtester, self.perf_agent)

        self._log(self.portfolio_agent.name,
                  f"Simulating portfolio at {allocation_pct:.0f}% allocation…")
        portfolio = self.portfolio_agent.run(
            bt["equity"], allocation_pct, risk.initial_capital)

        self._log("Reporting Agent", "Preparing exportable reports…")
        self._log("Dashboard Agent", "Rendering analytics dashboard…")
        self._log("Execution Supervisor Agent",
                  "Pipeline complete — ready for paper trading.", "success")

        conn = get_conn()
        conn.execute(
            "INSERT INTO strategy_runs(ts,idea,symbol,strategy_type,params,net_profit,sharpe,win_rate,max_dd,session_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), idea, plan.symbol, plan.strategy_type, json.dumps(asdict(plan)),
             perf["metrics"]["net_profit"], perf["metrics"]["sharpe"], perf["metrics"]["win_rate"],
             perf["metrics"]["max_dd"], st.session_state.session_id),
        )
        conn.commit()
        conn.close()

        result.update({
            "success": True, "plan": plan, "raw": raw, "clean": clean, "with_ind": with_ind,
            "with_sig": with_sig, "confidence": confidence, "backtest": bt, "risk_report": risk_report,
            "perf": perf, "optimization": optimization, "portfolio": portfolio, "log": self.log,
        })
        return result


# ═══════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    paper_bgcolor="#0c1120", plot_bgcolor="#0c1120",
    font=dict(family="JetBrains Mono, monospace", color="#8b98b4", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
)


def price_chart(df: pd.DataFrame, plan: StrategyPlan) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[
                        0.58, 0.2, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 increasing_line_color="#00e696", decreasing_line_color="#ff4a5a", name="Price"),
                  row=1, col=1)
    if plan.strategy_type in ("EMA Crossover",):
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_FAST"], line=dict(
            color="#4d8cff", width=1.3), name=f"EMA{plan.fast}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_SLOW"], line=dict(
            color="#ffb545", width=1.3), name=f"EMA{plan.slow}"), row=1, col=1)
    elif plan.strategy_type == "SMA Crossover":
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_FAST"], line=dict(
            color="#4d8cff", width=1.3), name=f"SMA{plan.fast}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SLOW"], line=dict(
            color="#ffb545", width=1.3), name=f"SMA{plan.slow}"), row=1, col=1)
    elif plan.strategy_type == "Bollinger Mean Reversion":
        for col, color in [("BB_UPPER", "#a06bff"), ("BB_MID", "#8b98b4"), ("BB_LOWER", "#a06bff")]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(
                color=color, width=1, dash="dot"), name=col), row=1, col=1)
    elif plan.strategy_type == "Breakout":
        fig.add_trace(go.Scatter(x=df.index, y=df["ROLL_HIGH"], line=dict(
            color="#00e696", width=1, dash="dot"), name="Rolling High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ROLL_LOW"], line=dict(
            color="#ff4a5a", width=1, dash="dot"), name="Rolling Low"), row=1, col=1)

    entries = df[df["LONG_ENTRY"]]
    exits = df[df["LONG_EXIT"]]
    fig.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers",
                             marker=dict(symbol="triangle-up", size=11, color="#00e696"), name="Entry"), row=1, col=1)
    fig.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers",
                             marker=dict(symbol="triangle-down", size=11, color="#ff4a5a"), name="Exit"), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], marker_color="#232f47", name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(
        color="#22d3ee", width=1.3), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff4a5a", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00e696", row=3, col=1)

    fig.update_layout(height=620, showlegend=True, **CHART_LAYOUT)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def equity_chart(equity: pd.Series, benchmark: Optional[pd.Series] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, line=dict(color="#00e696", width=2),
                             fill="tozeroy", fillcolor="rgba(0,230,150,0.06)", name="Strategy Equity"))
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, line=dict(color="#4d8cff", width=1.5, dash="dot"),
                                 name="Buy & Hold"))
    fig.update_layout(height=380, **CHART_LAYOUT)
    return fig


def drawdown_chart(equity: pd.Series) -> go.Figure:
    dd = (equity / equity.cummax() - 1.0) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, line=dict(color="#ff4a5a", width=1.5),
                             fill="tozeroy", fillcolor="rgba(255,74,90,0.08)", name="Drawdown %"))
    fig.update_layout(height=240, **CHART_LAYOUT)
    return fig


def optimization_heatmap(leaderboard: pd.DataFrame) -> Optional[go.Figure]:
    if leaderboard.empty or leaderboard.shape[1] < 3:
        return None
    p1, p2 = leaderboard.columns[0], leaderboard.columns[1]
    try:
        pivot = leaderboard.pivot_table(
            index=p1, columns=p2, values="test_sharpe", aggfunc="mean")
    except Exception:
        return None
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index],
                                    colorscale="Greens"))
    fig.update_layout(height=380, xaxis_title=p2,
                      yaxis_title=p1, **CHART_LAYOUT)
    return fig


def monthly_returns_chart(monthly: pd.DataFrame) -> Optional[go.Figure]:
    if monthly.empty:
        return None
    colors = ["#00e696" if v >= 0 else "#ff4a5a" for v in monthly["Return %"]]
    fig = go.Figure(
        go.Bar(x=monthly.index, y=monthly["Return %"], marker_color=colors))
    fig.update_layout(height=280, **CHART_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{int(time.time())}"
if "paper_broker" not in st.session_state:
    st.session_state.paper_broker = {"cash": 100_000.0, "positions": {}}
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "live_confirmed" not in st.session_state:
    st.session_state.live_confirmed = False

engine = OrchestrationEngine()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:800;color:#e6ecf6;">
            Quant<span style="color:#00e696;">Pilot</span> AI</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#4a5670;
            letter-spacing:0.18em;text-transform:uppercase;margin-top:2px;">Autonomous Quant Team</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Trading Idea</div>',
                unsafe_allow_html=True)
    idea = st.text_area(
        "Describe your strategy in one sentence", height=90,
        value=st.session_state.get(
            "last_idea", "Trade BTC using EMA 20/50 crossover with RSI confirmation"),
        placeholder="e.g. Create an intraday NIFTY breakout strategy",
        label_visibility="collapsed",
    )
    symbol_override = st.text_input(
        "Override symbol (optional)", "", placeholder="e.g. AAPL, BTC-USD, ^NSEI")

    st.markdown('<div class="sec-label">Capital & Risk</div>',
                unsafe_allow_html=True)
    initial_capital = st.number_input(
        "Initial Capital ($)", 1_000, 10_000_000, 100_000, 1_000)
    c1, c2 = st.columns(2)
    with c1:
        risk_per_trade = st.number_input("Risk / Trade %", 0.1, 10.0, 1.0, 0.1)
        stop_loss_pct = st.number_input("Stop Loss %", 0.5, 30.0, 3.0, 0.5)
        max_daily_loss = st.number_input(
            "Max Daily Loss %", 1.0, 50.0, 5.0, 0.5)
    with c2:
        take_profit_pct = st.number_input(
            "Take Profit %", 0.5, 100.0, 6.0, 0.5)
        trailing_stop_pct = st.number_input(
            "Trailing Stop %", 0.0, 30.0, 0.0, 0.5)
        max_drawdown_limit = st.number_input(
            "Max Drawdown %", 5.0, 80.0, 20.0, 1.0)

    with st.expander("Advanced execution settings"):
        commission_bps = st.slider("Commission (bps)", 0, 50, 5)
        slippage_bps = st.slider("Slippage (bps)", 0, 50, 3)
        max_exposure = st.slider("Max Portfolio Exposure %", 10, 100, 100)
        allocation_pct = st.slider(
            "Strategy Allocation of Portfolio %", 5, 100, 60)
        run_optimization = st.checkbox("Run parameter optimization", True)

    risk = RiskParams(
        initial_capital=float(initial_capital), risk_per_trade_pct=risk_per_trade,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, trailing_stop_pct=trailing_stop_pct,
        max_daily_loss_pct=max_daily_loss, max_drawdown_pct=max_drawdown_limit,
        commission_bps=commission_bps, slippage_bps=slippage_bps, max_portfolio_exposure_pct=max_exposure,
    )

    st.markdown('<div class="sec-label">Execution Mode</div>',
                unsafe_allow_html=True)
    st.markdown('<span class="pill pill-ok">PAPER TRADING (DEFAULT)</span>',
                unsafe_allow_html=True)
    live_toggle = st.checkbox(
        "Enable Live Order Submission (requires broker API — not connected)", False)
    if live_toggle:
        st.warning("No broker is connected. Live orders cannot be submitted; the platform will remain in paper "
                   "trading mode until a broker integration is configured and you explicitly confirm each order.")

    run_clicked = st.button("▶  RUN AUTONOMOUS PIPELINE",
                            use_container_width=True, type="primary")
    if st.button("↺  Reset Paper Broker", use_container_width=True):
        st.session_state.paper_broker = {
            "cash": float(initial_capital), "positions": {}}
        st.toast("Paper broker reset.")

if run_clicked:
    st.session_state.last_idea = idea
    with st.spinner("Autonomous agents collaborating on your strategy…"):
        st.session_state.pipeline_result = engine.run_pipeline(
            idea, symbol_override, risk, allocation_pct, run_optimization
        )
        st.session_state.paper_broker["cash"] = float(initial_capital)
        st.session_state.paper_broker["positions"] = {}

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.9rem;
    padding-bottom:0.7rem;border-bottom:1px solid var(--bdr);">
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:800;color:var(--t1);">
        Quant<span style="color:#00e696;">Pilot</span> AI
        <span style="font-size:0.6rem;color:var(--t3);font-weight:600;letter-spacing:0.14em;
            margin-left:0.8rem;font-family:'JetBrains Mono',monospace;">
            ENTERPRISE AUTONOMOUS TRADING PLATFORM</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:var(--t3);">
        {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
    </div>
</div>""", unsafe_allow_html=True)

result = st.session_state.pipeline_result

if result is None:
    st.info("Describe a trading idea in the sidebar and click **RUN AUTONOMOUS PIPELINE** to let the AI quant "
            "team build, backtest, optimize, and risk-check a complete strategy.")
    st.stop()

if not result.get("success"):
    st.error(result.get("error", "Pipeline failed."))
    with st.expander("Agent Execution Log"):
        for e in result.get("log", []):
            st.markdown(f'<div class="agent-row agent-run">[{e["ts"]}] {e["agent"]} — {e["message"]}</div>',
                        unsafe_allow_html=True)
    st.stop()

plan: StrategyPlan = result["plan"]
bt = result["backtest"]
perf = result["perf"]["metrics"]
monthly = result["perf"]["monthly"]
risk_report = result["risk_report"]
portfolio = result["portfolio"]
optimization = result["optimization"]
with_sig = result["with_sig"]

# Agent status panel
with st.expander("🤖  Autonomous Agent Execution Timeline", expanded=False):
    for e in result["log"]:
        cls = "agent-done" if e["level"] != "error" else "agent-run"
        st.markdown(f'<div class="agent-row {cls}">[{e["ts"]}]  {e["agent"]:<28}  {e["message"]}</div>',
                    unsafe_allow_html=True)

# KPI row
kc = st.columns(6)
with kc[0]:
    st.markdown(
        f'<div class="kpi"><div class="lab">Symbol</div><div class="val">{plan.symbol}</div></div>', unsafe_allow_html=True)
with kc[1]:
    st.markdown(
        f'<div class="kpi"><div class="lab">Strategy</div><div class="val" style="font-size:0.95rem;">{plan.strategy_type}</div></div>', unsafe_allow_html=True)
with kc[2]:
    npp = perf["net_profit_pct"]
    st.markdown(
        f'<div class="kpi"><div class="lab">Net Return</div><div class="val {"pos" if npp>=0 else "neg"}">{npp:+.2f}%</div></div>', unsafe_allow_html=True)
with kc[3]:
    st.markdown(
        f'<div class="kpi"><div class="lab">Sharpe</div><div class="val">{perf["sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
with kc[4]:
    st.markdown(
        f'<div class="kpi"><div class="lab">Win Rate</div><div class="val">{perf["win_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
with kc[5]:
    dd = perf["max_dd"]
    st.markdown(
        f'<div class="kpi"><div class="lab">Max Drawdown</div><div class="val neg">{dd:.2f}%</div></div>', unsafe_allow_html=True)

st.markdown("<div style='margin:0.7rem 0;'></div>", unsafe_allow_html=True)

tabs = st.tabs([
    "🧭 Overview", "📈 Chart & Signals", "🔁 Backtest", "🧪 Optimization",
    "🛡️ Risk", "💼 Portfolio", "🖊️ Paper Trading", "📊 Analytics",
    "📰 Market Data", "📑 Reports",
])

# ─────────────────────────── TAB: OVERVIEW ───────────────────────────
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Strategy Plan (Auto-Generated)")
        st.markdown(f"""
        <div class="card">
        <b>Idea:</b> <i>"{plan.idea}"</i><br><br>
        <b>Symbol:</b> {plan.symbol} &nbsp;|&nbsp; <b>Asset Class:</b> {plan.asset_class} &nbsp;|&nbsp;
        <b>Timeframe:</b> {plan.timeframe} ({plan.interval})<br>
        <b>Strategy Type:</b> {plan.strategy_type} &nbsp;|&nbsp; <b>Direction:</b> {plan.direction}<br>
        <b>Fast/Slow Periods:</b> {plan.fast} / {plan.slow} &nbsp;|&nbsp;
        <b>RSI Confirmation:</b> {plan.use_rsi_confirm} &nbsp;|&nbsp;
        <b>MACD Confirmation:</b> {plan.use_macd_confirm}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### AI Market Intelligence")
        conf = result["confidence"]
        conf_color = "var(--green)" if conf >= 55 else (
            "var(--red)" if conf <= 45 else "var(--amber)")
        st.markdown(f"""
        <div class="card">
        Short-term directional confidence (ML ensemble, next-bar up-probability):
        <span style="color:{conf_color};font-family:var(--mono);font-weight:700;font-size:1.1rem;"> {conf:.1f}%</span>
        <br><span style="color:var(--t3);font-size:0.78rem;">Advisory signal only — always confirmed by rule-based strategy logic and risk checks before any trade.</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("#### Risk Status")
        pill_cls = "pill-ok" if risk_report["status"] == "OK" else "pill-warn"
        st.markdown(
            f'<span class="pill {pill_cls}">{risk_report["status"]}</span>', unsafe_allow_html=True)
        if risk_report["breaches"]:
            for b in risk_report["breaches"]:
                st.warning(b)
        else:
            st.success("All risk limits within configured tolerance.")
        st.markdown("#### Quick Metrics")
        st.metric("Net Profit ($)", f"${perf['net_profit']:,.2f}")
        st.metric("Profit Factor", f"{perf['profit_factor']:.2f}")
        st.metric("Total Trades", perf["total_trades"])

# ─────────────────────────── TAB: CHART & SIGNALS ───────────────────────────
with tabs[1]:
    st.plotly_chart(price_chart(with_sig.tail(400), plan),
                    use_container_width=True)
    with st.expander("Detected candlestick patterns (most recent)"):
        pat_cols = ["PAT_DOJI", "PAT_BULL_ENGULF", "PAT_BEAR_ENGULF"]
        recent_pat = with_sig[pat_cols].tail(60)
        found = recent_pat[recent_pat.any(axis=1)]
        if found.empty:
            st.write(
                "No notable single/two-bar patterns detected in the recent window.")
        else:
            st.dataframe(found, use_container_width=True)
    st.caption(f"Trend regime (vs long EMA): **{with_sig['TREND'].iloc[-1]}**  ·  "
               f"Higher-timeframe (weekly) uptrend: **{bool(with_sig['HTF_UPTREND'].iloc[-1])}**")

# ─────────────────────────── TAB: BACKTEST ───────────────────────────
with tabs[2]:
    equity = bt["equity"]
    benchmark = None
    try:
        bench_shares = risk.initial_capital / result["clean"]["Close"].iloc[0]
        benchmark = result["clean"]["Close"] * bench_shares
    except Exception:
        pass
    st.plotly_chart(equity_chart(equity, benchmark), use_container_width=True)
    st.plotly_chart(drawdown_chart(equity), use_container_width=True)
    st.markdown("#### Trade Log")
    trades = bt["trades"]
    if trades.empty:
        st.info("No trades were triggered by this strategy over the selected history.")
    else:
        show = trades.copy()
        show["pnl"] = show["pnl"].round(2)
        show["return_pct"] = show["return_pct"].round(2)
        st.dataframe(show, use_container_width=True, height=320)

# ─────────────────────────── TAB: OPTIMIZATION ───────────────────────────
with tabs[3]:
    if optimization is None or optimization.empty:
        st.info("Enable 'Run parameter optimization' in the sidebar to search for better parameters "
                "using a walk-forward train/test split.")
    else:
        st.markdown(
            "#### Walk-Forward Parameter Leaderboard (ranked by out-of-sample Sharpe)")
        st.dataframe(optimization, use_container_width=True, height=320)
        heat = optimization_heatmap(optimization)
        if heat is not None:
            st.plotly_chart(heat, use_container_width=True)
        best = optimization.iloc[0].to_dict()
        st.success(f"Best out-of-sample parameters found: { {k: v for k, v in best.items() if k in optimization.columns[:2]} } "
                   f"— test Sharpe {best.get('test_sharpe', 0):.2f}, test return {best.get('test_return_%', 0):.2f}%")

# ─────────────────────────── TAB: RISK ───────────────────────────
with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Risk Configuration")
        st.markdown(f"""
        <div class="card">
        Initial Capital: <b>${risk.initial_capital:,.2f}</b><br>
        Risk per Trade: <b>{risk.risk_per_trade_pct}%</b><br>
        Stop Loss: <b>{risk.stop_loss_pct}%</b> &nbsp;|&nbsp; Take Profit: <b>{risk.take_profit_pct}%</b><br>
        Trailing Stop: <b>{risk.trailing_stop_pct}%</b><br>
        Max Daily Loss: <b>{risk.max_daily_loss_pct}%</b> &nbsp;|&nbsp;
        Max Drawdown Limit: <b>{risk.max_drawdown_pct}%</b><br>
        Commission: <b>{risk.commission_bps} bps</b> &nbsp;|&nbsp; Slippage: <b>{risk.slippage_bps} bps</b><br>
        Max Portfolio Exposure: <b>{risk.max_portfolio_exposure_pct}%</b>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("#### Live Risk Evaluation")
        st.metric("Realized Max Drawdown",
                  f"{risk_report.get('worst_drawdown_pct', 0):.2f}%")
        st.metric("Status", risk_report["status"])
        if not trades.empty:
            sample_price = float(with_sig["Close"].iloc[-1])
            sample_qty = engine.risk_agent.position_size(
                risk.initial_capital, sample_price, risk)
            st.markdown(f"""
            <div class="card">
            Example position sizing at current price (${sample_price:,.2f}):<br>
            Suggested quantity ≈ <b>{sample_qty:,.4f} units</b><br>
            Capital at risk this trade ≈ <b>${risk.initial_capital * risk.risk_per_trade_pct/100:,.2f}</b>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────── TAB: PORTFOLIO ───────────────────────────
with tabs[5]:
    port_eq = portfolio["portfolio_equity"]
    if port_eq is None or port_eq.empty:
        st.info("Portfolio simulation unavailable for this run.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Allocated Capital",
                  f"${portfolio['allocated_capital']:,.2f}")
        c2.metric("Cash Reserve", f"${portfolio['cash_reserve']:,.2f}")
        c3.metric("Ending Portfolio Value", f"${port_eq.iloc[-1]:,.2f}")
        st.plotly_chart(equity_chart(port_eq), use_container_width=True)
        st.caption("Portfolio equity = strategy allocation (scaled to its share of total capital) plus the "
                   "un-invested cash reserve held back per the allocation slider.")

# ─────────────────────────── TAB: PAPER TRADING ───────────────────────────
with tabs[6]:
    st.markdown("#### Simulated Broker (Paper Trading Only)")
    st.caption("No live broker is connected. All orders below are simulated and recorded to the local trade "
               "journal. Live order submission would require explicit per-order confirmation and a connected "
               "broker API — neither is enabled in this build.")
    book = st.session_state.paper_broker
    c1, c2, c3 = st.columns(3)
    c1.metric("Paper Cash", f"${book['cash']:,.2f}")
    c1.metric("Open Position",
              f"{book['positions'].get(plan.symbol, 0.0):,.4f} {plan.symbol}")
    last_price = float(with_sig["Close"].iloc[-1])
    c2.metric("Last Price", f"${last_price:,.2f}")
    mkt_val = book["positions"].get(plan.symbol, 0.0) * last_price
    c3.metric("Position Value", f"${mkt_val:,.2f}")

    oc1, oc2, oc3 = st.columns([1, 1, 1])
    with oc1:
        order_side = st.selectbox("Side", ["BUY", "SELL"])
    with oc2:
        order_qty = st.number_input("Quantity", 0.0, 1_000_000.0, 1.0, 0.1)
    with oc3:
        st.write("")
        st.write("")
        if st.button("Submit Paper Order", use_container_width=True):
            ok, msg = ExecutionSupervisorAgent().validate(
                order_side, order_qty, last_price, book["cash"], book["positions"], plan.symbol, risk
            )
            if ok:
                filled, fmsg = engine.paper_agent.place_order(
                    order_side, plan.symbol, order_qty, last_price, st.session_state.session_id
                )
                st.success(fmsg) if filled else st.error(fmsg)
            else:
                st.error(msg)

    st.markdown("#### Trade Journal")
    journal = engine.paper_agent.history(st.session_state.session_id)
    if journal.empty:
        st.info("No paper trades recorded yet this session.")
    else:
        st.dataframe(journal, use_container_width=True, height=280)

# ─────────────────────────── TAB: ANALYTICS ───────────────────────────
with tabs[7]:
    m = perf
    metric_grid = [
        ("Net Profit", f"${m['net_profit']:,.2f}"), ("Net Return",
                                                     f"{m['net_profit_pct']:+.2f}%"),
        ("Win Rate", f"{m['win_rate']:.1f}%"), ("Loss Rate",
                                                f"{m['loss_rate']:.1f}%"),
        ("Profit Factor",
         f"{m['profit_factor']:.2f}"), ("Sharpe Ratio", f"{m['sharpe']:.2f}"),
        ("Sortino Ratio",
         f"{m['sortino']:.2f}"), ("Calmar Ratio", f"{m['calmar']:.2f}"),
        ("Max Drawdown",
         f"{m['max_dd']:.2f}%"), ("Average Trade", f"${m['avg_trade']:,.2f}"),
        ("Expectancy",
         f"${m['expectancy']:,.2f}"), ("Risk/Reward", f"{m['risk_reward']:.2f}"),
    ]
    cols = st.columns(4)
    for i, (label, val) in enumerate(metric_grid):
        with cols[i % 4]:
            st.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{val}</div></div>',
                        unsafe_allow_html=True)
        if i % 4 == 3:
            st.write("")
    st.markdown("#### Monthly Returns")
    mchart = monthly_returns_chart(monthly)
    if mchart is not None:
        st.plotly_chart(mchart, use_container_width=True)
    else:
        st.info("Not enough history to compute monthly returns.")

# ─────────────────────────── TAB: MARKET DATA ───────────────────────────
with tabs[8]:
    st.markdown("#### Raw OHLCV (cleaned)")
    st.dataframe(result["clean"].tail(200).sort_index(
        ascending=False), use_container_width=True, height=340)
    st.markdown("#### Data Quality Summary")
    raw_rows, clean_rows = len(result["raw"]), len(result["clean"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Bars Downloaded", raw_rows)
    c2.metric("Bars After Cleaning", clean_rows)
    c3.metric("Rows Dropped", raw_rows - clean_rows)
    st.caption(f"Source: Yahoo Finance · Interval: {plan.interval} · Period: {plan.period} · "
               f"Asset Class: {plan.asset_class}")

# ─────────────────────────── TAB: REPORTS ───────────────────────────
with tabs[9]:
    st.markdown("#### Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇ Download Trades (CSV)", engine.reporter.trades_csv(trades if not trades.empty else pd.DataFrame()),
                           file_name=f"{plan.symbol}_trades.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇ Download Equity Curve (CSV)", engine.reporter.equity_csv(equity),
                           file_name=f"{plan.symbol}_equity.csv", mime="text/csv", use_container_width=True)
    with c3:
        pdf_bytes = engine.reporter.pdf_report(plan, risk, perf)
        if pdf_bytes is not None:
            st.download_button("⬇ Download PDF Report", pdf_bytes,
                               file_name=f"{plan.symbol}_report.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.info("Install `fpdf2` (pip install fpdf2) to enable one-click PDF report export. "
                    "CSV exports remain fully available.")

    st.markdown("#### Strategy Comparison — Run History")
    conn = get_conn()
    history_df = pd.read_sql_query(
        "SELECT ts, idea, symbol, strategy_type, net_profit, sharpe, win_rate, max_dd FROM strategy_runs "
        "WHERE session_id=? ORDER BY id DESC LIMIT 25", conn, params=(st.session_state.session_id,)
    )
    conn.close()
    if history_df.empty:
        st.info(
            "Run the pipeline with different ideas/symbols to build a comparison history.")
    else:
        st.dataframe(history_df, use_container_width=True, height=300)

st.markdown("<div style='margin-top:1.5rem;text-align:center;font-family:var(--mono);font-size:0.65rem;"
            "color:var(--t3);'>QuantPilot AI — Paper trading by default. No order is ever routed to a real "
            "broker without an explicit, per-order confirmation and a connected live brokerage API.</div>",
            unsafe_allow_html=True)
