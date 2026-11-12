# ============================================================
#  QuantPilot AI — Centralised Configuration & Defaults
# ============================================================
#
#  All constants, defaults, and tunable knobs live here.
#  Import this module anywhere in the project to avoid
#  magic numbers scattered across app.py.
# ============================================================

from __future__ import annotations

# ── App metadata ─────────────────────────────────────────────
APP_NAME        = "QuantPilot AI"
APP_SUBTITLE    = "Autonomous Quant Trading Platform"
APP_VERSION     = "2.0.0"
AUTHOR          = "Dheeraj Muley"
GITHUB_URL      = "https://github.com/dheeraj815/AI-Powered-Trading-Bot-with-ML"

# ── Database ──────────────────────────────────────────────────
DB_FILENAME     = ".quantpilot_ai.db"   # stored in user home directory

# ── Default Risk Parameters ───────────────────────────────────
DEFAULT_CAPITAL             = 100_000.0   # USD
DEFAULT_RISK_PER_TRADE_PCT  = 1.0         # % of capital at risk per trade
DEFAULT_STOP_LOSS_PCT       = 3.0         # % below entry
DEFAULT_TAKE_PROFIT_PCT     = 6.0         # % above entry
DEFAULT_TRAILING_STOP_PCT   = 0.0         # 0 = disabled
DEFAULT_MAX_DAILY_LOSS_PCT  = 5.0         # circuit-breaker: max daily loss
DEFAULT_MAX_DRAWDOWN_PCT    = 20.0        # circuit-breaker: portfolio drawdown
DEFAULT_COMMISSION_BPS      = 5           # basis points per trade
DEFAULT_SLIPPAGE_BPS        = 3           # basis points per trade
DEFAULT_MAX_EXPOSURE_PCT    = 100         # max % of capital deployed at once
DEFAULT_ALLOCATION_PCT      = 60          # % of portfolio allocated to strategy

# ── Default Strategy Parameters ──────────────────────────────
DEFAULT_EMA_FAST            = 20
DEFAULT_EMA_SLOW            = 50
DEFAULT_RSI_PERIOD          = 14
DEFAULT_RSI_LOW             = 30
DEFAULT_RSI_HIGH            = 70
DEFAULT_BB_PERIOD           = 20
DEFAULT_BB_STD              = 2.0
DEFAULT_BREAKOUT_LOOKBACK   = 20

# ── ML Model Parameters ───────────────────────────────────────
ML_N_ESTIMATORS             = 150
ML_MAX_DEPTH                = 5
ML_RANDOM_STATE             = 42
ML_MIN_TRAIN_BARS           = 120        # minimum bars needed for ML scoring
ML_TRAIN_SPLIT              = 0.8        # fraction used for training

# ── Data Settings ─────────────────────────────────────────────
DATA_RETRY_ATTEMPTS         = 3
DATA_RETRY_DELAY_S          = 0.4        # seconds between retries
DATA_CACHE_TTL_S            = 300        # 5 minutes Streamlit cache TTL
DATA_MIN_BARS               = 60         # minimum bars to proceed with backtest
INTRADAY_INTERVAL           = "15m"
INTRADAY_PERIOD             = "60d"
SWING_INTERVAL              = "1d"
SWING_PERIOD                = "2y"

# ── Performance / Analytics ───────────────────────────────────
RISK_FREE_RATE              = 0.04       # 4% annualised (for Sharpe / Sortino)
TRADING_DAYS_PER_YEAR       = 252

# ── Optimisation Grid ─────────────────────────────────────────
OPT_TRAIN_SPLIT             = 0.70       # 70% train, 30% test
OPT_EMA_FAST_VALUES         = [10, 20, 30, 50]
OPT_EMA_SLOW_VALUES         = [50, 100, 150, 200]
OPT_RSI_LOW_VALUES          = [20, 25, 30, 35]
OPT_RSI_HIGH_VALUES         = [65, 70, 75, 80]
OPT_BREAKOUT_VALUES         = [10, 20, 30, 40, 55]

# ── Chart Theme ───────────────────────────────────────────────
CHART_HEIGHT_MAIN           = 620
CHART_HEIGHT_EQUITY         = 380
CHART_HEIGHT_DRAWDOWN       = 240
CHART_HEIGHT_MONTHLY        = 280
CHART_HEIGHT_HEATMAP        = 380
CHART_BG_COLOR              = "#0c1120"
CHART_GRID_COLOR            = "rgba(255,255,255,0.05)"
CHART_FONT_FAMILY           = "JetBrains Mono, monospace"
CHART_FONT_COLOR            = "#8b98b4"
CHART_GREEN                 = "#00e696"
CHART_RED                   = "#ff4a5a"
CHART_BLUE                  = "#4d8cff"
CHART_AMBER                 = "#ffb545"
