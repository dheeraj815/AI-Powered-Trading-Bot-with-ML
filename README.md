<div align="center">

# 🧠 QuantPilot AI
### Autonomous Quant Trading Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/dheeraj815/AI-Powered-Trading-Bot-with-ML?style=social)](https://github.com/dheeraj815/AI-Powered-Trading-Bot-with-ML)

**A production-ready, AI-powered algorithmic trading platform** that converts a plain-English trading idea into a fully backtested, risk-managed, and paper-tradeable strategy — in seconds.

[🚀 Quick Start](#-quick-start) · [✨ Features](#-features) · [📸 Screenshots](#-screenshots) · [🧩 Architecture](#-architecture) · [⚙️ Configuration](#%EF%B8%8F-configuration) · [📋 Disclaimer](#-disclaimer)

</div>

---

## ✨ Features

| Category | Capabilities |
|---|---|
| 🧭 **Natural Language Input** | Describe any strategy in plain English — the AI parses it automatically |
| 📊 **Multi-Asset Support** | Stocks, Crypto, Forex, Indices, Commodities, ETFs |
| 🤖 **12 Autonomous Agents** | Each with a dedicated role: planning → data → indicators → signals → backtest → risk → reporting |
| 📈 **6 Strategy Types** | EMA/SMA Crossover, MACD, RSI Mean Reversion, Bollinger Bands, Breakout, Momentum |
| 🔬 **ML Confidence Scoring** | RandomForest ensemble for short-term directional probability |
| 📉 **Professional Backtesting** | Commission, slippage, trailing stop-loss, take-profit, position sizing |
| 📐 **Performance Analytics** | Sharpe, Sortino, Calmar, Profit Factor, Expectancy, Win Rate, Max Drawdown |
| 🧪 **Walk-Forward Optimization** | Grid-search parameters with train/test split to prevent curve-fitting |
| 🛡️ **Risk Management** | Per-trade risk %, daily loss limit, drawdown circuit-breaker, exposure cap |
| 💼 **Portfolio Simulation** | Capital allocation, cash reserve, strategy vs. portfolio equity curves |
| 🖊️ **Paper Trading** | Simulated broker with order book, trade journal, P&L tracking |
| 📑 **Exportable Reports** | CSV (trades + equity curve) · PDF report (optional, via `fpdf2`) |
| 🌙 **Dark Pro UI** | Custom JetBrains Mono / Space Grotesk themed Streamlit dashboard |

---

## 🛠 Tech Stack

```
Backend        Python 3.9+, Pandas, NumPy, scikit-learn (Random Forest)
Data           yfinance — real-time & historical OHLCV for 10,000+ tickers
Indicators     Custom native implementations: EMA, SMA, RSI, MACD, ATR, Bollinger Bands
Visualisation  Plotly (candlestick, equity, drawdown, heatmap, monthly returns)
UI             Streamlit 1.32+ with fully custom CSS theme
Persistence    SQLite — paper trade journal, strategy runs history, agent logs
Reporting      CSV export (always) · PDF export (fpdf2, optional)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/dheeraj815/AI-Powered-Trading-Bot-with-ML.git
cd AI-Powered-Trading-Bot-with-ML
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# Optional: enable PDF report export
pip install fpdf2
```

### 4. Launch the platform
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## 🧩 Architecture — 12 Autonomous Agents

```
┌─────────────────────────────────────────────────────────────────┐
│                     OrchestrationEngine                         │
│  (Central coordinator — runs every agent in sequence)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
     ┌─────────────────────▼──────────────────────┐
     │                                            │
  Agent 1              Agent 2 + 3           Agent 4
  Strategy Planner     Market Data +         Technical
  (NLP → plan)         Data Cleaning         Indicators
     │                      │                     │
     └──────────────────────▼─────────────────────┘
                            │
              Agent 5       │       ML Confidence
              Strategy      │       Scorer (RF)
              Builder        │
                            │
              Agent 6 ──────┤
              Risk Mgmt     │
                            │
              Agent 7 ──────┤
              Backtesting   │
                            │
              Agent 8 ──────┤
              Performance   │
              Analytics     │
                            │
              Agent 9 ──────┤
              Optimization  │
              (walk-fwd)    │
                            │
              Agent 10 ─────┤
              Portfolio Mgr │
                            │
              Agent 11 ─────┤
              Paper Trading │
              + Supervisor  │
                            │
              Agent 12 ─────┘
              Reporting
```

---

## 📸 Screenshots

> Launch the app locally to see the full interactive dashboard:

| Section | Description |
|---|---|
| **Sidebar** | Natural-language idea input + full risk configuration |
| **Overview Tab** | Auto-generated strategy plan + ML confidence score |
| **Chart & Signals** | Candlestick with EMA/BB/MACD overlays + buy/sell markers |
| **Backtest Tab** | Equity curve vs. buy-&-hold benchmark + trade log |
| **Optimization Tab** | Walk-forward leaderboard + parameter heatmap |
| **Risk Tab** | Live risk evaluation + position sizing calculator |
| **Paper Trading Tab** | Simulated order book + trade journal |
| **Analytics Tab** | 12 performance KPIs + monthly returns bar chart |
| **Reports Tab** | One-click CSV & PDF export |

---

## ⚙️ Configuration

All parameters are configurable **live from the sidebar** — no code changes required:

```
Capital & Risk
├── Initial Capital ($)         — default: $100,000
├── Risk per Trade (%)          — default: 1%
├── Stop Loss (%)               — default: 3%
├── Take Profit (%)             — default: 6%
├── Trailing Stop (%)           — default: 0% (disabled)
├── Max Daily Loss (%)          — default: 5%
└── Max Drawdown Limit (%)      — default: 20%

Advanced Execution
├── Commission (bps)            — default: 5 bps
├── Slippage (bps)              — default: 3 bps
├── Max Portfolio Exposure (%)  — default: 100%
├── Strategy Allocation (%)     — default: 60%
└── Run Parameter Optimization  — default: ON
```

---

## 🌐 Supported Assets (Examples)

| Type | Examples |
|---|---|
| **Stocks** | AAPL, TSLA, NVDA, MSFT, META, AMZN, GOOGL |
| **Indian Stocks** | RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS |
| **Crypto** | BTC-USD, ETH-USD, SOL-USD, DOGE-USD, XRP-USD |
| **Indices** | ^NSEI (Nifty 50), ^BSESN (Sensex), ^GSPC (S&P 500), ^IXIC (NASDAQ) |
| **Forex** | EURUSD=X, GBPUSD=X, USDINR=X, USDJPY=X |
| **Commodities** | GC=F (Gold), CL=F (Crude Oil), SI=F (Silver) |
| **ETFs** | SPY, QQQ, VOO, GLD, ARKK |

---

## 💡 Example Trading Ideas

```
"Trade BTC using EMA 20/50 crossover with RSI confirmation"
"Create an intraday NIFTY breakout strategy"
"Apply Bollinger Band mean reversion on TSLA"
"Build a MACD crossover swing strategy for AAPL"
"RSI oversold/overbought strategy on ETH with 25/75 thresholds"
"50-day breakout strategy on Gold futures"
```

---

## 📂 Project Structure

```
AI-Powered-Trading-Bot-with-ML/
├── app.py                    # Main application — all 12 agents + UI
├── requirements.txt          # Python dependencies
├── config.py                 # Centralised constants & defaults
├── .gitignore                # Excludes venv, __pycache__, .db, secrets
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
└── README.md                 # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
# Fork → Clone → Create a branch → Make changes → Open a PR
git checkout -b feature/your-feature-name
```

---

## 📋 Disclaimer

> **This platform is for educational and research purposes only.**
> It is **not** financial advice. Backtested performance does not guarantee future results.
> Never risk money you cannot afford to lose. Always consult a licensed financial advisor before trading.

---

## 📫 Connect

Built with ❤️ by **Dheeraj Muley**

[![GitHub](https://img.shields.io/badge/GitHub-dheeraj815-black?logo=github)](https://github.com/dheeraj815)

---

<div align="center">
⭐ <strong>Star this repo if you find it useful!</strong> ⭐
</div>
