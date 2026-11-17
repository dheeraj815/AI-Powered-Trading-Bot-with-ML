---
name: quantpilot-trading-bot
description: Autonomous 12-agent quant trading and market intelligence engine. Predicts buy/sell signals, generates multi-asset backtests, executes parameter walk-forward optimization, and manages risk limits.
version: 2.0.0
author: Dheeraj Muley
license: MIT
metadata:
  framework: python
  ui: streamlit
  strategy_types:
    - EMA_SMA_Crossover
    - MACD_Crossover
    - Bollinger_Mean_Reversion
    - RSI_Mean_Reversion
    - Breakout
    - Momentum
---

# QuantPilot AI — Autonomous Quant Trading Agent Skill

QuantPilot AI is an enterprise-grade algorithmic trading and quant intelligence skill designed for autonomous AI agents (Claude Code, Cursor, Codex, Antigravity).

## Capabilities for Calling Agents

When loaded into an AI agent, this skill grants autonomous capabilities to:
1. **Analyze Natural-Language Strategy Ideas**: Parses arbitrary trading concepts into structured quantitative execution plans.
2. **Fetch OHLCV Market Data**: Downloads historical and intraday feeds for Stocks (US & India), Crypto, Forex, Indices, and Commodities via Yahoo Finance API keys.
3. **Generate Technical Indicators**: Computes native EMA, SMA, RSI, MACD, ATR, Bollinger Bands, and candlestick pattern markers.
4. **Execute Backtests**: Runs simulations taking into account transaction commissions, execution slippage, trailing stops, and dynamic position sizing.
5. **Optimize Parameters**: Conducts walk-forward grid search validation on train/test historical splits to prevent overfitting.
6. **Risk Management Supervision**: Monitors maximum drawdown breaches, daily loss thresholds, and portfolio exposure limits.
7. **Paper Trading Simulation**: Maintains a local SQLite journal recording trades, fills, and cash balances.

## Prerequisites & System Requirements

The buyer's environment must satisfy:
- **Python 3.9+**
- Packages listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

## Agent Invocation & Script Execution

### 1. Launch Interactive Dashboard
To launch the full visual UI:
```bash
streamlit run app.py
```

### 2. Programmatic Strategy Execution
To run a strategy programmatically through Python:
```python
from app import OrchestrationEngine, RiskParams

engine = OrchestrationEngine()
risk = RiskParams(initial_capital=100000.0, risk_per_trade_pct=1.0, stop_loss_pct=3.0, take_profit_pct=6.0)

# Run complete 12-agent pipeline
result = engine.run_pipeline(
    idea="Trade BTC using EMA 20/50 crossover with RSI confirmation",
    symbol_override="BTC-USD",
    risk=risk,
    allocation_pct=60.0,
    run_optimization=True
)

if result["success"]:
    perf = result["perf"]["metrics"]
    print(f"Net Return: {perf['net_profit_pct']:.2f}% | Sharpe: {perf['sharpe']:.2f} | Win Rate: {perf['win_rate']:.1f}%")
```

## Supported Symbols
- **Crypto**: `BTC-USD`, `ETH-USD`, `SOL-USD`, `DOGE-USD`, `XRP-USD`
- **US Equities**: `AAPL`, `TSLA`, `NVDA`, `MSFT`, `SPY`, `QQQ`
- **Indian Equities**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `^NSEI` (Nifty 50)
- **Forex & Commodities**: `EURUSD=X`, `GBPUSD=X`, `GC=F` (Gold), `CL=F` (Crude Oil)

## License
MIT License — Copyright (c) 2026 Dheeraj Muley. Full commercial rights granted.
