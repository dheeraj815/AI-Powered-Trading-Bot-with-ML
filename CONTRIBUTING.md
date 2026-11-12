# Contributing to QuantPilot AI

Thank you for your interest in contributing! 🎉

## How to Contribute

### 1. Fork & Clone
```bash
git clone https://github.com/<your-username>/AI-Powered-Trading-Bot-with-ML.git
cd AI-Powered-Trading-Bot-with-ML
```

### 2. Set Up Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 4. Make Your Changes
- Follow the existing code style (clean, well-commented Python)
- Add a new agent class if adding significant new functionality
- Update `config.py` for any new configurable constants
- Test your changes by running `streamlit run app.py`

### 5. Commit & Push
```bash
git add .
git commit -m "feat: add <short description of your change>"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request
Go to the GitHub repo and open a Pull Request with a clear description of what you changed and why.

---

## Areas Where Contributions Are Welcome

| Area | Ideas |
|---|---|
| 🤖 **New Strategies** | VWAP, Ichimoku, Supertrend, Pivot Points |
| 📊 **New Indicators** | Stochastic RSI, Williams %R, OBV |
| 🌐 **Broker Integration** | Alpaca, Zerodha Kite, IBKR paper trading API |
| 🧪 **Testing** | Unit tests for agents, backtesting validation |
| 📱 **UI** | Mobile-friendly layout improvements |
| 📈 **ML Models** | XGBoost, LSTM, Gradient Boosting |

---

## Code Style

- Use **type hints** for all function signatures
- Write **docstrings** for all classes and public methods
- Keep functions **small and single-purpose**
- Avoid hardcoded values — use `config.py` constants

---

## Reporting Issues

Please open a [GitHub Issue](https://github.com/dheeraj815/AI-Powered-Trading-Bot-with-ML/issues) with:
- A clear title
- Steps to reproduce
- Expected vs. actual behaviour
- Python version + OS

---

## ⚠️ Important Note

This project is for **educational purposes only**. Contributions that enable real-money trading execution without explicit user confirmation will not be accepted.
