# Balanced Hybrid AI Trading Agent

## Project Overview

- **Goal**: Build a trustworthy, autonomous hybrid AI trading agent for crypto.
- **Hackathon**: AI Trading Agents with ERC-8004 - lablab.ai / Surge
- **Modules**: momentum trading, mean-reversion trading, yield optimizer, AI reasoning
- **Risk Management**: stop-loss, take-profit, max position, daily loss caps
- **Validation**: hash-based proof logs (simulating ERC-8004 validation)
- **Dashboard**: Streamlit/Flask visualization
- **Execution**: Hybrid AI + rule-based decision logic
- **Pre-Hackathon Objective**: Paper-trade using historical/sandbox data

## Architecture

```
balanced_hybrid_ai_agent/
├── main.py                    # Entry point - orchestrates all modules
├── config/
│   └── config.yaml            # Trading parameters, risk thresholds, module toggles
├── modules/
│   ├── momentum.py            # Momentum trading strategy
│   ├── mean_reversion.py      # Mean-reversion trading strategy
│   ├── yield_optimizer.py     # Yield optimization strategy
│   ├── ai_predictor.py        # AI-based prediction (weighted ensemble)
│   └── strategy_manager.py    # Combines all modules into final trade signal
├── risk/
│   └── risk_manager.py        # Risk management and trade gating
├── simulation/
│   └── paper_trader.py        # Executes trades with virtual funds and logs PnL
├── validation/
│   └── proof_logger.py        # SHA256 hash-based proof logging (ERC-8004 style)
├── dashboard/
│   └── dashboard.py           # Streamlit/Flask visualization
├── utils/
│   ├── config.py              # Configuration and constants
│   ├── data_loader.py         # Historical/sandbox data loading
│   ├── indicators.py          # MA, RSI, Bollinger Bands, z-score
│   ├── helpers.py             # General helper functions
│   └── logger.py              # Logging utility for trades and events
├── tests/
│   ├── test_momentum.py
│   ├── test_mean_reversion.py
│   ├── test_yield_optimizer.py
│   ├── test_risk_manager.py
│   └── test_proof_logger.py
└── data/
    ├── historical_prices.csv  # Sample historical crypto price data
    └── live_prices.json       # Optional live price API cache
```

## Workflow / Execution

1. **Load market data** (CSV/historical prices) via `data_loader.py`
2. **Compute indicators** via `indicators.py` (MA, RSI, Bollinger Bands, z-score)
3. **Modules generate signals** - momentum, mean-reversion, yield
4. **AI Predictor evaluates signals** - produces confidence score
5. **Strategy Manager combines results** - final trade signal
6. **Risk Manager checks signal** - validates stop-loss, max allocation, daily caps
7. **Paper Trader executes trade** - updates portfolio, logs PnL
8. **Proof Logger records hash** - input + decision + timestamp (ERC-8004 style)
9. **Dashboard updates** - visualize trades, portfolio, AI reasoning

## Tech Stack

- Python 3.10+
- IDE: VS Code
- Libraries: pandas, numpy, scikit-learn, matplotlib, streamlit, hashlib, yaml, requests
- AI Layer: Claude-generated reasoning module
- Testing: pytest
- Optional: Docker for isolated environment

## Project Instructions for Claude

- Always generate modular code following Python best practices.
- Always include risk management checks before trades.
- Always produce validation proof for every decision.
- Use `.claude/rules/` and `.claude/skills/` when applicable.
- Follow the rules defined in `.claude/rules/*.md`.
- When generating code, ensure each module is independently testable.
- Use type hints and docstrings for all public functions.
- Log all trade events with timestamps using Python's `logging` module.
- Never execute real trades — this is a paper-trading simulation only.
- Code should be fully runnable in VS Code.
- Pre-hackathon simulation must be fully operational without blockchain.

## Key Design Principles

1. **Hybrid Decision Making**: Every trade decision combines rule-based signals (momentum, mean-reversion) with AI predictions. Neither alone can trigger a trade.
2. **Risk-First Architecture**: No trade executes without passing all risk management checks.
3. **Provable Decisions**: Every decision is hashed and logged for ERC-8004 style on-chain verification.
4. **Modularity**: Each component can be tested, replaced, or upgraded independently.
5. **Transparency**: The dashboard provides real-time visibility into agent reasoning.
