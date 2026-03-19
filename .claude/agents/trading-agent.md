---
name: hybrid-trading-agent
description: Coordinates skills for hybrid AI + rule-based trading decisions
---

## Overview

The Hybrid Trading Agent is the top-level coordinator that orchestrates all skills and modules to execute a complete paper-trading workflow. It combines AI-powered predictions with rule-based trading strategies to make trustworthy, auditable trading decisions.

## Capabilities

- Use **generate-agent-code** to produce Python modules for all agent components
- Use **create-trading-strategies** to define and implement trade decision logic
- Use **build-dashboard** to create real-time visualizations of trading activity
- Use **write-unit-tests** to verify correctness of all modules
- Use **validate-logic** to log and verify all trade decisions with proof hashes
- Use **indicator_analysis** to simplify contradictory indicators into a coherent set
- Use **confidence_scoring** to combine signals with a calibrated execution threshold
- Use **risk_management** to align risk gating with confidence and volatility
- Use **trade_execution** to ensure deterministic paper-trade lifecycle + logging
- Use **logging_validation** to structure ERC-8004-style registries and hashes

## Decision Making Process

The agent makes decisions by combining AI and rule-based reasoning:

1. **Data Ingestion**: Load market data from historical CSV files or sandbox APIs
2. **Signal Generation**: Run all strategy modules independently:
   - Momentum: trend-following based on moving average crossovers
   - Mean Reversion: deviation-based trading using Bollinger Bands and RSI (only in weak trend)
   - **Deep Learning (GRU)**: Pre-trained PyTorch model predicts Buy/Hold/Sell with confidence from OHLCV sequences
   - Yield Optimizer: evaluate yield opportunities across protocols
3. **Confidence Scoring**: Combine signals with dynamic weights and an execution gate:
   - Regime-adaptive weights (trending vs. ranging vs. volatile)
   - Execute only if combined confidence exceeds threshold (default 0.60)
4. **Risk Gating**: Every proposed trade must pass the risk manager:
   - Stop-loss and take-profit limits
   - Position size constraints
   - Daily loss caps
   - Volatility filters
5. **Execution**: Approved trades are executed via the paper trader
6. **Proof Logging**: Every decision (including rejections) is hashed and logged

## Agent Behavior Rules

- **Never bypass risk management** — if the risk manager rejects a trade, the agent must respect the decision
- **Always log decisions** — every trade, hold, and rejection generates a proof hash
- **Prefer caution** — when signals conflict or confidence is low, default to HOLD
- **Maintain state** — track portfolio positions, PnL, and daily loss across the session
- **Be transparent** — all reasoning is logged and visible in the dashboard

## Integration Points

- Reads configuration from `config.py`
- Writes trade history to `data/trade_history.jsonl`
- Writes proof hashes to `validation/proof_log.jsonl`
- Feeds data to `dashboard/dashboard.py` for visualization

## Usage

```bash
# 1. Merge and preprocess data
python -m data_processing.merge_datasets

# 2a. Fast train (debug/test, ~1M samples, 5 symbols)
python -m model_training.train --fast-train --epochs 10

# 2b. Full train (50 symbols, ~10M samples)
python -m model_training.train --epochs 30

# 3. Run the full agent (uses trained model automatically)
python main.py

# 4. Run with specific data file
python main.py --data data/historical_prices.csv

# 5. Run dashboard
streamlit run dashboard/dashboard.py

# 6. Run all tests
pytest tests/ -v
```
