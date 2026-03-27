# Balanced Hybrid AI Trading Agent — ERC-8004

**AI Trading Agents with ERC-8004 Hackathon** | lablab.ai / Surge
**Track:** Verifiable Trust + Risk-Adjusted Performance

---

## What It Does

A hybrid AI + rule-based crypto trading agent that makes provably trustworthy decisions.  Every trade is validated through a multi-layer pipeline and logged with a SHA256 proof hash — ready for ERC-8004 on-chain submission.

**Key Capabilities:**
- **Bidirectional trading** — BUY and SELL signals in any market regime
- **Regime-adaptive parameters** — different confidence thresholds, position sizing, and SL/TP for trending vs choppy markets
- **Confidence-weighted signal combination** — momentum, mean-reversion, and indicator agreement scored with regime-aware weights
- **Comprehensive risk management** — ATR-based SL/TP, daily loss caps, consecutive-loss cooldowns, defensive mode
- **ERC-8004 proof logging** — every decision produces a deterministic SHA256 hash mapped to Identity, Reputation, and Validation registries

---

## ERC-8004 Integration

| ERC-8004 Component     | Implementation                                                              |
|------------------------|-----------------------------------------------------------------------------|
| **Identity Registry**  | Agent ID, strategy version, trading pair — logged in every proof record     |
| **Reputation Registry**| Portfolio snapshot (value, cash, daily PnL) at decision time                |
| **Validation Registry**| SHA256 proof hash of full decision record (tamper-proof, deterministic)     |
| **TradeIntent (EIP-712)** | Structured intent: pair, action, price, size, SL/TP, confidence, risk pass/fail |

Proof hashes are appended to `validation/proof_log.jsonl` in append-only JSONL format. Each entry can be independently verified by re-hashing the `full_record` field.

---

## Architecture

```
main.py                         # Entry point — orchestrates full pipeline
config.py                       # All tunable parameters (dataclass-based)
modules/
  momentum.py                   # EMA spread + MACD + RSI momentum strategy
  mean_reversion.py             # Bollinger Bands + RSI mean-reversion strategy
  ai_predictor.py               # Deterministic rule-based trade scorer
  confidence_scoring.py         # Regime-aware weighted signal combination
utils/
  data_loader.py                # CSV loading + synthetic data generation
  indicators.py                 # Centralized indicator computation (EMA, RSI, MACD, ATR, BB)
  helpers.py                    # Technical indicator primitives + utilities
risk/
  risk_manager.py               # Trade gating: confidence, sizing, SL/TP, caps, emergency rules
validation/
  proof_logger.py               # ERC-8004 proof hash generation + JSONL logging
dashboard/
  dashboard.py                  # Streamlit dashboard: PnL, Sharpe, drawdown, regime stats, proofs
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the trading agent
```bash
# On live Binance 4h data
python main.py --data data/eth_live_4h.csv
python main.py --data data/btc_live_4h.csv
python main.py --data data/sol_live_4h.csv

# On test datasets
python main.py --data data/d1.csv
python main.py --data data/d2.csv
python main.py --data data/d3.csv
```

### Clear logs before a fresh run
```bash
> validation/proof_log.jsonl
> data/trade_history.jsonl
```

### Launch the dashboard
```bash
streamlit run dashboard/dashboard.py
```

### Run tests
```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=modules --cov=risk --cov=validation --cov-report=term-missing
```

---

## Verifiable Trust

Every trade decision follows this pipeline:

1. **Market data** loaded and indicators computed
2. **Strategy modules** (momentum + mean-reversion) generate independent signals
3. **Rule-based scorer** evaluates indicator agreement
4. **Regime detection** classifies market as trending_up / trending_down / choppy
5. **Confidence scoring** combines signals with regime-aware weights
6. **Risk manager** validates against all rules (confidence threshold, position sizing, daily cap, cooldowns)
7. **Paper trader** executes approved trades with ATR-based SL/TP
8. **Proof logger** generates a SHA256 hash of the full decision record (ERC-8004 Validation artifact)

The proof hash is **deterministic** — the same inputs always produce the same hash. This enables independent verification and future on-chain anchoring.

---

## Risk-Adjusted Performance

| Metric         | Description                                                |
|----------------|------------------------------------------------------------|
| **Sharpe Ratio** | Risk-adjusted return (annualized, from trade PnLs)       |
| **Max Drawdown** | Largest peak-to-trough decline                           |
| **Win Rate**     | Percentage of profitable trades                          |
| **Regime Stats** | Distribution of decisions across market regimes          |

The agent adapts its behavior by regime:
- **Trending** — higher momentum weight, wider TP, lower confidence threshold
- **Choppy** — higher mean-reversion weight, tighter SL/TP, higher confidence threshold, smaller positions

---

## Tech Stack

- **Python 3.10+** — pandas, numpy, streamlit
- **Zero ML dependencies** — pure rule-based scorer (no sklearn, no torch)
- **SHA256 proof hashing** — stdlib `hashlib` only
- **Testing** — pytest with >80% coverage target

---

## Hackathon Submission Summary

This project implements a hybrid AI + rule-based crypto trading agent that demonstrates verifiable trust through ERC-8004 proof logging. Every trade decision — from signal generation through risk validation to execution — is captured in a deterministic SHA256 hash mapped to ERC-8004's Identity, Reputation, and Validation registries. The agent adapts to market regimes (trending/choppy) with dynamic confidence thresholds, ATR-based position sizing, and bidirectional trading. The Streamlit dashboard provides real-time visibility into PnL, Sharpe ratio, max drawdown, regime distribution, and proof audit trails. Built for the Surge Capital Sandbox with EIP-712 TradeIntent integration ready for Risk Router and Aerodrome DEX deployment on Base.
