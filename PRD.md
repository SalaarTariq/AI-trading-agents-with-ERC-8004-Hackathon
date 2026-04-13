# PRD: Hybrid AI Trading Agent with ERC-8004

**Hackathon**: AI Trading Agents with ERC-8004 (lablab.ai / Surge)
**Updated**: 2026-04-07 | **Status**: Pre-Hackathon (paper trading)

---

## What Is This?

An autonomous crypto trading agent that makes **verifiable, risk-managed decisions** by combining rule-based strategies with AI confidence scoring. Every trade is cryptographically hashed using the **ERC-8004 standard**, creating an auditable proof trail that can later be submitted on-chain.

No real money is involved. This is a paper-trading simulation using historical Binance data.

---

## How It Works

```
Market Data (CSV) 
  -> Compute Indicators (EMA, RSI, MACD, ATR, Bollinger Bands, ADX)
  -> Detect Market Regime (trending / ranging / choppy)
  -> Generate Signals (momentum in trends, mean-reversion in ranges)
  -> Score Confidence (regime-weighted blend of both strategies)
  -> Validate Risk (8 checks: loss caps, sizing, drawdown, volatility)
  -> Execute Paper Trade (if all checks pass)
  -> Log SHA256 Proof Hash (ERC-8004: Identity + Reputation + Intent + Validation)
  -> Display on Dashboard (Streamlit)
```

**Key rule**: No single signal can trigger a trade. Both strategies must contribute, risk manager must approve, and everything gets hashed.

---

## Core Components

| Module | What It Does |
|--------|-------------|
| `modules/strategy.py` | Detects regime, generates momentum + mean-reversion signals |
| `modules/confidence_scoring.py` | Blends signals with regime-aware weights (85/15 trending, 15/85 ranging) |
| `risk/risk_manager.py` | 8-layer validation: confidence gate, loss caps, cooldowns, ATR-based SL/TP, drawdown defense |
| `validation/proof_logger.py` | Builds ERC-8004 records (Identity, Reputation, Intent, Validation) and SHA256 hashes them |
| `dashboard/dashboard.py` | Streamlit UI showing portfolio KPIs, trade history, regime charts, proof audit |
| `utils/indicators.py` | Precomputes all technical indicators once, O(1) per-bar lookup |
| `main.py` | Orchestrates the full pipeline: load -> compute -> signal -> risk -> trade -> log |

---

## Risk Management

Every trade passes through these checks in order:

1. Confidence >= 0.67 (raised to 0.80 in high volatility)
2. Daily loss < 8% of portfolio
3. Not in cooldown (8-bar pause after 3 consecutive stop-losses)
4. Position size within 30% cap, cash available
5. Regime-adjusted sizing (choppy: 40%, defensive: 50%)
6. Drawdown defense (50% size cut after 15% drawdown from peak)
7. ATR-based dynamic stop-loss and take-profit
8. Risk-per-trade budget cap

If any check fails, the trade is rejected and the reason is logged.

---

## ERC-8004 Proof System

Every decision produces a record with four registries:

- **Identity**: agent ID, strategy version, trading pair
- **Reputation**: portfolio value, PnL, win rate, drawdown
- **Intent** (EIP-712 style): action, price, size, SL/TP, confidence
- **Validation**: SHA256 hash of the canonical JSON record

Proofs are append-only (`proof_log.jsonl`). Tamper detection built in. Designed for easy migration to on-chain submission via Web3.

---

## Tech Stack

Python 3.10+ | pandas, numpy | Streamlit | pytest | hashlib (stdlib)

No external AI/ML dependencies. Fully deterministic and reproducible.

---

## Data

7 datasets: BTC, ETH, SOL, AVAX (4h candles from Binance) + 3 historical test sets. Synthetic data fallback via Geometric Brownian Motion if no CSV available.

**Results so far**: 3,000+ trades executed, 3,200+ proof records generated.

---

## Current Status

**Working**: Full pipeline, regime detection, dual strategies, confidence scoring, 8-layer risk, ERC-8004 proofs, dashboard.

**Needs fixing**: Test suite (broken after module consolidation), `strategy.py` not yet git-tracked.

---

## Project Structure

```
├── main.py                  # Entry point
├── config.py                # All tunable parameters (dataclasses)
├── modules/
│   ├── strategy.py          # Regime detection + momentum + mean-reversion
│   ├── confidence_scoring.py
│   └── ai_predictor.py      # Supplementary indicator scoring
├── risk/risk_manager.py     # 8-layer trade validation
├── validation/proof_logger.py  # ERC-8004 hashing + logging
├── dashboard/dashboard.py   # Streamlit UI
├── utils/
│   ├── helpers.py           # Indicator math (SMA, EMA, RSI, ATR, MACD, ADX, etc.)
│   ├── indicators.py        # Bulk precomputation
│   └── data_loader.py       # CSV + synthetic data loading
├── tests/                   # pytest suite
└── data/                    # 7 OHLCV datasets + trade/proof logs
```

---

## Quick Start

```bash
pip install -r requirements.txt   # pandas, numpy, streamlit, pytest
python main.py                    # Run the trading simulation
streamlit run dashboard/dashboard.py  # Launch the dashboard
pytest tests/ -v                  # Run tests
```

---

## Roadmap

| Phase | Focus |
|-------|-------|
| **Pre-Hackathon** (now) | Fix tests, stabilize, backtest across all datasets |
| **Hackathon** | On-chain proof submission (Web3), live data feed, multi-pair trading, demo |
| **Post-Hackathon** | ML prediction layer, Docker deployment, alert system |
