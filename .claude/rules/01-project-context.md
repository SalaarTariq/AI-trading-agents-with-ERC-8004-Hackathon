# Project Context

## What This Project Is

This project builds an AI trading agent with hybrid decision-making for cryptocurrency markets. It combines traditional rule-based trading strategies (momentum, mean-reversion, yield optimization) with AI-powered predictions to make more robust and trustworthy trading decisions.

## Core Requirements

- Claude must always reason using **both** AI predictions and rule-based modules. A trade signal from only one source is insufficient — hybrid consensus is required.
- The project should be **self-contained** — all dependencies, data loaders, and configuration are included in the repository.
- The project must be **testable pre-hackathon** — every module has unit tests that can run without external API keys or live market connections.
- The project must be **modular** — each component (strategy, risk, validation, dashboard) lives in its own directory and can be developed/tested independently.

## Decision Flow

1. Market data is loaded (historical CSV or sandbox API)
2. Each strategy module (momentum, mean-reversion, yield) generates a signal with confidence
3. The AI predictor weighs all signals and market context to produce a final prediction
4. The risk manager validates the proposed trade against all risk rules
5. If approved, the trade is executed (paper trade) and logged
6. The proof logger creates a SHA256 hash of the full decision record
7. The dashboard displays the trade and agent reasoning in real-time

## Key Constraints

- **No real money** — this is paper trading only
- **No external API keys required for testing** — use historical data files
- **All decisions must be auditable** — every trade has a proof hash
- **Risk management is non-negotiable** — no trade bypasses the risk manager
