---
name: build-dashboard
description: Builds a dashboard to visualize trades, portfolio, and agent reasoning
---

## When to Use

- When asked to create a visualization interface for the trading agent
- When the user wants to monitor paper trading results
- When building reporting or monitoring features

## Instructions

1. Use **Streamlit** as the primary framework (fallback to Flask if requested)
2. Display the following sections:
   - **Portfolio Overview**: Current value, total PnL, PnL percentage
   - **Trade History**: Table of all executed trades with timestamps, prices, sizes
   - **AI Prediction Confidence**: Chart showing confidence scores over time
   - **Strategy Signals**: Visual breakdown of signals from each module
   - **Risk Manager Warnings**: Alerts for rejected trades, limit breaches
   - **Proof Log**: List of recent validation hashes
3. Update in real-time (auto-refresh every 5 seconds for Streamlit)
4. Use clear, readable charts (line charts for PnL, bar charts for signals)
5. Include filters for date range, strategy type, and trade outcome

## Dashboard Layout

```
┌──────────────────────────────────────────────┐
│  Portfolio Overview                          │
│  Value: $XX,XXX  |  PnL: +$X,XXX (+X.X%)   │
├──────────────────┬───────────────────────────┤
│  PnL Chart       │  Strategy Signals         │
│  (line over time)│  Momentum: BUY (0.72)     │
│                  │  MeanRev:  HOLD (0.45)    │
│                  │  AI Pred:  BUY (0.68)     │
├──────────────────┼───────────────────────────┤
│  Trade History   │  Risk Alerts              │
│  (scrollable     │  - Daily loss: 6.2%/10%   │
│   table)         │  - Max position: OK       │
│                  │  - Volatility: NORMAL     │
├──────────────────┴───────────────────────────┤
│  Validation Proof Log                        │
│  Hash: abc123... | Decision: BUY ETH @ $3200 │
└──────────────────────────────────────────────┘
```

## Data Sources

- Read trade history from `simulation/paper_trader.py` output
- Read proof hashes from `validation/proof_log.jsonl`
- Read portfolio state from the paper trader's portfolio tracker
- Read risk alerts from `risk/risk_manager.py` logs

## Example

**Input**: "Create dashboard for paper trading agent"

**Output**: `dashboard/dashboard.py` with:
- Streamlit app with all sections above
- Auto-refresh capability
- Data loading from trade logs and proof logs
- Charts using Streamlit's built-in plotting or matplotlib
- Responsive layout with columns and expanders
