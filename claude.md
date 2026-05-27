# Balanced Hybrid AI Trading Agent

## Project Overview

- **Goal**: Build a trustworthy, autonomous hybrid trading agent for crypto.
- **Hackathon**: AI Trading Agents with ERC-8004 — lablab.ai / Surge.
- **Decision model**: regime-aware blend of momentum + mean-reversion +
  a deterministic AI-style indicator scorer.
- **Risk Management**: confidence gate, ATR-based SL/TP, position caps,
  daily loss cap, drawdown defense, consecutive-loss cooldown,
  trailing stop.
- **Validation**: SHA256 proof hashes over canonical-JSON decision
  records, written append-only to `validation/proof_log.jsonl`
  (ERC-8004 envelope: identity / reputation / intent / validation).
- **Dashboard**: Streamlit visualization reading the proof and trade
  JSONLs.
- **Execution**: Paper-trading only — no real orders, no API keys.

## Architecture

```
.
├── main.py                    # Thin CLI wrapper around simulation.run_paper_trading
├── config.py                  # All tunable parameters as dataclasses (Portfolio, Strategy, Signal, Risk, Regime)
├── modules/
│   ├── strategy.py            # Regime detection + momentum + mean-reversion signals
│   ├── confidence_scoring.py  # combine_signals, effective_confidence_threshold (single source)
│   └── ai_predictor.py        # Deterministic indicator-agreement scorer (3rd voter)
├── risk/
│   └── risk_manager.py        # check_risk (decomposed gates+sizers), check_trailing_stop, update_after_trade
├── simulation/
│   ├── paper_trader.py        # PaperTrader, Position, TradeRecord, run_paper_trading, required_warmup_bars
│   └── __init__.py
├── validation/
│   ├── proof_logger.py        # SHA256 over canonical JSON, ERC-8004 record wrapper
│   └── proof_log.jsonl        # Append-only proof log
├── utils/
│   ├── data_loader.py         # CSV loader + synthetic OHLCV generator
│   ├── indicators.py          # precompute_all_indicators, indicators_at, Indicators dataclass
│   └── helpers.py             # EMA/RSI/MACD/ATR/ADX/BB primitives, setup_logging
├── dashboard/
│   └── dashboard.py           # Streamlit dashboard reading proof_log.jsonl + trade_history.jsonl
├── scripts/
│   └── backtest_datasets.py   # Multi-dataset backtest harness on top of run_paper_trading
├── tests/                     # pytest; run via: pytest tests/ -v --tb=short
└── data/                      # OHLCV CSVs and the trade_history.jsonl log
```

The previous CLAUDE.md described a `config/config.yaml`, separate
`modules/{momentum,mean_reversion,yield_optimizer,strategy_manager}.py`
files, a `simulation/paper_trader.py` that did not exist, and
`utils/{config,logger}.py`. None of those exist now; the structure
above is the ground truth as of the last refactor.

## Decision flow

1. **Load OHLCV** via `utils.data_loader.load_or_generate`.
2. **Precompute indicators once** via `utils.indicators.precompute_all_indicators`.
3. **Per bar**, after warmup (`simulation.required_warmup_bars`):
   1. `modules.strategy.generate_strategy_signal` → regime + momentum + mean-reversion signals.
   2. `modules.ai_predictor.predict_from_indicators` → third opinion (toggle: `SignalConfig.use_ai_predictor`).
   3. `modules.confidence_scoring.combine_signals` → blended action + confidence.
   4. `risk.risk_manager.check_risk` → gates (confidence/daily loss/cooldown) + sizers (caps/regime/vol/drawdown/risk-per-trade) + ATR SL/TP.
   5. `validation.proof_logger.log_decision` → SHA256 proof, appended to `proof_log.jsonl`.
   6. Open a `simulation.paper_trader.Position`; trailing stop tightens per bar.
4. **Exit** on SL/TP touch or end-of-simulation mark-to-market.

## Confidence-threshold rule

There is one shared volatility uplift in
`modules.confidence_scoring.effective_confidence_threshold`. Two callers
provide their own base:

- `combine_signals` uses `max(SignalConfig.execute_confidence_threshold, regime.conf_threshold)`.
- `risk_manager._confidence_gate` uses `max(RiskConfig.min_confidence, regime.conf_threshold)`.

Both then receive the same ATR-driven uplift, so the two gates cannot
silently disagree on *how* to react to volatility.

## Tech Stack

- Python 3.10+
- Libraries: `pandas`, `numpy`, `streamlit`, `pytest`. Hashing via stdlib
  `hashlib`. No `sklearn`/`torch` — the AI scorer is deterministic rules
  over indicators.

## Project Instructions for Claude

- Modular code; type hints on public signatures; docstrings only where
  the *why* is non-obvious.
- Every trade goes through the risk manager — no bypasses.
- Every approved trade produces a proof hash.
- Tests must run without network or API keys.
- Never execute real trades — paper trading only.
- Follow `.claude/rules/*.md`; risk and proof logger have a 95%+
  coverage expectation.

## Key Design Principles

1. **Hybrid consensus**: every approved trade has at least one of two
   rule signals firing; the AI voter participates with bounded weight
   (`SignalConfig.ai_predictor_weight`).
2. **Risk-first**: risk gates are pure (no side effects beyond the
   documented cooldown arming) and sizers are composable.
3. **Provable decisions**: deterministic canonical JSON → SHA256 →
   append-only JSONL.
4. **One source for thresholds**: confidence and SL/TP rules live in
   single functions reused by every caller.
5. **Transparent state**: `Position` / `TradeRecord` are dataclasses,
   not free-form dicts; logs are typed at the write site.
