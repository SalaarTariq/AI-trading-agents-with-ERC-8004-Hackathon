---
name: risk_management
description: Aligns risk gating (SL/TP, sizing, volatility, caps) with signal confidence to avoid blocking good trades
---

## When to Use

- When risk filters are too strict or misaligned with signal quality
- When adjusting dynamic SL/TP, sizing, volatility filters, or daily loss caps

## Instructions

1. Keep risk checks centralized in:
   - `risk/risk_manager.py` (`check_risk`, `check_trailing_stop`, `update_after_trade`)
2. Align minimum confidence to the signal execution gate:
   - `config.py`: `RiskConfig.min_confidence`
   - `config.py`: `SignalConfig.execute_confidence_threshold`
3. Prefer *size reduction* over outright rejection for volatility, except extremes:
   - `_check_volatility()` should mostly warn/reduce size
4. Ensure SL/TP are volatility-aware via ATR when enabled:
   - `RiskConfig.use_dynamic_sl_tp`

