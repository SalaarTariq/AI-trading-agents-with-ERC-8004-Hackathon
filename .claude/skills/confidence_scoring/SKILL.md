---
name: confidence_scoring
description: Implements confidence-weighted signal combination with dynamic weights and execution thresholds
---

## When to Use

- When multiple strategies produce conflicting signals
- When confidence scores are inconsistent or not comparable
- When you need an execution gate like confidence >= 0.60

## Instructions

1. Combine strategy outputs using unified scoring:
   - `modules/confidence_scoring.py`
2. Use regime-adaptive weights (trending/ranging/volatile) from:
   - `config.py` (`RegimeWeights`, `SignalConfig`)
3. Enforce execution gate:
   - `config.py`: `SignalConfig.execute_confidence_threshold`
4. Keep outputs compatible with `main.py` pipeline:
   - Combined: `{"action","confidence","score","details","buy_agreement","sell_agreement","regime"}`

