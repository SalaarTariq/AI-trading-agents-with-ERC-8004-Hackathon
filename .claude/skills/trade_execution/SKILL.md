---
name: trade_execution
description: Ensures trades are executed only when confidence and risk checks pass; logs trades deterministically
---

## When to Use

- When modifying the entry/exit lifecycle of paper trades
- When adjusting position open/close logic, SL/TP triggers, or trade history logging

## Instructions

1. Keep execution orchestration in:
   - `main.py` (`run_paper_trading`)
2. Ensure every bar produces a decision record and proof hash:
   - `validation/proof_logger.py`
3. Ensure every closed trade is appended to:
   - `data/trade_history.jsonl`
4. Preserve deterministic logging fields:
   - `dataset`, `pair`, `timestamp`, `action`, `entry_price`, `size`, `stop_loss`, `take_profit`, `proof_hash`

