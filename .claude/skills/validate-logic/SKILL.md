---
name: validate-logic
description: Validates all trade decisions and produces ERC-8004 style proof hashes
---

## When to Use

- When asked to verify agent output or trade decisions
- When generating validation reports for paper trading sessions
- When auditing the proof log for consistency

## Instructions

1. **Collect** all inputs for a trade decision:
   - Raw market data (prices, volumes, indicators)
   - Strategy module outputs (momentum, mean-reversion, yield signals)
   - AI predictor output (final prediction, confidence, weights)
   - Risk manager results (approved/rejected, reasons, adjusted size)
2. **Generate SHA256 hash** of the full decision record:
   ```python
   import hashlib
   import json

   def generate_proof_hash(decision_record: dict) -> str:
       canonical = json.dumps(decision_record, sort_keys=True, separators=(',', ':'))
       return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
   ```
3. **Store** the hash in `validation/proof_log.jsonl` with:
   - `proof_hash`: The SHA256 hash
   - `timestamp`: ISO 8601 UTC timestamp
   - `decision_summary`: Human-readable trade summary
   - `full_record`: Complete decision record for verification
4. **Output** a validation report for each trade:
   - Pass/Fail status for each risk check
   - Proof hash
   - Any warnings or anomalies detected

## Validation Checks

For each trade decision, verify:

- [ ] All required fields are present in the decision record
- [ ] Timestamp is valid ISO 8601 format
- [ ] Strategy signals are in valid format with confidence in [0.0, 1.0]
- [ ] Risk manager was consulted (not bypassed)
- [ ] If trade was executed, risk manager approved it
- [ ] Proof hash matches the decision record (re-hash to verify)
- [ ] Proof hash is unique (no duplicate hashes in the log)

## Verification Function

```python
def verify_proof(record: dict, expected_hash: str) -> bool:
    """Re-hash the record and compare with expected hash."""
    actual_hash = generate_proof_hash(record)
    return actual_hash == expected_hash
```

## Report Format

```
═══════════════════════════════════════════
 VALIDATION REPORT — Trade #42
═══════════════════════════════════════════
 Timestamp:    2026-03-17T14:30:00Z
 Action:       BUY ETH/USDC @ $3,200
 Size:         0.5 ETH ($1,600)
───────────────────────────────────────────
 Strategy Signals:
   Momentum:      BUY  (confidence: 0.72)
   Mean Reversion: HOLD (confidence: 0.45)
   AI Predictor:  BUY  (confidence: 0.68)
───────────────────────────────────────────
 Risk Checks:
   Stop-loss (4%):     ✓ SET at $3,072
   Take-profit (6%):   ✓ SET at $3,392
   Position size:      ✓ 16% of portfolio (< 30%)
   Daily loss cap:     ✓ 2.1% used (< 10%)
   Volatility filter:  ✓ Normal (1.2x avg)
───────────────────────────────────────────
 Proof Hash: a1b2c3d4e5f6...
 Verification: ✓ PASS
═══════════════════════════════════════════
```

## Example

**Input**: "Validate last 10 trades"

**Output**: List of 10 validation reports, each with:
- Trade summary
- All risk check pass/fail results
- Proof hash and verification status
- Any anomalies or warnings flagged
