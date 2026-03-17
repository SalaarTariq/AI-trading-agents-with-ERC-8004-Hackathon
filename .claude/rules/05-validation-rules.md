# Validation Guidelines

## Trade Decision Record

Every trade must produce a complete decision record containing:

1. **Timestamp**: ISO 8601 format UTC timestamp of the decision
2. **Module Inputs**: Raw market data fed to each strategy module
3. **Strategy Signals**: Output from each strategy (momentum, mean-reversion, yield)
4. **AI Prediction Outputs**: The AI predictor's final signal, confidence, and reasoning weights
5. **Risk Manager Checks**: Full risk assessment result including all rules evaluated
6. **Final Decision**: BUY / SELL / HOLD with execution details
7. **Portfolio State**: Portfolio value and positions at time of decision

## Hash Generation (ERC-8004 Style)

- Generate a **SHA256 hash** of the complete decision record
- The hash input must be a **deterministic JSON serialization** of the record:
  - Sort all dictionary keys alphabetically
  - Use consistent number formatting (no floating point drift)
  - Encode as UTF-8 before hashing
- The resulting hash serves as a unique, verifiable fingerprint of the decision

## Hash Format

```python
import hashlib
import json

def generate_proof_hash(decision_record: dict) -> str:
    """Generate SHA256 proof hash for a trade decision."""
    canonical = json.dumps(decision_record, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
```

## Logging Requirements

- All proof hashes must be **logged to a persistent file** (`validation/proof_log.jsonl`)
- Each log entry is a single JSON line containing:
  - `proof_hash`: The SHA256 hash
  - `timestamp`: When the proof was generated
  - `decision_summary`: Brief human-readable summary
  - `full_record`: The complete decision record (for verification)
- Proof logs must be **append-only** — never modify or delete existing entries

## Verification

- Provide a `verify_proof(record, expected_hash)` function that re-hashes the record and compares
- Include verification in unit tests to ensure hash reproducibility
- Paper trading reports must include all proof hashes for the session

## ERC-8004 Simulation

- In the hackathon context, these proof hashes simulate what would be submitted on-chain
- The validation module should be structured so that swapping in actual on-chain submission (via Web3) requires minimal changes
- Keep the hashing and logging logic separate from any future blockchain integration
