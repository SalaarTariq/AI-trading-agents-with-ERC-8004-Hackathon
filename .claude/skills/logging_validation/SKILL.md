---
name: logging_validation
description: Structures ERC-8004-style logs (identity, reputation, validation, intent) and produces deterministic hashes
---

## When to Use

- When updating proof log schemas or validation hashing
- When ensuring decisions are auditable and verifiable end-to-end

## Instructions

1. All decisions (BUY/SELL/HOLD + rejections) must be logged via:
   - `validation/proof_logger.py` (`log_decision`)
2. Ensure the logged `full_record` contains registry sections:
   - `identity`
   - `reputation`
   - `intent`
   - `validation`
3. Ensure hashing is deterministic and verification works:
   - `generate_proof_hash()`, `verify_proof()`, `verify_log_integrity()`

