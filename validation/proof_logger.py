"""
validation/proof_logger.py — SHA256 proof logging for ERC-8004 validation.


Every trade decision (executed or rejected) produces a deterministic
hash of its full record. Hashes are appended to a JSONL file for
auditability and future on-chain submission.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from config import CONFIG
from utils.helpers import utc_now_iso

logger = logging.getLogger(__name__)


def generate_proof_hash(decision_record: dict) -> str:
    """
    Generate a deterministic SHA256 hash of a decision record.

    The record is serialized to canonical JSON (sorted keys, compact
    separators) and encoded as UTF-8 before hashing.

    Args:
        decision_record: Full trade decision data.

    Returns:
        64-character hex SHA256 digest.
    """
    canonical = json.dumps(decision_record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_proof(record: dict, expected_hash: str) -> bool:
    """
    Re-hash a record and compare with the expected hash.

    Args:
        record: The decision record to verify.
        expected_hash: The hash to compare against.

    Returns:
        True if the hashes match.
    """
    actual = generate_proof_hash(record)
    match = actual == expected_hash
    if not match:
        logger.warning("Proof verification FAILED: expected %s, got %s", expected_hash, actual)
    return match


def log_decision(
    decision_record: dict,
    log_path: str | Path | None = None,
) -> str:
    """
    Hash a decision record and append it to the proof log.

    Args:
        decision_record: Full trade decision data including:
            - timestamp, strategy_signals, ai_prediction,
              risk_result, final_decision, portfolio_state
        log_path: Path to the JSONL log file (default from config).

    Returns:
        The proof hash string.
    """
    log_path = Path(log_path or CONFIG.proof_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proof_hash = generate_proof_hash(decision_record)

    entry = {
        "proof_hash": proof_hash,
        "timestamp": utc_now_iso(),
        "decision_summary": _summarize(decision_record),
        "full_record": decision_record,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    logger.info("Proof logged: %s — %s", proof_hash[:16], entry["decision_summary"])
    return proof_hash


def get_last_trades_hash(
    n: int = 10,
    log_path: str | Path | None = None,
) -> list[dict]:
    """
    Retrieve the last N trade proof entries from the log.

    Args:
        n: Number of recent entries to return.
        log_path: Path to the JSONL log file.

    Returns:
        List of proof log entries (most recent last).
    """
    log_path = Path(log_path or CONFIG.proof_log_path)

    if not log_path.exists():
        logger.warning("Proof log not found at %s", log_path)
        return []

    entries: list[dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries[-n:]


def load_all_proofs(log_path: str | Path | None = None) -> list[dict]:
    """Load all entries from the proof log."""
    log_path = Path(log_path or CONFIG.proof_log_path)
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def verify_log_integrity(log_path: str | Path | None = None) -> dict:
    """
    Verify all entries in the proof log by re-hashing.

    Returns:
        {"total": int, "valid": int, "invalid": int, "invalid_hashes": list[str]}
    """
    entries = load_all_proofs(log_path)
    valid = 0
    invalid = 0
    invalid_hashes: list[str] = []

    for entry in entries:
        record = entry.get("full_record", {})
        expected = entry.get("proof_hash", "")
        if verify_proof(record, expected):
            valid += 1
        else:
            invalid += 1
            invalid_hashes.append(expected)

    result = {
        "total": len(entries),
        "valid": valid,
        "invalid": invalid,
        "invalid_hashes": invalid_hashes,
    }
    logger.info("Proof log integrity: %d/%d valid", valid, len(entries))
    return result


def _summarize(record: dict) -> str:
    """Create a one-line human-readable summary of a decision record."""
    decision = record.get("final_decision", {})
    action = decision.get("action", "UNKNOWN")
    pair = decision.get("pair", "?")
    price = decision.get("entry_price", 0)
    size = decision.get("size", 0)

    if action in ("BUY", "SELL"):
        return f"{action} {pair} @ ${price:,.2f} (size ${size:,.0f})"
    return f"HOLD {pair} — no action"
