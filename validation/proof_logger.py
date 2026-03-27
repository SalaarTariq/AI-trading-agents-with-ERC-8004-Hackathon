"""
validation/proof_logger.py — SHA256 proof logging for ERC-8004 validation.

Every trade decision (executed or rejected) produces a deterministic
hash of its full record. Hashes are appended to a JSONL file for
auditability and future on-chain submission.

ERC-8004 Registry Mapping:
    Identity Registry  — Agent identity, strategy version, pair (who made the decision)
    Reputation Registry — Portfolio snapshot, rolling metrics (track record over time)
    Validation Registry — SHA256 proof hash of the full decision record (tamper-proof artifact)
    TradeIntent (EIP-712) — Structured intent: pair, action, price, size, SL/TP, confidence
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

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

    full_record = _build_full_record(decision_record)
    proof_hash = generate_proof_hash(full_record)

    entry = {
        "proof_hash": proof_hash,
        "timestamp": utc_now_iso(),
        "decision_summary": _summarize(full_record),
        "full_record": full_record,
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
    # Support both legacy shape (final_decision at top-level)
    # and ERC-8004 wrapper shape (decision.final_decision).
    decision = record.get("final_decision", {})
    if not decision and isinstance(record.get("decision"), dict):
        decision = record["decision"].get("final_decision", {}) or {}
    action = decision.get("action", "UNKNOWN")
    pair = decision.get("pair", "?")
    price = decision.get("entry_price", 0)
    size = decision.get("size", 0)

    if action in ("BUY", "SELL"):
        return f"{action} {pair} @ ${price:,.2f} (size ${size:,.0f})"
    return f"HOLD {pair} — no action"


def _build_full_record(decision_record: dict) -> dict:
    """
    Build an ERC-8004-aligned record wrapper around the raw decision record.

    The wrapper is deterministic (derived from decision_record only) so the
    proof hash is stable for the same decision inputs.
    """
    pair = decision_record.get("pair") or decision_record.get("final_decision", {}).get("pair") or "?"
    dataset = decision_record.get("dataset", "")

    # ── ERC-8004: Identity Registry ──────────────────────────────────
    # Maps to on-chain Identity Registry — uniquely identifies the agent,
    # its strategy version, and the market it operates on.
    identity = {
        "agent_id": "balanced_hybrid_ai_trading_agent",
        "strategy_version": "v2_confidence_weighted",
        "pair": pair,
        "dataset": dataset,
    }

    # ── ERC-8004: Reputation Registry ────────────────────────────────
    # Snapshot of agent performance at decision time.  On-chain, this
    # feeds the Reputation Score that other agents / protocols query.
    ps = decision_record.get("portfolio_state", {}) if isinstance(decision_record.get("portfolio_state"), dict) else {}
    reputation = {
        "portfolio_value": ps.get("total_value", None),
        "cash": ps.get("cash", None),
        "daily_pnl": ps.get("daily_pnl", None),
        "sample_count": None,
        "rolling_win_rate": None,
        "max_drawdown": None,
    }

    # ── ERC-8004: TradeIntent (EIP-712 structured data) ──────────────
    # This is the EIP-712 typed-data intent that will be signed and
    # submitted to the Risk Router / Aerodrome DEX on Base.
    rr = decision_record.get("risk_result", {}) if isinstance(decision_record.get("risk_result"), dict) else {}
    fd = decision_record.get("final_decision", {}) if isinstance(decision_record.get("final_decision"), dict) else {}
    intent = {
        "timestamp": decision_record.get("timestamp", ""),
        "pair": pair,
        "action": fd.get("action", "HOLD"),
        "entry_price": fd.get("entry_price", None),
        "size": fd.get("size", 0),
        "stop_loss": rr.get("stop_loss", None),
        "take_profit": rr.get("take_profit", None),
        "reasons": rr.get("reasons", []),
        "warnings": rr.get("warnings", []),
        "confidence": decision_record.get("combined_decision", {}).get("confidence", None)
        if isinstance(decision_record.get("combined_decision"), dict)
        else None,
        "indicators": decision_record.get("indicators", {}),
        "risk_pass_fail": rr.get("approved", False),
    }

    # ── ERC-8004: Validation Registry ────────────────────────────────
    # SHA256 hash of the raw decision inputs.  On-chain, this artifact
    # is stored in the Validation Registry for tamper-proof auditability.
    inputs_digest = generate_proof_hash(decision_record)
    validation = {
        "hash_algo": "sha256",
        "inputs_digest": inputs_digest,
    }

    return {
        "identity": identity,
        "reputation": reputation,
        "intent": intent,
        "validation": validation,
        "decision": decision_record,
    }
