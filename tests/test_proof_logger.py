"""
tests/test_proof_logger.py — Tests for the proof logging / validation system.

Tests cover hash determinism, format correctness, log integrity,
and verification functions.
"""

import json
import tempfile
from pathlib import Path

import pytest

from validation.proof_logger import (
    generate_proof_hash,
    verify_proof,
    log_decision,
    get_last_trades_hash,
    load_all_proofs,
    verify_log_integrity,
)


# ---------------------------------------------------------------------------
# Sample records
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "timestamp": "2025-06-15T14:30:00+00:00",
    "pair": "ETH/USDC",
    "current_price": 3200.0,
    "strategy_signals": {
        "momentum": {"signal": "BUY", "confidence": 0.72},
        "mean_reversion": {"signal": "HOLD", "confidence": 0.45},
    },
    "ai_prediction": {
        "signal": "BUY",
        "confidence": 0.68,
        "reasoning": "Strong trend with volume confirmation",
    },
    "risk_result": {
        "approved": True,
        "reasons": [],
        "adjusted_size": 10000,
    },
    "final_decision": {
        "action": "BUY",
        "pair": "ETH/USDC",
        "entry_price": 3200.0,
        "size": 10000,
    },
}

DIFFERENT_RECORD = {
    **SAMPLE_RECORD,
    "current_price": 3201.0,  # One dollar difference
}


class TestHashGeneration:
    """Verify SHA256 proof hash generation."""

    def test_hash_is_deterministic(self):
        hash1 = generate_proof_hash(SAMPLE_RECORD)
        hash2 = generate_proof_hash(SAMPLE_RECORD)
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        hash1 = generate_proof_hash(SAMPLE_RECORD)
        hash2 = generate_proof_hash(DIFFERENT_RECORD)
        assert hash1 != hash2

    def test_hash_is_valid_sha256(self):
        h = generate_proof_hash(SAMPLE_RECORD)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_string(self):
        h = generate_proof_hash(SAMPLE_RECORD)
        assert isinstance(h, str)

    def test_empty_record_still_hashes(self):
        h = generate_proof_hash({})
        assert len(h) == 64

    def test_key_order_does_not_matter(self):
        """sort_keys=True should make key order irrelevant."""
        record_a = {"b": 2, "a": 1}
        record_b = {"a": 1, "b": 2}
        assert generate_proof_hash(record_a) == generate_proof_hash(record_b)

    def test_nested_record_deterministic(self):
        nested = {"outer": {"z": 3, "a": 1}, "data": [1, 2, 3]}
        h1 = generate_proof_hash(nested)
        h2 = generate_proof_hash(nested)
        assert h1 == h2


class TestVerification:
    """Verify proof verification function."""

    def test_correct_hash_passes(self):
        h = generate_proof_hash(SAMPLE_RECORD)
        assert verify_proof(SAMPLE_RECORD, h) is True

    def test_wrong_hash_fails(self):
        assert verify_proof(SAMPLE_RECORD, "0" * 64) is False

    def test_tampered_record_fails(self):
        h = generate_proof_hash(SAMPLE_RECORD)
        tampered = {**SAMPLE_RECORD, "current_price": 9999.0}
        assert verify_proof(tampered, h) is False


class TestLogDecision:
    """Test proof logging to file."""

    def test_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            proof_hash = log_decision(SAMPLE_RECORD, log_path=log_path)
            assert log_path.exists()
            assert len(proof_hash) == 64

    def test_log_appends_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            log_decision(SAMPLE_RECORD, log_path=log_path)
            log_decision(DIFFERENT_RECORD, log_path=log_path)
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_logged_entry_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            log_decision(SAMPLE_RECORD, log_path=log_path)
            entry = json.loads(log_path.read_text().strip())
            assert "proof_hash" in entry
            assert "timestamp" in entry
            assert "decision_summary" in entry
            assert "full_record" in entry

    def test_logged_hash_matches_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            returned_hash = log_decision(SAMPLE_RECORD, log_path=log_path)
            entry = json.loads(log_path.read_text().strip())
            assert entry["proof_hash"] == returned_hash
            # Verify the full record re-hashes to the same value
            assert verify_proof(entry["full_record"], entry["proof_hash"])


class TestGetLastTrades:
    """Test retrieval of recent proof entries."""

    def test_returns_last_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            for i in range(5):
                record = {**SAMPLE_RECORD, "current_price": 3000 + i}
                log_decision(record, log_path=log_path)
            entries = get_last_trades_hash(n=3, log_path=log_path)
            assert len(entries) == 3

    def test_missing_file_returns_empty(self):
        entries = get_last_trades_hash(log_path="/nonexistent/path.jsonl")
        assert entries == []


class TestLogIntegrity:
    """Test full log verification."""

    def test_all_valid_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            for i in range(3):
                record = {**SAMPLE_RECORD, "current_price": 3000 + i}
                log_decision(record, log_path=log_path)
            result = verify_log_integrity(log_path=log_path)
            assert result["total"] == 3
            assert result["valid"] == 3
            assert result["invalid"] == 0

    def test_tampered_entry_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "proof_log.jsonl"
            log_decision(SAMPLE_RECORD, log_path=log_path)

            # Tamper with the logged record
            lines = log_path.read_text().strip().split("\n")
            entry = json.loads(lines[0])
            entry["full_record"]["current_price"] = 9999.0
            log_path.write_text(json.dumps(entry) + "\n")

            result = verify_log_integrity(log_path=log_path)
            assert result["invalid"] == 1
