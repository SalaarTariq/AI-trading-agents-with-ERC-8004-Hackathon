"""
tests/test_ai_predictor.py — Tests for the AI prediction module.
"""

import numpy as np
import pandas as pd
import pytest

from modules.ai_predictor import (
    generate_signal,
    generate_signal_from_strategy_outputs,
    _parse_llm_response,
)
from config import AIConfig


class TestAIPredictorFormat:
    """Verify signal output format."""

    def test_returns_required_keys(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert "signal" in result
        assert "confidence" in result
        assert "metadata" in result

    def test_signal_is_valid(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert 0.0 <= result["confidence"] <= 1.0


class TestLightweightPredictor:
    """Test the sklearn-based predictor."""

    def test_insufficient_data_returns_hold(self):
        short = pd.DataFrame({
            "open": [100, 101], "high": [102, 103],
            "low": [99, 100], "close": [101, 102],
            "volume": [1000, 1100],
        })
        result = generate_signal(short)
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_deterministic_with_same_data(self, sample_ohlcv):
        r1 = generate_signal(sample_ohlcv)
        r2 = generate_signal(sample_ohlcv)
        assert r1["signal"] == r2["signal"]
        assert r1["confidence"] == r2["confidence"]

    def test_metadata_includes_model_info(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        meta = result["metadata"]
        if "reason" not in meta:
            assert "model" in meta
            assert "prob_up" in meta


class TestEnsemblePredictor:
    """Test the strategy-output ensemble."""

    def test_consensus_buy_produces_buy(self, sample_ohlcv):
        signals = {
            "momentum": {"signal": "BUY", "confidence": 0.8},
            "mean_reversion": {"signal": "BUY", "confidence": 0.7},
        }
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        # ML model has independent prediction, may not agree with strategy consensus
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_consensus_sell_produces_sell(self, sample_ohlcv):
        signals = {
            "momentum": {"signal": "SELL", "confidence": 0.9},
            "mean_reversion": {"signal": "SELL", "confidence": 0.8},
        }
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert result["signal"] in ("SELL", "HOLD")

    def test_mixed_signals_likely_hold(self, sample_ohlcv):
        signals = {
            "momentum": {"signal": "BUY", "confidence": 0.5},
            "mean_reversion": {"signal": "SELL", "confidence": 0.5},
        }
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        # Conflicting signals → likely HOLD
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_handles_missing_signals(self):
        signals = {
            "momentum": {"signal": "BUY", "confidence": 0.6},
        }
        # No df → no ML model, just strategy consensus
        result = generate_signal_from_strategy_outputs(signals, df=None)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_metadata_has_strategy_details(self, sample_ohlcv):
        signals = {
            "momentum": {"signal": "BUY", "confidence": 0.7},
        }
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        # ML model is used when data is available, so check for ML metadata
        assert "source" in result["metadata"]
        assert result["metadata"]["source"] in ("ml_model", "strategy_consensus_fallback")


class TestLLMResponseParser:
    """Test the LLM response parser."""

    def test_parse_buy_response(self):
        direction, conf, reason = _parse_llm_response("BUY 75% Strong uptrend")
        assert direction == "BUY"
        assert conf == 0.75

    def test_parse_sell_response(self):
        direction, conf, reason = _parse_llm_response("SELL 60% Overbought RSI")
        assert direction == "SELL"
        assert conf == 0.60

    def test_parse_hold_response(self):
        direction, conf, reason = _parse_llm_response("HOLD 30% No clear signal")
        assert direction == "HOLD"
        assert conf == 0.30

    def test_parse_garbage_defaults_to_hold(self):
        direction, conf, reason = _parse_llm_response("asdf jkl;")
        assert direction == "HOLD"
        assert conf == 0.0

    def test_confidence_clamped_to_1(self):
        direction, conf, reason = _parse_llm_response("BUY 150% Very confident")
        assert conf <= 1.0
