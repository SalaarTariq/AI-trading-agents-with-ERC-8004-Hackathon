"""
tests/test_ai_predictor.py — Tests for the rule-based AI scorer.
"""

import numpy as np
import pandas as pd
import pytest

from modules.ai_predictor import generate_signal_from_strategy_outputs


class TestOutputFormat:
    """Verify signal output format."""

    def test_returns_required_keys(self, sample_ohlcv):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.5},
                   "mean_reversion": {"signal": 0, "raw_strength": 0.0}}
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert "signal" in result
        assert "confidence" in result
        assert "metadata" in result
        assert "prob_up" in result
        assert "rolling_accuracy" in result

    def test_signal_is_valid(self, sample_ohlcv):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.6}}
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self, sample_ohlcv):
        signals = {"momentum": {"signal": -1, "raw_strength": 0.8}}
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_prob_up_in_range(self, sample_ohlcv):
        signals = {"momentum": {"signal": 1, "raw_strength": 1.0},
                   "mean_reversion": {"signal": 1, "raw_strength": 1.0}}
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert 0.05 <= result["prob_up"] <= 0.95


class TestDeterministic:
    """Rule-based scorer must be deterministic."""

    def test_same_input_same_output(self, sample_ohlcv):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.7},
                   "mean_reversion": {"signal": -1, "raw_strength": 0.3}}
        r1 = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        r2 = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert r1["signal"] == r2["signal"]
        assert r1["confidence"] == r2["confidence"]
        assert r1["prob_up"] == r2["prob_up"]


class TestDirectionLogic:
    """Test that strong agreement produces expected directions."""

    def test_strong_buy_agreement(self, trending_up_data):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.9},
                   "mean_reversion": {"signal": 1, "raw_strength": 0.8}}
        result = generate_signal_from_strategy_outputs(signals, trending_up_data)
        assert result["prob_up"] > 0.5
        assert result["signal"] in ("BUY", "HOLD")

    def test_strong_sell_agreement(self, trending_down_data):
        signals = {"momentum": {"signal": -1, "raw_strength": 0.9},
                   "mean_reversion": {"signal": -1, "raw_strength": 0.8}}
        result = generate_signal_from_strategy_outputs(signals, trending_down_data)
        assert result["prob_up"] < 0.5
        assert result["signal"] in ("SELL", "HOLD")

    def test_conflicting_signals_near_neutral(self, sample_ohlcv):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.5},
                   "mean_reversion": {"signal": -1, "raw_strength": 0.5}}
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        # Conflicting → prob_up near 0.5
        assert 0.3 <= result["prob_up"] <= 0.7


class TestEdgeCases:
    """Test edge cases and fallbacks."""

    def test_no_df_still_works(self):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.8},
                   "mean_reversion": {"signal": 1, "raw_strength": 0.6}}
        result = generate_signal_from_strategy_outputs(signals, df=None)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_short_df_still_works(self):
        short = pd.DataFrame({
            "open": [100, 101], "high": [102, 103],
            "low": [99, 100], "close": [101, 102],
            "volume": [1000, 1100],
        })
        signals = {"momentum": {"signal": 0, "raw_strength": 0.0}}
        result = generate_signal_from_strategy_outputs(signals, short)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_empty_strategy_signals(self, sample_ohlcv):
        result = generate_signal_from_strategy_outputs({}, sample_ohlcv)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_metadata_has_source(self, sample_ohlcv):
        signals = {"momentum": {"signal": 1, "raw_strength": 0.5}}
        result = generate_signal_from_strategy_outputs(signals, sample_ohlcv)
        assert result["metadata"]["source"] == "rule_based_scorer"
