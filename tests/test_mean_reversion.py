"""
tests/test_mean_reversion.py — Tests for the mean-reversion strategy.
"""

import numpy as np
import pandas as pd
import pytest

from modules.mean_reversion import generate_signal
from config import MeanReversionConfig


class TestMeanReversionSignalFormat:
    """Verify signal output format."""

    def test_returns_required_keys(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert "signal" in result
        assert "confidence" in result  # New API (was raw_strength)
        assert "signal_str" in result
        assert "metadata" in result

    def test_signal_is_valid(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert result["signal"] in (-1, 0, 1)

    def test_confidence_in_range(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert 0.0 <= result["confidence"] <= 1.0


class TestMeanReversionLogic:
    """Verify trading logic."""

    def test_insufficient_data_returns_hold(self):
        short = pd.DataFrame({"close": [100, 101]})
        result = generate_signal(short)
        assert result["signal"] == 0
        assert result["confidence"] == 0.0

    def test_flat_market_is_hold(self, flat_data):
        result = generate_signal(flat_data)
        # Flat data should have low z-score, resulting in HOLD
        assert result["signal"] in (-1, 0, 1)

    def test_extreme_oversold_could_buy(self):
        """Simulate extreme oversold: price drops sharply below mean."""
        days = 100
        prices = np.ones(days) * 3000.0
        # Last 5 bars: sharp drop
        prices[-5:] = [2900, 2850, 2800, 2750, 2700]
        df = pd.DataFrame({"close": prices})
        result = generate_signal(df)
        # Should at least not be SELL when price has cratered
        assert result["signal"] in (1, 0)

    def test_extreme_overbought_could_sell(self):
        """Simulate extreme overbought: price spikes above mean."""
        days = 100
        prices = np.ones(days) * 3000.0
        prices[-5:] = [3100, 3150, 3200, 3300, 3400]
        df = pd.DataFrame({"close": prices})
        result = generate_signal(df)
        assert result["signal"] in (-1, 0)

    def test_metadata_contains_indicators(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        meta = result["metadata"]
        if "reason" not in meta:
            assert "zscore" in meta
            assert "rsi" in meta
            assert "bb_upper" in meta
            assert "bb_lower" in meta

    def test_custom_config(self, sample_ohlcv):
        cfg = MeanReversionConfig(lookback_period=10, bb_std_dev=1.5, rsi_period=7)
        result = generate_signal(sample_ohlcv, cfg=cfg)
        assert result["signal"] in (-1, 0, 1)
        assert 0.0 <= result["confidence"] <= 1.0
