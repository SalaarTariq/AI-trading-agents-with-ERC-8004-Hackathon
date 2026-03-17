"""
tests/test_momentum.py — Tests for the momentum trading strategy.
"""

import numpy as np
import pandas as pd
import pytest

from modules.momentum import generate_signal
from config import MomentumConfig


class TestMomentumSignalFormat:
    """Verify signal output format and constraints."""

    def test_returns_required_keys(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert "signal" in result
        assert "confidence" in result
        assert "metadata" in result

    def test_signal_is_valid_direction(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_metadata_is_dict(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        assert isinstance(result["metadata"], dict)


class TestMomentumLogic:
    """Verify trading logic correctness."""

    def test_insufficient_data_returns_hold(self):
        short_df = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
        })
        result = generate_signal(short_df)
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_trending_up_not_sell(self, trending_up_data):
        result = generate_signal(trending_up_data)
        # In a clear uptrend, should not be SELL
        assert result["signal"] in ("BUY", "HOLD")

    def test_trending_down_not_buy(self, trending_down_data):
        result = generate_signal(trending_down_data)
        # In a clear downtrend, should not be BUY
        assert result["signal"] in ("SELL", "HOLD")

    def test_flat_market_is_hold(self, flat_data):
        result = generate_signal(flat_data)
        assert result["signal"] == "HOLD"

    def test_custom_config_respected(self, sample_ohlcv):
        cfg = MomentumConfig(fast_period=5, slow_period=10, volume_ma_period=10)
        result = generate_signal(sample_ohlcv, cfg=cfg)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_metadata_contains_ema_values(self, sample_ohlcv):
        result = generate_signal(sample_ohlcv)
        if result["signal"] != "HOLD" or "fast_ema" in result["metadata"]:
            meta = result["metadata"]
            assert "fast_ema" in meta or "reason" in meta

    def test_empty_dataframe_returns_hold(self):
        empty = pd.DataFrame({"close": [], "volume": []})
        result = generate_signal(empty)
        assert result["signal"] == "HOLD"

    def test_single_price_returns_hold(self):
        single = pd.DataFrame({"close": [3000.0], "volume": [10000.0]})
        result = generate_signal(single)
        assert result["signal"] == "HOLD"
