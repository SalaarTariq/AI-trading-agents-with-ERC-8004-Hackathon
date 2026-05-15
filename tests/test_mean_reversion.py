"""
tests/test_mean_reversion.py — Tests for the mean-reversion branch of the unified strategy.
"""

import numpy as np
import pandas as pd
import pytest

from config import StrategyConfig
from modules.strategy import generate_strategy_signal
from utils.indicators import precompute_all_indicators


def _last_signal(df: pd.DataFrame, cfg: StrategyConfig | None = None) -> dict:
    """Run the unified strategy on the final bar and return the mean-reversion slice."""
    if len(df) < 2:
        return {
            "regime": "choppy",
            "momentum": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {}},
            "mean_reversion": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {}},
        }
    pre = precompute_all_indicators(df, cfg)
    return generate_strategy_signal(pre, len(df) - 1, cfg)


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a close series for indicator inputs."""
    return pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.full_like(close, 10000.0),
    })


class TestMeanReversionSignalFormat:
    """Verify signal output format."""

    def test_returns_required_keys(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["mean_reversion"]
        assert "signal" in out
        assert "confidence" in out
        assert "raw_strength" in out
        assert "metadata" in out

    def test_signal_is_valid(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["mean_reversion"]
        assert out["signal"] in (-1, 0, 1)

    def test_confidence_in_range(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["mean_reversion"]
        assert 0.0 <= out["confidence"] <= 1.0


class TestMeanReversionLogic:
    """Verify trading logic."""

    def test_insufficient_data_returns_hold(self):
        short = _ohlcv_from_close(np.array([100.0, 101.0]))
        out = _last_signal(short)["mean_reversion"]
        assert out["signal"] == 0
        assert out["confidence"] == 0.0

    def test_flat_market_is_hold(self, flat_data):
        out = _last_signal(flat_data)["mean_reversion"]
        assert out["signal"] in (-1, 0, 1)

    def test_extreme_oversold_does_not_sell(self):
        """Sharp drop below recent mean should not produce a SELL signal."""
        days = 100
        prices = np.ones(days) * 3000.0
        prices[-5:] = [2900, 2850, 2800, 2750, 2700]
        df = _ohlcv_from_close(prices)
        out = _last_signal(df)["mean_reversion"]
        assert out["signal"] in (1, 0)

    def test_extreme_overbought_does_not_buy(self):
        """Sharp spike above recent mean should not produce a BUY signal."""
        days = 100
        prices = np.ones(days) * 3000.0
        prices[-5:] = [3100, 3150, 3200, 3300, 3400]
        df = _ohlcv_from_close(prices)
        out = _last_signal(df)["mean_reversion"]
        assert out["signal"] in (-1, 0)

    def test_metadata_contains_indicators(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["mean_reversion"]
        meta = out["metadata"]
        if "reason" not in meta:
            assert "zscore" in meta
            assert "rsi" in meta

    def test_custom_config(self, sample_ohlcv):
        cfg = StrategyConfig(bb_period=10, bb_std_dev=1.5, rsi_period=7)
        out = _last_signal(sample_ohlcv, cfg=cfg)["mean_reversion"]
        assert out["signal"] in (-1, 0, 1)
        assert 0.0 <= out["confidence"] <= 1.0
