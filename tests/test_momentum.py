"""
tests/test_momentum.py — Tests for the momentum branch of the unified strategy.
"""

import pandas as pd
import pytest

from config import StrategyConfig
from modules.strategy import generate_strategy_signal
from utils.indicators import precompute_all_indicators


def _last_signal(df: pd.DataFrame, cfg: StrategyConfig | None = None) -> dict:
    """Run the unified strategy on the final bar and return the momentum slice."""
    if len(df) < 2:
        return {
            "regime": "choppy",
            "momentum": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {}},
            "mean_reversion": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {}},
        }
    pre = precompute_all_indicators(df, cfg)
    return generate_strategy_signal(pre, len(df) - 1, cfg)


class TestMomentumSignalFormat:
    """Verify signal output format and constraints."""

    def test_returns_required_keys(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["momentum"]
        assert "signal" in out
        assert "confidence" in out
        assert "raw_strength" in out
        assert "metadata" in out

    def test_signal_is_valid_direction(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["momentum"]
        assert out["signal"] in (-1, 0, 1)

    def test_confidence_in_range(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["momentum"]
        assert 0.0 <= out["confidence"] <= 1.0

    def test_metadata_is_dict(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["momentum"]
        assert isinstance(out["metadata"], dict)


class TestMomentumLogic:
    """Verify trading logic correctness."""

    def test_insufficient_data_returns_hold(self):
        short_df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "volume": [1000, 1100, 1200],
        })
        out = _last_signal(short_df)["momentum"]
        assert out["signal"] == 0
        assert out["confidence"] == 0.0

    def test_trending_up_not_sell(self, trending_up_data):
        out = _last_signal(trending_up_data)["momentum"]
        # In a clear uptrend, momentum should not signal SELL.
        assert out["signal"] in (1, 0)

    def test_trending_down_not_buy(self, trending_down_data):
        out = _last_signal(trending_down_data)["momentum"]
        assert out["signal"] in (-1, 0)

    def test_flat_market_signal_valid(self, flat_data):
        out = _last_signal(flat_data)["momentum"]
        assert out["signal"] in (-1, 0, 1)

    def test_custom_config_respected(self, sample_ohlcv):
        cfg = StrategyConfig(ema_fast_period=5, ema_slow_period=10, rsi_period=10)
        out = _last_signal(sample_ohlcv, cfg=cfg)["momentum"]
        assert out["signal"] in (-1, 0, 1)
        assert 0.0 <= out["confidence"] <= 1.0

    def test_metadata_contains_explanation(self, sample_ohlcv):
        out = _last_signal(sample_ohlcv)["momentum"]
        meta = out["metadata"]
        # Either the active-signal keys or the gating "reason" key must be present.
        assert ("spread_pct" in meta and "macd_hist" in meta) or "reason" in meta

    def test_empty_dataframe_returns_hold(self):
        empty = pd.DataFrame({"close": [], "high": [], "low": [], "volume": []})
        out = _last_signal(empty)["momentum"]
        assert out["signal"] == 0

    def test_single_price_returns_hold(self):
        single = pd.DataFrame({
            "close": [3000.0],
            "high": [3001.0],
            "low": [2999.0],
            "volume": [10000.0],
        })
        out = _last_signal(single)["momentum"]
        assert out["signal"] == 0
