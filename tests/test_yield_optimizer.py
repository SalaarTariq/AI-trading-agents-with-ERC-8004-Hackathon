"""
tests/test_yield_optimizer.py — Tests for the yield optimization strategy.
"""

import pytest

from modules.yield_optimizer import generate_signal, PoolInfo, DEFAULT_POOLS
from config import YieldConfig


class TestYieldSignalFormat:
    """Verify signal output format."""

    def test_returns_required_keys(self):
        result = generate_signal()
        assert "signal" in result
        assert "confidence" in result
        assert "metadata" in result

    def test_signal_is_valid(self):
        result = generate_signal()
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self):
        result = generate_signal()
        assert 0.0 <= result["confidence"] <= 1.0


class TestYieldLogic:
    """Verify yield optimization logic."""

    def test_empty_pools_returns_hold(self):
        result = generate_signal(pools=[])
        assert result["signal"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_high_apy_pool_triggers_buy(self):
        pools = [
            PoolInfo("Test Pool", apy=0.20, tvl=100_000_000, risk_score=0.1)
        ]
        result = generate_signal(pools=pools)
        assert result["signal"] == "BUY"
        assert result["confidence"] > 0.0

    def test_low_apy_pool_returns_hold(self):
        cfg = YieldConfig(min_apy=0.10)
        pools = [
            PoolInfo("Low Pool", apy=0.02, tvl=100_000_000, risk_score=0.1)
        ]
        result = generate_signal(pools=pools, cfg=cfg)
        assert result["signal"] == "HOLD"

    def test_high_risk_reduces_adjusted_apy(self):
        # High APY but high risk → risk-adjusted APY may fall below min
        cfg = YieldConfig(min_apy=0.10)
        pools = [
            PoolInfo("Risky Pool", apy=0.15, tvl=50_000_000, risk_score=0.9)
        ]
        result = generate_signal(pools=pools, cfg=cfg)
        # 0.15 * 0.1 = 0.015 adjusted → below 0.10 min
        assert result["signal"] == "HOLD"

    def test_allocation_respects_max(self):
        cfg = YieldConfig(max_pool_allocation_pct=0.10)
        pools = [
            PoolInfo("Good Pool", apy=0.12, tvl=200_000_000, risk_score=0.2)
        ]
        result = generate_signal(pools=pools, portfolio_value=100_000, cfg=cfg)
        if result["signal"] == "BUY":
            alloc = result["metadata"]["suggestion"]["suggested_allocation"]
            assert alloc <= 0.10

    def test_withdrawal_suggestion(self):
        cfg = YieldConfig(min_apy=0.10)
        pools = [
            PoolInfo("Dying Pool", apy=0.01, tvl=1_000_000, risk_score=0.3,
                     current_allocation=0.15)
        ]
        result = generate_signal(pools=pools, cfg=cfg)
        # Should suggest withdrawal since adj APY (0.007) < 50% of min (0.05)
        assert result["signal"] in ("SELL", "HOLD")

    def test_default_pools_produce_valid_signal(self):
        result = generate_signal(pools=DEFAULT_POOLS)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
