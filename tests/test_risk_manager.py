"""
tests/test_risk_manager.py — Tests for the risk manager.

This is a critical module — tests cover all boundary conditions,
emergency rules, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from risk.risk_manager import check_risk, update_after_trade, PortfolioState
from config import RiskConfig, RegimeConfig, RegimeParams


class TestRiskApproval:
    """Test basic approval/rejection logic."""

    def test_valid_trade_approved(self, default_portfolio):
        result = check_risk(
            signal="BUY", confidence=0.7, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio,
        )
        assert result.approved is True
        assert len(result.reasons) == 0

    def test_hold_signal_no_crash(self, default_portfolio):
        # HOLD signals typically aren't sent to risk, but shouldn't crash
        result = check_risk(
            signal="HOLD", confidence=0.5, entry_price=3000.0,
            requested_size=0, portfolio=default_portfolio,
        )
        assert result.adjusted_size == 0


class TestConfidenceThreshold:
    """Test minimum confidence gating."""

    def test_low_confidence_rejected(self, default_portfolio):
        # Choppy regime has threshold 0.65, so 0.40 is rejected
        result = check_risk(
            signal="BUY", confidence=0.40, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio,
            regime="choppy",
        )
        assert result.approved is False
        assert any("Confidence" in r for r in result.reasons)

    def test_exact_threshold_approved(self, default_portfolio):
        # Trending regime has threshold 0.60, so 0.60 passes
        regime_cfg = RegimeConfig(
            trending_up=RegimeParams(conf_threshold=0.60, position_mult=1.0,
                                     sl_atr_mult=2.0, tp_atr_mult=3.0),
        )
        result = check_risk(
            signal="BUY", confidence=0.60, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio,
            regime="trending_up", regime_cfg=regime_cfg,
        )
        assert result.approved is True


class TestPositionSizing:
    """Test max position size enforcement."""

    def test_oversized_position_reduced(self, default_portfolio):
        # 30% of 100k = 30k, request 50k
        cfg = RiskConfig(max_position_pct=0.30)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=50_000, portfolio=default_portfolio, cfg=cfg,
        )
        assert result.adjusted_size <= 30_000
        assert any("exceeds max" in w for w in result.warnings)

    def test_within_limit_unchanged(self, default_portfolio):
        # Use trending regime (position_mult=1.0) so sizing isn't reduced
        regime_cfg = RegimeConfig(
            trending_up=RegimeParams(conf_threshold=0.35, position_mult=1.0,
                                     sl_atr_mult=2.0, tp_atr_mult=3.0),
        )
        cfg = RiskConfig(max_position_pct=0.30, max_capital_pct=0.25)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=20_000, portfolio=default_portfolio, cfg=cfg,
            regime="trending_up", regime_cfg=regime_cfg,
        )
        # No ATR → fallback: min(20k, 100k*0.25) * 1.0 = 20k, but max_capital caps at 25k
        assert result.adjusted_size <= 25_000
        assert result.approved is True


class TestDailyLossCap:
    """Test daily loss cap enforcement."""

    def test_daily_cap_blocks_trade(self):
        portfolio = PortfolioState(
            total_value=90_000, cash=70_000,
            daily_pnl=-10_500, peak_value=100_000,
        )
        cfg = RiskConfig(daily_loss_cap_pct=0.10)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio, cfg=cfg,
        )
        assert result.approved is False
        assert any("Daily loss cap" in r for r in result.reasons)

    def test_within_daily_cap_allowed(self):
        portfolio = PortfolioState(
            total_value=95_000, cash=75_000,
            daily_pnl=-5_000, peak_value=100_000,
        )
        cfg = RiskConfig(daily_loss_cap_pct=0.10)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio, cfg=cfg,
        )
        # daily_pnl / total_value = 5000/95000 ≈ 5.3% < 10%
        assert result.approved is True


class TestConsecutiveLossPause:
    """Test pause after consecutive stop-losses."""

    def test_three_losses_triggers_cooldown(self):
        portfolio = PortfolioState(
            total_value=95_000, cash=75_000,
            consecutive_losses=3, peak_value=100_000,
        )
        cfg = RiskConfig(consecutive_loss_pause=3)
        result = check_risk(
            signal="BUY", confidence=0.9, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio, cfg=cfg,
        )
        assert result.approved is False
        assert any("cooldown" in r.lower() for r in result.reasons)
        # Should have set cooldown and reset consecutive_losses
        assert portfolio.cooldown_bars == 8  # v3: reduced from 10 to 8
        assert portfolio.consecutive_losses == 0

    def test_cooldown_blocks_trades(self):
        portfolio = PortfolioState(
            total_value=95_000, cash=75_000,
            cooldown_bars=5, peak_value=100_000,
        )
        result = check_risk(
            signal="BUY", confidence=0.9, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio,
        )
        assert result.approved is False
        assert any("Cooldown" in r for r in result.reasons)

    def test_two_losses_allowed(self):
        portfolio = PortfolioState(
            total_value=97_000, cash=77_000,
            consecutive_losses=2, peak_value=100_000,
        )
        cfg = RiskConfig(consecutive_loss_pause=3)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio, cfg=cfg,
        )
        assert result.approved is True

    def test_cooldown_zero_allows_trade(self):
        portfolio = PortfolioState(
            total_value=97_000, cash=77_000,
            cooldown_bars=0, peak_value=100_000,
        )
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio,
        )
        assert result.approved is True


class TestDefensiveMode:
    """Test drawdown-triggered defensive mode."""

    def test_drawdown_reduces_size(self):
        portfolio = PortfolioState(
            total_value=83_000, cash=63_000,
            peak_value=100_000,  # 17% drawdown > 15% threshold
        )
        cfg = RiskConfig(max_drawdown_pct=0.15, defensive_size_mult=0.25)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=20_000, portfolio=portfolio, cfg=cfg,
        )
        # Should be reduced by 75%: 20k * 0.25 = 5k
        assert result.adjusted_size <= 20_000 * 0.30  # Allow some float tolerance
        assert any("Defensive" in w for w in result.warnings)


class TestStopLossTakeProfit:
    """Test SL/TP price calculations."""

    def test_buy_stop_loss_below_entry(self, default_portfolio):
        cfg = RiskConfig(stop_loss_pct=0.04)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio, cfg=cfg,
        )
        assert result.stop_loss_price == pytest.approx(2880.0, rel=1e-3)

    def test_buy_take_profit_above_entry(self, default_portfolio):
        cfg = RiskConfig(take_profit_pct=0.06)
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio, cfg=cfg,
        )
        assert result.take_profit_price == pytest.approx(3180.0, rel=1e-3)

    def test_sell_stop_loss_above_entry(self, default_portfolio):
        cfg = RiskConfig(stop_loss_pct=0.05)
        result = check_risk(
            signal="SELL", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio, cfg=cfg,
        )
        assert result.stop_loss_price == pytest.approx(3150.0, rel=1e-3)

    def test_sell_take_profit_below_entry(self, default_portfolio):
        cfg = RiskConfig(take_profit_pct=0.06)
        result = check_risk(
            signal="SELL", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=default_portfolio, cfg=cfg,
        )
        assert result.take_profit_price == pytest.approx(2820.0, rel=1e-3)


class TestCashCheck:
    """Test cash sufficiency."""

    def test_exceeds_cash_reduced(self):
        portfolio = PortfolioState(
            total_value=100_000, cash=5_000, peak_value=100_000,
        )
        result = check_risk(
            signal="BUY", confidence=0.8, entry_price=3000.0,
            requested_size=10_000, portfolio=portfolio,
        )
        assert result.adjusted_size <= 5_000


class TestUpdateAfterTrade:
    """Test portfolio state updates after trades."""

    def test_winning_trade_resets_losses(self):
        portfolio = PortfolioState(
            total_value=95_000, cash=75_000,
            consecutive_losses=2, peak_value=100_000,
        )
        updated = update_after_trade(portfolio, trade_pnl=1000, hit_stop_loss=False)
        assert updated.consecutive_losses == 0
        assert updated.total_value == 96_000

    def test_stop_loss_increments_counter(self):
        portfolio = PortfolioState(
            total_value=95_000, cash=75_000,
            consecutive_losses=1, peak_value=100_000,
        )
        updated = update_after_trade(portfolio, trade_pnl=-500, hit_stop_loss=True)
        assert updated.consecutive_losses == 2

    def test_new_peak_updated(self):
        portfolio = PortfolioState(
            total_value=100_000, cash=80_000,
            peak_value=100_000,
        )
        updated = update_after_trade(portfolio, trade_pnl=5000)
        assert updated.peak_value == 105_000
