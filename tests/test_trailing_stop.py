"""Tests for risk_manager.check_trailing_stop and Position.update_trailing_stop."""

from __future__ import annotations

import pytest

from config import RiskConfig
from risk.risk_manager import check_trailing_stop
from simulation.paper_trader import Position


def _buy_position(entry: float = 100.0, sl: float = 96.0, tp: float = 110.0) -> dict:
    return {
        "action": "BUY",
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
    }


def _sell_position(entry: float = 100.0, sl: float = 104.0, tp: float = 90.0) -> dict:
    return {
        "action": "SELL",
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
    }


class TestBuyTrailingStop:
    def test_no_progress_keeps_stop(self):
        cfg = RiskConfig(use_trailing_stop=True)
        pos = _buy_position()
        updated = check_trailing_stop(pos, current_price=101.0, cfg=cfg)
        # Only 10% of the way to TP — below the 60% breakeven trigger.
        assert updated["stop_loss"] == 96.0

    def test_activates_at_breakeven_threshold(self):
        cfg = RiskConfig(
            use_trailing_stop=True,
            trailing_breakeven_pct=0.60,
            trailing_lock_pct=0.70,
        )
        pos = _buy_position()
        # 60% of (110 - 100) = 6 → current_price = 106.
        updated = check_trailing_stop(pos, current_price=106.0, cfg=cfg)
        # Locks 70% of realized move: entry + 6 * 0.7 = 104.2.
        assert updated["stop_loss"] == pytest.approx(104.2, rel=1e-3)

    def test_never_loosens_stop(self):
        cfg = RiskConfig(use_trailing_stop=True)
        # Stop already tight (above breakeven from a previous tightening).
        pos = _buy_position(sl=105.0)
        updated = check_trailing_stop(pos, current_price=106.0, cfg=cfg)
        # New computed SL = 104.2, but existing is 105 — must keep existing.
        assert updated["stop_loss"] == 105.0

    def test_disabled_is_no_op(self):
        cfg = RiskConfig(use_trailing_stop=False)
        pos = _buy_position()
        updated = check_trailing_stop(pos, current_price=109.0, cfg=cfg)
        assert updated["stop_loss"] == 96.0


class TestSellTrailingStop:
    def test_activates_for_short(self):
        cfg = RiskConfig(
            use_trailing_stop=True,
            trailing_breakeven_pct=0.60,
            trailing_lock_pct=0.70,
        )
        pos = _sell_position()
        # 60% of (100 - 90) = 6 → current_price = 94.
        updated = check_trailing_stop(pos, current_price=94.0, cfg=cfg)
        # Locks 70% from entry downward: 100 - 6 * 0.7 = 95.8.
        assert updated["stop_loss"] == pytest.approx(95.8, rel=1e-3)

    def test_never_loosens_short_stop(self):
        cfg = RiskConfig(use_trailing_stop=True)
        pos = _sell_position(sl=95.0)
        updated = check_trailing_stop(pos, current_price=94.0, cfg=cfg)
        assert updated["stop_loss"] == 95.0


class TestPositionDataclassTrailing:
    """The typed Position wrapper must apply the same rule as the dict path."""

    def test_buy_position_updates_stop(self):
        cfg = RiskConfig(use_trailing_stop=True)
        pos = Position(
            action="BUY",
            pair="ETH/USDC",
            entry_price=100.0,
            size=5_000.0,
            stop_loss=96.0,
            take_profit=110.0,
            timestamp="2025-01-01T00:00:00",
            proof_hash="x",
        )
        pos.update_trailing_stop(current_price=108.0, cfg=cfg)
        assert pos.stop_loss > 96.0  # tightened
        assert pos.stop_loss <= 108.0  # but not above current price
