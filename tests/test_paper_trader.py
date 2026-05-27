"""Tests for the simulation.paper_trader loop.

Covers:
- ``required_warmup_bars`` derivation from StrategyConfig.
- End-of-simulation mark-to-market closing of an open position.
- The summary dict shape stays stable for the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig, StrategyConfig
from simulation.paper_trader import (
    PaperTrader,
    Position,
    TradeRecord,
    required_warmup_bars,
    run_paper_trading,
)


def _make_trending_ohlcv(n: int = 250, seed: int = 1) -> pd.DataFrame:
    """Strong uptrend so the agent has a reason to enter a long."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 0.30, n)  # 30% over the run
    noise = rng.normal(0, 0.002, n)
    log_returns = drift / n + noise
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(1_000, 5_000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestWarmup:
    def test_default_warmup_floor_is_50(self):
        # Default StrategyConfig values keep the longest lookback at 35
        # (MACD slow+signal); the function should still return at least 50.
        assert required_warmup_bars() >= 50

    def test_warmup_scales_with_longer_indicators(self):
        cfg = StrategyConfig(macd_slow=200, macd_signal=20)
        assert required_warmup_bars(cfg) >= 220


class TestSummaryShape:
    def test_summary_keys_stable(self, tmp_path):
        df = _make_trending_ohlcv(250)
        out = run_paper_trading(
            df,
            dataset_label="trend",
            trade_log_path=tmp_path / "trades.jsonl",
        )
        for key in (
            "initial_balance",
            "final_value",
            "total_pnl",
            "pnl_pct",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "proof_hashes_generated",
            "trades",
        ):
            assert key in out

    def test_trade_records_serializable(self, tmp_path):
        df = _make_trending_ohlcv(200)
        out = run_paper_trading(
            df,
            dataset_label="trend",
            trade_log_path=tmp_path / "trades.jsonl",
        )
        # Each trade must be a plain dict (asdict result) — no dataclasses or
        # numpy types — so the dashboard's json.loads can read it back.
        for trade in out["trades"]:
            assert isinstance(trade, dict)
            assert isinstance(trade["pnl"], (int, float))
            assert trade["dataset"] == "trend"


class TestEndOfSimulationClose:
    """A position still open on the final bar must be closed via mark-to-market."""

    def test_open_position_closed_at_end(self, tmp_path):
        cfg = AppConfig()
        # Loose enough thresholds that an entry will fire on a clean trend.
        cfg.signal.execute_confidence_threshold = 0.50
        cfg.risk.min_confidence = 0.50
        cfg.regime.trending_up.conf_threshold = 0.50
        cfg.regime.trending_down.conf_threshold = 0.50
        cfg.regime.ranging.conf_threshold = 0.50
        cfg.regime.choppy.conf_threshold = 0.50

        df = _make_trending_ohlcv(300, seed=3)

        trader = PaperTrader(
            df=df,
            cfg=cfg,
            dataset_label="eos",
            trade_log_path=tmp_path / "trades.jsonl",
        )
        # Manually open a position near the end so it cannot hit SL or TP
        # before the final bar — forcing the end-of-sim branch.
        last_idx = len(df) - 2
        last_price = float(df["close"].iloc[last_idx])
        trader.open_position = Position(
            action="BUY",
            pair="ETH/USDC",
            entry_price=last_price,
            size=1_000.0,
            stop_loss=last_price * 0.5,  # impossible to hit
            take_profit=last_price * 1.5,  # impossible to hit
            timestamp=str(df.index[last_idx]),
            proof_hash="manual-test-hash",
        )
        trader.portfolio.cash -= 1_000.0

        # Run the close-at-end branch directly.
        trader._close_at_end()

        assert trader.open_position is None
        assert len(trader.trades) == 1
        closed = trader.trades[0]
        assert closed.exit_reason == "end_of_simulation"
        assert closed.exit_timestamp == str(df.index[-1])
