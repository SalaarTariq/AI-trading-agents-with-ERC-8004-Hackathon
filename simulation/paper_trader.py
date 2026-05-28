"""Paper-trading simulation loop for the ERC-8004 hybrid agent.

Extracted from main.py so the orchestrator stays a thin CLI and the
trade-loop is independently testable. Behavior matches the previous
in-line implementation; only the structure has changed:

- `Position` and `TradeRecord` replace ad-hoc dicts.
- `PaperTrader` owns one full backtest run; `run_paper_trading` is a
  convenience wrapper around it that preserves the prior return shape.
- The required warmup is derived from `StrategyConfig` instead of a
  hard-coded 50.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from config import AppConfig, CONFIG, RiskConfig, StrategyConfig
from modules.ai_predictor import predict_from_indicators
from modules.confidence_scoring import combine_signals
from modules.strategy import generate_strategy_signal
from risk.risk_manager import (
    PortfolioState,
    RiskResult,
    check_risk,
    compute_trailing_stop,
    update_after_trade,
)
from utils.indicators import indicators_at, precompute_all_indicators
from validation.proof_logger import log_decision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """An open paper-trading position."""

    action: str  # "BUY" or "SELL"
    pair: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    timestamp: str
    proof_hash: str

    def update_trailing_stop(self, current_price: float, cfg: RiskConfig) -> None:
        """Tighten the stop in place using the risk manager rule."""
        self.stop_loss = compute_trailing_stop(
            action=self.action,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            current_price=current_price,
            cfg=cfg,
        )

    def check_exit(self, high: float, low: float) -> tuple[bool, float]:
        """Return (closed, pnl) using SL/TP touches against the bar range."""
        entry = self.entry_price
        if self.action == "BUY":
            if low <= self.stop_loss:
                return True, (self.stop_loss - entry) / entry * self.size
            if high >= self.take_profit:
                return True, (self.take_profit - entry) / entry * self.size
        elif self.action == "SELL":
            if high >= self.stop_loss:
                return True, (entry - self.stop_loss) / entry * self.size
            if low <= self.take_profit:
                return True, (entry - self.take_profit) / entry * self.size
        return False, 0.0

    def mark_to_market(self, exit_price: float) -> float:
        """PnL if closed at exit_price (no SL/TP check)."""
        entry = self.entry_price
        if self.action == "BUY":
            return (exit_price - entry) / entry * self.size
        return (entry - exit_price) / entry * self.size


@dataclass
class TradeRecord:
    """A closed trade ready to be appended to the trade history log."""

    action: str
    pair: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    timestamp: str
    proof_hash: str
    exit_price: float
    exit_timestamp: str
    pnl: float
    exit_reason: str
    dataset: str

    @classmethod
    def from_position(
        cls,
        position: Position,
        *,
        exit_price: float,
        exit_timestamp: str,
        pnl: float,
        exit_reason: str,
        dataset: str,
    ) -> "TradeRecord":
        return cls(
            action=position.action,
            pair=position.pair,
            entry_price=round(position.entry_price, 8),
            size=round(position.size, 2),
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            timestamp=position.timestamp,
            proof_hash=position.proof_hash,
            exit_price=round(exit_price, 8),
            exit_timestamp=exit_timestamp,
            pnl=round(pnl, 2),
            exit_reason=exit_reason,
            dataset=dataset,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def required_warmup_bars(cfg: StrategyConfig | None = None) -> int:
    """Minimum bars before per-bar indicators are populated.

    Driven by the longest *required* lookback: MACD (slow + signal = 35) plus
    a small buffer. The 120-bar ATR quantile uses ``min_periods=30`` so it is
    available well before bar 120; trades that need it are gated through
    ``indicators_at`` returning None, not through the warmup count.
    """
    cfg = cfg or CONFIG.strategy
    longest_indicator = max(
        cfg.ema_slow_period,
        cfg.rsi_period,
        cfg.macd_slow + cfg.macd_signal,
        cfg.atr_period,
        cfg.bb_period,
        cfg.adx_period,
    )
    return max(50, longest_indicator + 5)


# ---------------------------------------------------------------------------
# Paper trader
# ---------------------------------------------------------------------------


class PaperTrader:
    """One full backtest run over a precomputed indicator frame."""

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: AppConfig | None = None,
        *,
        dataset_label: str = "",
        trade_log_path: Path | str | None = None,
        warmup: int | None = None,
    ) -> None:
        self.df = df
        self.cfg = cfg or CONFIG
        self.dataset_label = dataset_label
        self.trade_log_path = Path(trade_log_path or self.cfg.trade_log_path)
        self.warmup = warmup if warmup is not None else required_warmup_bars(self.cfg.strategy)

        self.pre = precompute_all_indicators(df, self.cfg.strategy)
        self.portfolio = PortfolioState(
            total_value=self.cfg.portfolio.initial_balance,
            cash=self.cfg.portfolio.initial_balance,
            peak_value=self.cfg.portfolio.initial_balance,
        )
        self.pair = (
            self.cfg.portfolio.trading_pairs[0]
            if self.cfg.portfolio.trading_pairs
            else "ETH/USDC"
        )

        self.open_position: Position | None = None
        self.trades: list[TradeRecord] = []
        self.proof_hashes: list[str] = []
        self._last_date: str = ""

    # --- main loop -------------------------------------------------------

    def run(self) -> dict:
        for idx in range(self.warmup, len(self.df)):
            self._step(idx)
        self._close_at_end()
        return self.summary()

    def _step(self, idx: int) -> None:
        row = self.df.iloc[idx]
        timestamp = str(self.df.index[idx])
        price = float(row["close"])

        self._daily_reset(timestamp)

        if self.open_position is not None:
            self._manage_open_position(
                price=price,
                high=float(row["high"]),
                low=float(row["low"]),
                timestamp=timestamp,
            )
            if self.open_position is not None:
                return  # still in the trade — no new entries

        if self.portfolio.cooldown_bars > 0:
            self.portfolio.cooldown_bars -= 1

        self._try_open_position(idx=idx, price=price, timestamp=timestamp)

    # --- per-bar helpers -------------------------------------------------

    def _daily_reset(self, timestamp: str) -> None:
        today = timestamp[:10]
        if today != self._last_date:
            self.portfolio.daily_pnl = 0.0
            self._last_date = today

    def _manage_open_position(
        self,
        *,
        price: float,
        high: float,
        low: float,
        timestamp: str,
    ) -> None:
        assert self.open_position is not None
        self.open_position.update_trailing_stop(price, self.cfg.risk)
        closed, pnl = self.open_position.check_exit(high=high, low=low)
        if not closed:
            return

        hit_sl = pnl < 0
        self.portfolio.cash += self.open_position.size + pnl
        self.portfolio = update_after_trade(self.portfolio, pnl, hit_stop_loss=hit_sl)

        trade = TradeRecord.from_position(
            self.open_position,
            exit_price=price,
            exit_timestamp=timestamp,
            pnl=pnl,
            exit_reason="stop_loss" if hit_sl else "take_profit",
            dataset=self.dataset_label,
        )
        self.trades.append(trade)
        self._append_trade_log(trade)
        self.open_position = None

    def _try_open_position(self, *, idx: int, price: float, timestamp: str) -> None:
        ind = indicators_at(self.pre, idx)
        if ind is None:
            return

        strategy_out = generate_strategy_signal(self.pre, idx, self.cfg.strategy)
        ai_out = (
            predict_from_indicators(strategy_out, ind, price)
            if self.cfg.signal.use_ai_predictor
            else None
        )
        combined = combine_signals(
            strategy_out,
            current_atr_norm=ind.atr_norm_14,
            cfg=self.cfg.signal,
            regime_cfg=self.cfg.regime,
            ai_out=ai_out,
        )

        action = combined["action"]
        regime = combined["regime"]

        if action in ("BUY", "SELL"):
            risk = check_risk(
                signal=action,
                confidence=float(combined["confidence"]),
                entry_price=price,
                requested_size=self.portfolio.cash * self.cfg.risk.max_capital_pct,
                portfolio=self.portfolio,
                cfg=self.cfg.risk,
                regime=regime,
                regime_cfg=self.cfg.regime,
                pre_ind=ind,
            )
        else:
            risk = RiskResult(
                approved=False,
                reasons=["Signal is HOLD"],
                adjusted_size=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
            )

        decision_record = self._build_decision_record(
            timestamp=timestamp,
            price=price,
            strategy_out=strategy_out,
            combined=combined,
            indicators=ind,
            risk=risk,
            action=action,
        )

        if not (risk.approved and action in ("BUY", "SELL")):
            return

        proof_hash = log_decision(decision_record)
        self.proof_hashes.append(proof_hash)

        size = float(risk.adjusted_size)
        self.portfolio.cash -= size
        self.open_position = Position(
            action=action,
            pair=self.pair,
            entry_price=round(price, 8),
            size=round(size, 2),
            stop_loss=risk.stop_loss_price,
            take_profit=risk.take_profit_price,
            timestamp=timestamp,
            proof_hash=proof_hash,
        )

    def _build_decision_record(
        self,
        *,
        timestamp: str,
        price: float,
        strategy_out: dict,
        combined: dict,
        indicators,
        risk: RiskResult,
        action: str,
    ) -> dict:
        ind = indicators
        return {
            "timestamp": timestamp,
            "pair": self.pair,
            "current_price": round(price, 8),
            "strategy_signals": strategy_out,
            "combined_decision": combined,
            "indicators": {
                "ema_spread": ind.ema_spread,
                "ema_spread_pct": ind.ema_spread_pct,
                "rsi": ind.rsi_14,
                "macd_hist": ind.macd_hist,
                "atr": ind.atr_14,
                "atr_norm": ind.atr_norm_14,
                "bb_zscore": ind.bb_zscore,
                "adx": ind.adx_14,
            },
            "risk_result": {
                "approved": risk.approved,
                "reasons": risk.reasons,
                "warnings": risk.warnings,
                "adjusted_size": risk.adjusted_size,
                "stop_loss": risk.stop_loss_price,
                "take_profit": risk.take_profit_price,
            },
            "final_decision": {
                "action": action if risk.approved else "HOLD",
                "pair": self.pair,
                "entry_price": round(price, 8),
                "size": risk.adjusted_size if risk.approved else 0.0,
            },
            "portfolio_state": {
                "total_value": round(self.portfolio.total_value, 2),
                "cash": round(self.portfolio.cash, 2),
                "daily_pnl": round(self.portfolio.daily_pnl, 2),
            },
            "dataset": self.dataset_label,
        }

    # --- end-of-run handling ---------------------------------------------

    def _close_at_end(self) -> None:
        if self.open_position is None:
            return

        final_price = float(self.df["close"].iloc[-1])
        pnl = self.open_position.mark_to_market(final_price)

        self.portfolio.cash += self.open_position.size + pnl
        self.portfolio = update_after_trade(self.portfolio, pnl, hit_stop_loss=pnl < 0)

        trade = TradeRecord.from_position(
            self.open_position,
            exit_price=final_price,
            exit_timestamp=str(self.df.index[-1]),
            pnl=pnl,
            exit_reason="end_of_simulation",
            dataset=self.dataset_label,
        )
        self.trades.append(trade)
        self._append_trade_log(trade)
        self.open_position = None

    # --- IO --------------------------------------------------------------

    def _append_trade_log(self, trade: TradeRecord) -> None:
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trade_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trade.to_dict(), default=str) + "\n")

    # --- summary ---------------------------------------------------------

    def summary(self) -> dict:
        total_pnl = self.portfolio.total_value - self.cfg.portfolio.initial_balance
        wins = sum(1 for t in self.trades if t.pnl > 0)
        n = len(self.trades)
        return {
            "initial_balance": self.cfg.portfolio.initial_balance,
            "final_value": round(self.portfolio.total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(total_pnl / self.cfg.portfolio.initial_balance * 100.0, 2),
            "total_trades": n,
            "winning_trades": wins,
            "losing_trades": n - wins,
            "win_rate": round(wins / n * 100.0, 2) if n else 0.0,
            "proof_hashes_generated": len(self.proof_hashes),
            "trades": [t.to_dict() for t in self.trades],
        }


# ---------------------------------------------------------------------------
# Backwards-compatible convenience wrapper
# ---------------------------------------------------------------------------


def run_paper_trading(
    df: pd.DataFrame,
    cfg: AppConfig | None = None,
    warmup: int | None = None,
    dataset_label: str = "",
    trade_log_path: Path | str | None = None,
) -> dict:
    """Run full paper-trading simulation on OHLCV data.

    Wrapper around `PaperTrader.run` kept so existing imports
    (`from main import run_paper_trading`) continue to work.
    """
    trader = PaperTrader(
        df=df,
        cfg=cfg,
        dataset_label=dataset_label,
        trade_log_path=trade_log_path,
        warmup=warmup,
    )
    return trader.run()
