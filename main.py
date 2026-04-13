"""Clean main entry point for the ERC-8004 hybrid trading agent.

Pipeline kept intentionally simple:
1) Load data
2) Precompute indicators once
3) Generate merged strategy signal
4) Compute confidence/action
5) Run risk check
6) Execute paper trade and log validation proof
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from config import AppConfig, CONFIG
from modules.confidence_scoring import combine_signals
from modules.strategy import generate_strategy_signal
from risk.risk_manager import (
    PortfolioState,
    RiskResult,
    check_risk,
    check_trailing_stop,
    update_after_trade,
)
from utils.data_loader import load_or_generate
from utils.helpers import setup_logging
from utils.indicators import indicators_at, precompute_all_indicators
from validation.proof_logger import log_decision

logger = logging.getLogger(__name__)
TRADE_LOG = Path("data/trade_history.jsonl")


def _check_exit(position: dict, high: float, low: float) -> tuple[bool, float]:
    """Close on stop-loss or take-profit for both long and short positions."""
    entry = float(position["entry_price"])
    sl = float(position["stop_loss"])
    tp = float(position["take_profit"])
    size = float(position["size"])
    action = str(position["action"])

    if action == "BUY":
        if low <= sl:
            return True, (sl - entry) / entry * size
        if high >= tp:
            return True, (tp - entry) / entry * size
    elif action == "SELL":
        if high >= sl:
            return True, (entry - sl) / entry * size
        if low <= tp:
            return True, (entry - tp) / entry * size

    return False, 0.0


def _append_trade_log(record: dict) -> None:
    """Persist trade records for dashboard/reporting."""
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRADE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def run_paper_trading(
    df: pd.DataFrame,
    cfg: AppConfig | None = None,
    warmup: int = 50,
    dataset_label: str = "",
) -> dict:
    """Run full paper-trading simulation on OHLCV data."""
    cfg = cfg or CONFIG
    pre = precompute_all_indicators(df, cfg.strategy)

    portfolio = PortfolioState(
        total_value=cfg.portfolio.initial_balance,
        cash=cfg.portfolio.initial_balance,
        peak_value=cfg.portfolio.initial_balance,
    )

    pair = cfg.portfolio.trading_pairs[0] if cfg.portfolio.trading_pairs else "ETH/USDC"
    open_position: dict | None = None
    trades: list[dict] = []
    proof_hashes: list[str] = []

    last_date = ""
    for idx in range(warmup, len(df)):
        row = df.iloc[idx]
        timestamp = str(df.index[idx])
        price = float(row["close"])

        # Daily reset for daily-loss-cap enforcement.
        today = timestamp[:10]
        if today != last_date:
            portfolio.daily_pnl = 0.0
            last_date = today

        # Manage open trade first.
        if open_position is not None:
            open_position = check_trailing_stop(open_position, price, cfg.risk)
            closed, pnl = _check_exit(
                open_position,
                high=float(row["high"]),
                low=float(row["low"]),
            )
            if closed:
                hit_sl = pnl < 0
                portfolio.cash += float(open_position["size"]) + pnl
                portfolio = update_after_trade(portfolio, pnl, hit_stop_loss=hit_sl)

                trade = {
                    **open_position,
                    "exit_price": round(price, 8),
                    "exit_timestamp": timestamp,
                    "pnl": round(pnl, 2),
                    "exit_reason": "stop_loss" if hit_sl else "take_profit",
                    "dataset": dataset_label,
                }
                trades.append(trade)
                _append_trade_log(trade)
                open_position = None

        if open_position is not None:
            continue

        if portfolio.cooldown_bars > 0:
            portfolio.cooldown_bars -= 1

        ind = indicators_at(pre, idx)
        if ind is None:
            continue

        strategy_out = generate_strategy_signal(pre, idx, cfg.strategy)
        combined = combine_signals(
            strategy_out,
            current_atr_norm=ind.atr_norm_14,
            cfg=cfg.signal,
        )

        action = combined["action"]
        regime = combined["regime"]

        if action in ("BUY", "SELL"):
            risk = check_risk(
                signal=action,
                confidence=float(combined["confidence"]),
                entry_price=price,
                requested_size=portfolio.cash * cfg.risk.max_capital_pct,
                portfolio=portfolio,
                cfg=cfg.risk,
                regime=regime,
                regime_cfg=cfg.regime,
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

        decision_record = {
            "timestamp": timestamp,
            "pair": pair,
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
                "pair": pair,
                "entry_price": round(price, 8),
                "size": risk.adjusted_size if risk.approved else 0.0,
            },
            "portfolio_state": {
                "total_value": round(portfolio.total_value, 2),
                "cash": round(portfolio.cash, 2),
                "daily_pnl": round(portfolio.daily_pnl, 2),
            },
            "dataset": dataset_label,
        }

        if risk.approved and action in ("BUY", "SELL"):
            proof_hash = log_decision(decision_record)
            proof_hashes.append(proof_hash)

            size = float(risk.adjusted_size)
            portfolio.cash -= size
            open_position = {
                "action": action,
                "pair": pair,
                "entry_price": round(price, 8),
                "size": round(size, 2),
                "stop_loss": risk.stop_loss_price,
                "take_profit": risk.take_profit_price,
                "timestamp": timestamp,
                "proof_hash": proof_hash,
            }

    # Mark-to-market close for any open position.
    if open_position is not None:
        final_price = float(df["close"].iloc[-1])
        entry = float(open_position["entry_price"])
        size = float(open_position["size"])
        if open_position["action"] == "BUY":
            pnl = (final_price - entry) / entry * size
        else:
            pnl = (entry - final_price) / entry * size

        portfolio.cash += size + pnl
        portfolio = update_after_trade(portfolio, pnl, hit_stop_loss=pnl < 0)

        end_trade = {
            **open_position,
            "exit_price": round(final_price, 8),
            "exit_timestamp": str(df.index[-1]),
            "pnl": round(pnl, 2),
            "exit_reason": "end_of_simulation",
            "dataset": dataset_label,
        }
        trades.append(end_trade)
        _append_trade_log(end_trade)

    total_pnl = portfolio.total_value - cfg.portfolio.initial_balance
    wins = sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0)

    return {
        "initial_balance": cfg.portfolio.initial_balance,
        "final_value": round(portfolio.total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / cfg.portfolio.initial_balance * 100.0, 2),
        "total_trades": len(trades),
        "winning_trades": wins,
        "losing_trades": len(trades) - wins,
        "win_rate": round(wins / len(trades) * 100.0, 2) if trades else 0.0,
        "proof_hashes_generated": len(proof_hashes),
        "trades": trades,
    }


def main() -> None:
    """CLI wrapper for running the simplified paper-trading loop."""
    parser = argparse.ArgumentParser(description="ERC-8004 Hybrid Trading Agent")
    parser.add_argument("--data", type=str, default=None, help="OHLCV CSV path")
    parser.add_argument("--days", type=int, default=365, help="Synthetic days if --data omitted")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    df = load_or_generate(args.data, days=args.days)
    dataset_label = Path(args.data).stem if args.data else "synthetic"
    summary = run_paper_trading(df, dataset_label=dataset_label)

    print("\n" + "=" * 60)
    print("PAPER TRADING SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        if key != "trades":
            print(f"  {key:.<30} {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
