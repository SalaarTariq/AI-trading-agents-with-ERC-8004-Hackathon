"""
main.py — Entry point for the Balanced Hybrid AI Trading Agent.

Orchestrates all modules: loads market data, runs strategy modules,
combines signals via rule-based scorer, validates through risk manager,
executes paper trades, and logs proof hashes.

Usage:
    python main.py
    python main.py --data data/historical_prices.csv
    python main.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, AppConfig
from utils.helpers import setup_logging, ema, atr
from utils.data_loader import load_or_generate
from utils.regime import detect_regime as _detect_regime, should_skip_trade, RegimeResult
from modules.momentum import generate_signal as momentum_signal
from modules.mean_reversion import generate_signal as mean_reversion_signal
from modules.ai_predictor import generate_signal_from_strategy_outputs
from modules.confidence_scoring import compute_confidence
from utils.indicators import compute_indicators
from risk.risk_manager import (
    check_risk, update_after_trade, check_trailing_stop, PortfolioState, RiskResult,
)
from validation.proof_logger import log_decision

logger = logging.getLogger(__name__)

TRADE_LOG = Path("data/trade_history.jsonl")


def detect_regime(df: pd.DataFrame) -> RegimeResult:
    """
    Detect current market regime (delegated to utils/regime.py).

    Returns:
        RegimeResult with regime string, metrics, and filter flags.
    """
    return _detect_regime(df)


def combine_signals(
    strategy_signals: dict[str, dict],
    cfg: AppConfig | None = None,
    regime: str = "choppy",
) -> dict:
    """
    Combine strategy signals using regime-aware confidence scoring.
    """
    cfg = cfg or CONFIG

    mom = strategy_signals.get("momentum", {"signal": 0, "raw_strength": 0.0})
    mr = strategy_signals.get("mean_reversion", {"signal": 0, "raw_strength": 0.0})
    ai = strategy_signals.get("ai_ensemble", {"prob_up": 0.5, "rolling_accuracy": None})

    rp = cfg.regime.get(regime)

    # Regime-adaptive signal weights + counter-trend suppression
    conf, action = compute_confidence(
        mom, mr, ai, None,
        conf_threshold=rp.conf_threshold,
        w_mom=rp.w_mom,
        w_mr=rp.w_mr,
        w_ai=rp.w_ai,
        regime=regime,
    )

    details = {
        "momentum": {"signal": mom.get("signal"), "raw_strength": round(float(mom.get("raw_strength", 0.0)), 4)},
        "mean_reversion": {"signal": mr.get("signal"), "raw_strength": round(float(mr.get("raw_strength", 0.0)), 4)},
        "ai": {
            "prob_up": round(float(ai.get("prob_up", 0.5)), 4),
            "rolling_accuracy": ai.get("rolling_accuracy", None),
        },
    }

    return {
        "action": action,
        "confidence": round(float(conf), 4),
        "score": round(float(conf), 4),
        "details": details,
        "regime": regime,
        "buy_agreement": int(mom.get("signal", 0) == 1) + int(mr.get("signal", 0) == 1),
        "sell_agreement": int(mom.get("signal", 0) == -1) + int(mr.get("signal", 0) == -1),
    }


def run_paper_trading(
    df: pd.DataFrame,
    cfg: AppConfig | None = None,
    warmup: int = 30,
    dataset_label: str = "",
) -> dict:
    """
    Run a paper-trading simulation over historical data.

    Args:
        df: OHLCV DataFrame.
        cfg: App config.
        warmup: Number of initial bars to skip (for indicator warm-up).
        dataset_label: Label for the dataset in logs.

    Returns:
        Summary dict with PnL, trade count, and proof hashes.
    """
    cfg = cfg or CONFIG

    portfolio = PortfolioState(
        total_value=cfg.portfolio.initial_balance,
        cash=cfg.portfolio.initial_balance,
        peak_value=cfg.portfolio.initial_balance,
    )

    trades: list[dict] = []
    proof_hashes: list[str] = []
    pair = cfg.portfolio.trading_pairs[0] if cfg.portfolio.trading_pairs else "ETH/USDC"

    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

    total_bars = len(df)
    logger.info(
        "Starting paper trading: %d bars, warmup=%d, pair=%s, balance=$%.0f",
        total_bars, warmup, pair, portfolio.total_value,
    )

    open_position: dict | None = None
    last_date: str = ""

    for i in range(warmup, total_bars):
        window = df.iloc[:i + 1]
        current_price = window["close"].iloc[-1]
        timestamp = str(window.index[-1])

        # --- Daily PnL reset (detect new calendar day) ---
        current_date = timestamp[:10]
        if current_date != last_date:
            if last_date:
                logger.debug("Daily reset: PnL was $%.2f", portfolio.daily_pnl)
            portfolio.daily_pnl = 0.0
            last_date = current_date

        # --- Update trailing stop and check exit ---
        if open_position:
            open_position = check_trailing_stop(open_position, current_price, cfg.risk)
            high = window["high"].iloc[-1]
            low = window["low"].iloc[-1]
            closed, pnl = _check_exit(open_position, high, low, current_price)
            if closed:
                hit_sl = pnl < 0
                portfolio.cash += open_position["size"] + pnl
                portfolio = update_after_trade(portfolio, pnl, hit_stop_loss=hit_sl)

                trade_record = {
                    **open_position,
                    "exit_price": round(current_price, 8),
                    "pnl": round(pnl, 2),
                    "exit_timestamp": timestamp,
                    "exit_reason": "stop_loss" if hit_sl else "take_profit",
                    "dataset": dataset_label,
                }
                trades.append(trade_record)
                _append_trade_log(trade_record)

                logger.info(
                    "Closed %s %s @ %g → PnL $%.2f (%s)",
                    open_position["action"], pair, current_price, pnl,
                    trade_record["exit_reason"],
                )
                open_position = None
                # Post-loss cooldown: avoid re-entering immediately after SL
                if hit_sl and portfolio.cooldown_bars < 5:
                    portfolio.cooldown_bars = max(portfolio.cooldown_bars, 5)

        if open_position:
            continue

        if portfolio.cooldown_bars > 0:
            portfolio.cooldown_bars -= 1

        # --- Generate strategy signals ---
        mom_sig = momentum_signal(window, cfg.momentum)
        mr_sig = mean_reversion_signal(window, cfg.mean_reversion)

        ind = compute_indicators(window)

        strategy_signals = {
            "momentum": mom_sig,
            "mean_reversion": mr_sig,
        }

        # --- Rule-based scorer ---
        ai_sig = generate_signal_from_strategy_outputs(strategy_signals, window)
        strategy_signals["ai_ensemble"] = ai_sig

        # --- Detect regime (strengthened: spread + ATR + volume + HTF) ---
        regime_result = detect_regime(window)
        regime = regime_result.regime

        # --- Volume / HTF filters: skip trade if filters fail ---
        skip_trade, skip_reason = should_skip_trade(regime_result)

        # --- Combine all signals ---
        combined = combine_signals(strategy_signals, cfg, regime=regime)

        # --- Risk check with regime ---
        if skip_trade:
            risk_result = RiskResult(
                approved=False,
                reasons=[f"Regime filter: {skip_reason}"],
                adjusted_size=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
            )
        elif combined["action"] in ("BUY", "SELL"):
            requested_size = portfolio.cash * 0.2
            risk_result = check_risk(
                signal=combined["action"],
                confidence=combined["confidence"],
                entry_price=current_price,
                requested_size=requested_size,
                portfolio=portfolio,
                df=window,
                cfg=cfg.risk,
                regime=regime,
                regime_cfg=cfg.regime,
            )
        else:
            risk_result = RiskResult(
                approved=False,
                reasons=["Signal is HOLD — no trade"],
                adjusted_size=0.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
            )

        # --- Build decision record ---
        decision_record = {
            "timestamp": timestamp,
            "pair": pair,
            "current_price": round(current_price, 8),
            "strategy_signals": {
                k: (
                    {"signal": v.get("signal"), "raw_strength": round(float(v.get("raw_strength", 0.0)), 4)}
                    if isinstance(v, dict) and "raw_strength" in v
                    else v
                )
                for k, v in strategy_signals.items()
            },
            "ai_prediction": {
                "signal": ai_sig["signal"],
                "confidence": round(ai_sig["confidence"], 4),
                "reasoning": ai_sig.get("metadata", {}).get("reasoning", ""),
            },
            "combined_decision": combined,
            "indicators": (
                {
                    "ema_spread": ind.ema_spread,
                    "rsi": ind.rsi_14,
                    "macd_hist": ind.macd_hist,
                    "atr": ind.atr_14,
                    "atr_norm": ind.atr_norm_14,
                }
                if ind is not None
                else {}
            ),
            "regime_detail": {
                "regime": regime_result.regime,
                "ema_spread_norm": regime_result.ema_spread_norm,
                "atr_norm": regime_result.atr_norm,
                "volume_ratio": regime_result.volume_ratio,
                "htf_trend": regime_result.htf_trend,
                "volume_ok": regime_result.volume_ok,
                "htf_confirms": regime_result.htf_confirms,
            },
            "risk_result": {
                "approved": risk_result.approved,
                "reasons": risk_result.reasons,
                "warnings": risk_result.warnings,
                "adjusted_size": risk_result.adjusted_size,
                "stop_loss": risk_result.stop_loss_price,
                "take_profit": risk_result.take_profit_price,
            },
            "final_decision": {
                "action": combined["action"] if risk_result.approved else "HOLD",
                "pair": pair,
                "entry_price": round(current_price, 8),
                "size": risk_result.adjusted_size if risk_result.approved else 0,
            },
            "portfolio_state": {
                "total_value": round(portfolio.total_value, 2),
                "cash": round(portfolio.cash, 2),
                "daily_pnl": round(portfolio.daily_pnl, 2),
            },
            "dataset": dataset_label,
        }

        # --- Execute trade and log proof ---
        if risk_result.approved and combined["action"] in ("BUY", "SELL"):
            proof_hash = log_decision(decision_record)
            proof_hashes.append(proof_hash)

            size = risk_result.adjusted_size
            portfolio.cash -= size

            open_position = {
                "action": combined["action"],
                "pair": pair,
                "entry_price": round(current_price, 8),
                "size": round(size, 2),
                "stop_loss": risk_result.stop_loss_price,
                "take_profit": risk_result.take_profit_price,
                "timestamp": timestamp,
                "proof_hash": proof_hash,
                "dataset": dataset_label,
            }

            logger.info(
                "Opened %s %s @ %g size=$%.0f (SL=%g TP=%g) regime=%s vol_ratio=%.2f htf=%s",
                combined["action"], pair, current_price, size,
                risk_result.stop_loss_price, risk_result.take_profit_price, regime,
                regime_result.volume_ratio, regime_result.htf_trend,
            )

    # --- Close any remaining position at last price ---
    if open_position:
        final_price = df["close"].iloc[-1]
        entry = open_position["entry_price"]
        if open_position["action"] == "BUY":
            pnl = (final_price - entry) / entry * open_position["size"]
        else:
            pnl = (entry - final_price) / entry * open_position["size"]

        portfolio.cash += open_position["size"] + pnl
        portfolio = update_after_trade(portfolio, pnl)
        trades.append({
            **open_position,
            "exit_price": round(final_price, 8),
            "pnl": round(pnl, 2),
            "exit_timestamp": str(df.index[-1]),
            "exit_reason": "end_of_simulation",
        })
        _append_trade_log(trades[-1])

    # --- Summary ---
    total_pnl = portfolio.total_value - cfg.portfolio.initial_balance
    win_count = sum(1 for t in trades if t.get("pnl", 0) > 0)

    summary = {
        "initial_balance": cfg.portfolio.initial_balance,
        "final_value": round(portfolio.total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / cfg.portfolio.initial_balance * 100, 2),
        "total_trades": len(trades),
        "winning_trades": win_count,
        "losing_trades": len(trades) - win_count,
        "win_rate": round(win_count / len(trades) * 100, 1) if trades else 0,
        "proof_hashes_generated": len(proof_hashes),
        "trades": trades,
    }

    logger.info("=" * 60)
    logger.info("PAPER TRADING COMPLETE")
    logger.info("  PnL: $%.2f (%.1f%%)", total_pnl, summary["pnl_pct"])
    logger.info("  Trades: %d (W:%d / L:%d)", len(trades), win_count, len(trades) - win_count)
    logger.info("  Proofs: %d hashes generated", len(proof_hashes))
    logger.info("=" * 60)

    return summary


def _check_exit(
    position: dict, high: float, low: float, close: float
) -> tuple[bool, float]:
    """Check if an open position hits stop-loss or take-profit."""
    entry = position["entry_price"]
    sl = position["stop_loss"]
    tp = position["take_profit"]
    size = position["size"]

    if position["action"] == "BUY":
        if low <= sl:
            return True, (sl - entry) / entry * size
        if high >= tp:
            return True, (tp - entry) / entry * size
    elif position["action"] == "SELL":
        if high >= sl:
            return True, (entry - sl) / entry * size
        if low <= tp:
            return True, (entry - tp) / entry * size

    return False, 0.0


def _append_trade_log(trade: dict) -> None:
    """Append a trade record to the trade history JSONL file."""
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def main():
    """Parse CLI args and run the trading agent."""
    parser = argparse.ArgumentParser(
        description="Balanced Hybrid AI Trading Agent — Paper Trading Simulator"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to OHLCV CSV file (generates synthetic data if omitted)",
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of synthetic data days to generate (default: 365)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Balanced Hybrid AI Trading Agent")
    logger.info("=" * 60)

    df = load_or_generate(args.data, days=args.days)
    logger.info("Data loaded: %d rows, range %s to %s",
                len(df), df.index[0], df.index[-1])

    dataset_label = Path(args.data).stem if args.data else "synthetic"
    summary = run_paper_trading(df, dataset_label=dataset_label)

    print("\n" + "=" * 60)
    print("PAPER TRADING SUMMARY")
    print("=" * 60)
    for key, val in summary.items():
        print(f"  {key:.<30} {val}")
    print("=" * 60)
    print(f"\nProof log: {CONFIG.proof_log_path}")
    print(f"Trade log: {TRADE_LOG}")
    print("Dashboard: streamlit run dashboard/dashboard.py")


if __name__ == "__main__":
    main()
