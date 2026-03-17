"""
main.py — Entry point for the Balanced Hybrid AI Trading Agent.


Orchestrates all modules: loads market data, runs strategy modules,
combines signals via AI predictor, validates through risk manager,
executes paper trades, and logs proof hashes.

Usage:
    # Run paper trading simulation on synthetic data
    python main.py

    # Run on a specific CSV file
    python main.py --data data/historical_prices.csv

    # Run with debug logging
    python main.py --log-level DEBUG

    # Launch the dashboard (separate terminal)
    streamlit run dashboard/dashboard.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, AppConfig
from utils.helpers import setup_logging, utc_now_iso, adx, bollinger_bands
from utils.data_loader import load_or_generate
from modules.momentum import generate_signal as momentum_signal
from modules.mean_reversion import generate_signal as mean_reversion_signal
from modules.yield_optimizer import generate_signal as yield_signal
from modules.ai_predictor import generate_signal_from_strategy_outputs
from risk.risk_manager import (
    check_risk, update_after_trade, check_trailing_stop, PortfolioState,
)
from validation.proof_logger import log_decision

logger = logging.getLogger(__name__)

# Trade history log path
TRADE_LOG = Path("data/trade_history.jsonl")


def detect_regime(df: pd.DataFrame) -> str:
    """
    Detect current market regime using ADX and Bollinger Band width.

    Returns:
        One of "trending_up", "trending_down", "ranging", "volatile".
    """
    close = df["close"]

    if len(df) < 30 or not {"high", "low"}.issubset(df.columns):
        return "ranging"  # Default when insufficient data

    adx_series = adx(df["high"], df["low"], close, 14)
    adx_val = adx_series.iloc[-1]

    _, upper, lower = bollinger_bands(close, 20, 2.0)
    bb_width = ((upper - lower) / close).iloc[-1] if close.iloc[-1] > 0 else 0.0

    # High volatility: BB width > 6%
    if bb_width > 0.06:
        return "volatile"

    # Strong trend: ADX > 25
    if not np.isnan(adx_val) and adx_val > 25:
        # Determine direction by recent returns
        ret_10d = close.pct_change(10).iloc[-1] if len(close) > 10 else 0
        if ret_10d > 0:
            return "trending_up"
        else:
            return "trending_down"

    return "ranging"


def _get_regime_weights(regime: str, cfg: AppConfig) -> dict[str, float]:
    """Get strategy weights based on current market regime."""
    rw = cfg.regime_weights
    if regime.startswith("trending"):
        return {
            "momentum": rw.trending_momentum,
            "mean_reversion": rw.trending_mean_reversion,
            "ai_predictor": rw.trending_ai,
        }
    elif regime == "volatile":
        return {
            "momentum": rw.volatile_momentum,
            "mean_reversion": rw.volatile_mean_reversion,
            "ai_predictor": rw.volatile_ai,
        }
    else:  # ranging
        return {
            "momentum": rw.ranging_momentum,
            "mean_reversion": rw.ranging_mean_reversion,
            "ai_predictor": rw.ranging_ai,
        }


def combine_signals(
    strategy_signals: dict[str, dict],
    cfg: AppConfig | None = None,
    regime: str = "ranging",
) -> dict:
    """
    Combine strategy signals using regime-adaptive weighted voting.

    Each strategy's signal is converted to a numeric score
    (BUY=+1, SELL=-1, HOLD=0) weighted by its confidence and
    the regime-adapted weight. Requires minimum strategy agreement.

    Args:
        strategy_signals: Dict of strategy name → signal dict.
        cfg: App config with weight settings.
        regime: Current market regime.

    Returns:
        {"action": "BUY"|"SELL"|"HOLD", "confidence": float, "details": dict}
    """
    cfg = cfg or CONFIG

    # Use regime-adaptive weights if enabled, otherwise use static weights
    if cfg.signal.use_regime_detection:
        weights = _get_regime_weights(regime, cfg)
    else:
        weights = {
            "momentum": cfg.weights.momentum,
            "mean_reversion": cfg.weights.mean_reversion,
            "ai_predictor": cfg.weights.ai_predictor,
        }
    weights["yield_optimizer"] = cfg.weights.yield_optimizer

    signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

    weighted_sum = 0.0
    total_weight = 0.0
    details: dict = {}
    # Track direction agreement
    buy_count = 0
    sell_count = 0

    for name, sig in strategy_signals.items():
        if name == "ai_ensemble":
            continue
        w = weights.get(name, 0.1)
        if w <= 0:
            continue
        direction_val = signal_map.get(sig["signal"], 0.0)
        conf = sig["confidence"]
        contribution = direction_val * conf * w
        weighted_sum += contribution
        total_weight += w * conf
        details[name] = {
            "signal": sig["signal"],
            "confidence": round(conf, 3),
            "weight": w,
            "contribution": round(contribution, 4),
        }
        if sig["signal"] == "BUY":
            buy_count += 1
        elif sig["signal"] == "SELL":
            sell_count += 1

    # Add AI ensemble signal
    ai_sig = strategy_signals.get("ai_ensemble")
    if ai_sig:
        ai_w = weights.get("ai_predictor", cfg.weights.ai_predictor)
        ai_val = signal_map.get(ai_sig["signal"], 0.0)
        ai_conf = ai_sig["confidence"]
        ai_contrib = ai_val * ai_conf * ai_w
        weighted_sum += ai_contrib
        total_weight += ai_w * ai_conf
        details["ai_ensemble"] = {
            "signal": ai_sig["signal"],
            "confidence": round(ai_conf, 3),
            "weight": ai_w,
            "contribution": round(ai_contrib, 4),
        }
        if ai_sig["signal"] == "BUY":
            buy_count += 1
        elif ai_sig["signal"] == "SELL":
            sell_count += 1

    # Normalize
    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    threshold = cfg.signal.signal_threshold

    if final_score > threshold:
        action = "BUY"
    elif final_score < -threshold:
        action = "SELL"
    else:
        action = "HOLD"

    # Consensus requirement: need min_agreement strategies to agree
    min_agree = cfg.signal.min_agreement
    if action == "BUY" and buy_count < min_agree:
        action = "HOLD"
    elif action == "SELL" and sell_count < min_agree:
        action = "HOLD"

    confidence = min(abs(final_score) * 2.5, 1.0)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "score": round(final_score, 4),
        "details": details,
        "regime": regime,
        "buy_agreement": buy_count,
        "sell_agreement": sell_count,
    }


def run_paper_trading(
    df: pd.DataFrame,
    cfg: AppConfig | None = None,
    warmup: int = 30,
    dataset_label: str = "",
) -> dict:
    """
    Run a paper-trading simulation over historical data.

    Iterates through each bar (after warmup period), generates signals
    from all modules, combines them, checks risk, executes paper trades,
    and logs proof hashes.

    Args:
        df: OHLCV DataFrame.
        cfg: App config.
        warmup: Number of initial bars to skip (for indicator warm-up).

    Returns:
        Summary dict with PnL, trade count, and proof hashes.
    """
    cfg = cfg or CONFIG

    # Initialize portfolio
    portfolio = PortfolioState(
        total_value=cfg.portfolio.initial_balance,
        cash=cfg.portfolio.initial_balance,
        peak_value=cfg.portfolio.initial_balance,
    )

    trades: list[dict] = []
    proof_hashes: list[str] = []
    pair = cfg.portfolio.trading_pairs[0] if cfg.portfolio.trading_pairs else "ETH/USDC"

    # Ensure trade log directory exists
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

    total_bars = len(df)
    logger.info(
        "Starting paper trading: %d bars, warmup=%d, pair=%s, balance=$%.0f",
        total_bars, warmup, pair, portfolio.total_value,
    )

    open_position: dict | None = None  # Track one open position at a time

    for i in range(warmup, total_bars):
        window = df.iloc[:i + 1]  # All data up to current bar
        current_price = window["close"].iloc[-1]
        timestamp = str(window.index[-1])

        # --- Update trailing stop and check if open position hits SL/TP ---
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

        # --- Skip if we already have an open position ---
        if open_position:
            continue

        # --- Decrement cooldown timer ---
        if portfolio.cooldown_bars > 0:
            portfolio.cooldown_bars -= 1

        # --- Generate strategy signals ---
        mom_sig = momentum_signal(window, cfg.momentum)
        mr_sig = mean_reversion_signal(window, cfg.mean_reversion)

        strategy_signals = {
            "momentum": mom_sig,
            "mean_reversion": mr_sig,
        }

        # Only include yield optimizer if weight > 0
        if cfg.weights.yield_optimizer > 0:
            yld_sig = yield_signal(window, portfolio_value=portfolio.total_value, cfg=cfg.yield_opt)
            strategy_signals["yield_optimizer"] = yld_sig

        # --- AI ensemble prediction ---
        ai_sig = generate_signal_from_strategy_outputs(strategy_signals, window, cfg.ai)
        strategy_signals["ai_ensemble"] = ai_sig

        # --- Detect regime ---
        regime = detect_regime(window)

        # --- Combine all signals ---
        combined = combine_signals(strategy_signals, cfg, regime=regime)

        # --- Risk check ---
        if combined["action"] in ("BUY", "SELL"):
            requested_size = portfolio.cash * 0.2  # 20% of available cash per trade
            risk_result = check_risk(
                signal=combined["action"],
                confidence=combined["confidence"],
                entry_price=current_price,
                requested_size=requested_size,
                portfolio=portfolio,
                df=window,
                cfg=cfg.risk,
            )
        else:
            risk_result = _skip_risk_result()

        # --- Build decision record ---
        decision_record = {
            "timestamp": timestamp,
            "pair": pair,
            "current_price": round(current_price, 8),
            "strategy_signals": {
                k: {"signal": v["signal"], "confidence": round(v["confidence"], 4)}
                for k, v in strategy_signals.items()
            },
            "ai_prediction": {
                "signal": ai_sig["signal"],
                "confidence": round(ai_sig["confidence"], 4),
                "reasoning": ai_sig.get("metadata", {}).get("reasoning", ""),
            },
            "combined_decision": combined,
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

        # --- Log proof ---
        proof_hash = log_decision(decision_record)
        proof_hashes.append(proof_hash)

        # --- Execute trade ---
        if risk_result.approved and combined["action"] in ("BUY", "SELL"):
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
                "Opened %s %s @ %g size=$%.0f (SL=%g TP=%g)",
                combined["action"], pair, current_price, size,
                risk_result.stop_loss_price, risk_result.take_profit_price,
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
            pnl = (sl - entry) / entry * size
            return True, pnl
        if high >= tp:
            pnl = (tp - entry) / entry * size
            return True, pnl
    elif position["action"] == "SELL":
        if high >= sl:
            pnl = (entry - sl) / entry * size
            return True, pnl
        if low <= tp:
            pnl = (entry - tp) / entry * size
            return True, pnl

    return False, 0.0


def _skip_risk_result():
    """Return a dummy risk result for HOLD signals."""
    from risk.risk_manager import RiskResult
    return RiskResult(
        approved=False,
        reasons=["Signal is HOLD — no trade"],
        adjusted_size=0.0,
        stop_loss_price=0.0,
        take_profit_price=0.0,
    )


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

    # Load data
    df = load_or_generate(args.data, days=args.days)
    logger.info("Data loaded: %d rows, range %s to %s",
                len(df), df.index[0], df.index[-1])

    # Run simulation
    summary = run_paper_trading(df)

    # Print summary
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
