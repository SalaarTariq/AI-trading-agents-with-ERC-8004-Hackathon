"""
main.py — Entry point for the Balanced Hybrid AI Trading Agent.

Orchestrates all modules: loads market data, runs strategy modules,
combines signals via regime-aware scorer, validates through risk manager,
executes paper trades, and logs proof hashes.

KEY IMPROVEMENTS (v3 — profitability overhaul):
- Smart regime detection: ADX + BB width + EMA spread → 4 regimes
- Momentum: ADX-gated, MACD crossover + pullback entries in trends
- Mean-reversion: BB width ranging detection, RSI reversal confirmation
- Asymmetric R:R: TP targets 2-3x SL distance
- Time-based exits: close stale positions after max_hold_bars
- Sharpe ratio and max drawdown in summary stats

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
from utils.helpers import setup_logging
from utils.data_loader import load_or_generate
from utils.indicators import Indicators, precompute_all_indicators
from modules.confidence_scoring import compute_confidence
from risk.risk_manager import (
    check_risk, update_after_trade, check_trailing_stop, PortfolioState, RiskResult,
)
from validation.proof_logger import log_decision

logger = logging.getLogger(__name__)

TRADE_LOG = Path("data/trade_history.jsonl")

# Maximum bars to hold a position before forced exit (avoid stale trades)
MAX_HOLD_BARS = 30


# ---------------------------------------------------------------------------
# Per-bar indicator lookup from precomputed arrays
# ---------------------------------------------------------------------------

def _indicators_at(pre: pd.DataFrame, idx: int, close: float) -> Indicators | None:
    """Build an Indicators dataclass from precomputed arrays at bar `idx`."""
    row = pre.iloc[idx]
    if np.isnan(row["ema_fast"]) or np.isnan(row["rsi_14"]) or np.isnan(row["macd_hist"]):
        return None
    if close == 0:
        return None
    atr_v = row["atr_14"] if not np.isnan(row["atr_14"]) else None
    atr_norm_v = row["atr_norm_14"] if not np.isnan(row["atr_norm_14"]) else None
    bb_width_v = row["bb_width"] if not np.isnan(row["bb_width"]) else 0.0
    return Indicators(
        ema_fast=float(row["ema_fast"]),
        ema_slow=float(row["ema_slow"]),
        ema_spread=float(row["ema_spread"]),
        ema_spread_norm=float(row["ema_spread_norm"]),
        rsi_14=float(row["rsi_14"]),
        macd_line=float(row["macd_line"]),
        macd_signal=float(row["macd_signal"]),
        macd_hist=float(row["macd_hist"]),
        macd_hist_norm=float(row["macd_hist_norm"]),
        atr_14=float(atr_v) if atr_v is not None else None,
        atr_norm_14=float(atr_norm_v) if atr_norm_v is not None else None,
        bb_width=float(bb_width_v),
    )


# ---------------------------------------------------------------------------
# Smart regime detection (v3)
# ---------------------------------------------------------------------------
# Uses ADX for trend strength, BB width for ranging detection, and
# EMA spread direction. Four regimes: trending_up, trending_down, ranging, choppy.

ADX_TREND_STRONG = 25.0    # Strong trend
ADX_TREND_WEAK = 18.0      # Weak but real trend
ADX_RANGING_MAX = 20.0     # Below this = potential range
BB_WIDTH_RANGING = 0.06    # Narrow BB = ranging market
BB_WIDTH_CHOPPY = 0.10     # Wide BB = volatile/choppy


def _regime_at(pre: pd.DataFrame, df: pd.DataFrame, idx: int) -> dict:
    """Market regime detection using ADX + BB width + EMA spread.

    Regime classification:
    - trending_up/down: ADX >= 18, EMA spread confirms direction, not too volatile
    - ranging: ADX < 20 AND BB width narrow — ideal for mean-reversion
    - choppy: everything else — trade cautiously

    Also determines HTF trend via EMA50 to filter counter-trend entries.
    """
    close = float(df["close"].iloc[idx])
    if close == 0 or idx < 30:
        return {"regime": "choppy", "ema_spread_norm": 0.0, "atr_norm": 0.0,
                "volume_ratio": 1.0, "htf_trend": "neutral", "volume_ok": True,
                "htf_confirms": True, "trend_strength": "none",
                "adx": 0.0, "bb_width": 0.0}

    row = pre.iloc[idx]
    spread_norm = float(row["ema_spread"]) / close if close > 0 else 0.0
    atr_norm = float(row["atr_norm_14"]) if not np.isnan(row["atr_norm_14"]) else 0.0
    adx_val = float(row["adx_14"]) if not np.isnan(row["adx_14"]) else 0.0
    bb_width = float(row["bb_width"]) if not np.isnan(row["bb_width"]) else 0.0

    abs_spread = abs(spread_norm)
    trend_strength = "none"
    regime = "choppy"

    # ── Trending: ADX confirms directional movement ──────────────────
    if adx_val >= ADX_TREND_STRONG and abs_spread >= 0.002:
        regime = "trending_up" if spread_norm > 0 else "trending_down"
        trend_strength = "strong"
    elif adx_val >= ADX_TREND_WEAK and abs_spread >= 0.0015:
        regime = "trending_up" if spread_norm > 0 else "trending_down"
        trend_strength = "moderate"

    # ── Ranging: low ADX + narrow Bollinger Bands ────────────────────
    elif adx_val < ADX_RANGING_MAX and bb_width < BB_WIDTH_RANGING:
        regime = "ranging"
        trend_strength = "none"

    # ── Choppy: high volatility, no clear direction ──────────────────
    # (default — everything that doesn't fit trending or ranging)

    # Higher-timeframe trend via EMA50
    ema50 = float(row["ema50"]) if not np.isnan(row["ema50"]) else close
    htf_trend = (
        "bullish" if close > ema50 * 1.008
        else "bearish" if close < ema50 * 0.992
        else "neutral"
    )

    return {
        "regime": regime, "ema_spread_norm": round(spread_norm, 6),
        "atr_norm": round(atr_norm, 6), "volume_ratio": 1.0,
        "htf_trend": htf_trend, "volume_ok": True, "htf_confirms": True,
        "trend_strength": trend_strength,
        "adx": round(adx_val, 2), "bb_width": round(bb_width, 4),
    }


# ---------------------------------------------------------------------------
# Momentum signal (v3)
# ---------------------------------------------------------------------------
# KEY CHANGES:
# - ADX threshold lowered to 18 (was 20+)
# - MACD crossover window extended to 4 bars
# - Pullback entries get confidence bonus
# - Trend-aligned RSI ranges are wider (more entries)
# - Volatility penalty reduced (less over-filtering)

def _momentum_signal(pre: pd.DataFrame, df: pd.DataFrame, idx: int, cfg) -> dict:
    """Momentum signal — ADX-gated with MACD crossover and pullback entries."""
    row = pre.iloc[idx]
    close = float(df["close"].iloc[idx])
    if np.isnan(row["ema_fast"]) or np.isnan(row["rsi_14"]) or close == 0:
        return {"signal": 0, "raw_strength": 0.0, "signal_str": "HOLD", "confidence": 0.0,
                "metadata": {"reason": "warming up"}}

    atr_norm = float(row["atr_norm_14"]) if not np.isnan(row["atr_norm_14"]) else 0.001
    ema_spread_pct = float(row["ema_spread"]) / close
    macd_hist_pct = float(row["macd_hist"]) / close
    rsi_now = float(row["rsi_14"])
    ema_fast_v = float(row["ema_fast"])
    adx_val = float(row["adx_14"]) if not np.isnan(row["adx_14"]) else 0.0

    trend_strength = min(abs(float(row["ema_spread_norm"])) / 0.008, 1.0)
    macd_strength = min(abs(float(row["macd_hist_norm"])) / 0.6, 1.0)
    # Lighter volatility penalty — was over-filtering good signals
    volatility_penalty = float(np.clip(max(atr_norm - 0.020, 0.0) * 5.0, 0.0, 0.15))

    # MACD direction confirmation: histogram in same direction as trade
    # (removed strict crossover requirement — too many missed continuation entries)
    macd_confirms_buy = float(row["macd_hist"]) > 0
    macd_confirms_sell = float(row["macd_hist"]) < 0

    # Also check for recent crossover (bonus, not required)
    has_macd_cross_buy = False
    has_macd_cross_sell = False
    for lookback in range(0, min(4, idx)):
        mc = float(pre.iloc[idx - lookback]["macd_cross"]) if not np.isnan(pre.iloc[idx - lookback]["macd_cross"]) else 0
        if mc == 1:
            has_macd_cross_buy = True
        elif mc == -1:
            has_macd_cross_sell = True

    # ADX filter: lowered to 18 — capture more valid trends
    adx_ok = adx_val >= 18.0

    # Pullback detection: RSI dipped then recovering, or price near EMA
    rsi_high_5 = float(row["rsi_14_high_5"]) if not np.isnan(row["rsi_14_high_5"]) else rsi_now
    rsi_low_5 = float(row["rsi_14_low_5"]) if not np.isnan(row["rsi_14_low_5"]) else rsi_now
    has_pullback_buy = (rsi_high_5 - rsi_now >= 3.0) or (close <= ema_fast_v * 1.003)
    has_pullback_sell = (rsi_now - rsi_low_5 >= 3.0) or (close >= ema_fast_v * 0.997)

    # Wider RSI ranges to capture more trend entries
    rsi_ok_buy = (38.0 if has_pullback_buy else 42.0) <= rsi_now <= 72.0
    rsi_ok_sell = 28.0 <= rsi_now <= (62.0 if has_pullback_sell else 58.0)

    signal = 0
    # BUY: uptrend confirmed by EMA spread + MACD crossover + ADX
    # MACD crossover is required to avoid chasing extended moves
    if (ema_spread_pct > 0.0020 and macd_confirms_buy and rsi_ok_buy
            and trend_strength >= 0.25 and has_macd_cross_buy and adx_ok):
        signal = 1
    # SELL: downtrend confirmed
    elif (ema_spread_pct < -0.0020 and macd_confirms_sell and rsi_ok_sell
            and trend_strength >= 0.25 and has_macd_cross_sell and adx_ok):
        signal = -1

    if signal != 0:
        if signal == 1:
            rsi_strength = min(max((72.0 - rsi_now) / 30.0, 0.0), 1.0)
        else:
            rsi_strength = min(max((rsi_now - 28.0) / 30.0, 0.0), 1.0)
        pullback_bonus = 0.06 if (signal == 1 and has_pullback_buy) or (signal == -1 and has_pullback_sell) else 0.0
        # ADX strength bonus: stronger trend = higher strength
        adx_bonus = min(max(adx_val - 18.0, 0.0) / 30.0, 0.12)
        # MACD crossover bonus: recent crossover is extra confirmation
        cross_bonus = 0.06 if (signal == 1 and has_macd_cross_buy) or (signal == -1 and has_macd_cross_sell) else 0.0
        raw_strength = float(np.clip(
            0.35 * trend_strength + 0.25 * macd_strength + 0.15 * rsi_strength + pullback_bonus + adx_bonus + cross_bonus,
            0.0, 1.0,
        ))
    else:
        raw_strength = 0.0

    confidence = float(np.clip(0.55 + 0.35 * raw_strength - volatility_penalty, 0.50, 0.90)) if signal != 0 else 0.0
    signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

    return {
        "signal": signal, "raw_strength": raw_strength,
        "signal_str": signal_str, "confidence": confidence,
        "metadata": {
            "rsi": round(rsi_now, 2),
            "trend_strength": round(trend_strength, 4),
            "macd_strength": round(macd_strength, 4),
            "volatility_penalty": round(volatility_penalty, 4),
            "has_pullback_buy": has_pullback_buy,
            "has_pullback_sell": has_pullback_sell,
            "has_macd_cross_buy": has_macd_cross_buy,
            "has_macd_cross_sell": has_macd_cross_sell,
            "adx": round(adx_val, 2),
        },
    }


# ---------------------------------------------------------------------------
# Mean-reversion signal (v3)
# ---------------------------------------------------------------------------
# KEY CHANGES:
# - Ranging detection uses BB width (< 0.06) AND ADX (< 20), not just EMA spread
# - BB-touch entries are strongest (price at band edge)
# - RSI reversal velocity confirms mean-reversion is underway
# - Wider BB z-score entry range: -0.20 instead of -0.25
# - Reduced volatility penalty overlap

def _mean_reversion_signal(pre: pd.DataFrame, df: pd.DataFrame, idx: int, cfg) -> dict:
    """Mean-reversion signal — BB width + ADX ranging detection with reversal confirmation."""
    row = pre.iloc[idx]
    close = float(df["close"].iloc[idx])
    if np.isnan(row["bb_mid"]) or np.isnan(row["rsi_14"]) or close == 0:
        return {"signal": 0, "raw_strength": 0.0, "signal_str": "HOLD", "confidence": 0.0,
                "metadata": {"reason": "warming up"}}

    bb_mid = float(row["bb_mid"])
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])
    rsi_now = float(row["rsi_14"])
    adx_val = float(row["adx_14"]) if not np.isnan(row["adx_14"]) else 0.0
    bb_width = float(row["bb_width"]) if not np.isnan(row["bb_width"]) else 0.0

    # ── Ranging detection: ADX low + BB narrow ───────────────────────
    # This is much more reliable than EMA spread alone
    is_ranging = adx_val < ADX_RANGING_MAX and bb_width < BB_WIDTH_RANGING

    # Also accept moderate ranges with wider BB (but ADX must be very low)
    if not is_ranging and adx_val < 15.0 and bb_width < BB_WIDTH_CHOPPY:
        is_ranging = True

    bb_width_abs = max(bb_upper - bb_lower, 1e-8)
    bb_z = (close - bb_mid) / bb_width_abs
    atr_norm = float(row["atr_norm_14"]) if not np.isnan(row["atr_norm_14"]) else 0.0
    volatility_penalty = float(np.clip(max(atr_norm - 0.025, 0.0) * 5.0, 0.0, 0.15))

    # RSI reversion velocity: RSI moving toward 50 in last 3 bars
    rsi_delta_3 = float(row["rsi_delta_3"]) if not np.isnan(row["rsi_delta_3"]) else 0.0
    reversion_buy = rsi_delta_3 > 3.0    # RSI rising toward 50
    reversion_sell = rsi_delta_3 < -3.0   # RSI falling toward 50

    # BB band touch: strongest MR signal
    bb_touch_buy = close <= bb_lower * 1.002   # Within 0.2% of lower band
    bb_touch_sell = close >= bb_upper * 0.998   # Within 0.2% of upper band

    signal = 0
    if is_ranging:
        # Strong signal: BB touch (no RSI requirement — price at extremes)
        if bb_touch_buy:
            signal = 1
        elif bb_z < -0.20 and rsi_now <= 48:
            signal = 1
        elif bb_touch_sell:
            signal = -1
        elif bb_z > 0.20 and rsi_now >= 52:
            signal = -1

    if signal != 0:
        band_strength = min(abs(bb_z) / 0.7, 1.0)
        rsi_strength = min(abs(rsi_now - 50.0) / 15.0, 1.0)
        reversion_bonus = 0.10 if (signal == 1 and reversion_buy) or (signal == -1 and reversion_sell) else 0.0
        touch_bonus = 0.12 if (signal == 1 and bb_touch_buy) or (signal == -1 and bb_touch_sell) else 0.0
        raw_strength = float(np.clip(
            0.50 * band_strength + 0.30 * rsi_strength + reversion_bonus + touch_bonus,
            0.0, 1.0,
        ))
    else:
        raw_strength = 0.0

    confidence = float(np.clip(0.52 + 0.38 * raw_strength - volatility_penalty, 0.50, 0.90)) if signal != 0 else 0.0
    signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

    return {
        "signal": signal, "raw_strength": raw_strength,
        "signal_str": signal_str, "confidence": confidence,
        "metadata": {
            "bb_z": round(bb_z, 4),
            "rsi": round(rsi_now, 2),
            "is_ranging": is_ranging,
            "adx": round(adx_val, 2),
            "bb_width": round(bb_width, 4),
            "volatility_penalty": round(volatility_penalty, 4),
            "bb_touch_buy": bb_touch_buy,
            "bb_touch_sell": bb_touch_sell,
            "reversion_velocity": round(rsi_delta_3, 2),
        },
    }


def _ai_scorer(strategy_signals: dict, ind: Indicators | None) -> dict:
    """Rule-based AI scorer using pre-computed Indicators."""
    indicator_score = 0.0
    ind_details: dict[str, float] = {}

    if ind is not None:
        ema_contrib = float(np.clip(ind.ema_spread_norm, -1.0, 1.0))
        indicator_score += ema_contrib * 0.4
        ind_details["ema_spread_norm"] = round(ema_contrib, 4)

        macd_contrib = float(np.clip(ind.macd_hist_norm, -1.0, 1.0))
        indicator_score += macd_contrib * 0.3
        ind_details["macd_hist_norm"] = round(macd_contrib, 4)

        rsi_val = ind.rsi_14
        ema_bearish = ema_contrib < -0.1
        ema_bullish = ema_contrib > 0.1
        if rsi_val < 35:
            rsi_contrib = (35 - rsi_val) / 35 if not ema_bearish else 0.0
        elif rsi_val > 65:
            rsi_contrib = -(rsi_val - 65) / 35 if not ema_bullish else 0.0
        else:
            rsi_contrib = 0.0
        indicator_score += rsi_contrib * 0.3
        ind_details["rsi_contrib"] = round(rsi_contrib, 4)

    mom = strategy_signals.get("momentum", {"signal": 0, "raw_strength": 0.0})
    mr = strategy_signals.get("mean_reversion", {"signal": 0, "raw_strength": 0.0})
    mom_val = int(mom.get("signal", 0)) * float(mom.get("raw_strength", 0.0))
    mr_val = int(mr.get("signal", 0)) * float(mr.get("raw_strength", 0.0))

    score = 0.40 * indicator_score + 0.35 * mom_val + 0.25 * mr_val
    prob_up = float(np.clip(0.5 + score * 0.5, 0.05, 0.95))

    if prob_up > 0.52:
        direction = "BUY"
    elif prob_up < 0.48:
        direction = "SELL"
    else:
        direction = "HOLD"

    confidence = abs(prob_up - 0.5) * 2.0

    return {
        "signal": direction, "confidence": confidence,
        "metadata": {
            "source": "rule_based_scorer",
            "indicator_score": round(indicator_score, 4),
            "combined_score": round(score, 4),
            "indicators": ind_details,
            "reasoning": f"Rule scorer: prob_up={prob_up:.3f} (ind={indicator_score:+.3f} mom={mom_val:+.3f} mr={mr_val:+.3f})",
        },
        "prob_up": prob_up, "rolling_accuracy": None,
    }


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------

def combine_signals(
    strategy_signals: dict[str, dict],
    cfg: AppConfig | None = None,
    regime: str = "choppy",
    current_atr_norm: float | None = None,
) -> dict:
    """Combine strategy signals using regime-aware confidence scoring."""
    cfg = cfg or CONFIG
    mom = strategy_signals.get("momentum", {"signal": 0, "raw_strength": 0.0})
    mr = strategy_signals.get("mean_reversion", {"signal": 0, "raw_strength": 0.0})
    ai = strategy_signals.get("ai_ensemble", {"prob_up": 0.5, "rolling_accuracy": None})

    rp = cfg.regime.get(regime)
    conf, action = compute_confidence(
        mom, mr, ai, current_atr_norm,
        conf_threshold=rp.conf_threshold,
        w_mom=rp.w_mom, w_mr=rp.w_mr, w_ai=rp.w_ai,
        regime=regime,
    )

    return {
        "action": action,
        "confidence": round(float(conf), 4),
        "score": round(float(conf), 4),
        "details": {
            "momentum": {"signal": mom.get("signal"), "raw_strength": round(float(mom.get("raw_strength", 0.0)), 4)},
            "mean_reversion": {"signal": mr.get("signal"), "raw_strength": round(float(mr.get("raw_strength", 0.0)), 4)},
            "ai": {"prob_up": round(float(ai.get("prob_up", 0.5)), 4), "rolling_accuracy": ai.get("rolling_accuracy")},
        },
        "regime": regime,
        "buy_agreement": int(mom.get("signal", 0) == 1) + int(mr.get("signal", 0) == 1),
        "sell_agreement": int(mom.get("signal", 0) == -1) + int(mr.get("signal", 0) == -1),
    }


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_paper_trading(
    df: pd.DataFrame,
    cfg: AppConfig | None = None,
    warmup: int = 30,
    dataset_label: str = "",
) -> dict:
    """
    Run a paper-trading simulation over historical data.

    Pre-computes all indicators once (O(n)), then iterates bars in O(1) per bar.
    Includes time-based exits and Sharpe/drawdown tracking.
    """
    cfg = cfg or CONFIG

    # Pre-compute all indicators on the full dataset
    pre = precompute_all_indicators(df)

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

    # Track bar-by-bar returns for Sharpe calculation
    bar_returns: list[float] = []
    prev_portfolio_value = portfolio.total_value
    max_drawdown = 0.0

    for i in range(warmup, total_bars):
        current_price = float(df["close"].iloc[i])
        timestamp = str(df.index[i])

        # --- Daily PnL reset ---
        current_date = timestamp[:10]
        if current_date != last_date:
            if last_date:
                logger.debug("Daily reset: PnL was $%.2f", portfolio.daily_pnl)
            portfolio.daily_pnl = 0.0
            last_date = current_date

        # --- Track bar returns for Sharpe ---
        current_total = portfolio.cash
        if open_position:
            # Mark-to-market the open position
            entry = open_position["entry_price"]
            size = open_position["size"]
            if open_position["action"] == "BUY":
                current_total += size + (current_price - entry) / entry * size
            else:
                current_total += size + (entry - current_price) / entry * size
        bar_ret = (current_total - prev_portfolio_value) / max(prev_portfolio_value, 1.0)
        bar_returns.append(bar_ret)
        prev_portfolio_value = current_total

        # Track max drawdown
        if portfolio.peak_value > 0:
            dd = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
            max_drawdown = max(max_drawdown, dd)

        # --- Check exit on open position ---
        if open_position:
            open_position = check_trailing_stop(open_position, current_price, cfg.risk)
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])
            closed, pnl = _check_exit(open_position, high, low, current_price)

            # Time-based exit: close stale positions that are losing or flat
            # (winning positions are protected by trailing stop instead)
            bars_held = i - open_position.get("entry_bar", i)
            entry = open_position["entry_price"]
            size = open_position["size"]
            if open_position["action"] == "BUY":
                unrealized_pnl = (current_price - entry) / entry * size
            else:
                unrealized_pnl = (entry - current_price) / entry * size

            if not closed and bars_held >= MAX_HOLD_BARS and unrealized_pnl <= size * 0.01:
                # Only force-close if not meaningfully in profit
                pnl = unrealized_pnl
                closed = True
                exit_reason = "time_exit"
            else:
                exit_reason = None

            if closed:
                if exit_reason is None:
                    hit_sl = pnl < 0
                    exit_reason = "stop_loss" if hit_sl else "take_profit"
                else:
                    hit_sl = pnl < 0

                portfolio.cash += open_position["size"] + pnl
                portfolio = update_after_trade(portfolio, pnl, hit_stop_loss=(hit_sl and exit_reason == "stop_loss"))

                trade_record = {
                    **open_position,
                    "exit_price": round(current_price, 8),
                    "pnl": round(pnl, 2),
                    "exit_timestamp": timestamp,
                    "exit_reason": exit_reason,
                    "dataset": dataset_label,
                    "bars_held": bars_held,
                }
                trades.append(trade_record)
                _append_trade_log(trade_record)

                logger.info(
                    "Closed %s %s @ %g -> PnL $%.2f (%s, %d bars)",
                    open_position["action"], pair, current_price, pnl,
                    exit_reason, bars_held,
                )
                open_position = None
                # Reduced cooldown: only 4 bars after SL (was 8)
                if hit_sl and exit_reason == "stop_loss" and portfolio.cooldown_bars < 4:
                    portfolio.cooldown_bars = max(portfolio.cooldown_bars, 4)

        if open_position:
            continue

        if portfolio.cooldown_bars > 0:
            portfolio.cooldown_bars -= 1

        # --- Get indicators for this bar (O(1) lookup) ---
        ind = _indicators_at(pre, i, current_price)

        # --- Strategy signals (O(1) from precomputed arrays) ---
        mom_sig = _momentum_signal(pre, df, i, cfg.momentum)
        mr_sig = _mean_reversion_signal(pre, df, i, cfg.mean_reversion)

        strategy_signals = {"momentum": mom_sig, "mean_reversion": mr_sig}

        # --- Rule-based scorer ---
        ai_sig = _ai_scorer(strategy_signals, ind)
        strategy_signals["ai_ensemble"] = ai_sig

        # --- Regime detection (O(1) from precomputed arrays) ---
        regime_info = _regime_at(pre, df, i)
        regime = regime_info["regime"]

        # --- Combine signals ---
        combined = combine_signals(
            strategy_signals,
            cfg,
            regime=regime,
            current_atr_norm=(ind.atr_norm_14 if ind is not None else None),
        )

        # Block counter-trend trades using HTF filter AND regime direction
        htf = regime_info.get("htf_trend", "neutral")
        if combined["action"] == "BUY" and htf == "bearish":
            logger.debug("BUY blocked by bearish HTF trend | conf=%.3f", combined["confidence"])
            combined["action"] = "HOLD"
        elif combined["action"] == "SELL" and htf == "bullish":
            logger.debug("SELL blocked by bullish HTF trend | conf=%.3f", combined["confidence"])
            combined["action"] = "HOLD"
        # Also block: buying in trending_down or selling in trending_up (regime alignment)
        elif combined["action"] == "BUY" and regime == "trending_down":
            logger.debug("BUY blocked by trending_down regime | conf=%.3f", combined["confidence"])
            combined["action"] = "HOLD"
        elif combined["action"] == "SELL" and regime == "trending_up":
            logger.debug("SELL blocked by trending_up regime | conf=%.3f", combined["confidence"])
            combined["action"] = "HOLD"

        logger.info(
            "Signal %s | regime=%s(%s) conf=%.3f mom=%s mr=%s adx=%.1f",
            combined["action"],
            regime,
            regime_info.get("trend_strength", "?"),
            combined["confidence"],
            mom_sig.get("signal", 0),
            mr_sig.get("signal", 0),
            regime_info.get("adx", 0.0),
        )

        # --- Risk check ---
        if combined["action"] in ("BUY", "SELL"):
            requested_size = portfolio.cash * 0.2
            risk_result = check_risk(
                signal=combined["action"],
                confidence=combined["confidence"],
                entry_price=current_price,
                requested_size=requested_size,
                portfolio=portfolio,
                cfg=cfg.risk,
                regime=regime,
                regime_cfg=cfg.regime,
                pre_ind=ind,
            )
        else:
            risk_result = RiskResult(
                approved=False, reasons=["Signal is HOLD -- no trade"],
                adjusted_size=0.0, stop_loss_price=0.0, take_profit_price=0.0,
            )

        # --- Build decision record ---
        decision_record = {
            "timestamp": timestamp,
            "pair": pair,
            "current_price": round(current_price, 8),
            "strategy_signals": {
                k: {"signal": v.get("signal"), "raw_strength": round(float(v.get("raw_strength", 0.0)), 4)}
                if isinstance(v, dict) and "raw_strength" in v else v
                for k, v in strategy_signals.items()
            },
            "ai_prediction": {
                "signal": ai_sig["signal"],
                "confidence": round(ai_sig["confidence"], 4),
                "reasoning": ai_sig.get("metadata", {}).get("reasoning", ""),
            },
            "combined_decision": combined,
            "indicators": (
                {"ema_spread": ind.ema_spread, "rsi": ind.rsi_14,
                 "macd_hist": ind.macd_hist, "atr": ind.atr_14, "atr_norm": ind.atr_norm_14}
                if ind is not None else {}
            ),
            "regime_detail": regime_info,
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
                "entry_bar": i,  # Track for time-based exits
            }

            logger.info(
                "Opened %s %s @ %g size=$%.0f (SL=%g TP=%g) regime=%s(%s)",
                combined["action"], pair, current_price, size,
                risk_result.stop_loss_price, risk_result.take_profit_price,
                regime, regime_info.get("trend_strength", "?"),
            )
        elif combined["action"] in ("BUY", "SELL"):
            logger.info(
                "Trade skipped | action=%s conf=%.3f reasons=%s",
                combined["action"],
                combined["confidence"],
                "; ".join(risk_result.reasons) if risk_result.reasons else "none",
            )

    # --- Close any remaining position at last price ---
    if open_position:
        final_price = float(df["close"].iloc[-1])
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

    # --- Summary with Sharpe ratio and max drawdown ---
    total_pnl = portfolio.total_value - cfg.portfolio.initial_balance
    win_count = sum(1 for t in trades if t.get("pnl", 0) > 0)

    # Sharpe ratio calculation
    if len(bar_returns) > 1:
        mean_ret = np.mean(bar_returns)
        std_ret = np.std(bar_returns, ddof=1)
        # Annualize: assume 6 bars/day (4h candles), 365 days
        bars_per_year = 6 * 365
        sharpe = (mean_ret / max(std_ret, 1e-10)) * np.sqrt(bars_per_year) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    # Average win/loss for profit factor
    wins = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
    losses = [abs(t["pnl"]) for t in trades if t.get("pnl", 0) < 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    profit_factor = sum(wins) / max(sum(losses), 1.0) if losses else float("inf")

    summary = {
        "initial_balance": cfg.portfolio.initial_balance,
        "final_value": round(portfolio.total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "pnl_pct": round(total_pnl / cfg.portfolio.initial_balance * 100, 2),
        "total_trades": len(trades),
        "winning_trades": win_count,
        "losing_trades": len(trades) - win_count,
        "win_rate": round(win_count / len(trades) * 100, 1) if trades else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "proof_hashes_generated": len(proof_hashes),
        "trades": trades,
    }

    logger.info("=" * 60)
    logger.info("PAPER TRADING COMPLETE")
    logger.info("  PnL: $%.2f (%.1f%%)", total_pnl, summary["pnl_pct"])
    logger.info("  Trades: %d (W:%d / L:%d) WR=%.1f%%", len(trades), win_count,
                len(trades) - win_count, summary["win_rate"])
    logger.info("  Avg Win: $%.0f | Avg Loss: $%.0f | PF: %.2f",
                avg_win, avg_loss, profit_factor)
    logger.info("  Sharpe: %.2f | Max DD: %.1f%%", sharpe, max_drawdown * 100)
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
        description="Balanced Hybrid AI Trading Agent -- Paper Trading Simulator"
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
    logger.info("Balanced Hybrid AI Trading Agent v3")
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
        if key != "trades":
            print(f"  {key:.<30} {val}")
    print("=" * 60)
    print(f"\nProof log: {CONFIG.proof_log_path}")
    print(f"Trade log: {TRADE_LOG}")
    print("Dashboard: streamlit run dashboard/dashboard.py")


if __name__ == "__main__":
    main()
