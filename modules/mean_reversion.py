"""
modules/mean_reversion.py — Mean-reversion trading strategy.


Identifies overbought/oversold conditions using Bollinger Bands,
z-score, RSI, and Stochastic Oscillator. Requires N-of-M condition
agreement and suppresses signals during strong trends (ADX guard).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MeanReversionConfig, CONFIG
from utils.helpers import (
    bollinger_bands, zscore, rsi, stochastic, adx, normalize_confidence,
)

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    cfg: MeanReversionConfig | None = None,
) -> dict:
    """
    Produce a mean-reversion signal for the most recent bar.

    Logic (2-of-4 by default):
        BUY conditions:
          1. Price < lower Bollinger Band
          2. Z-score < entry threshold
          3. RSI < oversold threshold
          4. Stochastic %K < stoch_oversold

        SELL conditions:
          1. Price > upper Bollinger Band
          2. Z-score > abs(entry threshold)
          3. RSI > overbought threshold
          4. Stochastic %K > stoch_overbought

        Signal fires when >= min_conditions are met.
        Suppressed when ADX > adx_max_threshold (strong trend).

    Args:
        df: OHLCV DataFrame with at least a `close` column.
        cfg: Optional MeanReversionConfig override.

    Returns:
        Standardized signal dict.
    """
    cfg = cfg or CONFIG.mean_reversion
    min_rows = cfg.lookback_period + 2

    if len(df) < min_rows:
        logger.warning("Not enough data for mean-reversion (%d rows, need %d)",
                       len(df), min_rows)
        return _hold_signal("insufficient data")

    close = df["close"]
    current_price = close.iloc[-1]

    # Indicators
    mid, upper, lower = bollinger_bands(close, cfg.lookback_period, cfg.bb_std_dev)
    z = zscore(close, cfg.lookback_period)
    rsi_vals = rsi(close, cfg.rsi_period)

    z_now = z.iloc[-1]
    rsi_now = rsi_vals.iloc[-1]
    bb_mid = mid.iloc[-1]
    bb_upper = upper.iloc[-1]
    bb_lower = lower.iloc[-1]

    # Handle NaN from indicator warm-up
    if np.isnan(z_now) or np.isnan(rsi_now):
        return _hold_signal("indicators warming up")

    # Stochastic (optional — requires high/low)
    stoch_k_now = np.nan
    stoch_d_now = np.nan
    if {"high", "low"}.issubset(df.columns) and len(df) >= cfg.stoch_k_period + cfg.stoch_d_period:
        stoch_k, stoch_d = stochastic(
            df["high"], df["low"], close, cfg.stoch_k_period, cfg.stoch_d_period
        )
        stoch_k_now = stoch_k.iloc[-1]
        stoch_d_now = stoch_d.iloc[-1]

    # ADX guard
    adx_val = np.nan
    if {"high", "low"}.issubset(df.columns) and len(df) >= cfg.adx_period + 2:
        adx_series = adx(df["high"], df["low"], close, cfg.adx_period)
        adx_val = adx_series.iloc[-1]

    adx_blocked = not np.isnan(adx_val) and adx_val > cfg.adx_max_threshold

    # --- Count BUY conditions ---
    buy_conditions = 0
    if current_price < bb_lower:
        buy_conditions += 1
    if z_now < cfg.zscore_entry:
        buy_conditions += 1
    if rsi_now < cfg.rsi_oversold:
        buy_conditions += 1
    if not np.isnan(stoch_k_now) and stoch_k_now < cfg.stoch_oversold:
        buy_conditions += 1

    # --- Count SELL conditions ---
    sell_conditions = 0
    if current_price > bb_upper:
        sell_conditions += 1
    if z_now > abs(cfg.zscore_entry):
        sell_conditions += 1
    if rsi_now > cfg.rsi_overbought:
        sell_conditions += 1
    if not np.isnan(stoch_k_now) and stoch_k_now > cfg.stoch_overbought:
        sell_conditions += 1

    # --- Signal logic: N-of-M ---
    direction = "HOLD"
    conditions_met = 0

    if buy_conditions >= cfg.min_conditions:
        direction = "BUY"
        conditions_met = buy_conditions
    elif sell_conditions >= cfg.min_conditions:
        direction = "SELL"
        conditions_met = sell_conditions

    # ADX guard: suppress in strong trends
    if direction != "HOLD" and adx_blocked:
        logger.debug("MeanReversion %s suppressed — ADX %.1f > %.1f (strong trend)",
                      direction, adx_val, cfg.adx_max_threshold)
        direction = "HOLD"

    # Confidence: how extreme the deviation is (z-score depth + RSI extremity)
    # Bonus for more conditions met
    z_strength = min(abs(z_now) / 3.0, 1.0)
    if direction == "BUY":
        rsi_strength = max((cfg.rsi_oversold - rsi_now) / cfg.rsi_oversold, 0.0)
    elif direction == "SELL":
        rsi_strength = max((rsi_now - cfg.rsi_overbought) / (100 - cfg.rsi_overbought), 0.0)
    else:
        rsi_strength = 0.0

    # More conditions = higher confidence
    condition_bonus = min(conditions_met / 4.0, 1.0) if conditions_met > 0 else 0.0

    confidence = normalize_confidence(
        (z_strength * 0.35 + rsi_strength * 0.35 + condition_bonus * 0.30)
    )

    metadata = {
        "price": round(current_price, 4),
        "bb_upper": round(bb_upper, 4),
        "bb_mid": round(bb_mid, 4),
        "bb_lower": round(bb_lower, 4),
        "zscore": round(z_now, 4),
        "rsi": round(rsi_now, 4),
        "stoch_k": round(stoch_k_now, 4) if not np.isnan(stoch_k_now) else None,
        "stoch_d": round(stoch_d_now, 4) if not np.isnan(stoch_d_now) else None,
        "adx": round(adx_val, 2) if not np.isnan(adx_val) else None,
        "adx_blocked": adx_blocked,
        "buy_conditions_met": buy_conditions,
        "sell_conditions_met": sell_conditions,
        "min_conditions_required": cfg.min_conditions,
    }

    logger.info(
        "MeanReversion signal=%s confidence=%.3f z=%.3f rsi=%.1f "
        "stoch_k=%.1f adx=%.1f conds=%d/%d price=%.2f [%.2f / %.2f]",
        direction, confidence, z_now, rsi_now,
        stoch_k_now if not np.isnan(stoch_k_now) else 0.0,
        adx_val if not np.isnan(adx_val) else 0.0,
        conditions_met, cfg.min_conditions,
        current_price, bb_lower, bb_upper,
    )

    return {
        "signal": direction,
        "confidence": confidence,
        "metadata": metadata,
    }


def _hold_signal(reason: str) -> dict:
    """Return a neutral HOLD signal."""
    return {
        "signal": "HOLD",
        "confidence": 0.0,
        "metadata": {"reason": reason},
    }
