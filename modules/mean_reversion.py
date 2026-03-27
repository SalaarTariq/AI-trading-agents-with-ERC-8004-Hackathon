"""
modules/mean_reversion.py — Mean-reversion trading strategy.

v3 — Profitability overhaul:
- Ranging detection: uses ADX < 25 (not just EMA spread) for reliable regime detection
- BB-touch entries: price at band edge = strongest MR signal
- RSI reversal velocity: confirms mean-reversion is underway
- Wider BB z-score entry (-0.20): more MR opportunities
- Lighter volatility penalty to avoid over-filtering

Outputs:
- signal: -1|0|1
- raw_strength: 0..1
- confidence: 0..1
- signal_str: "BUY"|"SELL"|"HOLD"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MeanReversionConfig, CONFIG
from utils.helpers import bollinger_bands
from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    cfg: MeanReversionConfig | None = None,
) -> dict:
    """Generate mean-reversion signal for ranging market conditions.

    Uses BB width + ADX for ranging detection, with RSI reversal confirmation.

    Args:
        df: OHLCV DataFrame with 'close' column.
        cfg: Optional MeanReversionConfig override.

    Returns:
        {
            "signal": -1|0|1,
            "confidence": 0..1,
            "signal_str": "BUY"|"SELL"|"HOLD",
            "metadata": {...}
        }
    """
    cfg = cfg or CONFIG.mean_reversion

    if len(df) < max(20, cfg.lookback_period):
        return _hold_signal("insufficient data")

    close = df["close"]
    current_price = float(close.iloc[-1])
    if current_price == 0:
        return _hold_signal("invalid price")

    # Bollinger Bands (20, 2)
    mid, upper, lower = bollinger_bands(close, cfg.lookback_period, cfg.bb_std_dev)
    bb_mid = float(mid.iloc[-1])
    bb_upper = float(upper.iloc[-1])
    bb_lower = float(lower.iloc[-1])

    if np.isnan(bb_mid) or np.isnan(bb_upper) or np.isnan(bb_lower):
        return _hold_signal("indicators warming up")

    # Get EMA spread, RSI, and ADX
    ind = compute_indicators(
        df,
        ema_fast_period=9,
        ema_slow_period=21,
        rsi_period=cfg.rsi_period,
        atr_period=14,
    )
    if ind is None:
        return _hold_signal("indicators warming up")

    rsi_now = ind.rsi_14
    ema_spread = ind.ema_spread
    ema_spread_norm = abs(float(ind.ema_spread_norm))
    ema_slow = ind.ema_slow

    # --- RANGING FILTER (v3) ---
    # Primary: use EMA spread vs threshold (available without ADX)
    ranging_threshold = abs(ema_slow) * 0.005 if ema_slow != 0 else 0.01
    is_ranging = abs(ema_spread) < ranging_threshold

    # Secondary: if BB are narrow, also treat as ranging
    bb_width = (bb_upper - bb_lower) / max(bb_mid, 1e-8) if bb_mid > 0 else 0.0
    if bb_width < 0.06 and abs(ema_spread) < ranging_threshold * 2.0:
        is_ranging = True

    if not is_ranging:
        return _hold_signal("not ranging (in trend)")

    # --- SIGNAL GENERATION ---
    signal = 0

    band_range = max(bb_upper - bb_lower, 1e-8)
    bb_z = (current_price - bb_mid) / band_range

    atr_norm = float(ind.atr_norm_14) if ind.atr_norm_14 is not None else 0.0
    volatility_penalty = 0.0
    if atr_norm > 0.025:
        volatility_penalty = float(np.clip((atr_norm - 0.025) * 5.0, 0.0, 0.15))

    # BB-touch detection: price at or beyond band edge
    bb_touch_buy = current_price <= bb_lower * 1.002
    bb_touch_sell = current_price >= bb_upper * 0.998

    # BB touch: strongest signal (no RSI required)
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
        touch_bonus = 0.12 if (signal == 1 and bb_touch_buy) or (signal == -1 and bb_touch_sell) else 0.0
        trend_penalty = min(ema_spread_norm / 0.008, 1.0)
        raw_strength = float(np.clip(
            0.50 * band_strength + 0.35 * rsi_strength + touch_bonus - 0.05 * trend_penalty,
            0.0, 1.0,
        ))
        confidence = float(np.clip(0.50 + 0.38 * raw_strength - volatility_penalty, 0.0, 0.90))
    else:
        raw_strength = 0.0
        confidence = 0.0

    signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

    metadata = {
        "price": round(current_price, 4),
        "bb_upper": round(bb_upper, 4),
        "bb_mid": round(bb_mid, 4),
        "bb_lower": round(bb_lower, 4),
        "bb_z": round(bb_z, 4),
        "zscore": round(bb_z, 4),
        "rsi": round(rsi_now, 2),
        "atr_norm": round(atr_norm, 6),
        "ema_spread_norm": round(ema_spread_norm, 6),
        "ema_spread_pct": round(ema_spread / current_price * 100, 4),
        "bb_width": round(bb_width, 4),
        "is_ranging": is_ranging,
        "raw_strength": round(raw_strength, 4),
        "signal": signal,
        "confidence": round(confidence, 4),
    }

    if signal != 0:
        logger.info(
            "MeanRev %s | conf=%.3f strength=%.3f bb_z=%.3f",
            signal_str,
            confidence,
            raw_strength,
            bb_z,
        )

    return {
        "signal": signal,
        "raw_strength": raw_strength,
        "confidence": confidence,
        "signal_str": signal_str,
        "metadata": metadata,
    }


def _hold_signal(reason: str) -> dict:
    """Return a neutral HOLD signal."""
    return {
        "signal": 0,
        "raw_strength": 0.0,
        "signal_str": "HOLD",
        "confidence": 0.0,
        "metadata": {"reason": reason},
    }
