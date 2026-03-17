"""
modules/momentum.py — Momentum trading strategy.


Generates BUY/SELL/HOLD signals based on moving-average crossovers
confirmed by MACD, ADX trend strength, and above-average volume.
Returns a standardized signal dict with a confidence score in [0, 1].
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MomentumConfig, CONFIG
from utils.helpers import ema, sma, macd, adx, normalize_confidence

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    cfg: MomentumConfig | None = None,
) -> dict:
    """
    Produce a momentum signal for the most recent bar.

    Logic:
        - BUY when fast EMA crosses above slow EMA, MACD histogram > 0,
          ADX > threshold, and volume > avg volume.
        - SELL when fast EMA crosses below slow EMA, MACD histogram < 0,
          ADX > threshold, and volume > avg volume.
        - HOLD otherwise.

    Confidence formula:
        (crossover_strength * 0.3) + (adx_strength * 0.3)
        + (macd_strength * 0.2) + (volume_boost * 0.2)

    Args:
        df: OHLCV DataFrame (needs at least `close` and `volume` columns).
        cfg: Optional MomentumConfig override.

    Returns:
        {
            "signal": "BUY" | "SELL" | "HOLD",
            "confidence": float,
            "metadata": { ... }
        }
    """
    cfg = cfg or CONFIG.momentum

    if len(df) < cfg.slow_period + 2:
        logger.warning("Not enough data for momentum (%d rows, need %d)",
                       len(df), cfg.slow_period + 2)
        return _hold_signal("insufficient data")

    close = df["close"]
    volume = df["volume"]

    # Compute EMAs
    fast = ema(close, cfg.fast_period)
    slow = ema(close, cfg.slow_period)
    vol_ma = sma(volume, cfg.volume_ma_period)

    # Current and previous crossover values
    spread_now = (fast.iloc[-1] - slow.iloc[-1]) / slow.iloc[-1]
    spread_prev = (fast.iloc[-2] - slow.iloc[-2]) / slow.iloc[-2]
    vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 0.0

    volume_confirmed = vol_ratio > 1.0

    # --- MACD confirmation ---
    macd_line, signal_line, histogram = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    hist_now = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0.0
    macd_bullish = hist_now > 0
    macd_bearish = hist_now < 0

    # --- ADX trend strength filter ---
    adx_val = np.nan
    if {"high", "low"}.issubset(df.columns) and len(df) >= cfg.adx_period + 2:
        adx_series = adx(df["high"], df["low"], close, cfg.adx_period)
        adx_val = adx_series.iloc[-1]

    adx_strong = not np.isnan(adx_val) and adx_val >= cfg.adx_threshold
    adx_weak = not np.isnan(adx_val) and adx_val < cfg.adx_weak_threshold

    # Determine signal direction
    if spread_now > cfg.signal_threshold and spread_prev <= cfg.signal_threshold:
        direction = "BUY"
    elif spread_now < -cfg.signal_threshold and spread_prev >= -cfg.signal_threshold:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Require volume confirmation for actionable signals
    if direction != "HOLD" and not volume_confirmed:
        logger.debug("Momentum %s signal suppressed — low volume (ratio %.2f)",
                      direction, vol_ratio)
        direction = "HOLD"

    # MACD must confirm direction
    if direction == "BUY" and not macd_bullish:
        logger.debug("Momentum BUY suppressed — MACD histogram negative (%.4f)", hist_now)
        direction = "HOLD"
    elif direction == "SELL" and not macd_bearish:
        logger.debug("Momentum SELL suppressed — MACD histogram positive (%.4f)", hist_now)
        direction = "HOLD"

    # ADX filter: suppress in weak/no trend
    if direction != "HOLD" and adx_weak:
        logger.debug("Momentum %s suppressed — ADX too low (%.1f < %.1f)",
                      direction, adx_val, cfg.adx_weak_threshold)
        direction = "HOLD"

    # Confidence calculation
    crossover_strength = min(abs(spread_now) / (cfg.signal_threshold * 5), 1.0)
    adx_strength = min(adx_val / 50.0, 1.0) if not np.isnan(adx_val) else 0.3
    macd_strength = min(abs(hist_now) / (close.iloc[-1] * 0.005), 1.0)
    volume_boost = min(vol_ratio / 2.0, 1.0) if volume_confirmed else 0.0

    raw_confidence = (
        crossover_strength * 0.3
        + adx_strength * 0.3
        + macd_strength * 0.2
        + volume_boost * 0.2
    )

    # Reduce confidence when ADX is in the weak zone (20-25)
    if not np.isnan(adx_val) and cfg.adx_weak_threshold <= adx_val < cfg.adx_threshold:
        raw_confidence *= 0.5

    confidence = normalize_confidence(raw_confidence)

    metadata = {
        "fast_ema": round(fast.iloc[-1], 4),
        "slow_ema": round(slow.iloc[-1], 4),
        "spread": round(spread_now, 6),
        "spread_prev": round(spread_prev, 6),
        "volume_ratio": round(vol_ratio, 4),
        "volume_confirmed": volume_confirmed,
        "macd_histogram": round(hist_now, 4),
        "macd_bullish": macd_bullish,
        "adx": round(adx_val, 2) if not np.isnan(adx_val) else None,
        "adx_strong": adx_strong,
    }

    logger.info("Momentum signal=%s confidence=%.3f spread=%.5f vol=%.2f macd_hist=%.4f adx=%.1f",
                direction, confidence, spread_now, vol_ratio, hist_now,
                adx_val if not np.isnan(adx_val) else 0.0)

    return {
        "signal": direction,
        "confidence": confidence,
        "metadata": metadata,
    }


def _hold_signal(reason: str) -> dict:
    """Return a neutral HOLD signal with zero confidence."""
    return {
        "signal": "HOLD",
        "confidence": 0.0,
        "metadata": {"reason": reason},
    }
