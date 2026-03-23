"""
utils/regime.py — Strengthened market regime detection.

Uses EMA9-EMA21 normalized spread + ATR14 normalized + volume filter
to classify the market as trending_up, trending_down, or choppy.

Includes:
- Multi-timeframe confirmation (simulated 1h EMA from bar data)
- Volume/liquidity filter (skip low-volume environments)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.helpers import ema, atr, sma

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeResult:
    """Result of regime detection."""
    regime: str                # "trending_up", "trending_down", "choppy"
    ema_spread_norm: float     # Normalized EMA spread
    atr_norm: float            # ATR / price
    volume_ratio: float        # Current vol / 20-bar avg vol
    htf_trend: str             # Higher TF trend: "up", "down", "neutral"
    volume_ok: bool            # Passes liquidity filter
    htf_confirms: bool         # Higher TF confirms signal direction


# ---------------------------------------------------------------------------
# Thresholds — tuned for real crypto DEX data (4h bars, high vol)
# ---------------------------------------------------------------------------
SPREAD_THRESHOLD = 0.004       # |norm spread| > 0.4% = trending
ATR_NORM_CAP = 0.08            # ATR/price < 8% = allow trends in real crypto
VOLUME_MIN_RATIO = 0.50        # Current vol must be >= 50% of 20-bar avg
HTF_EMA_PERIOD = 50            # Simulated higher-TF EMA (50-bar ~ 1h from 5m)


def detect_regime(
    df: pd.DataFrame,
    spread_threshold: float = SPREAD_THRESHOLD,
    atr_cap: float = ATR_NORM_CAP,
    volume_min_ratio: float = VOLUME_MIN_RATIO,
) -> RegimeResult:
    """
    Detect current market regime.

    Logic:
        1. Compute EMA9-EMA21 spread, normalize by price
        2. Compute ATR14, normalize by price
        3. Compute volume ratio (current / 20-bar SMA)
        4. Compute higher-TF EMA50 trend direction
        5. Classify:
           - |spread| > threshold AND atr_norm < cap AND volume_ok → trending
           - Otherwise → choppy

    Args:
        df: OHLCV DataFrame with at least 'close' column.
        spread_threshold: Min |spread| to classify as trending.
        atr_cap: Max ATR norm to allow trend classification.
        volume_min_ratio: Min volume ratio to pass liquidity filter.

    Returns:
        RegimeResult with regime classification and supporting metrics.
    """
    if len(df) < 30 or "close" not in df.columns:
        return _default_result()

    close = df["close"]
    current_price = float(close.iloc[-1])
    if current_price == 0:
        return _default_result()

    # --- EMA spread (normalized) ---
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema9_v = float(ema9.iloc[-1])
    ema21_v = float(ema21.iloc[-1])

    if np.isnan(ema9_v) or np.isnan(ema21_v):
        return _default_result()

    spread_norm = (ema9_v - ema21_v) / current_price

    # --- ATR normalized ---
    atr_norm = 0.0
    if {"high", "low"}.issubset(df.columns):
        atr_series = atr(df["high"], df["low"], close, 14)
        atr_val = float(atr_series.iloc[-1])
        if not np.isnan(atr_val):
            atr_norm = atr_val / current_price

    # --- Volume filter ---
    volume_ratio = 1.0
    volume_ok = True
    if "volume" in df.columns and len(df) >= 20:
        vol = df["volume"]
        current_vol = float(vol.iloc[-1])
        avg_vol = float(sma(vol, 20).iloc[-1])
        if avg_vol > 0 and not np.isnan(avg_vol):
            volume_ratio = current_vol / avg_vol
            volume_ok = volume_ratio >= volume_min_ratio
        else:
            volume_ok = True  # can't compute, don't block

    # --- Higher-TF trend (simulated via EMA50) ---
    htf_trend = "neutral"
    htf_confirms = True  # default: don't block if insufficient data
    if len(close) >= HTF_EMA_PERIOD + 5:
        ema50 = ema(close, HTF_EMA_PERIOD)
        ema50_v = float(ema50.iloc[-1])
        ema50_prev = float(ema50.iloc[-5])  # slope over last 5 bars
        if not np.isnan(ema50_v) and not np.isnan(ema50_prev):
            slope = (ema50_v - ema50_prev) / max(current_price, 1e-12)
            # Use ATR-adaptive slope threshold (small for low-vol, larger for high-vol)
            slope_threshold = max(0.0005, atr_norm * 0.02)
            if slope > slope_threshold:
                htf_trend = "up"
            elif slope < -slope_threshold:
                htf_trend = "down"
            else:
                htf_trend = "neutral"

    # --- Adaptive ATR cap: use rolling median ATR for extreme-vol assets ---
    effective_atr_cap = atr_cap
    if {"high", "low"}.issubset(df.columns) and len(df) >= 30:
        atr_series = atr(df["high"], df["low"], close, 14)
        atr_norm_series = atr_series / close
        atr_norm_series = atr_norm_series.dropna()
        if len(atr_norm_series) >= 20:
            rolling_median = float(atr_norm_series.iloc[-20:].median())
            # If the asset's typical volatility exceeds the static cap,
            # adapt the cap to 2× the rolling median (don't block everything)
            if rolling_median > atr_cap * 0.7:
                effective_atr_cap = rolling_median * 2.0

    # --- Regime classification ---
    abs_spread = abs(spread_norm)

    if abs_spread > spread_threshold and atr_norm < effective_atr_cap and volume_ok:
        if spread_norm > 0:
            regime = "trending_up"
            htf_confirms = htf_trend in ("up", "neutral")
        else:
            regime = "trending_down"
            htf_confirms = htf_trend in ("down", "neutral")
    else:
        regime = "choppy"
        htf_confirms = True  # irrelevant for choppy

    result = RegimeResult(
        regime=regime,
        ema_spread_norm=round(spread_norm, 6),
        atr_norm=round(atr_norm, 6),
        volume_ratio=round(volume_ratio, 4),
        htf_trend=htf_trend,
        volume_ok=volume_ok,
        htf_confirms=htf_confirms,
    )

    logger.debug(
        "Regime=%s spread=%.4f atr_norm=%.4f vol_ratio=%.2f htf=%s confirms=%s",
        result.regime, spread_norm, atr_norm, volume_ratio,
        htf_trend, htf_confirms,
    )

    return result


def should_skip_trade(regime_result: RegimeResult) -> tuple[bool, str]:
    """
    Check if a trade should be skipped based on regime filters.

    Returns:
        (skip: bool, reason: str)
    """
    if not regime_result.volume_ok:
        return True, f"Volume too low: ratio={regime_result.volume_ratio:.2f} < {VOLUME_MIN_RATIO}"

    if not regime_result.htf_confirms and regime_result.regime != "choppy":
        return True, (
            f"HTF trend ({regime_result.htf_trend}) does not confirm "
            f"regime ({regime_result.regime})"
        )

    return False, ""


def _default_result() -> RegimeResult:
    """Return a safe default (choppy) when data is insufficient."""
    return RegimeResult(
        regime="choppy",
        ema_spread_norm=0.0,
        atr_norm=0.0,
        volume_ratio=1.0,
        htf_trend="neutral",
        volume_ok=True,
        htf_confirms=True,
    )
