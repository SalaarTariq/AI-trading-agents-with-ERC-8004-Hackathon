"""
utils/indicators.py — Centralized indicator computation.

Per your spec, this module exposes ONE shared function that computes:
- EMA_fast (default 9)
- EMA_slow (default 21)
- EMA_spread = EMA_fast - EMA_slow
- RSI(14)
- MACD(12,26,9): line, signal, histogram
- ATR(14)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.helpers import atr, ema, macd, rsi, bollinger_bands


@dataclass(frozen=True)
class Indicators:
    ema_fast: float
    ema_slow: float
    ema_spread: float
    ema_spread_norm: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_hist_norm: float
    atr_14: float | None
    atr_norm_14: float | None
    bb_width: float | None


def compute_indicators(
    df: pd.DataFrame,
    *,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14,
) -> Indicators | None:
    """
    Compute a single, shared indicator bundle for the latest bar.

    Returns None when inputs are insufficient or values are not computable.
    """
    if df is None or len(df) < max(ema_slow_period, rsi_period, macd_slow, atr_period) + 2:
        return None
    close_s = df["close"]
    close = float(close_s.iloc[-1])
    if close == 0:
        return None

    ema_fast_s = ema(close_s, ema_fast_period)
    ema_slow_s = ema(close_s, ema_slow_period)
    ema_fast_v = float(ema_fast_s.iloc[-1])
    ema_slow_v = float(ema_slow_s.iloc[-1])
    if np.isnan(ema_fast_v) or np.isnan(ema_slow_v):
        return None

    rsi_s = rsi(close_s, rsi_period)
    rsi_v = float(rsi_s.iloc[-1])
    if np.isnan(rsi_v):
        return None

    macd_line_s, macd_signal_s, macd_hist_s = macd(close_s, macd_fast, macd_slow, macd_signal)
    macd_line_v = float(macd_line_s.iloc[-1])
    macd_signal_v = float(macd_signal_s.iloc[-1])
    macd_hist_v = float(macd_hist_s.iloc[-1])
    if np.isnan(macd_line_v) or np.isnan(macd_signal_v) or np.isnan(macd_hist_v):
        return None

    atr_v = None
    atr_norm_v = None
    if {"high", "low"}.issubset(df.columns):
        atr_s = atr(df["high"], df["low"], close_s, atr_period)
        atr_val = float(atr_s.iloc[-1])
        if not np.isnan(atr_val):
            atr_v = atr_val
            atr_norm_v = atr_val / close if close != 0 else None

    ema_spread_norm = float(np.tanh(float(ema_fast_v - ema_slow_v) / max(close * 0.01, 1e-12)))
    macd_hist_norm = float(np.tanh(macd_hist_v / max(close * 0.005, 1e-12)))

    mid, upper, lower = bollinger_bands(close_s, 20, 2.0)
    mid_v = float(mid.iloc[-1])
    upp_v = float(upper.iloc[-1])
    low_v = float(lower.iloc[-1])
    bb_width_v = float((upp_v - low_v) / mid_v) if mid_v > 0 and not np.isnan(mid_v) else 0.0

    return Indicators(
        ema_fast=ema_fast_v,
        ema_slow=ema_slow_v,
        ema_spread=float(ema_fast_v - ema_slow_v),
        ema_spread_norm=ema_spread_norm,
        rsi_14=float(rsi_v),
        macd_line=float(macd_line_v),
        macd_signal=float(macd_signal_v),
        macd_hist=float(macd_hist_v),
        macd_hist_norm=macd_hist_norm,
        atr_14=atr_v,
        atr_norm_14=atr_norm_v,
        bb_width=bb_width_v,
    )

