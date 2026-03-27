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

from utils.helpers import adx, atr, ema, macd, rsi, bollinger_bands


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


def precompute_all_indicators(
    df: pd.DataFrame,
    *,
    ema_fast_period: int = 9,
    ema_slow_period: int = 21,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """
    Pre-compute all indicator series on the full DataFrame at once.

    Returns a DataFrame with the same index as `df` plus indicator columns.
    Values before the warm-up period will be NaN.
    """
    close_s = df["close"]

    ema_fast_s = ema(close_s, ema_fast_period)
    ema_slow_s = ema(close_s, ema_slow_period)
    ema_spread_s = ema_fast_s - ema_slow_s
    rsi_s = rsi(close_s, rsi_period)
    macd_line_s, macd_sig_s, macd_hist_s = macd(close_s, macd_fast, macd_slow, macd_signal_period)
    mid_s, upper_s, lower_s = bollinger_bands(close_s, bb_period, bb_std)

    # EMA5/EMA13 for momentum fast crossover
    ema5_s = ema(close_s, 5)
    ema13_s = ema(close_s, 13)

    # EMA50 for HTF trend
    ema50_s = ema(close_s, 50)

    result = pd.DataFrame({
        "ema_fast": ema_fast_s,
        "ema_slow": ema_slow_s,
        "ema_spread": ema_spread_s,
        "rsi_14": rsi_s,
        "macd_line": macd_line_s,
        "macd_signal": macd_sig_s,
        "macd_hist": macd_hist_s,
        "bb_mid": mid_s,
        "bb_upper": upper_s,
        "bb_lower": lower_s,
        "ema5": ema5_s,
        "ema13": ema13_s,
        "ema50": ema50_s,
    }, index=df.index)

    # ATR + ADX
    if {"high", "low"}.issubset(df.columns):
        atr_s = atr(df["high"], df["low"], close_s, atr_period)
        result["atr_14"] = atr_s
        result["atr_norm_14"] = atr_s / close_s
        result["adx_14"] = adx(df["high"], df["low"], close_s, 14)
    else:
        result["atr_14"] = np.nan
        result["atr_norm_14"] = np.nan
        result["adx_14"] = np.nan

    # RSI rolling lookbacks for pullback detection
    result["rsi_14_high_5"] = result["rsi_14"].rolling(5, min_periods=1).max()
    result["rsi_14_low_5"] = result["rsi_14"].rolling(5, min_periods=1).min()

    # Close vs EMA9 distance (proximity-based entries)
    result["close_vs_ema9"] = (close_s - ema_fast_s) / close_s.clip(lower=1e-12)

    # MACD crossover detection: 1 = bullish cross, -1 = bearish, 0 = none
    macd_sign = np.sign(macd_hist_s)
    macd_sign_prev = macd_sign.shift(1)
    result["macd_cross"] = 0
    result.loc[(macd_sign > 0) & (macd_sign_prev <= 0), "macd_cross"] = 1
    result.loc[(macd_sign < 0) & (macd_sign_prev >= 0), "macd_cross"] = -1

    # ADX slope: 3-bar change in ADX (rising = strengthening trend)
    if "adx_14" in result.columns:
        result["adx_slope"] = result["adx_14"] - result["adx_14"].shift(3)
    else:
        result["adx_slope"] = 0.0

    # RSI 3-bar change (reversion velocity for MR)
    result["rsi_delta_3"] = result["rsi_14"] - result["rsi_14"].shift(3)

    # Volume SMA for regime detection
    if "volume" in df.columns:
        from utils.helpers import sma as _sma
        result["vol_sma20"] = _sma(df["volume"], 20)

    # Normalized indicators
    result["ema_spread_norm"] = np.tanh(ema_spread_s / (close_s * 0.01).clip(lower=1e-12))
    result["macd_hist_norm"] = np.tanh(macd_hist_s / (close_s * 0.005).clip(lower=1e-12))

    # BB width
    result["bb_width"] = (upper_s - lower_s) / mid_s.replace(0, np.nan)

    return result

