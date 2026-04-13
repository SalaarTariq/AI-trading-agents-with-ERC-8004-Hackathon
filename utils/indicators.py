"""Shared indicator calculations for the cleaned trading pipeline.

All indicators are computed once per dataset in `precompute_all_indicators`,
which avoids repeated per-bar recomputation and keeps strategy logic fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import StrategyConfig, CONFIG
from utils.helpers import adx, atr, bollinger_bands, ema, macd, rsi


@dataclass(frozen=True)
class Indicators:
    """Latest-bar indicator bundle consumed by strategy and risk modules."""

    ema_fast: float
    ema_slow: float
    ema_spread: float
    ema_spread_pct: float
    rsi_14: float
    macd_hist: float
    atr_14: float | None
    atr_norm_14: float | None
    bb_mid: float
    bb_upper: float
    bb_lower: float
    bb_zscore: float
    adx_14: float


def precompute_all_indicators(
    df: pd.DataFrame,
    cfg: StrategyConfig | None = None,
) -> pd.DataFrame:
    """Precompute all indicators needed by the strategy in one pass."""
    cfg = cfg or CONFIG.strategy

    close = df["close"]
    ema_fast_s = ema(close, cfg.ema_fast_period)
    ema_slow_s = ema(close, cfg.ema_slow_period)
    ema_spread_s = ema_fast_s - ema_slow_s

    rsi_s = rsi(close, cfg.rsi_period)
    _, _, macd_hist_s = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    bb_mid, bb_upper, bb_lower = bollinger_bands(close, cfg.bb_period, cfg.bb_std_dev)

    out = pd.DataFrame(index=df.index)
    out["ema_fast"] = ema_fast_s
    out["ema_slow"] = ema_slow_s
    out["ema_spread"] = ema_spread_s
    out["ema_spread_pct"] = ema_spread_s / close.replace(0, np.nan)
    out["rsi_14"] = rsi_s
    out["macd_hist"] = macd_hist_s

    if {"high", "low"}.issubset(df.columns):
        atr_s = atr(df["high"], df["low"], close, cfg.atr_period)
        adx_s = adx(df["high"], df["low"], close, cfg.adx_period)
        out["atr_14"] = atr_s
        out["atr_norm_14"] = atr_s / close.replace(0, np.nan)
        out["adx_14"] = adx_s
    else:
        out["atr_14"] = np.nan
        out["atr_norm_14"] = np.nan
        out["adx_14"] = np.nan

    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    # Convert BB channel to z-score proxy using std implied by BB width.
    bb_std = (bb_upper - bb_lower) / (2.0 * cfg.bb_std_dev)
    out["bb_zscore"] = (close - bb_mid) / bb_std.replace(0, np.nan)

    # Used by confidence scoring for top-20% volatility penalties.
    out["atr_norm_q80_120"] = out["atr_norm_14"].rolling(120, min_periods=30).quantile(0.8)

    # Reversal velocity input for mean-reversion confidence boost.
    out["rsi_delta_3"] = out["rsi_14"] - out["rsi_14"].shift(3)

    return out


def indicators_at(pre: pd.DataFrame, idx: int) -> Indicators | None:
    """Build an Indicators object from precomputed rows at index `idx`."""
    row = pre.iloc[idx]

    required = ["ema_fast", "ema_slow", "ema_spread", "ema_spread_pct", "rsi_14", "macd_hist", "bb_mid", "bb_upper", "bb_lower", "bb_zscore"]
    if any(np.isnan(row.get(col, np.nan)) for col in required):
        return None

    atr_val = None if np.isnan(row.get("atr_14", np.nan)) else float(row["atr_14"])
    atr_norm = None if np.isnan(row.get("atr_norm_14", np.nan)) else float(row["atr_norm_14"])
    adx_val = 0.0 if np.isnan(row.get("adx_14", np.nan)) else float(row["adx_14"])

    return Indicators(
        ema_fast=float(row["ema_fast"]),
        ema_slow=float(row["ema_slow"]),
        ema_spread=float(row["ema_spread"]),
        ema_spread_pct=float(row["ema_spread_pct"]),
        rsi_14=float(row["rsi_14"]),
        macd_hist=float(row["macd_hist"]),
        atr_14=atr_val,
        atr_norm_14=atr_norm,
        bb_mid=float(row["bb_mid"]),
        bb_upper=float(row["bb_upper"]),
        bb_lower=float(row["bb_lower"]),
        bb_zscore=float(row["bb_zscore"]),
        adx_14=adx_val,
    )
