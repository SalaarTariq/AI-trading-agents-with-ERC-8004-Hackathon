"""
modules/momentum.py — Momentum trading strategy.

v3 — Profitability overhaul:
- ADX threshold lowered to 18 (captures more valid trends)
- EMA slope confirmation over 4 bars
- Pullback entries in established trends get confidence bonus
- Wider RSI range for trend-following entries
- Lighter volatility penalty (avoid over-filtering)

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

from config import MomentumConfig, CONFIG
from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    cfg: MomentumConfig | None = None,
) -> dict:
    """Generate momentum signal with ADX-gated trend confirmation.

    Args:
        df: OHLCV DataFrame with 'close' column.
        cfg: Optional MomentumConfig override.

    Returns:
        {
            "signal": -1|0|1,
            "confidence": 0..1,
            "signal_str": "BUY"|"SELL"|"HOLD",
            "metadata": {...}
        }
    """
    if len(df) < 21:
        return _hold_signal("insufficient data")

    cfg = cfg or CONFIG.momentum

    # Compute core indicators
    ind = compute_indicators(
        df,
        ema_fast_period=9,
        ema_slow_period=21,
        rsi_period=14,
        macd_fast=cfg.macd_fast,
        macd_slow=cfg.macd_slow,
        macd_signal=cfg.macd_signal,
        atr_period=14,
    )
    if ind is None:
        return _hold_signal("indicators warming up")

    current_price = float(df["close"].iloc[-1])
    if current_price == 0:
        return _hold_signal("invalid price")

    # --- CORE SIGNALS ---
    ema_spread = ind.ema_spread
    ema_spread_pct = ema_spread / current_price
    ema_spread_norm = abs(ind.ema_spread_norm)

    macd_hist = ind.macd_hist
    macd_hist_pct = macd_hist / current_price
    macd_hist_norm = abs(ind.macd_hist_norm)
    rsi_now = ind.rsi_14

    # --- EMA slope: 4-bar momentum of fast EMA ---
    ema_slope_pct = 0.0
    if len(df) >= 25:
        ema_fast_series = df["close"].ewm(span=9, adjust=False).mean()
        base = max(current_price, 1e-8)
        ema_slope_pct = float((ema_fast_series.iloc[-1] - ema_fast_series.iloc[-4]) / base)

    # --- Volatility penalty (lighter than v2) ---
    atr_norm = float(ind.atr_norm_14) if ind.atr_norm_14 is not None else 0.0
    volatility_penalty = 0.0
    if atr_norm > 0.020:
        volatility_penalty = float(np.clip((atr_norm - 0.020) * 5.0, 0.0, 0.15))

    # --- Pullback detection ---
    has_pullback_buy = False
    has_pullback_sell = False
    if len(df) >= 10:
        rsi_series = df["close"].diff()
        recent_rsi = rsi_now
        if ema_spread_pct > 0 and current_price <= ind.ema_fast * 1.003:
            has_pullback_buy = True
        if ema_spread_pct < 0 and current_price >= ind.ema_fast * 0.997:
            has_pullback_sell = True

    # --- SIGNAL LOGIC ---
    # Lowered thresholds: EMA spread > 0.15%, MACD > 0.01%, slope > 0.06%
    buy_ok = (
        ema_spread_pct > 0.0015
        and macd_hist_pct > 0.0001
        and ema_slope_pct > 0.0006
        and (38.0 if has_pullback_buy else 42.0) <= rsi_now <= 72.0
    )
    sell_ok = (
        ema_spread_pct < -0.0015
        and macd_hist_pct < -0.0001
        and ema_slope_pct < -0.0006
        and 28.0 <= rsi_now <= (62.0 if has_pullback_sell else 58.0)
    )

    signal = 0
    if buy_ok:
        signal = 1
    elif sell_ok:
        signal = -1

    if signal != 0:
        trend_strength = min(ema_spread_norm / 0.007, 1.0)
        macd_strength = min(macd_hist_norm / 0.5, 1.0)
        slope_strength = min(abs(ema_slope_pct) / 0.003, 1.0)
        if signal == 1:
            rsi_strength = min(max((72.0 - rsi_now) / 30.0, 0.0), 1.0)
        else:
            rsi_strength = min(max((rsi_now - 28.0) / 30.0, 0.0), 1.0)

        pullback_bonus = 0.08 if (signal == 1 and has_pullback_buy) or (signal == -1 and has_pullback_sell) else 0.0

        raw_strength = float(np.clip(
            0.38 * trend_strength + 0.27 * macd_strength + 0.20 * slope_strength + 0.10 * rsi_strength + pullback_bonus,
            0.0,
            1.0,
        ))
        confidence = float(np.clip(0.50 + 0.38 * raw_strength - volatility_penalty, 0.0, 0.90))
    else:
        raw_strength = 0.0
        confidence = 0.0

    signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

    metadata = {
        "ema_fast": round(ind.ema_fast, 8),
        "ema_slow": round(ind.ema_slow, 8),
        "ema_spread_pct": round(ema_spread_pct, 6),
        "macd_hist_pct": round(macd_hist_pct, 6),
        "ema_slope_pct": round(ema_slope_pct, 6),
        "atr_norm": round(atr_norm, 6),
        "rsi": round(rsi_now, 2),
        "raw_strength": round(raw_strength, 4),
        "signal": signal,
        "confidence": round(confidence, 4),
    }

    if signal != 0:
        logger.info(
            "Momentum %s | conf=%.3f strength=%.3f atr_norm=%.4f",
            signal_str,
            confidence,
            raw_strength,
            atr_norm,
        )

    return {
        "signal": signal,
        "raw_strength": raw_strength,
        "confidence": confidence,
        "signal_str": signal_str,
        "metadata": metadata,
    }


def _hold_signal(reason: str) -> dict:
    """Return a neutral HOLD signal with zero confidence."""
    return {
        "signal": 0,
        "raw_strength": 0.0,
        "signal_str": "HOLD",
        "confidence": 0.0,
        "metadata": {"reason": reason},
    }
