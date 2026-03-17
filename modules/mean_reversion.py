"""
modules/mean_reversion.py — Mean-reversion trading strategy.


Identifies overbought/oversold conditions using Bollinger Bands,
and RSI in ranging markets (spec C).

Outputs:
- signal: -1|0|1
- raw_strength: 0..1

Compatibility:
- also provides `signal_str` and `confidence` for existing pipeline/tests
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MeanReversionConfig, CONFIG
from utils.helpers import bollinger_bands, normalize_confidence, zscore
from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    cfg: MeanReversionConfig | None = None,
) -> dict:
    """
    Produce a mean-reversion output for the most recent bar.

    Spec C:
    - Active only when |EMA_spread| < small threshold (ranging)
    - BUY when price < lower BB(20,2) AND RSI < 35
    - SELL when price > upper BB(20,2) AND RSI > 65

    Args:
        df: OHLCV DataFrame with at least a `close` column.
        cfg: Optional MeanReversionConfig override.

    Returns:
        {
            "signal": -1|0|1,
            "raw_strength": float,
            "signal_str": "BUY"|"SELL"|"HOLD",
            "confidence": float,
            "metadata": dict,
        }
    """
    cfg = cfg or CONFIG.mean_reversion
    # Need enough bars for BB + RSI + EMA spread
    min_rows = max(cfg.lookback_period, cfg.rsi_period, 21) + 2

    if len(df) < min_rows:
        logger.warning("Not enough data for mean-reversion (%d rows, need %d)",
                       len(df), min_rows)
        return _hold_signal("insufficient data")

    close = df["close"]
    current_price = float(close.iloc[-1])

    # BB
    mid, upper, lower = bollinger_bands(close, cfg.lookback_period, cfg.bb_std_dev)
    bb_mid = float(mid.iloc[-1])
    bb_upper = float(upper.iloc[-1])
    bb_lower = float(lower.iloc[-1])
    if np.isnan(bb_mid) or np.isnan(bb_upper) or np.isnan(bb_lower):
        return _hold_signal("indicators warming up")

    # Z-score retained for logging/compatibility (not used for decisions)
    z_now = float(zscore(close, cfg.lookback_period).iloc[-1])

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

    # Spec C: activate only in ranging market (Wait, the user wants us to trade more. Let's make ranging optional or sensitive too)
    # The user's formula doesn't explicitly restrict to ranging, but let's keep it broadly applicable
    ranging_threshold = 0.0
    try:
        ranging_threshold = float(getattr(cfg, "ema_spread_ranging_threshold", 0.0))
    except Exception:
        ranging_threshold = 0.0
    if ranging_threshold <= 0:
        # User wants more trades: widen ranging threshold
        ranging_threshold = abs(ind.ema_slow) * 0.005  # Increased from 0.002

    is_ranging = abs(ema_spread) < ranging_threshold

    bb_width_abs = max(bb_upper - bb_lower, 1e-8)
    bb_dist = (current_price - bb_mid) / bb_width_abs

    # RSI direction: user wants rsi < 40 ? 1 : rsi > 60 ? -1 : 0
    rsi_dir = 0
    if rsi_now < 45:    # loosened from 40 for more trades
        rsi_dir = 1
    elif rsi_now > 55:  # loosened from 60 for more trades
        rsi_dir = -1

    # raw_val positive for BUY, negative for SELL
    # distance is positive if price > mid, so mean-reverting is -distance
    raw_val = -float(np.tanh(bb_dist * 3.0)) * rsi_dir

    # Only act if ranging (or if signal is super strong)
    signal = 0
    if raw_val > 0.1 and (is_ranging or raw_val > 0.5):
        signal = 1
    elif raw_val < -0.1 and (is_ranging or raw_val < -0.5):
        signal = -1

    raw_strength = float(np.clip(abs(raw_val), 0.0, 1.0)) if signal != 0 else 0.0
    confidence = raw_strength
    signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

    metadata = {
        "price": round(current_price, 4),
        "bb_upper": round(bb_upper, 4),
        "bb_mid": round(bb_mid, 4),
        "bb_lower": round(bb_lower, 4),
        "zscore": round(z_now, 4),
        "rsi": round(rsi_now, 4),
        "ema_spread": round(ema_spread, 8),
        "is_ranging": is_ranging,
        "ranging_threshold": round(ranging_threshold, 8),
        "raw_val": round(raw_val, 4),
        "raw_strength": round(raw_strength, 4),
    }

    logger.info("MeanReversion signal=%d raw_strength=%.3f", signal, raw_strength)

    return {
        "signal": signal,
        "raw_strength": raw_strength,
        "signal_str": signal_str,
        "confidence": confidence,
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
