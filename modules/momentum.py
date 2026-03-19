"""
modules/momentum.py — Momentum trading strategy.


Simplified momentum strategy (spec B).

Keeps only:
- EMA_spread > 0 (uptrend) or < 0 (downtrend)
- MACD histogram confirms direction
- RSI between 35–75 (avoid extreme chase/panic)

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

from config import MomentumConfig, CONFIG
from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    cfg: MomentumConfig | None = None,
) -> dict:
    """
    Produce a momentum output for the most recent bar.

    Args:
        df: OHLCV DataFrame (needs at least `close`).
        cfg: Optional MomentumConfig override.

    Returns:
        {
            "signal": -1|0|1,
            "raw_strength": float,
            "signal_str": "BUY"|"SELL"|"HOLD",
            "confidence": float,
            "metadata": { ... },
        }
    """
    cfg = cfg or CONFIG.momentum

    ind = compute_indicators(
        df,
        ema_fast_period=getattr(cfg, "ema_fast_period", 9),
        ema_slow_period=getattr(cfg, "ema_slow_period", 21),
        rsi_period=14,
        macd_fast=cfg.macd_fast,
        macd_slow=cfg.macd_slow,
        macd_signal=cfg.macd_signal,
        atr_period=14,
    )
    if ind is None:
        return _hold_signal("indicators warming up")

    ema_spread = ind.ema_spread
    macd_hist = ind.macd_hist
    rsi_now = ind.rsi_14

    current_price = float(df["close"].iloc[-1])
    atr_norm = ind.atr_norm_14 if ind.atr_norm_14 else 0.001

    ema_spread_pct = ema_spread / current_price
    macd_hist_pct = macd_hist / current_price

    # Spec B: sensitive momentum signal via tanh
    raw_val = float(np.tanh(ema_spread_pct / max(2 * atr_norm, 1e-8)) * (1 + macd_hist_pct / max(atr_norm, 1e-8)))

    rsi_ok = 25.0 <= rsi_now <= 75.0  # broadened for more trades
    signal = 0
    if raw_val > 0.1 and rsi_ok:
        signal = 1
    elif raw_val < -0.1 and rsi_ok:
        signal = -1

    raw_strength = float(np.clip(abs(raw_val), 0.0, 1.0)) if signal != 0 else 0.0
    confidence = raw_strength
    signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"

    metadata = {
        "ema_fast": round(ind.ema_fast, 8),
        "ema_slow": round(ind.ema_slow, 8),
        "ema_spread": round(ema_spread, 8),
        "macd_hist": round(macd_hist, 8),
        "rsi": round(rsi_now, 2),
        "rsi_ok": rsi_ok,
        "raw_val": round(raw_val, 4),
        "raw_strength": round(raw_strength, 4),
    }

    logger.info("Momentum signal=%d raw_strength=%.3f", signal, raw_strength)

    return {
        "signal": signal,
        "raw_strength": raw_strength,
        "signal_str": signal_str,
        "confidence": confidence,
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
