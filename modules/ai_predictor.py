"""
modules/ai_predictor.py — Deterministic rule-based trade scoring.

Replaces the ML ensemble with indicator agreement scoring.
No sklearn, no torch, no external model dependencies.

Outputs a standardized signal dict with prob_up and rolling_accuracy
keys for consumption by confidence_scoring.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)


def generate_signal_from_strategy_outputs(
    strategy_signals: dict[str, dict],
    df: pd.DataFrame | None = None,
    cfg: object | None = None,  # noqa: ARG001 — kept for API compatibility
) -> dict:
    """
    Generate a trade signal from indicator agreement scoring.

    Scores:
      - EMA spread direction (from indicators)
      - MACD histogram sign
      - RSI zone (oversold=bullish, overbought=bearish, neutral=0)

    Combined with momentum and mean-reversion raw_strength.

    Args:
        strategy_signals: {"momentum": {...}, "mean_reversion": {...}, ...}
        df: OHLCV DataFrame for indicator computation.
        cfg: Unused, kept for API compatibility.

    Returns:
        {signal, confidence, metadata, prob_up, rolling_accuracy}
    """
    ind = compute_indicators(df) if df is not None and len(df) > 28 else None

    # Indicator agreement score [-1, 1]
    indicator_score = 0.0
    ind_details: dict[str, float] = {}

    if ind is not None:
        # EMA spread: positive = bullish
        ema_contrib = float(np.clip(ind.ema_spread_norm, -1.0, 1.0))
        indicator_score += ema_contrib * 0.4
        ind_details["ema_spread_norm"] = round(ema_contrib, 4)

        # MACD histogram: positive = bullish
        macd_contrib = float(np.clip(ind.macd_hist_norm, -1.0, 1.0))
        indicator_score += macd_contrib * 0.3
        ind_details["macd_hist_norm"] = round(macd_contrib, 4)

        # RSI zone: <35 = bullish (+1), >65 = bearish (-1), else neutral
        rsi_val = ind.rsi_14
        if rsi_val < 35:
            rsi_contrib = (35 - rsi_val) / 35  # 0..1
        elif rsi_val > 65:
            rsi_contrib = -(rsi_val - 65) / 35  # -1..0
        else:
            rsi_contrib = 0.0
        indicator_score += rsi_contrib * 0.3
        ind_details["rsi_contrib"] = round(rsi_contrib, 4)

    # Strategy signal contributions
    mom = strategy_signals.get("momentum", {"signal": 0, "raw_strength": 0.0})
    mr = strategy_signals.get("mean_reversion", {"signal": 0, "raw_strength": 0.0})

    mom_val = int(mom.get("signal", 0)) * float(mom.get("raw_strength", 0.0))
    mr_val = int(mr.get("signal", 0)) * float(mr.get("raw_strength", 0.0))

    # Weighted combination: 40% indicators, 35% momentum, 25% mean-reversion
    score = 0.40 * indicator_score + 0.35 * mom_val + 0.25 * mr_val

    # Map score to prob_up [0, 1]
    prob_up = float(np.clip(0.5 + score * 0.5, 0.05, 0.95))

    if prob_up > 0.52:
        direction = "BUY"
    elif prob_up < 0.48:
        direction = "SELL"
    else:
        direction = "HOLD"

    confidence = abs(prob_up - 0.5) * 2.0  # 0..1

    metadata = {
        "source": "rule_based_scorer",
        "indicator_score": round(indicator_score, 4),
        "mom_val": round(mom_val, 4),
        "mr_val": round(mr_val, 4),
        "combined_score": round(score, 4),
        "indicators": ind_details,
        "reasoning": f"Rule scorer: prob_up={prob_up:.3f} (ind={indicator_score:+.3f} mom={mom_val:+.3f} mr={mr_val:+.3f})",
    }

    logger.info(
        "AI-RuleScorer signal=%s confidence=%.3f prob_up=%.3f",
        direction, confidence, prob_up,
    )

    return {
        "signal": direction,
        "confidence": confidence,
        "metadata": metadata,
        "prob_up": prob_up,
        "rolling_accuracy": None,
    }
