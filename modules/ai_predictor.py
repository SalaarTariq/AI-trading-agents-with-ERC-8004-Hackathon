"""Deterministic indicator-agreement scorer used as the agent's
"AI" voter inside the hybrid pipeline.

Two entry points:

- ``predict_from_indicators(strategy_signals, ind, close_price)`` — fast
  path used by the live trading loop, takes an already-computed
  :class:`Indicators` bundle.
- ``generate_signal_from_strategy_outputs(strategy_signals, df, cfg)`` —
  convenience wrapper that computes indicators from a DataFrame. Kept
  for tests and standalone use.

Both return the same shape, so the predictor can be swapped to a real
ML model later without touching callers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.indicators import Indicators, indicators_at, precompute_all_indicators

logger = logging.getLogger(__name__)


def _signal_dict(
    direction: str,
    confidence: float,
    prob_up: float,
    *,
    indicator_score: float,
    mom_val: float,
    mr_val: float,
    score: float,
    ind_details: dict[str, float],
) -> dict:
    return {
        "signal": direction,
        "confidence": confidence,
        "metadata": {
            "source": "rule_based_scorer",
            "indicator_score": round(indicator_score, 4),
            "mom_val": round(mom_val, 4),
            "mr_val": round(mr_val, 4),
            "combined_score": round(score, 4),
            "indicators": ind_details,
            "reasoning": (
                f"Rule scorer: prob_up={prob_up:.3f} "
                f"(ind={indicator_score:+.3f} mom={mom_val:+.3f} mr={mr_val:+.3f})"
            ),
        },
        "prob_up": prob_up,
        "rolling_accuracy": None,
    }


def _score_indicators(ind: Indicators | None, close_price: float) -> tuple[float, dict]:
    """Return (indicator_score in [-1, 1], detail dict)."""
    if ind is None:
        return 0.0, {}

    # EMA spread direction normalized to a 2% reference band.
    ema_norm = float(np.clip(ind.ema_spread_pct / 0.02, -1.0, 1.0))

    # MACD histogram as a fraction of price, normalized to a 0.5% band.
    macd_pct = (ind.macd_hist / close_price) if close_price > 0 else 0.0
    macd_norm = float(np.clip(macd_pct / 0.005, -1.0, 1.0))

    # RSI: oversold is bullish only if trend is not bearish, and vice versa.
    rsi_val = ind.rsi_14
    ema_bearish = ema_norm < -0.1
    ema_bullish = ema_norm > 0.1
    if rsi_val < 35:
        rsi_contrib = (35 - rsi_val) / 35 if not ema_bearish else 0.0
    elif rsi_val > 65:
        rsi_contrib = -(rsi_val - 65) / 35 if not ema_bullish else 0.0
    else:
        rsi_contrib = 0.0

    score = 0.4 * ema_norm + 0.3 * macd_norm + 0.3 * rsi_contrib
    details = {
        "ema_spread_norm": round(ema_norm, 4),
        "macd_hist_norm": round(macd_norm, 4),
        "rsi_contrib": round(rsi_contrib, 4),
    }
    return float(score), details


def predict_from_indicators(
    strategy_signals: dict[str, dict],
    ind: Indicators | None,
    close_price: float,
) -> dict:
    """Score the bar using indicators + the upstream strategy signals.

    Pure function over already-computed inputs — suitable for use inside
    the per-bar trading loop.
    """
    indicator_score, ind_details = _score_indicators(ind, close_price)

    mom = strategy_signals.get("momentum", {"signal": 0, "raw_strength": 0.0})
    mr = strategy_signals.get("mean_reversion", {"signal": 0, "raw_strength": 0.0})
    mom_val = int(mom.get("signal", 0)) * float(mom.get("raw_strength", 0.0))
    mr_val = int(mr.get("signal", 0)) * float(mr.get("raw_strength", 0.0))

    score = 0.40 * indicator_score + 0.35 * mom_val + 0.25 * mr_val
    prob_up = float(np.clip(0.5 + score * 0.5, 0.05, 0.95))

    if prob_up > 0.52:
        direction = "BUY"
    elif prob_up < 0.48:
        direction = "SELL"
    else:
        direction = "HOLD"
    confidence = abs(prob_up - 0.5) * 2.0

    logger.info(
        "AI-RuleScorer signal=%s confidence=%.3f prob_up=%.3f",
        direction,
        confidence,
        prob_up,
    )
    return _signal_dict(
        direction,
        confidence,
        prob_up,
        indicator_score=indicator_score,
        mom_val=mom_val,
        mr_val=mr_val,
        score=score,
        ind_details=ind_details,
    )


def generate_signal_from_strategy_outputs(
    strategy_signals: dict[str, dict],
    df: pd.DataFrame | None = None,
    cfg: object | None = None,  # noqa: ARG001 — kept for API parity
) -> dict:
    """DataFrame-based wrapper around :func:`predict_from_indicators`.

    Recomputes indicators from ``df`` (use the per-bar entry point inside
    the trading loop to avoid this cost).
    """
    ind: Indicators | None = None
    close_price = 0.0
    if df is not None and len(df) > 28:
        pre = precompute_all_indicators(df)
        ind = indicators_at(pre, len(df) - 1)
        close_price = float(df["close"].iloc[-1])

    return predict_from_indicators(strategy_signals, ind, close_price)
