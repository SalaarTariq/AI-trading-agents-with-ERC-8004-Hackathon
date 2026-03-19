"""
modules/confidence_scoring.py — Confidence scoring.

Combines momentum, mean-reversion, and indicator agreement signals
into a single confidence score and action. No tanh squashing.
Regime-aware threshold is passed in from caller.
"""

from __future__ import annotations

import numpy as np


def compute_confidence(
    momentum_out: dict,
    mean_rev_out: dict,
    ai_out: dict,
    current_atr_norm: float | None,
    *,
    conf_threshold: float = 0.45,
    w_mom: float = 0.45,
    w_mr: float = 0.30,
    w_ai: float = 0.25,
) -> tuple[float, str]:
    """
    Compute combined confidence and action.

    Args:
        momentum_out: {signal: -1|0|1, raw_strength: 0..1}
        mean_rev_out: {signal: -1|0|1, raw_strength: 0..1}
        ai_out: {prob_up: 0..1}
        current_atr_norm: Normalized ATR (unused now, kept for API compat).
        conf_threshold: Minimum absolute score to trigger BUY/SELL.
        w_mom: Weight for momentum signal.
        w_mr: Weight for mean-reversion signal.
        w_ai: Weight for AI/indicator agreement signal.

    Returns:
        (confidence, action) where confidence is in [0, 1].
    """
    m_sig = int(momentum_out.get("signal", 0))
    m_str = float(momentum_out.get("raw_strength", 0.0))
    r_sig = int(mean_rev_out.get("signal", 0))
    r_str = float(mean_rev_out.get("raw_strength", 0.0))

    prob_up = float(ai_out.get("prob_up", 0.5))

    # Directional values: [-1, 1]
    m_val = m_sig * m_str
    r_val = r_sig * r_str
    ai_val = (prob_up - 0.5) * 2.0  # map [0,1] → [-1,1]

    # Weighted sum: [-1, 1]
    raw_score = w_mom * m_val + w_mr * r_val + w_ai * ai_val

    # Confidence = absolute magnitude
    confidence = float(np.clip(abs(raw_score), 0.0, 1.0))

    if raw_score > conf_threshold:
        action = "BUY"
    elif raw_score < -conf_threshold:
        action = "SELL"
    else:
        action = "HOLD"
        confidence = 0.0

    return confidence, action
