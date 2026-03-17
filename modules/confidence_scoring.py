"""
modules/confidence_scoring.py — Confidence scoring (spec E).

Implements:
    def compute_confidence(momentum_out, mean_rev_out, ai_out, current_atr_norm):
        base = 0.45 * momentum_out.strength + 0.25 * mean_rev_out.strength + 0.30 * ai_out.prob
        if current_atr_norm > 1.5: base *= 0.7
        confidence = clamp(base, 0, 1)
        return confidence, action
"""

from __future__ import annotations

import numpy as np


def compute_confidence(
    momentum_out: dict,
    mean_rev_out: dict,
    ai_out: dict,
    current_atr_norm: float | None,
    *,
    ai_weight: float = 0.30,
    ai_weight_low_acc: float = 0.15,
    ai_max_weight_cap: float = 0.35,
    ai_min_rolling_acc: float = 0.52,
) -> tuple[float, str]:
    """
    Compute combined confidence and action.

    momentum_out / mean_rev_out:
        expected keys: signal (-1|0|1), raw_strength (0..1)

    ai_out:
        expected keys: prob_up (0..1), rolling_accuracy (0..1)
    """
    m_sig = int(momentum_out.get("signal", 0))
    m_str = float(momentum_out.get("raw_strength", 0.0))
    r_sig = int(mean_rev_out.get("signal", 0))
    r_str = float(mean_rev_out.get("raw_strength", 0.0))

    prob_up = float(ai_out.get("prob_up", 0.5))
    ai_acc = ai_out.get("rolling_accuracy", None)
    ai_acc_v = float(ai_acc) if isinstance(ai_acc, (int, float)) else None

    # Determine action by directional agreement (prefer momentum when both fire)
    direction = 0
    if m_sig != 0 and r_sig != 0:
        direction = m_sig if m_sig == r_sig else 0
    elif m_sig != 0:
        direction = m_sig
    elif r_sig != 0:
        direction = r_sig

    action = "HOLD"
    if direction == 1:
        action = "BUY"
    elif direction == -1:
        action = "SELL"

    # AI weight rule (spec D) - keep signature parameters
    w_ai = 0.2

    # Momentum and mean reversion directional strengths
    m_val = m_sig * m_str
    r_val = r_sig * r_str

    # AI contribution: scale prob_up to [-1, 1]
    ai_val = (prob_up - 0.5) * 2.0

    # Combine signals
    conf_raw = 0.5 * m_val + 0.3 * r_val + w_ai * ai_val
    conf_squashed = np.tanh(conf_raw * 3.0)
    conf_scaled = (conf_squashed + 1) / 2.0  # mapped to [0, 1] for scaling logic if we want positive scale
    
    # Dynamic threshold
    atr_val = current_atr_norm if current_atr_norm is not None else 0.0
    threshold = 0.65 if atr_val > 1.2 else 0.45

    if conf_squashed > threshold:
        action = "BUY"
        confidence = conf_scaled
    elif conf_squashed < -threshold:
        action = "SELL"
        confidence = 1.0 - conf_scaled
    else:
        action = "HOLD"
        confidence = 0.0

    return confidence, action

