"""Confidence scoring for hybrid strategy outputs.

KEY IMPROVEMENTS (v3 — profitability overhaul):
- Regime-aware dynamic weighting: trending → momentum only, ranging → MR only
- Trend-quality multiplier: ADX strength amplifies confidence in confirmed trends
- Agreement bonus when both modules align on direction
- Volatility penalty properly calibrated (not double-penalizing)
- Lower execution threshold in trending regimes to capture more trend moves
- Higher threshold in choppy regimes to avoid whipsaw losses
"""

from __future__ import annotations

import numpy as np


CONF_MIN = 0.0
CONF_MAX = 0.90  # Raised cap: allow very high-quality setups to express full confidence


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _strength(out: dict) -> float:
    """Get normalized signal strength from strategy output."""
    if "raw_strength" in out:
        return _clip01(float(out.get("raw_strength", 0.0)))
    if "strength" in out:
        return _clip01(float(out.get("strength", 0.0)))
    return _clip01(float(out.get("confidence", 0.0)))


def get_execution_threshold(
    current_atr_norm: float | None,
    base_threshold: float = 0.60,
    regime: str = "choppy",
) -> float:
    """Return execution threshold, adapted by regime and volatility.

    Key insight: in trending regimes we WANT to trade more (lower threshold),
    in choppy regimes we want to be highly selective (higher threshold).
    """
    # Regime-based baseline
    if regime in ("trending_up", "trending_down"):
        base = min(base_threshold, 0.58)  # More permissive in trends
    elif regime == "ranging":
        base = min(base_threshold, 0.60)  # MR opportunities in ranges
    else:
        base = max(base_threshold, 0.65)  # Strict in choppy markets

    if current_atr_norm is None:
        return float(np.clip(base, 0.55, 0.75))

    atr = max(0.0, float(current_atr_norm))
    # Only raise threshold in extreme volatility
    if atr >= 0.04:
        return max(base, 0.78)
    if atr >= 0.03:
        return max(base, 0.70)

    return float(np.clip(base, 0.55, 0.75))


def compute_confidence(
    momentum_out: dict,
    mean_rev_out: dict,
    ai_out: dict | None = None,
    current_atr_norm: float | None = None,
    *,
    conf_threshold: float = 0.60,
    w_mom: float = 0.60,
    w_mr: float = 0.40,
    w_ai: float = 0.0,
    regime: str = "choppy",
) -> tuple[float, str]:
    """Compute final confidence and action with regime-aware weighting.

    Returns (confidence, action) where action is BUY, SELL, or HOLD.
    """
    m_sig = int(momentum_out.get("signal", 0))
    r_sig = int(mean_rev_out.get("signal", 0))

    m_strength = _strength(momentum_out)
    r_strength = _strength(mean_rev_out)

    # ── Regime-based weight routing ──────────────────────────────────
    # Strong regime separation: trend -> momentum, range -> MR, choppy -> conservative blend
    if regime in ("trending_up", "trending_down"):
        mom_w, mr_w = 0.90, 0.10  # Trending: momentum dominates
    elif regime == "ranging":
        mom_w, mr_w = 0.10, 0.90  # Ranging: mean-reversion dominates
    else:
        mom_w, mr_w = 0.45, 0.55  # Choppy: slight MR bias (contrarian edge)

    # ── Score accumulation ───────────────────────────────────────────
    buy_score = 0.0
    sell_score = 0.0
    if m_sig == 1:
        buy_score += mom_w * m_strength
    elif m_sig == -1:
        sell_score += mom_w * m_strength
    if r_sig == 1:
        buy_score += mr_w * r_strength
    elif r_sig == -1:
        sell_score += mr_w * r_strength

    if buy_score == 0.0 and sell_score == 0.0:
        return 0.0, "HOLD"

    direction = "BUY" if buy_score > sell_score else "SELL"
    dominant = buy_score if direction == "BUY" else sell_score
    opposite = sell_score if direction == "BUY" else buy_score

    dominant_strength = _clip01(dominant)
    disagreement = _clip01(opposite)

    # ── Conflict detection ───────────────────────────────────────────
    # Only block if both modules actively disagree with meaningful strength
    if m_sig != 0 and r_sig != 0 and m_sig != r_sig:
        if min(m_strength, r_strength) >= 0.35:
            return 0.0, "HOLD"

    # Minimum signal quality gate
    if dominant_strength < 0.20:
        return 0.0, "HOLD"

    # ── Confidence calculation ───────────────────────────────────────
    # Base confidence from dominant signal quality
    quality = _clip01(0.70 * dominant_strength + 0.30 * (1.0 - disagreement))

    # Agreement bonus: both momentum and MR agree on direction
    agreement_bonus = 0.0
    if m_sig != 0 and r_sig != 0 and m_sig == r_sig:
        agreement_bonus = 0.10  # Strong bonus for consensus

    # Regime-quality bonus: trending regimes with momentum get a boost
    regime_bonus = 0.0
    if regime in ("trending_up", "trending_down") and m_sig != 0 and m_strength >= 0.40:
        regime_bonus = 0.05
    elif regime == "ranging" and r_sig != 0 and r_strength >= 0.40:
        regime_bonus = 0.05

    # Volatility penalty (lighter than before — avoid double-penalizing)
    atr = max(0.0, float(current_atr_norm)) if current_atr_norm is not None else 0.0
    volatility_penalty = 0.0
    if atr > 0.025:
        volatility_penalty = float(np.clip((atr - 0.025) * 6.0, 0.0, 0.20))

    # Final confidence assembly
    confidence = 0.50 + 0.40 * quality + agreement_bonus + regime_bonus - volatility_penalty
    confidence = float(np.clip(confidence, CONF_MIN, CONF_MAX))

    # ── Execution threshold check ────────────────────────────────────
    exec_threshold = get_execution_threshold(
        current_atr_norm, base_threshold=conf_threshold, regime=regime,
    )
    if confidence < exec_threshold:
        return confidence, "HOLD"

    return confidence, direction
