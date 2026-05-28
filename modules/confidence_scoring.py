"""Regime-aware confidence scoring for merged strategy signals.

Why this module remains:
- It keeps confidence math separate from strategy logic.
- It makes execution thresholds explicit and easy to tune.
"""

from __future__ import annotations

import numpy as np

from config import CONFIG, RegimeConfig, SignalConfig


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def effective_confidence_threshold(
    *,
    base: float,
    current_atr_norm: float | None,
    signal_cfg: SignalConfig | None = None,
) -> float:
    """Apply the shared volatility uplift to a caller-supplied base.

    The base is whatever the caller considers authoritative:

    - ``combine_signals`` passes its own ``execute_confidence_threshold``
      (signal-quality view).
    - The risk manager passes ``max(min_confidence, regime.conf_threshold)``
      (trade-safety view).

    Volatility uplift logic lives here so the two paths can't drift on
    *how* they react to high-ATR conditions, only on *what* base they
    consider acceptable in calm markets.
    """
    cfg = signal_cfg or CONFIG.signal
    if current_atr_norm is None:
        return base
    atr = max(0.0, float(current_atr_norm))
    if atr >= cfg.atr_norm_high:
        return max(base, cfg.threshold_floor_high_atr)
    if atr >= cfg.atr_norm_medium:
        return max(base, cfg.threshold_floor_medium_atr)
    return base


def _ai_signal_int(ai_out: dict | None) -> int:
    """Map an ai_predictor output to {-1, 0, +1}."""
    if not ai_out:
        return 0
    sig = str(ai_out.get("signal", "HOLD")).upper()
    if sig == "BUY":
        return 1
    if sig == "SELL":
        return -1
    return 0


def combine_signals(
    strategy_out: dict,
    current_atr_norm: float | None,
    cfg: SignalConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
    ai_out: dict | None = None,
) -> dict:
    """Combine the three voters into one action/confidence output.

    Voters:
    - momentum (rule)
    - mean-reversion (rule)
    - ai_predictor (deterministic rule-based scorer; optional, weight in
      ``SignalConfig.ai_predictor_weight``).

    The AI voter is included to satisfy the hybrid-consensus requirement
    described in ``.claude/rules/01-project-context.md``. Its weight is
    deliberately small so it cannot override two agreeing rule voters.
    """
    cfg = cfg or CONFIG.signal

    regime = str(strategy_out.get("regime", "choppy"))
    mom = strategy_out.get("momentum", {})
    mr = strategy_out.get("mean_reversion", {})

    m_sig = int(mom.get("signal", 0))
    r_sig = int(mr.get("signal", 0))
    m_strength = _clip01(float(mom.get("raw_strength", 0.0)))
    r_strength = _clip01(float(mr.get("raw_strength", 0.0)))

    ai_enabled = cfg.use_ai_predictor and ai_out is not None
    a_sig = _ai_signal_int(ai_out) if ai_enabled else 0
    a_strength = _clip01(float(ai_out.get("confidence", 0.0))) if ai_enabled else 0.0

    if regime in ("trending_up", "trending_down"):
        w_mom, w_mr = cfg.trend_momentum_weight, cfg.trend_meanrev_weight
    elif regime == "ranging":
        w_mom, w_mr = cfg.range_momentum_weight, cfg.range_meanrev_weight
    else:
        w_mom, w_mr = cfg.choppy_momentum_weight, cfg.choppy_meanrev_weight

    # Renormalize rule weights down to leave room for the AI voter.
    w_ai = cfg.ai_predictor_weight if ai_enabled else 0.0
    scale = max(0.0, 1.0 - w_ai)
    w_mom *= scale
    w_mr *= scale

    buy_score = 0.0
    sell_score = 0.0
    if m_sig == 1:
        buy_score += w_mom * m_strength
    elif m_sig == -1:
        sell_score += w_mom * m_strength

    if r_sig == 1:
        buy_score += w_mr * r_strength
    elif r_sig == -1:
        sell_score += w_mr * r_strength

    if a_sig == 1:
        buy_score += w_ai * a_strength
    elif a_sig == -1:
        sell_score += w_ai * a_strength

    if buy_score == 0.0 and sell_score == 0.0:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "score": 0.0,
            "regime": regime,
            "details": {"momentum": mom, "mean_reversion": mr, "ai": ai_out or {}},
            "buy_agreement": int(m_sig == 1) + int(r_sig == 1) + int(a_sig == 1),
            "sell_agreement": int(m_sig == -1) + int(r_sig == -1) + int(a_sig == -1),
        }

    action = "BUY" if buy_score > sell_score else "SELL"
    dominant = buy_score if action == "BUY" else sell_score

    # Scale base with dominant strength so near-zero signals can't ride
    # agreement/regime bonuses up to the execution threshold.
    dom = _clip01(dominant)
    confidence = 0.30 + 0.55 * dom

    if m_sig != 0 and r_sig != 0 and m_sig == r_sig:
        confidence += cfg.agreement_bonus

    # Small extra nudge when the AI voter also agrees with the action.
    if ai_enabled and (
        (action == "BUY" and a_sig == 1) or (action == "SELL" and a_sig == -1)
    ):
        confidence += cfg.ai_agreement_bonus

    if regime in ("trending_up", "trending_down") and ((m_sig == 1 and action == "BUY") or (m_sig == -1 and action == "SELL")):
        confidence += cfg.regime_quality_bonus
    if regime == "ranging" and ((r_sig == 1 and action == "BUY") or (r_sig == -1 and action == "SELL")):
        confidence += cfg.regime_quality_bonus

    atr_rank = float(strategy_out.get("atr_percentile_rank", 0.5) or 0.5)
    if atr_rank >= 0.80:
        confidence -= cfg.high_volatility_penalty

    confidence = _clip01(confidence)

    regime_cfg = regime_cfg or CONFIG.regime
    base_threshold = max(
        cfg.execute_confidence_threshold,
        regime_cfg.get(regime).conf_threshold,
    )
    threshold = effective_confidence_threshold(
        base=base_threshold,
        current_atr_norm=current_atr_norm,
        signal_cfg=cfg,
    )
    has_support = False
    if action == "BUY":
        has_support = (
            (m_sig == 1 and m_strength >= cfg.strong_support_min_strength)
            or (r_sig == 1 and r_strength >= cfg.strong_support_min_strength)
        )
    else:
        has_support = (
            (m_sig == -1 and m_strength >= cfg.strong_support_min_strength)
            or (r_sig == -1 and r_strength >= cfg.strong_support_min_strength)
        )

    if confidence < threshold or not has_support:
        action = "HOLD"

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "score": round(confidence, 4),
        "regime": regime,
        "details": {
            "momentum": {
                "signal": m_sig,
                "raw_strength": round(m_strength, 4),
            },
            "mean_reversion": {
                "signal": r_sig,
                "raw_strength": round(r_strength, 4),
            },
            "ai": {
                "signal": a_sig,
                "raw_strength": round(a_strength, 4),
                "enabled": ai_enabled,
            },
        },
        "buy_agreement": int(m_sig == 1) + int(r_sig == 1) + int(a_sig == 1),
        "sell_agreement": int(m_sig == -1) + int(r_sig == -1) + int(a_sig == -1),
    }


