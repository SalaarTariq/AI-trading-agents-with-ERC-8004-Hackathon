"""Regime-aware confidence scoring for merged strategy signals.

Why this module remains:
- It keeps confidence math separate from strategy logic.
- It makes execution thresholds explicit and easy to tune.
"""

from __future__ import annotations

import numpy as np

from config import SignalConfig, CONFIG


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def get_execution_threshold(
    current_atr_norm: float | None,
    cfg: SignalConfig | None = None,
) -> float:
    """Return execution threshold with a mild volatility uplift."""
    cfg = cfg or CONFIG.signal
    base = cfg.execute_confidence_threshold

    if current_atr_norm is None:
        return base

    atr = max(0.0, float(current_atr_norm))
    if atr >= 0.035:
        return max(base, 0.80)
    if atr >= 0.025:
        return max(base, 0.72)
    return base


def combine_signals(
    strategy_out: dict,
    current_atr_norm: float | None,
    cfg: SignalConfig | None = None,
) -> dict:
    """Combine momentum + mean-reversion into one action/confidence output."""
    cfg = cfg or CONFIG.signal

    regime = str(strategy_out.get("regime", "choppy"))
    mom = strategy_out.get("momentum", {})
    mr = strategy_out.get("mean_reversion", {})

    m_sig = int(mom.get("signal", 0))
    r_sig = int(mr.get("signal", 0))
    m_strength = _clip01(float(mom.get("raw_strength", 0.0)))
    r_strength = _clip01(float(mr.get("raw_strength", 0.0)))

    if regime in ("trending_up", "trending_down"):
        w_mom, w_mr = cfg.trend_momentum_weight, cfg.trend_meanrev_weight
    elif regime == "ranging":
        w_mom, w_mr = cfg.range_momentum_weight, cfg.range_meanrev_weight
    else:
        w_mom, w_mr = cfg.choppy_momentum_weight, cfg.choppy_meanrev_weight

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

    if buy_score == 0.0 and sell_score == 0.0:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "score": 0.0,
            "regime": regime,
            "details": {"momentum": mom, "mean_reversion": mr},
            "buy_agreement": int(m_sig == 1) + int(r_sig == 1),
            "sell_agreement": int(m_sig == -1) + int(r_sig == -1),
        }

    action = "BUY" if buy_score > sell_score else "SELL"
    dominant = buy_score if action == "BUY" else sell_score

    confidence = 0.42 + 0.45 * _clip01(dominant)

    if m_sig != 0 and r_sig != 0 and m_sig == r_sig:
        confidence += cfg.agreement_bonus

    if regime in ("trending_up", "trending_down") and ((m_sig == 1 and action == "BUY") or (m_sig == -1 and action == "SELL")):
        confidence += cfg.regime_quality_bonus
    if regime == "ranging" and ((r_sig == 1 and action == "BUY") or (r_sig == -1 and action == "SELL")):
        confidence += cfg.regime_quality_bonus

    atr_rank = float(strategy_out.get("atr_percentile_rank", 0.5) or 0.5)
    if atr_rank >= 0.80:
        confidence -= cfg.high_volatility_penalty

    confidence = _clip01(confidence)

    threshold = get_execution_threshold(current_atr_norm, cfg)
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
        },
        "buy_agreement": int(m_sig == 1) + int(r_sig == 1),
        "sell_agreement": int(m_sig == -1) + int(r_sig == -1),
    }


def compute_confidence(
    momentum_out: dict,
    mean_rev_out: dict,
    ai_out: dict | None = None,
    current_atr_norm: float | None = None,
    atr_percentile_rank: float | None = None,
    *,
    conf_threshold: float = 0.67,
    regime: str = "choppy",
    **_: object,
) -> tuple[float, str]:
    """Compatibility shim for older callers expecting `(confidence, action)`.

    The cleaned architecture uses `combine_signals`, but this wrapper keeps
    the old interface operational for transitional scripts.
    """
    del ai_out

    out = {
        "regime": regime,
        "momentum": momentum_out,
        "mean_reversion": mean_rev_out,
        "atr_percentile_rank": atr_percentile_rank if atr_percentile_rank is not None else 0.50,
    }
    # Temporary override without mutating global config.
    local_cfg = SignalConfig(**vars(CONFIG.signal))
    local_cfg.execute_confidence_threshold = conf_threshold

    combined = combine_signals(out, current_atr_norm=current_atr_norm, cfg=local_cfg)
    return float(combined["confidence"]), str(combined["action"])

