"""Risk manager for the ERC-8004 trading agent.

This file is the trade-gating layer. Behavior is unchanged from the
previous monolithic implementation; the logic has been decomposed into
named gates (which only read portfolio state) and sizers (which only
modify ``adjusted_size``). The single mutation of ``portfolio`` —
arming the consecutive-loss cooldown — is isolated to ``_cooldown_gate``
and documented there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import RiskConfig, RegimeConfig, RegimeParams, CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PortfolioState:
    """Minimal portfolio snapshot used by risk checks."""

    total_value: float
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_value: float = 0.0
    consecutive_losses: int = 0
    is_defensive: bool = False
    cooldown_bars: int = 0


@dataclass
class RiskResult:
    """Output of risk validation for one trade intent."""

    approved: bool
    reasons: list[str]
    adjusted_size: float
    stop_loss_price: float
    take_profit_price: float
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gates (read-only checks that may append to ``reasons``)
# ---------------------------------------------------------------------------


def _confidence_gate(
    confidence: float,
    cfg: RiskConfig,
    rp: RegimeParams,
    atr_norm: float,
) -> str | None:
    """Reject trades below the per-bar safety threshold.

    Base is ``max(cfg.min_confidence, regime.conf_threshold)``. The shared
    volatility uplift from ``effective_confidence_threshold`` is then
    applied so the risk gate raises in step with the signal-side gate
    when ATR spikes — but the *base* stays a pure safety value and is
    not influenced by the signal config.
    """
    # Local import keeps the risk module import-cycle-free at module load.
    from modules.confidence_scoring import effective_confidence_threshold

    base = max(cfg.min_confidence, rp.conf_threshold)
    threshold = effective_confidence_threshold(base=base, current_atr_norm=atr_norm)
    if confidence < threshold:
        return f"Confidence {confidence:.3f} below threshold {threshold:.3f}"
    return None


def _daily_loss_gate(portfolio: PortfolioState, cfg: RiskConfig) -> str | None:
    daily_loss_pct = abs(portfolio.daily_pnl) / max(portfolio.total_value, 1.0)
    if portfolio.daily_pnl < 0 and daily_loss_pct >= cfg.daily_loss_cap_pct:
        return (
            f"Daily loss cap reached: {daily_loss_pct:.1%} >= {cfg.daily_loss_cap_pct:.0%}"
        )
    return None


def _cooldown_gate(portfolio: PortfolioState, cfg: RiskConfig) -> str | None:
    """Block during an active cooldown; arm one after N consecutive losses.

    This is the one place ``check_risk`` mutates ``portfolio``: when
    ``consecutive_losses`` crosses the threshold, we arm the cooldown
    counter and reset the loss tally. Keeping it here means the trigger
    fires exactly once at the same bar as the rejection.
    """
    if portfolio.cooldown_bars > 0:
        return f"Cooldown active: {portfolio.cooldown_bars} bars remaining"
    if portfolio.consecutive_losses >= cfg.consecutive_loss_pause:
        portfolio.cooldown_bars = 8
        portfolio.consecutive_losses = 0
        return "Consecutive loss protection triggered (8-bar cooldown)"
    return None


# ---------------------------------------------------------------------------
# Sizers (transform ``size``, may append to ``warnings``)
# ---------------------------------------------------------------------------


def _apply_position_caps(
    size: float,
    portfolio: PortfolioState,
    cfg: RiskConfig,
    warnings: list[str],
) -> float:
    max_position_size = portfolio.total_value * cfg.max_position_pct
    if size > max_position_size:
        warnings.append(
            f"Requested size exceeds max_position_pct ({cfg.max_position_pct:.0%}); clamped"
        )
        size = max_position_size
    return min(size, portfolio.cash)


def _apply_regime_and_volatility(
    size: float,
    regime: str,
    rp: RegimeParams,
    atr_norm: float,
    cfg: RiskConfig,
    warnings: list[str],
) -> float:
    mult = rp.position_mult
    if regime == "trending_down":
        mult = min(mult, 0.60)
    elif regime == "choppy":
        mult = min(mult, 0.40)

    if atr_norm > cfg.atr_volatility_reduce_threshold:
        warnings.append(f"High volatility ATR_norm={atr_norm:.3f}; reducing size")
        mult *= 0.60

    return size * mult


def _apply_drawdown_defense(
    size: float,
    portfolio: PortfolioState,
    cfg: RiskConfig,
    warnings: list[str],
) -> float:
    if portfolio.peak_value <= 0:
        return size
    drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
    if drawdown >= cfg.max_drawdown_pct:
        portfolio.is_defensive = True
    if portfolio.is_defensive:
        warnings.append(
            f"Defensive mode active; size reduced by {(1 - cfg.defensive_size_mult) * 100:.0f}%"
        )
        return size * cfg.defensive_size_mult
    return size


def _stop_loss_take_profit_distances(
    entry_price: float,
    atr_val: float | None,
    cfg: RiskConfig,
) -> tuple[float, float]:
    if cfg.use_dynamic_sl_tp and atr_val and atr_val > 0:
        sl_dist = float(
            np.clip(
                cfg.atr_sl_multiplier * atr_val,
                entry_price * cfg.min_sl_pct,
                entry_price * cfg.max_sl_pct,
            )
        )
        tp_dist = float(
            np.clip(
                cfg.atr_tp_multiplier * atr_val,
                entry_price * cfg.min_tp_pct,
                entry_price * cfg.max_tp_pct,
            )
        )
        return sl_dist, tp_dist
    return entry_price * cfg.stop_loss_pct, entry_price * cfg.take_profit_pct


def _apply_risk_per_trade(
    size: float,
    entry_price: float,
    sl_dist: float,
    portfolio: PortfolioState,
    cfg: RiskConfig,
    warnings: list[str],
) -> float:
    stop_pct = max(sl_dist / max(entry_price, 1e-12), 1e-6)
    max_risk_amt = portfolio.total_value * cfg.risk_per_trade_pct
    max_size_by_risk = max_risk_amt / stop_pct
    if size > max_size_by_risk:
        warnings.append("Risk-per-trade cap applied")
        return max_size_by_risk
    return size


def _stop_loss_take_profit_prices(
    signal: str, entry_price: float, sl_dist: float, tp_dist: float
) -> tuple[float, float]:
    if signal == "BUY":
        return entry_price - sl_dist, entry_price + tp_dist
    if signal == "SELL":
        return entry_price + sl_dist, entry_price - tp_dist
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check_risk(
    signal: str,
    confidence: float,
    entry_price: float,
    requested_size: float,
    portfolio: PortfolioState,
    cfg: RiskConfig | None = None,
    regime: str = "choppy",
    regime_cfg: RegimeConfig | None = None,
    pre_ind: object | None = None,
) -> RiskResult:
    """Validate risk constraints before opening a position."""
    cfg = cfg or CONFIG.risk
    regime_cfg = regime_cfg or CONFIG.regime
    rp: RegimeParams = regime_cfg.get(regime)

    reasons: list[str] = []
    warnings: list[str] = []

    atr_val = (
        float(pre_ind.atr_14)
        if pre_ind is not None and getattr(pre_ind, "atr_14", None)
        else None
    )
    atr_norm = (
        float(pre_ind.atr_norm_14)
        if pre_ind is not None and getattr(pre_ind, "atr_norm_14", None)
        else 0.0
    )

    # 1) Gates — collect rejection reasons but keep evaluating so the caller
    #    sees the full picture (warnings, sizing) for logging/replay.
    for reason in (
        _confidence_gate(confidence, cfg, rp, atr_norm),
        _daily_loss_gate(portfolio, cfg),
        _cooldown_gate(portfolio, cfg),
    ):
        if reason:
            reasons.append(reason)

    # 2) Sizing — apply caps, regime mult, volatility reduction, drawdown.
    size = max(float(requested_size), 0.0)
    size = _apply_position_caps(size, portfolio, cfg, warnings)
    size = _apply_regime_and_volatility(size, regime, rp, atr_norm, cfg, warnings)
    size = _apply_drawdown_defense(size, portfolio, cfg, warnings)

    # 3) SL/TP distances and final risk-per-trade clamp.
    sl_dist, tp_dist = _stop_loss_take_profit_distances(entry_price, atr_val, cfg)
    size = _apply_risk_per_trade(size, entry_price, sl_dist, portfolio, cfg, warnings)
    stop_loss, take_profit = _stop_loss_take_profit_prices(
        signal, entry_price, sl_dist, tp_dist
    )

    approved = (
        len(reasons) == 0 and size > 0 and signal in ("BUY", "SELL")
    )

    result = RiskResult(
        approved=approved,
        reasons=reasons,
        adjusted_size=round(max(size, 0.0), 2),
        stop_loss_price=round(stop_loss, 8),
        take_profit_price=round(take_profit, 8),
        warnings=warnings,
    )

    if approved:
        logger.info(
            "Risk APPROVED | %s size=$%.0f SL=%g TP=%g regime=%s",
            signal,
            result.adjusted_size,
            result.stop_loss_price,
            result.take_profit_price,
            regime,
        )
    else:
        logger.warning("Risk REJECTED | signal=%s reasons=%s", signal, reasons)

    return result


# ---------------------------------------------------------------------------
# Trailing stop + post-trade updates
# ---------------------------------------------------------------------------


def check_trailing_stop(
    position: dict,
    current_price: float,
    cfg: RiskConfig | None = None,
) -> dict:
    """Tighten stop as a trade moves toward take-profit.

    Activates at 60% of the path to TP and locks in 70% of the realized
    move from entry. Operates in place on the ``position`` dict for
    backwards compatibility; ``Position.update_trailing_stop`` is the
    typed wrapper.
    """
    cfg = cfg or CONFIG.risk
    if not cfg.use_trailing_stop:
        return position

    entry = float(position["entry_price"])
    sl = float(position["stop_loss"])
    tp = float(position["take_profit"])
    action = str(position["action"])

    if action == "BUY":
        tp_distance = tp - entry
        progress = current_price - entry
    else:
        tp_distance = entry - tp
        progress = entry - current_price

    if tp_distance <= 0:
        return position

    progress_pct = progress / tp_distance
    if progress_pct >= cfg.trailing_breakeven_pct:
        if action == "BUY":
            new_sl = entry + progress * cfg.trailing_lock_pct
            position["stop_loss"] = max(sl, round(new_sl, 8))
        else:
            new_sl = entry - progress * cfg.trailing_lock_pct
            position["stop_loss"] = min(sl, round(new_sl, 8))

    return position


def update_after_trade(
    portfolio: PortfolioState,
    trade_pnl: float,
    hit_stop_loss: bool = False,
) -> PortfolioState:
    """Update portfolio health fields after closing a trade."""
    portfolio.total_value += trade_pnl
    portfolio.daily_pnl += trade_pnl

    if portfolio.total_value > portfolio.peak_value:
        portfolio.peak_value = portfolio.total_value

    if hit_stop_loss:
        portfolio.consecutive_losses += 1
    else:
        portfolio.consecutive_losses = 0

    if portfolio.peak_value > 0:
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        portfolio.is_defensive = drawdown >= CONFIG.risk.max_drawdown_pct

    return portfolio
