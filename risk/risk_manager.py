"""Risk manager for the cleaned ERC-8004 trading agent.

Why this file remains strict:
- It is the primary protection layer against overtrading and drawdown.
- It enforces confidence gates, capital limits, SL/TP, and daily stop rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import RiskConfig, RegimeConfig, RegimeParams, CONFIG

logger = logging.getLogger(__name__)


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
    adjusted_size = float(max(requested_size, 0.0))

    atr_val = float(pre_ind.atr_14) if pre_ind is not None and getattr(pre_ind, "atr_14", None) else None
    atr_norm = float(pre_ind.atr_norm_14) if pre_ind is not None and getattr(pre_ind, "atr_norm_14", None) else 0.0

    # 1) Confidence gate
    threshold = max(cfg.min_confidence, rp.conf_threshold)
    if confidence < threshold:
        reasons.append(f"Confidence {confidence:.3f} below threshold {threshold:.3f}")

    # 2) Daily loss cap
    daily_loss_pct = abs(portfolio.daily_pnl) / max(portfolio.total_value, 1.0)
    if portfolio.daily_pnl < 0 and daily_loss_pct >= cfg.daily_loss_cap_pct:
        reasons.append(
            f"Daily loss cap reached: {daily_loss_pct:.1%} >= {cfg.daily_loss_cap_pct:.0%}"
        )

    # 3) Consecutive-loss cooldown
    if portfolio.cooldown_bars > 0:
        reasons.append(f"Cooldown active: {portfolio.cooldown_bars} bars remaining")
    elif portfolio.consecutive_losses >= cfg.consecutive_loss_pause:
        portfolio.cooldown_bars = 8
        portfolio.consecutive_losses = 0
        reasons.append("Consecutive loss protection triggered (8-bar cooldown)")

    # 4) Position cap and cash cap
    max_position_size = portfolio.total_value * cfg.max_position_pct
    adjusted_size = min(adjusted_size, max_position_size)
    adjusted_size = min(adjusted_size, portfolio.cash)

    # 5) Regime/volatility sizing controls
    regime_mult = rp.position_mult
    if regime == "trending_down":
        regime_mult = min(regime_mult, 0.60)
    elif regime == "choppy":
        regime_mult = min(regime_mult, 0.40)

    if atr_norm > cfg.atr_volatility_reduce_threshold:
        warnings.append(f"High volatility ATR_norm={atr_norm:.3f}; reducing size")
        regime_mult *= 0.60

    adjusted_size *= regime_mult

    # 6) Drawdown defenses
    if portfolio.peak_value > 0:
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        if drawdown >= cfg.max_drawdown_pct:
            portfolio.is_defensive = True
        if portfolio.is_defensive:
            adjusted_size *= 0.50
            warnings.append("Defensive mode active; size reduced by 50%")

    # 7) ATR-based SL/TP (risk-first asymmetric R:R)
    if cfg.use_dynamic_sl_tp and atr_val and atr_val > 0:
        sl_dist = cfg.atr_sl_multiplier * atr_val
        tp_dist = cfg.atr_tp_multiplier * atr_val

        sl_dist = min(max(sl_dist, entry_price * cfg.min_sl_pct), entry_price * cfg.max_sl_pct)
        tp_dist = min(max(tp_dist, entry_price * cfg.min_tp_pct), entry_price * cfg.max_tp_pct)
    else:
        sl_dist = entry_price * cfg.stop_loss_pct
        tp_dist = entry_price * cfg.take_profit_pct

    # 8) Risk budget per trade based on stop distance
    stop_pct = max(sl_dist / max(entry_price, 1e-12), 1e-6)
    max_risk_amt = portfolio.total_value * cfg.risk_per_trade_pct
    max_size_by_risk = max_risk_amt / stop_pct
    if adjusted_size > max_size_by_risk:
        adjusted_size = max_size_by_risk
        warnings.append("Risk-per-trade cap applied")

    if signal == "BUY":
        stop_loss = entry_price - sl_dist
        take_profit = entry_price + tp_dist
    elif signal == "SELL":
        stop_loss = entry_price + sl_dist
        take_profit = entry_price - tp_dist
    else:
        stop_loss = 0.0
        take_profit = 0.0

    approved = (len(reasons) == 0) and adjusted_size > 0 and signal in ("BUY", "SELL")

    result = RiskResult(
        approved=approved,
        reasons=reasons,
        adjusted_size=round(max(adjusted_size, 0.0), 2),
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


def check_trailing_stop(
    position: dict,
    current_price: float,
    cfg: RiskConfig | None = None,
) -> dict:
    """Tighten stop as trade moves toward take-profit.

    Rule kept:
    - Activate trailing at 60% progress to TP.
    - Lock 70% of realized move from entry.
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
