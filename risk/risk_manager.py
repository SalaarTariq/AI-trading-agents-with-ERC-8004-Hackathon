"""
risk/risk_manager.py — Risk management and trade gating.

Every proposed trade must pass through `check_risk()` before execution.
The risk manager enforces stop-loss, take-profit, position sizing,
daily loss caps, volatility filters, and emergency rules.

KEY IMPROVEMENTS (v3 — profitability overhaul):
- Asymmetric R:R: TP = 2-3x SL distance via ATR multipliers
- Regime-aware ATR scaling: tighter SL in trends, wider TP to let winners run
- Confidence-scaled sizing: higher confidence = larger position
- Trailing stop locks profit at 50% of TP progress (was 65%)
- Lighter cooldown after stop-loss (4 bars vs 8)

EIP-712 / Surge Integration Note:
    During the hackathon, approved trades will be wrapped as EIP-712
    TradeIntent structs and submitted to the Surge Risk Router on Base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import RiskConfig, RegimeConfig, RegimeParams, CONFIG
from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of current portfolio for risk evaluation."""
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
    """Result of a risk check."""
    approved: bool
    reasons: list[str]
    adjusted_size: float
    stop_loss_price: float
    take_profit_price: float
    warnings: list[str] = field(default_factory=list)
    dynamic_sl_pct: float = 0.0
    dynamic_tp_pct: float = 0.0


def check_risk(
    signal: str,
    confidence: float,
    entry_price: float,
    requested_size: float,
    portfolio: PortfolioState,
    df: pd.DataFrame | None = None,
    cfg: RiskConfig | None = None,
    regime: str = "choppy",
    regime_cfg: RegimeConfig | None = None,
    pre_ind: object | None = None,
) -> RiskResult:
    """
    Validate a proposed trade against all risk rules.

    Args:
        signal: "BUY" or "SELL".
        confidence: Combined confidence score [0, 1].
        entry_price: Proposed entry price.
        requested_size: Requested position size in base currency.
        portfolio: Current portfolio state.
        df: Optional OHLCV data for volatility calculations.
        cfg: Optional RiskConfig override.
        regime: Market regime string.
        regime_cfg: Optional RegimeConfig override.
        pre_ind: Pre-computed Indicators object (skips recomputation if provided).

    Returns:
        RiskResult with approval status, reasons, and adjusted sizing.
    """
    cfg = cfg or CONFIG.risk
    regime_cfg = regime_cfg or CONFIG.regime
    rp: RegimeParams = regime_cfg.get(regime)

    reasons: list[str] = []
    warnings: list[str] = []
    adjusted_size = requested_size

    ind = pre_ind if pre_ind is not None else (compute_indicators(df) if df is not None else None)
    atr_norm = ind.atr_norm_14 if ind is not None and ind.atr_norm_14 is not None else 0.0
    atr_val = ind.atr_14 if ind is not None else None

    # --- 1. Regime-aware confidence threshold ---
    dyn_threshold = max(rp.conf_threshold, cfg.min_confidence)
    if confidence < dyn_threshold:
        reasons.append(
            f"Confidence {confidence:.3f} below regime threshold {dyn_threshold} ({regime})"
        )

    # --- 2. Daily loss cap ---
    daily_loss_pct = abs(portfolio.daily_pnl) / max(portfolio.total_value, 1)
    if portfolio.daily_pnl < 0 and daily_loss_pct >= cfg.daily_loss_cap_pct:
        reasons.append(
            f"Daily loss cap reached: {daily_loss_pct:.1%} >= {cfg.daily_loss_cap_pct:.0%}"
        )

    # --- 3. Cooldown / Consecutive loss pause ---
    if portfolio.cooldown_bars > 0:
        reasons.append(
            f"Cooldown active: {portfolio.cooldown_bars} bars remaining after consecutive losses"
        )
    elif portfolio.consecutive_losses >= cfg.consecutive_loss_pause:
        portfolio.cooldown_bars = 8  # Shorter cooldown (was 10)
        portfolio.consecutive_losses = 0
        reasons.append(
            "Consecutive stop-loss pause triggered: entering 8-bar cooldown"
        )

    # --- 4. Max position size ---
    max_position_value = portfolio.total_value * cfg.max_position_pct
    if requested_size > max_position_value:
        warnings.append(
            f"Requested size ${requested_size:,.0f} exceeds max "
            f"${max_position_value:,.0f} — reducing"
        )
        adjusted_size = max_position_value

    # --- 5. Drawdown and loss-streak protections ---
    if portfolio.peak_value > 0:
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        if drawdown >= 0.10 and not portfolio.is_defensive:
            portfolio.is_defensive = True
            warnings.append(
                f"Defensive mode activated: drawdown {drawdown:.1%} >= 10%"
            )

        if portfolio.is_defensive:
            adjusted_size *= 0.50
            warnings.append(
                f"Defensive mode: drawdown {drawdown:.1%} — size reduced 50%"
            )

        if 0.05 <= drawdown < 0.10:
            adjusted_size *= 0.75
            warnings.append(f"Drawdown control active ({drawdown:.1%}) — size reduced 25%")

    if portfolio.consecutive_losses >= 2:
        adjusted_size *= 0.75
        warnings.append("Loss streak protection: consecutive losses >= 2 — size reduced 25%")

    # --- 6. ATR-based sizing with regime multiplier ---
    # Confidence-scaled: higher confidence = position up to full allocation
    confidence_scale = float(np.clip((confidence - 0.55) / 0.30, 0.5, 1.0))

    vol_adjustment = 1.0
    atr_reduce_threshold = cfg.atr_volatility_reduce_threshold
    if atr_reduce_threshold > 1.0:
        atr_reduce_threshold = atr_reduce_threshold / 100.0

    if atr_norm > atr_reduce_threshold > 0:
        warnings.append(f"High volatility (ATR_norm={atr_norm:.3f}) — reducing size 40%")
        vol_adjustment = 0.60

    if atr_norm > 0:
        # Inverse-volatility sizing: lower vol = larger position
        size_pct = 0.015 / (2.0 * atr_norm)
        size_pct = min(size_pct, 0.15)
        adjusted_size = portfolio.total_value * size_pct * vol_adjustment * rp.position_mult * confidence_scale
    else:
        adjusted_size = min(adjusted_size, portfolio.total_value * cfg.max_capital_pct) * vol_adjustment * rp.position_mult * confidence_scale

    # --- 7. Cash check ---
    if adjusted_size > portfolio.cash:
        if adjusted_size > portfolio.cash * 1.01:
            warnings.append(
                f"Size ${adjusted_size:,.0f} exceeds cash ${portfolio.cash:,.0f} — reducing"
            )
        adjusted_size = min(adjusted_size, portfolio.cash)

    # --- 8. Dynamic SL/TP via ATR with asymmetric R:R ---
    # KEY CHANGE: TP is always 2-3x SL for positive expectancy
    if atr_val is not None and atr_val > 0:
        sl_dist = rp.sl_atr_mult * atr_val   # Tight SL (1.0-1.5x ATR)
        tp_dist = rp.tp_atr_mult * atr_val    # Wide TP (2.0-3.5x ATR)

        # Clamp to min/max percentage of entry price
        sl_dist = max(sl_dist, entry_price * cfg.min_sl_pct)
        sl_dist = min(sl_dist, entry_price * cfg.max_sl_pct)
        tp_dist = max(tp_dist, entry_price * cfg.min_tp_pct)
        tp_dist = min(tp_dist, entry_price * cfg.max_tp_pct)

        # Ensure R:R >= 1.3:1 — if TP < 1.3 * SL, widen TP modestly
        if tp_dist < sl_dist * 1.3:
            tp_dist = sl_dist * 1.3
            tp_dist = min(tp_dist, entry_price * cfg.max_tp_pct)
    else:
        sl_dist = entry_price * cfg.stop_loss_pct
        tp_dist = entry_price * cfg.take_profit_pct

    # --- 9. Per-trade risk cap by stop distance ---
    stop_loss_pct = max(sl_dist / max(entry_price, 1e-8), 1e-6)
    risk_budget_pct = float(np.clip(cfg.risk_per_trade_pct, 0.01, 0.02))
    max_risk_amount = portfolio.total_value * risk_budget_pct
    max_size_by_risk = max_risk_amount / stop_loss_pct
    if adjusted_size > max_size_by_risk:
        warnings.append(
            f"Risk per trade cap: max ${max_size_by_risk:,.0f} at SL {stop_loss_pct:.2%}"
        )
        adjusted_size = max_size_by_risk

    if signal == "BUY":
        stop_loss_price = entry_price - sl_dist
        take_profit_price = entry_price + tp_dist
    elif signal == "SELL":
        stop_loss_price = entry_price + sl_dist
        take_profit_price = entry_price - tp_dist
    else:
        stop_loss_price = 0.0
        take_profit_price = 0.0

    approved = len(reasons) == 0 and adjusted_size > 0

    result = RiskResult(
        approved=approved,
        reasons=reasons,
        adjusted_size=round(adjusted_size, 2),
        stop_loss_price=round(stop_loss_price, 8),
        take_profit_price=round(take_profit_price, 8),
        warnings=warnings,
    )

    if approved:
        rr_ratio = round(tp_dist / max(sl_dist, 1e-8), 1)
        logger.info(
            "Risk APPROVED: %s $%.0f @ %g | SL=%g TP=%g R:R=%.1f | regime=%s conf=%.2f",
            signal, adjusted_size, entry_price,
            stop_loss_price, take_profit_price, rr_ratio, regime, confidence,
        )
    else:
        logger.warning(
            "Risk REJECTED: %s $%.0f @ %g | regime=%s | reasons=%s",
            signal, requested_size, entry_price, regime, reasons,
        )

    return result


def check_trailing_stop(
    position: dict,
    current_price: float,
    cfg: RiskConfig | None = None,
) -> dict:
    """
    Update stop-loss based on trailing stop logic.

    v3: Earlier activation to lock profits sooner.
    - Move to breakeven at 50% of TP distance (was 65%)
    - Lock 50% of profit at 70% of TP distance (was 85%)
    """
    cfg = cfg or CONFIG.risk
    if not cfg.use_trailing_stop:
        return position

    entry = position["entry_price"]
    sl = position["stop_loss"]
    tp = position["take_profit"]
    action = position["action"]

    if action == "BUY":
        tp_distance = tp - entry
        price_progress = current_price - entry
    else:
        tp_distance = entry - tp
        price_progress = entry - current_price

    if tp_distance <= 0:
        return position

    progress_pct = price_progress / tp_distance

    if progress_pct >= cfg.trailing_lock_pct:
        # Lock 60% of profit at 80% TP progress
        if action == "BUY":
            new_sl = entry + price_progress * 0.60
            position["stop_loss"] = max(sl, round(new_sl, 8))
        else:
            new_sl = entry - price_progress * 0.60
            position["stop_loss"] = min(sl, round(new_sl, 8))
    elif progress_pct >= cfg.trailing_breakeven_pct:
        # Move to slight profit (entry + 20% of progress) at 60% TP progress
        if action == "BUY":
            new_sl = entry + price_progress * 0.20
            position["stop_loss"] = max(sl, round(new_sl, 8))
        else:
            new_sl = entry - price_progress * 0.20
            position["stop_loss"] = min(sl, round(new_sl, 8))

    return position


def update_after_trade(
    portfolio: PortfolioState,
    trade_pnl: float,
    hit_stop_loss: bool = False,
) -> PortfolioState:
    """
    Update portfolio state after a trade closes.

    Args:
        portfolio: Current portfolio state.
        trade_pnl: Realized PnL from the closed trade.
        hit_stop_loss: Whether the trade was closed by stop-loss.

    Returns:
        Updated PortfolioState.
    """
    portfolio.total_value += trade_pnl
    portfolio.daily_pnl += trade_pnl

    if portfolio.total_value > portfolio.peak_value:
        portfolio.peak_value = portfolio.total_value

    if hit_stop_loss:
        portfolio.consecutive_losses += 1
        logger.warning("Stop-loss hit. Consecutive losses: %d", portfolio.consecutive_losses)
    else:
        portfolio.consecutive_losses = 0

    if portfolio.peak_value > 0:
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        portfolio.is_defensive = drawdown >= CONFIG.risk.max_drawdown_pct

    return portfolio
