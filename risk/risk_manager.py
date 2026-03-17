"""
risk/risk_manager.py — Risk management and trade gating.


Every proposed trade must pass through `check_risk()` before execution.
The risk manager enforces stop-loss, take-profit, position sizing,
daily loss caps, volatility filters, and emergency rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import RiskConfig, CONFIG
from utils.helpers import atr, sma, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Snapshot of current portfolio for risk evaluation."""
    total_value: float
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # symbol → value
    daily_pnl: float = 0.0
    peak_value: float = 0.0
    consecutive_losses: int = 0
    is_defensive: bool = False
    cooldown_bars: int = 0  # Bars remaining in cooldown after consecutive losses


@dataclass
class RiskResult:
    """Result of a risk check."""
    approved: bool
    reasons: list[str]
    adjusted_size: float     # May be reduced from requested size
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
) -> RiskResult:
    """
    Validate a proposed trade against all risk rules.

    Args:
        signal: "BUY" or "SELL".
        confidence: Combined confidence score [0, 1].
        entry_price: Proposed entry price.
        requested_size: Requested position size in base currency (e.g. USD).
        portfolio: Current portfolio state.
        df: Optional OHLCV data for volatility calculations.
        cfg: Optional RiskConfig override.

    Returns:
        RiskResult with approval status, reasons, and adjusted sizing.
    """
    cfg = cfg or CONFIG.risk
    reasons: list[str] = []
    warnings: list[str] = []
    adjusted_size = requested_size

    # --- 1. Minimum confidence ---
    if confidence < cfg.min_confidence:
        reasons.append(
            f"Confidence {confidence:.3f} below minimum {cfg.min_confidence}"
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
        # Trigger a 10-bar cooldown and reset the counter
        portfolio.cooldown_bars = 10
        portfolio.consecutive_losses = 0
        reasons.append(
            f"Consecutive stop-loss pause triggered: entering 10-bar cooldown"
        )

    # --- 4. Max position size ---
    max_position_value = portfolio.total_value * cfg.max_position_pct
    if requested_size > max_position_value:
        warnings.append(
            f"Requested size ${requested_size:,.0f} exceeds max "
            f"${max_position_value:,.0f} — reducing"
        )
        adjusted_size = max_position_value

    # --- 5. Defensive mode (drawdown) ---
    if portfolio.peak_value > 0:
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        if drawdown >= cfg.max_drawdown_pct:
            portfolio.is_defensive = True
            warnings.append(
                f"Defensive mode active: drawdown {drawdown:.1%} >= {cfg.max_drawdown_pct:.0%}"
            )
            adjusted_size *= cfg.defensive_size_mult

    # --- 6. Volatility filter ---
    vol_adjustment = 1.0
    if df is not None and len(df) >= 44:
        vol_blocked, vol_warning = _check_volatility(df, cfg)
        if vol_blocked:
            reasons.append(vol_warning)
        elif vol_warning:
            warnings.append(vol_warning)
            adjusted_size *= 0.5
            vol_adjustment = 0.5

    # --- 7. Confidence-based position sizing ---
    # Replace fixed 20% with dynamic sizing
    if adjusted_size > 0 and confidence > 0:
        base_pct = 0.20
        conf_factor = 0.5 + confidence  # [0.5, 1.5]
        dynamic_size = portfolio.cash * base_pct * conf_factor * vol_adjustment
        if dynamic_size < adjusted_size:
            adjusted_size = dynamic_size

    # --- 8. Cash check ---
    if adjusted_size > portfolio.cash:
        if adjusted_size > portfolio.cash * 1.01:  # allow tiny float error
            warnings.append(
                f"Size ${adjusted_size:,.0f} exceeds cash ${portfolio.cash:,.0f} — reducing"
            )
        adjusted_size = min(adjusted_size, portfolio.cash)

    # --- Calculate stop-loss / take-profit prices ---
    sl_pct = cfg.stop_loss_pct
    tp_pct = cfg.take_profit_pct

    if cfg.use_dynamic_sl_tp and df is not None and {"high", "low"}.issubset(df.columns):
        dynamic_sl, dynamic_tp = _compute_dynamic_sl_tp(df, entry_price, cfg)
        if dynamic_sl > 0:
            sl_pct = dynamic_sl
        if dynamic_tp > 0:
            tp_pct = dynamic_tp

    if signal == "BUY":
        stop_loss_price = entry_price * (1 - sl_pct)
        take_profit_price = entry_price * (1 + tp_pct)
    elif signal == "SELL":
        stop_loss_price = entry_price * (1 + sl_pct)
        take_profit_price = entry_price * (1 - tp_pct)
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
        dynamic_sl_pct=round(sl_pct, 4),
        dynamic_tp_pct=round(tp_pct, 4),
    )

    if approved:
        logger.info(
            "Risk APPROVED: %s $%.0f @ %g | SL=%g (%.1f%%) TP=%g (%.1f%%) | warnings=%s",
            signal, adjusted_size, entry_price,
            stop_loss_price, sl_pct * 100, take_profit_price, tp_pct * 100,
            warnings or "none",
        )
    else:
        logger.warning(
            "Risk REJECTED: %s $%.0f @ %g | reasons=%s",
            signal, requested_size, entry_price, reasons,
        )

    return result


def _compute_dynamic_sl_tp(
    df: pd.DataFrame, entry_price: float, cfg: RiskConfig
) -> tuple[float, float]:
    """
    Compute ATR-based dynamic stop-loss and take-profit percentages.

    Returns:
        (sl_pct, tp_pct) clamped to configured min/max bounds.
    """
    if len(df) < 15:
        return 0.0, 0.0

    close = df["close"]
    atr_series = atr(df["high"], df["low"], close, 14)
    atr_val = atr_series.iloc[-1]

    if np.isnan(atr_val) or entry_price == 0:
        return 0.0, 0.0

    sl_pct = (atr_val * cfg.atr_sl_multiplier) / entry_price
    tp_pct = (atr_val * cfg.atr_tp_multiplier) / entry_price

    # Clamp to bounds
    sl_pct = float(np.clip(sl_pct, cfg.min_sl_pct, cfg.max_sl_pct))
    tp_pct = float(np.clip(tp_pct, cfg.min_tp_pct, cfg.max_tp_pct))

    return sl_pct, tp_pct


def check_trailing_stop(
    position: dict,
    current_price: float,
    cfg: RiskConfig | None = None,
) -> dict:
    """
    Update stop-loss based on trailing stop logic.

    Args:
        position: Open position dict with entry_price, stop_loss, take_profit, action.
        current_price: Current market price.
        cfg: Risk config.

    Returns:
        Updated position dict (stop_loss may be tightened).
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
    else:  # SELL
        tp_distance = entry - tp
        price_progress = entry - current_price

    if tp_distance <= 0:
        return position

    progress_pct = price_progress / tp_distance

    if progress_pct >= cfg.trailing_lock_pct:
        # Lock 50% of gains
        if action == "BUY":
            new_sl = entry + price_progress * 0.5
            position["stop_loss"] = max(sl, round(new_sl, 8))
        else:
            new_sl = entry - price_progress * 0.5
            position["stop_loss"] = min(sl, round(new_sl, 8))
    elif progress_pct >= cfg.trailing_breakeven_pct:
        # Move SL to breakeven
        if action == "BUY":
            position["stop_loss"] = max(sl, entry)
        else:
            position["stop_loss"] = min(sl, entry)

    return position


def _check_volatility(
    df: pd.DataFrame, cfg: RiskConfig
) -> tuple[bool, str]:
    """
    Check if current volatility exceeds the threshold.

    Returns:
        (blocked, message) — blocked=True means trade should be rejected.
    """
    close = df["close"]
    # Current 14-day realized vol vs 30-day average vol
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        return False, ""

    recent_vol = returns.iloc[-14:].std() * np.sqrt(252)  # annualized
    avg_vol = returns.iloc[-30:].std() * np.sqrt(252)

    if avg_vol == 0:
        return False, ""

    vol_ratio = recent_vol / avg_vol

    if vol_ratio > cfg.volatility_threshold:
        return True, (
            f"Volatility too high: {vol_ratio:.2f}x average "
            f"(threshold {cfg.volatility_threshold:.1f}x)"
        )
    elif vol_ratio > cfg.volatility_threshold * 0.75:
        return False, (
            f"Elevated volatility: {vol_ratio:.2f}x average — reducing size"
        )
    return False, ""


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

    # Check defensive mode
    if portfolio.peak_value > 0:
        drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
        portfolio.is_defensive = drawdown >= CONFIG.risk.max_drawdown_pct

    return portfolio
