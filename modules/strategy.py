"""Unified strategy module combining momentum and mean-reversion.

Why this file exists:
- Removes duplicated logic spread across multiple modules.
- Keeps one clear regime detector and one signal-generation interface.
- Makes signal behavior consistent and easier to tune.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import StrategyConfig, CONFIG

logger = logging.getLogger(__name__)


def detect_regime(row: pd.Series, cfg: StrategyConfig | None = None) -> str:
    """Classify regime from indicator snapshot.

    Regimes:
    - trending_up
    - trending_down
    - ranging
    - choppy
    """
    cfg = cfg or CONFIG.strategy

    spread_pct = float(row.get("ema_spread_pct", 0.0) or 0.0)
    adx_val = float(row.get("adx_14", 0.0) or 0.0)
    atr_norm = float(row.get("atr_norm_14", 0.0) or 0.0)
    atr_q80 = float(row.get("atr_norm_q80_120", 0.0) or 0.0)

    if spread_pct > cfg.regime_trend_up_spread_pct and adx_val > cfg.regime_trend_adx_min:
        return "trending_up"
    if spread_pct < cfg.regime_trend_down_spread_pct and adx_val > cfg.regime_trend_adx_min:
        return "trending_down"
    if abs(spread_pct) < cfg.regime_ranging_abs_spread_pct or adx_val < cfg.regime_ranging_adx_max:
        return "ranging"
    if atr_q80 > 0 and atr_norm >= atr_q80:
        return "choppy"
    return "choppy"


def _momentum_signal(row: pd.Series, regime: str, cfg: StrategyConfig) -> dict:
    """Generate momentum signal in trending regimes only."""
    if regime not in ("trending_up", "trending_down"):
        return {
            "signal": 0,
            "raw_strength": 0.0,
            "confidence": 0.0,
            "metadata": {"reason": "momentum disabled outside trending regimes"},
        }

    spread_pct = float(row.get("ema_spread_pct", 0.0) or 0.0)
    macd_hist = float(row.get("macd_hist", 0.0) or 0.0)
    rsi_val = float(row.get("rsi_14", 50.0) or 50.0)

    in_rsi_band = cfg.momentum_rsi_min <= rsi_val <= cfg.momentum_rsi_max
    signal = 0
    if (
        regime == "trending_up"
        and spread_pct > cfg.momentum_spread_pct_min
        and macd_hist > cfg.momentum_macd_hist_abs_min
        and in_rsi_band
    ):
        signal = 1
    elif (
        regime == "trending_down"
        and spread_pct < -cfg.momentum_spread_pct_min
        and macd_hist < -cfg.momentum_macd_hist_abs_min
        and in_rsi_band
    ):
        signal = -1

    if signal == 0:
        return {
            "signal": 0,
            "raw_strength": 0.0,
            "confidence": 0.0,
            "metadata": {"reason": "momentum thresholds not met"},
        }

    spread_strength = min(abs(spread_pct) / (cfg.momentum_spread_pct_min * 2.0), 1.0)
    macd_strength = min(abs(macd_hist) / (cfg.momentum_macd_hist_abs_min * 3.0), 1.0)
    rsi_center = 1.0 - min(abs(rsi_val - 50.0) / 5.0, 1.0)

    raw_strength = float(np.clip(0.50 * spread_strength + 0.35 * macd_strength + 0.15 * rsi_center, 0.0, 1.0))
    confidence = float(np.clip(0.50 + 0.35 * raw_strength, 0.0, 0.90))

    return {
        "signal": signal,
        "raw_strength": raw_strength,
        "confidence": confidence,
        "metadata": {
            "spread_pct": round(spread_pct, 6),
            "macd_hist": round(macd_hist, 6),
            "rsi": round(rsi_val, 2),
        },
    }


def _mean_reversion_signal(row: pd.Series, regime: str, cfg: StrategyConfig) -> dict:
    """Generate mean-reversion signal in ranging regimes only."""
    if regime != "ranging":
        return {
            "signal": 0,
            "raw_strength": 0.0,
            "confidence": 0.0,
            "metadata": {"reason": "mean reversion disabled outside ranging regime"},
        }

    zscore = float(row.get("bb_zscore", 0.0) or 0.0)
    rsi_val = float(row.get("rsi_14", 50.0) or 50.0)
    macd_hist = float(row.get("macd_hist", 0.0) or 0.0)
    macd_prev = float(row.get("macd_hist_prev", macd_hist) or macd_hist)
    rsi_delta_3 = float(row.get("rsi_delta_3", 0.0) or 0.0)

    macd_flatten_up = macd_hist >= macd_prev
    macd_flatten_down = macd_hist <= macd_prev

    signal = 0
    if zscore <= -cfg.meanrev_zscore_entry and rsi_val < cfg.meanrev_rsi_buy_max and (macd_flatten_up or macd_hist > 0):
        signal = 1
    elif zscore >= cfg.meanrev_zscore_entry and rsi_val > cfg.meanrev_rsi_sell_min and (macd_flatten_down or macd_hist < 0):
        signal = -1

    if signal == 0:
        return {
            "signal": 0,
            "raw_strength": 0.0,
            "confidence": 0.0,
            "metadata": {"reason": "mean-reversion thresholds not met"},
        }

    z_strength = min(abs(zscore) / (cfg.meanrev_zscore_entry * 1.8), 1.0)
    rsi_strength = min(abs(rsi_val - 50.0) / 20.0, 1.0)
    raw_strength = float(np.clip(0.58 * z_strength + 0.42 * rsi_strength, 0.0, 1.0))

    velocity_bonus = 0.0
    if signal == 1 and rsi_val < 40.0 and rsi_delta_3 >= cfg.meanrev_velocity_bonus_trigger:
        velocity_bonus = min(0.10, 0.02 * (rsi_delta_3 - cfg.meanrev_velocity_bonus_trigger + 1.0))

    confidence = float(np.clip(0.44 + 0.34 * raw_strength + velocity_bonus, 0.0, 0.88))

    return {
        "signal": signal,
        "raw_strength": raw_strength,
        "confidence": confidence,
        "metadata": {
            "zscore": round(zscore, 4),
            "rsi": round(rsi_val, 2),
            "macd_hist": round(macd_hist, 6),
            "rsi_delta_3": round(rsi_delta_3, 4),
            "velocity_bonus": round(velocity_bonus, 4),
        },
    }


def generate_strategy_signal(pre: pd.DataFrame, idx: int, cfg: StrategyConfig | None = None) -> dict:
    """Generate merged strategy output for bar `idx`.

    Returns one structure containing regime, momentum and mean-reversion outputs.
    """
    cfg = cfg or CONFIG.strategy

    if idx <= 0:
        return {
            "regime": "choppy",
            "momentum": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {"reason": "warmup"}},
            "mean_reversion": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {"reason": "warmup"}},
        }

    row = pre.iloc[idx].copy()
    prev_row = pre.iloc[idx - 1]
    row["macd_hist_prev"] = prev_row.get("macd_hist", np.nan)

    critical = ["ema_spread_pct", "rsi_14", "macd_hist", "bb_zscore"]
    if any(pd.isna(row.get(c)) for c in critical):
        return {
            "regime": "choppy",
            "momentum": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {"reason": "indicator warmup"}},
            "mean_reversion": {"signal": 0, "raw_strength": 0.0, "confidence": 0.0, "metadata": {"reason": "indicator warmup"}},
        }

    regime = detect_regime(row, cfg)
    mom = _momentum_signal(row, regime, cfg)
    mr = _mean_reversion_signal(row, regime, cfg)

    logger.debug(
        "Strategy idx=%d regime=%s mom=%s mr=%s",
        idx,
        regime,
        mom["signal"],
        mr["signal"],
    )

    return {
        "regime": regime,
        "momentum": mom,
        "mean_reversion": mr,
        "atr_percentile_rank": 0.80
        if (float(row.get("atr_norm_14", 0.0) or 0.0) >= float(row.get("atr_norm_q80_120", np.inf) or np.inf))
        else 0.50,
    }
