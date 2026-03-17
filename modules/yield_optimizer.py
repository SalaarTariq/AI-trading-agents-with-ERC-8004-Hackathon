"""
modules/yield_optimizer.py — Yield optimization strategy.


Evaluates simulated DeFi yield/LP opportunities and suggests
allocation or rebalancing actions. In paper-trading mode this
uses synthetic pool data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import YieldConfig, CONFIG
from utils.helpers import normalize_confidence, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass
class PoolInfo:
    """Simulated liquidity pool metadata."""
    name: str
    apy: float              # Annualized percentage yield (0.05 = 5%)
    tvl: float              # Total value locked in USD
    risk_score: float       # 0 (safe) to 1 (risky)
    current_allocation: float = 0.0  # Current allocation fraction


# Default simulated pools for paper trading
DEFAULT_POOLS: list[PoolInfo] = [
    PoolInfo("ETH-USDC Uniswap V3",  apy=0.08,  tvl=50_000_000, risk_score=0.2),
    PoolInfo("SOL-USDC Raydium",      apy=0.12,  tvl=15_000_000, risk_score=0.4),
    PoolInfo("BTC-USDC Curve",        apy=0.05,  tvl=80_000_000, risk_score=0.1),
    PoolInfo("MATIC-USDC Quickswap",  apy=0.15,  tvl=8_000_000,  risk_score=0.5),
]


def generate_signal(
    df: pd.DataFrame | None = None,
    pools: list[PoolInfo] | None = None,
    portfolio_value: float = 100_000.0,
    cfg: YieldConfig | None = None,
) -> dict:
    """
    Evaluate yield opportunities and produce a signal.

    Logic:
        - Rank pools by risk-adjusted APY (apy * (1 - risk_score)).
        - BUY (allocate) if the best pool exceeds min APY threshold.
        - SELL (withdraw) if current allocation is in a pool whose
          risk-adjusted APY has dropped below threshold.
        - HOLD if no actionable opportunity exists.

    Args:
        df: OHLCV data (used for volatility context; optional).
        pools: List of pool opportunities to evaluate.
        portfolio_value: Current portfolio value for sizing.
        cfg: Optional YieldConfig override.

    Returns:
        Standardized signal dict with allocation suggestions.
    """
    cfg = cfg or CONFIG.yield_opt
    pools = pools if pools is not None else DEFAULT_POOLS

    if not pools:
        return _hold_signal("no pools available")

    # Compute risk-adjusted APY for each pool
    scored = []
    for pool in pools:
        adj_apy = pool.apy * (1.0 - pool.risk_score)
        max_alloc = min(cfg.max_pool_allocation_pct, pool.tvl / max(portfolio_value, 1))
        scored.append((pool, adj_apy, max_alloc))

    # Sort by risk-adjusted APY descending
    scored.sort(key=lambda x: x[1], reverse=True)
    best_pool, best_adj_apy, best_max_alloc = scored[0]

    # Determine signal
    if best_adj_apy >= cfg.min_apy:
        direction = "BUY"
        confidence = normalize_confidence(best_adj_apy / 0.15)  # 15% adj APY = max confidence
        suggestion = {
            "pool": best_pool.name,
            "apy": round(best_pool.apy, 4),
            "risk_adjusted_apy": round(best_adj_apy, 4),
            "suggested_allocation": round(best_max_alloc, 4),
            "suggested_amount": round(portfolio_value * best_max_alloc, 2),
        }
    else:
        direction = "HOLD"
        confidence = 0.0
        suggestion = {"reason": "no pool meets minimum APY threshold"}

    # Check if any current allocations should be withdrawn
    withdrawals = []
    for pool in pools:
        if pool.current_allocation > 0:
            adj = pool.apy * (1.0 - pool.risk_score)
            if adj < cfg.min_apy * 0.5:  # Below 50% of min threshold → suggest exit
                withdrawals.append(pool.name)

    if withdrawals and direction == "HOLD":
        direction = "SELL"
        confidence = 0.3

    metadata = {
        "top_pools": [
            {"name": p.name, "apy": round(p.apy, 4), "risk_adj_apy": round(a, 4)}
            for p, a, _ in scored[:3]
        ],
        "suggestion": suggestion,
        "withdrawals": withdrawals,
        "timestamp": utc_now_iso(),
    }

    logger.info("Yield signal=%s confidence=%.3f best_pool=%s adj_apy=%.3f",
                direction, confidence, best_pool.name, best_adj_apy)

    return {
        "signal": direction,
        "confidence": confidence,
        "metadata": metadata,
    }


def _hold_signal(reason: str) -> dict:
    """Return neutral HOLD signal."""
    return {
        "signal": "HOLD",
        "confidence": 0.0,
        "metadata": {"reason": reason},
    }
