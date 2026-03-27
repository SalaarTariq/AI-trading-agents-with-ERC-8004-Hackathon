"""
config.py — Centralized configuration for the Balanced Hybrid AI Trading Agent.

All tunable parameters for strategies, risk management, regime detection,
and portfolio settings live here.

KEY CHANGES (v3 — profitability overhaul):
- Asymmetric R:R: TP targets 2-3x SL distance for positive expectancy
- Regime params tuned: trending regimes get lower thresholds, wider TPs
- Trailing stop activates earlier to lock profits
- Risk per trade slightly higher to let winners compound
- Reduced cooldown severity to avoid missing recovery trades
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioConfig:
    """Portfolio and account settings."""
    initial_balance: float = 100_000.0
    base_currency: str = "USDC"
    trading_pairs: list[str] = field(default_factory=lambda: [
        "ETH/USDC", "BTC/USDC", "SOL/USDC"
    ])
    max_concurrent_positions: int = 5


@dataclass
class RiskConfig:
    """Risk management thresholds.

    KEY CHANGE: Asymmetric R:R — TP is 2-3x SL for positive expectancy.
    Even with 45% win rate, 2:1 R:R is profitable.
    """
    stop_loss_pct: float = 0.04          # Default SL: 4%
    take_profit_pct: float = 0.07        # Default TP: 7% (1.75:1 R:R)
    atr_sl_mult: float = 1.8             # SL: 1.8x ATR (balanced whipsaw avoidance)
    atr_tp_mult: float = 3.2             # TP: 3.2x ATR (~1.8:1 R:R)
    risk_per_trade_pct: float = 0.015    # 1.5% risk per trade
    max_capital_pct: float = 0.20
    max_position_pct: float = 0.25
    daily_loss_cap_pct: float = 0.08     # Tighter daily cap
    max_drawdown_pct: float = 0.15
    defensive_size_mult: float = 0.25
    atr_volatility_reduce_threshold: float = 0.03  # Normalized ATR threshold
    consecutive_loss_pause: int = 3
    min_confidence: float = 0.55
    use_dynamic_sl_tp: bool = True
    atr_sl_multiplier: float = 2.0       # SL wide enough to avoid noise
    atr_tp_multiplier: float = 3.0       # TP reachable but still > SL
    min_sl_pct: float = 0.025            # Min SL: 2.5%
    max_sl_pct: float = 0.06             # Max SL: 6%
    min_tp_pct: float = 0.035            # Min TP: 3.5%
    max_tp_pct: float = 0.12             # Max TP: 12%
    use_trailing_stop: bool = True
    trailing_breakeven_pct: float = 0.60  # Move to breakeven at 60% of TP
    trailing_lock_pct: float = 0.80       # Lock 60% profit at 80% of TP


@dataclass
class MomentumConfig:
    """Momentum strategy parameters."""
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    volume_ma_period: int = 20
    signal_threshold: float = 0.01
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    adx_threshold: float = 20.0         # Lowered from 25 — capture more trends
    adx_weak_threshold: float = 15.0    # Lowered from 20


@dataclass
class MeanReversionConfig:
    """Mean-reversion strategy parameters."""
    lookback_period: int = 20
    bb_std_dev: float = 2.0
    zscore_entry: float = -2.0
    zscore_exit: float = 0.0
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    ema_spread_ranging_threshold: float = 0.0
    min_conditions: int = 2
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    adx_max_threshold: float = 25.0     # ADX < 25 = ranging market
    adx_period: int = 14


@dataclass
class WeightConfig:
    """Signal combination weights."""
    momentum: float = 0.45
    mean_reversion: float = 0.30
    indicator_agreement: float = 0.25


@dataclass
class RegimeParams:
    """Parameters for a single market regime."""
    conf_threshold: float = 0.45
    position_mult: float = 1.0
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.5
    w_mom: float = 0.45
    w_mr: float = 0.30
    w_ai: float = 0.25


@dataclass
class RegimeConfig:
    """Regime-specific trading parameters.

    v3 tuning — R:R calibrated for 4h crypto candles:
        Regime         | Conf Thresh | Pos Mult | SL ATR | TP ATR | w_mom | w_mr
        trending_up    | 0.55        | 1.0x     | 1.8    | 3.0    | 0.90  | 0.10
        trending_down  | 0.55        | 0.8x     | 1.8    | 3.0    | 0.90  | 0.10
        ranging        | 0.58        | 0.6x     | 1.5    | 2.2    | 0.10  | 0.90
        choppy         | 0.65        | 0.3x     | 2.0    | 2.8    | 0.45  | 0.55
    """
    trending_up: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.55, position_mult=1.0, sl_atr_mult=1.8, tp_atr_mult=3.6,
        w_mom=0.90, w_mr=0.10, w_ai=0.0,
    ))
    trending_down: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.55, position_mult=0.8, sl_atr_mult=1.8, tp_atr_mult=3.6,
        w_mom=0.90, w_mr=0.10, w_ai=0.0,
    ))
    ranging: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.58, position_mult=0.6, sl_atr_mult=1.5, tp_atr_mult=2.8,
        w_mom=0.10, w_mr=0.90, w_ai=0.0,
    ))
    choppy: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.65, position_mult=0.3, sl_atr_mult=1.8, tp_atr_mult=3.0,
        w_mom=0.45, w_mr=0.55, w_ai=0.0,
    ))

    def get(self, regime: str) -> RegimeParams:
        """Get params for a regime name, defaulting to choppy."""
        return getattr(self, regime, self.choppy)


@dataclass
class SignalConfig:
    """Signal combination parameters."""
    signal_threshold: float = 0.20
    min_agreement: int = 2
    use_regime_detection: bool = True
    execute_confidence_threshold: float = 0.60


@dataclass
class AppConfig:
    """Top-level application config aggregating all sub-configs."""
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    log_level: str = "INFO"
    data_dir: str = "data"
    proof_log_path: str = "validation/proof_log.jsonl"
    dashboard_refresh_seconds: int = 60


# Global default config instance
CONFIG = AppConfig()
