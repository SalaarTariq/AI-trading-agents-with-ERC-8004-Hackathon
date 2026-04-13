"""Centralized configuration for the cleaned ERC-8004 trading agent.

This file intentionally keeps all tunable settings in one place so strategy,
risk, and execution behavior can be changed without touching business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioConfig:
    """Portfolio/account-level settings."""

    initial_balance: float = 100_000.0
    base_currency: str = "USDC"
    trading_pairs: list[str] = field(default_factory=lambda: ["ETH/USDC"])
    max_concurrent_positions: int = 5


@dataclass
class StrategyConfig:
    """Strategy and regime thresholds used by modules/strategy.py."""

    # Indicator periods
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std_dev: float = 2.0
    adx_period: int = 14

    # Regime detection thresholds
    regime_trend_up_spread_pct: float = 0.0025
    regime_trend_down_spread_pct: float = -0.0025
    regime_ranging_abs_spread_pct: float = 0.0015
    regime_trend_adx_min: float = 25.0
    regime_ranging_adx_max: float = 20.0

    # Momentum entry thresholds (used in trending regimes)
    momentum_spread_pct_min: float = 0.0025
    momentum_macd_hist_abs_min: float = 0.03
    momentum_rsi_min: float = 45.0
    momentum_rsi_max: float = 55.0

    # Mean-reversion entry thresholds (used in ranging regime)
    meanrev_zscore_entry: float = 1.8
    meanrev_rsi_buy_max: float = 35.0
    meanrev_rsi_sell_min: float = 65.0
    meanrev_velocity_bonus_trigger: float = 5.0


@dataclass
class SignalConfig:
    """Confidence-combining and execution gate settings."""

    execute_confidence_threshold: float = 0.67
    strong_support_min_strength: float = 0.55

    # Regime-aware blending between momentum and mean-reversion strengths.
    trend_momentum_weight: float = 0.85
    trend_meanrev_weight: float = 0.15
    range_momentum_weight: float = 0.15
    range_meanrev_weight: float = 0.85
    choppy_momentum_weight: float = 0.55
    choppy_meanrev_weight: float = 0.45

    agreement_bonus: float = 0.15
    regime_quality_bonus: float = 0.10
    high_volatility_penalty: float = 0.20


@dataclass
class RegimeParams:
    """Risk execution multipliers per regime."""

    conf_threshold: float = 0.67
    position_mult: float = 1.0
    sl_atr_mult: float = 2.2
    tp_atr_mult: float = 3.8


@dataclass
class RegimeConfig:
    """Risk defaults for each regime."""

    trending_up: RegimeParams = field(
        default_factory=lambda: RegimeParams(conf_threshold=0.67, position_mult=1.00)
    )
    trending_down: RegimeParams = field(
        default_factory=lambda: RegimeParams(conf_threshold=0.67, position_mult=0.60)
    )
    ranging: RegimeParams = field(
        default_factory=lambda: RegimeParams(conf_threshold=0.67, position_mult=0.80)
    )
    choppy: RegimeParams = field(
        default_factory=lambda: RegimeParams(conf_threshold=0.70, position_mult=0.40)
    )

    def get(self, regime: str) -> RegimeParams:
        return getattr(self, regime, self.choppy)


@dataclass
class RiskConfig:
    """Risk-manager controls kept strict for capital protection."""

    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.06

    min_confidence: float = 0.67
    daily_loss_cap_pct: float = 0.08
    max_drawdown_pct: float = 0.15

    max_position_pct: float = 0.25
    max_capital_pct: float = 0.20
    risk_per_trade_pct: float = 0.015

    atr_volatility_reduce_threshold: float = 0.03

    use_dynamic_sl_tp: bool = True
    atr_sl_multiplier: float = 2.2
    atr_tp_multiplier: float = 3.8
    min_sl_pct: float = 0.025
    max_sl_pct: float = 0.06
    min_tp_pct: float = 0.035
    max_tp_pct: float = 0.12

    use_trailing_stop: bool = True
    trailing_breakeven_pct: float = 0.60
    trailing_lock_pct: float = 0.70

    consecutive_loss_pause: int = 3


@dataclass
class AppConfig:
    """Top-level app config used by the main pipeline."""

    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    log_level: str = "INFO"
    data_dir: str = "data"
    proof_log_path: str = "validation/proof_log.jsonl"


CONFIG = AppConfig()
