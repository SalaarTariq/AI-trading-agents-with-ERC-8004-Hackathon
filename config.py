"""
config.py — Centralized configuration for the Balanced Hybrid AI Trading Agent.

All tunable parameters for strategies, risk management, regime detection,
and portfolio settings live here.
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
    """Risk management thresholds."""
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.06
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 3.0
    risk_per_trade_pct: float = 0.02
    max_capital_pct: float = 0.25
    max_position_pct: float = 0.30
    daily_loss_cap_pct: float = 0.10
    max_drawdown_pct: float = 0.15
    defensive_size_mult: float = 0.25
    atr_volatility_reduce_threshold: float = 2.0
    consecutive_loss_pause: int = 3
    min_confidence: float = 0.60
    use_dynamic_sl_tp: bool = True
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    min_sl_pct: float = 0.02
    max_sl_pct: float = 0.08
    min_tp_pct: float = 0.03
    max_tp_pct: float = 0.12
    use_trailing_stop: bool = False
    trailing_breakeven_pct: float = 0.65
    trailing_lock_pct: float = 0.85


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
    adx_threshold: float = 25.0
    adx_weak_threshold: float = 20.0


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
    adx_max_threshold: float = 30.0
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
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    w_mom: float = 0.45
    w_mr: float = 0.30
    w_ai: float = 0.25


@dataclass
class RegimeConfig:
    """Regime-specific trading parameters.

    Tuned for real crypto data (live Binance 4h):
        Regime         | Conf Thresh | Pos Mult | SL ATR | TP ATR | w_mom | w_mr
        trending_up    | 0.40        | 1.0x     | 1.0    | 1.8    | 0.55  | 0.15
        trending_down  | 0.25        | 1.0x     | 1.0    | 1.8    | 0.60  | 0.10
        choppy         | 0.65        | 0.3x     | 1.2    | 1.5    | 0.20  | 0.50
    """
    trending_up: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.40, position_mult=1.0, sl_atr_mult=1.0, tp_atr_mult=1.8,
        w_mom=0.55, w_mr=0.15, w_ai=0.30,
    ))
    trending_down: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.25, position_mult=1.0, sl_atr_mult=1.0, tp_atr_mult=1.8,
        w_mom=0.60, w_mr=0.10, w_ai=0.30,
    ))
    choppy: RegimeParams = field(default_factory=lambda: RegimeParams(
        conf_threshold=0.65, position_mult=0.3, sl_atr_mult=1.2, tp_atr_mult=1.5,
        w_mom=0.20, w_mr=0.50, w_ai=0.30,
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
