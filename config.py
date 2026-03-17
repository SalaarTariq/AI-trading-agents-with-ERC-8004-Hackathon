"""
config.py — Centralized configuration for the Balanced Hybrid AI Trading Agent.


All tunable parameters for strategies, risk management, AI weighting,
and portfolio settings live here. Import this module wherever you need
configuration values.
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
    stop_loss_pct: float = 0.04          # 4% default stop-loss per trade
    take_profit_pct: float = 0.06        # 6% default take-profit per trade
    max_position_pct: float = 0.30       # 30% max of portfolio per position
    daily_loss_cap_pct: float = 0.10     # 10% daily loss cap
    max_drawdown_pct: float = 0.15       # 15% drawdown triggers defensive mode
    defensive_size_mult: float = 0.25    # 75% reduction in defensive mode
    volatility_threshold: float = 2.0    # Skip if vol > 2x 30-day avg
    consecutive_loss_pause: int = 3      # Pause after N consecutive stop-losses
    min_confidence: float = 0.55         # Minimum combined confidence to trade
    # Dynamic SL/TP (ATR-based)
    use_dynamic_sl_tp: bool = True       # Enable ATR-based SL/TP
    atr_sl_multiplier: float = 2.0       # SL = ATR * this
    atr_tp_multiplier: float = 3.0       # TP = ATR * this
    min_sl_pct: float = 0.02            # Floor: 2% minimum SL
    max_sl_pct: float = 0.08            # Ceiling: 8% maximum SL
    min_tp_pct: float = 0.03            # Floor: 3% minimum TP
    max_tp_pct: float = 0.12            # Ceiling: 12% maximum TP
    # Trailing stop
    use_trailing_stop: bool = True
    trailing_breakeven_pct: float = 0.50  # At 50% of TP, tighten SL to breakeven
    trailing_lock_pct: float = 0.75       # At 75% of TP, lock 50% of gains


@dataclass
class MomentumConfig:
    """Momentum strategy parameters."""
    fast_period: int = 12
    slow_period: int = 26
    volume_ma_period: int = 20
    signal_threshold: float = 0.01       # Min crossover spread to trigger signal
    # MACD confirmation
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # ADX trend strength filter
    adx_period: int = 14
    adx_threshold: float = 25.0          # Strong trend threshold
    adx_weak_threshold: float = 20.0     # Below this = no trend → HOLD


@dataclass
class MeanReversionConfig:
    """Mean-reversion strategy parameters."""
    lookback_period: int = 20
    bb_std_dev: float = 2.0              # Bollinger Band standard deviations
    zscore_entry: float = -2.0           # Z-score threshold to enter (buy)
    zscore_exit: float = 0.0             # Z-score threshold to exit
    rsi_period: int = 14
    rsi_oversold: float = 35.0           # Widened from 30
    rsi_overbought: float = 65.0         # Widened from 70
    # 2-of-N condition logic
    min_conditions: int = 2              # Min conditions to trigger (out of 4)
    # Stochastic confirmer
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    # ADX guard: suppress mean reversion in strong trends
    adx_max_threshold: float = 30.0      # Suppress when ADX > this
    adx_period: int = 14


@dataclass
class YieldConfig:
    """Yield optimizer parameters."""
    min_apy: float = 0.05               # 5% minimum APY to consider
    max_pool_allocation_pct: float = 0.20  # 20% max per pool
    rebalance_threshold: float = 0.02    # 2% drift triggers rebalance


@dataclass
class AIConfig:
    """AI predictor configuration."""
    model_type: str = "lightweight"      # "lightweight" (sklearn) or "llm" (GPT4All)
    llm_model_name: str = "orca-mini-3b-gguf2-q4_0.gguf"
    confidence_weight: float = 0.35      # AI weight in combined decision
    use_llm: bool = False                # Set True to use GPT4All (requires download)


@dataclass
class WeightConfig:
    """Signal combination weights (must sum to 1.0)."""
    momentum: float = 0.30
    mean_reversion: float = 0.30
    yield_optimizer: float = 0.0
    ai_predictor: float = 0.40


@dataclass
class RegimeWeights:
    """Regime-adaptive signal weights."""
    # Trending market: favor momentum + AI
    trending_momentum: float = 0.45
    trending_mean_reversion: float = 0.10
    trending_ai: float = 0.45
    # Ranging market: favor mean reversion + AI
    ranging_momentum: float = 0.10
    ranging_mean_reversion: float = 0.45
    ranging_ai: float = 0.45
    # Volatile market: reduce all, lean on AI
    volatile_momentum: float = 0.20
    volatile_mean_reversion: float = 0.20
    volatile_ai: float = 0.60


@dataclass
class SignalConfig:
    """Signal combination parameters."""
    signal_threshold: float = 0.20       # Min score to trigger BUY/SELL (raised from 0.12)
    min_agreement: int = 2               # Min strategies agreeing on direction
    use_regime_detection: bool = True


@dataclass
class AppConfig:
    """Top-level application config aggregating all sub-configs."""
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    yield_opt: YieldConfig = field(default_factory=YieldConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    regime_weights: RegimeWeights = field(default_factory=RegimeWeights)
    signal: SignalConfig = field(default_factory=SignalConfig)
    log_level: str = "INFO"
    data_dir: str = "data"
    proof_log_path: str = "validation/proof_log.jsonl"
    dashboard_refresh_seconds: int = 60


# Global default config instance
CONFIG = AppConfig()
