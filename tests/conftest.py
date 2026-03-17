"""
tests/conftest.py — Shared fixtures for all test modules.

Provides reusable synthetic OHLCV data, portfolio states, and
configuration overrides for deterministic testing.
"""

import numpy as np
import pandas as pd
import pytest

from config import AppConfig, RiskConfig, MomentumConfig, MeanReversionConfig
from risk.risk_manager import PortfolioState


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 200 rows of deterministic synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    days = 200
    start = 3000.0

    returns = rng.normal(0.0002, 0.02, days)
    prices = start * np.exp(np.cumsum(returns))

    opens = np.roll(prices, 1)
    opens[0] = start
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0, 0.015, days))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0, 0.015, days))
    volumes = rng.uniform(5000, 30000, days)

    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=dates)


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Generate data with a clear uptrend."""
    days = 100
    prices = 3000.0 + np.arange(days) * 15.0  # Steady climb
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 5, days)
    prices = prices + noise

    volumes = rng.uniform(10000, 30000, days)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "open": np.roll(prices, 1),
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": volumes,
    }, index=dates)


@pytest.fixture
def trending_down_data() -> pd.DataFrame:
    """Generate data with a clear downtrend."""
    days = 100
    prices = 5000.0 - np.arange(days) * 15.0  # Steady decline
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 5, days)
    prices = prices + noise

    volumes = rng.uniform(10000, 30000, days)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "open": np.roll(prices, 1),
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": volumes,
    }, index=dates)


@pytest.fixture
def flat_data() -> pd.DataFrame:
    """Generate sideways / flat price data."""
    days = 100
    rng = np.random.default_rng(99)
    prices = 3000.0 + rng.normal(0, 2, days)  # Very low variance
    volumes = rng.uniform(10000, 20000, days)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "open": np.roll(prices, 1),
        "high": prices + 5,
        "low": prices - 5,
        "close": prices,
        "volume": volumes,
    }, index=dates)


@pytest.fixture
def default_portfolio() -> PortfolioState:
    """Default portfolio for risk testing."""
    return PortfolioState(
        total_value=100_000.0,
        cash=80_000.0,
        positions={"ETH": 20_000.0},
        daily_pnl=0.0,
        peak_value=100_000.0,
        consecutive_losses=0,
    )


@pytest.fixture
def default_config() -> AppConfig:
    """Default app config."""
    return AppConfig()
