"""
utils/data_loader.py — Market data loading and preprocessing.


Loads historical price data from CSV files and generates synthetic
data for testing when no real data is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Expected columns: timestamp (or date), open, high, low, close, volume.
    Column names are case-insensitive.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with lowercase column names and a DatetimeIndex.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify and set the datetime index
    time_col = None
    for candidate in ("timestamp", "date", "datetime", "time"):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def generate_synthetic_ohlcv(
    days: int = 365,
    start_price: float = 3000.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Uses a geometric Brownian motion model to simulate realistic
    price movements.

    Args:
        days: Number of daily candles to generate.
        start_price: Starting close price.
        volatility: Daily return standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)

    # Generate daily log returns
    returns = rng.normal(loc=0.0002, scale=volatility, size=days)
    prices = start_price * np.exp(np.cumsum(returns))

    # Simulate OHLCV from close prices
    opens = np.roll(prices, 1)
    opens[0] = start_price
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0, 0.015, size=days))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0, 0.015, size=days))
    volumes = rng.uniform(1_000, 50_000, size=days) * (prices / start_price)

    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq="D")

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=dates)
    df.index.name = "timestamp"

    logger.info("Generated %d synthetic OHLCV rows (seed=%d)", days, seed)
    return df


def load_or_generate(filepath: str | Path | None = None, **kwargs) -> pd.DataFrame:
    """
    Load data from CSV if available, otherwise generate synthetic data.

    Args:
        filepath: Optional path to a CSV file.
        **kwargs: Passed to generate_synthetic_ohlcv if generating.

    Returns:
        OHLCV DataFrame.
    """
    if filepath and Path(filepath).exists():
        return load_csv(filepath)
    logger.warning("No data file found — generating synthetic data for testing.")
    return generate_synthetic_ohlcv(**kwargs)
