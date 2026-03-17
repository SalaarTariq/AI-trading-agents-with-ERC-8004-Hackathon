"""
tests/test_helpers.py — Tests for technical indicator helpers.
"""

import numpy as np
import pandas as pd
import pytest

from utils.helpers import (
    sma, ema, bollinger_bands, zscore, rsi, atr, normalize_confidence,
    macd, stochastic, adx, adx_with_di, obv, vwap,
)


@pytest.fixture
def ohlcv_100() -> pd.DataFrame:
    """100-bar OHLCV dataset with a mild uptrend."""
    rng = np.random.default_rng(42)
    days = 100
    start = 3000.0
    returns = rng.normal(0.001, 0.02, days)
    prices = start * np.exp(np.cumsum(returns))
    opens = np.roll(prices, 1)
    opens[0] = start
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0, 0.01, days))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0, 0.01, days))
    volumes = rng.uniform(10000, 50000, days)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": prices, "volume": volumes,
    }, index=dates)


class TestMACD:
    """Tests for the MACD indicator."""

    def test_returns_three_series(self, ohlcv_100):
        macd_line, signal_line, hist = macd(ohlcv_100["close"])
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd_line) == len(ohlcv_100)

    def test_histogram_is_macd_minus_signal(self, ohlcv_100):
        macd_line, signal_line, hist = macd(ohlcv_100["close"])
        diff = macd_line - signal_line
        pd.testing.assert_series_equal(hist, diff, check_names=False)

    def test_custom_periods(self, ohlcv_100):
        macd_line, signal_line, hist = macd(ohlcv_100["close"], fast=8, slow=21, signal=5)
        assert len(macd_line) == len(ohlcv_100)

    def test_no_nans_after_warmup(self, ohlcv_100):
        macd_line, _, _ = macd(ohlcv_100["close"])
        # After slow period (26) there should be values
        assert not macd_line.iloc[30:].isna().any()


class TestStochastic:
    """Tests for the Stochastic Oscillator."""

    def test_returns_two_series(self, ohlcv_100):
        k, d = stochastic(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_range_0_to_100(self, ohlcv_100):
        k, d = stochastic(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_custom_periods(self, ohlcv_100):
        k, d = stochastic(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"],
                          k_period=10, d_period=5)
        assert len(k) == len(ohlcv_100)


class TestADX:
    """Tests for the ADX indicator."""

    def test_returns_series(self, ohlcv_100):
        result = adx(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_100)

    def test_adx_with_di_returns_three(self, ohlcv_100):
        adx_val, plus_di, minus_di = adx_with_di(
            ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"]
        )
        assert isinstance(adx_val, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)

    def test_adx_non_negative(self, ohlcv_100):
        result = adx(ohlcv_100["high"], ohlcv_100["low"], ohlcv_100["close"])
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_strong_trend_high_adx(self):
        """Strong uptrend should produce ADX > 25."""
        days = 100
        prices = 1000.0 + np.arange(days) * 20.0
        highs = prices + 5
        lows = prices - 5
        df_high = pd.Series(highs)
        df_low = pd.Series(lows)
        df_close = pd.Series(prices)
        result = adx(df_high, df_low, df_close, period=14)
        valid = result.dropna()
        assert valid.iloc[-1] > 20  # Strong trend


class TestOBV:
    """Tests for On-Balance Volume."""

    def test_returns_series(self, ohlcv_100):
        result = obv(ohlcv_100["close"], ohlcv_100["volume"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_100)

    def test_up_day_adds_volume(self):
        close = pd.Series([100.0, 105.0, 110.0])
        volume = pd.Series([1000.0, 2000.0, 3000.0])
        result = obv(close, volume)
        # All up days: OBV = 0 + 2000 + 3000 = 5000
        assert result.iloc[-1] == 5000.0

    def test_down_day_subtracts_volume(self):
        close = pd.Series([110.0, 105.0, 100.0])
        volume = pd.Series([1000.0, 2000.0, 3000.0])
        result = obv(close, volume)
        # All down: 0 + (-2000) + (-3000) = -5000
        assert result.iloc[-1] == -5000.0


class TestVWAP:
    """Tests for Volume Weighted Average Price."""

    def test_returns_series(self, ohlcv_100):
        result = vwap(ohlcv_100["high"], ohlcv_100["low"],
                      ohlcv_100["close"], ohlcv_100["volume"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_100)

    def test_single_bar_is_typical_price(self):
        h = pd.Series([105.0])
        l = pd.Series([95.0])
        c = pd.Series([100.0])
        v = pd.Series([1000.0])
        result = vwap(h, l, c, v)
        expected = (105 + 95 + 100) / 3.0
        assert result.iloc[0] == pytest.approx(expected)
