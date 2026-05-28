"""Tests for utils/data_loader.py — CSV loading and synthetic generation."""

import pandas as pd
import pytest

from utils.data_loader import generate_synthetic_ohlcv, load_csv, load_or_generate


class TestLoadCsv:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "does_not_exist.csv")

    def test_loads_ohlcv_round_trip(self, tmp_path, sample_ohlcv):
        path = tmp_path / "ohlcv.csv"
        sample_ohlcv.to_csv(path, index_label="timestamp")
        loaded = load_csv(path)
        assert set(loaded.columns) >= {"open", "high", "low", "close", "volume"}
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert len(loaded) == len(sample_ohlcv)

    def test_case_insensitive_columns(self, tmp_path):
        # Mixed case headers should be normalized to lowercase.
        path = tmp_path / "ohlcv.csv"
        path.write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2025-01-01,100,110,90,105,1000\n"
            "2025-01-02,105,115,100,110,1200\n"
        )
        df = load_csv(path)
        assert "close" in df.columns and "Close" not in df.columns

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("timestamp,open,high,low,close\n2025-01-01,1,2,0.5,1.5\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(path)


class TestGenerateSynthetic:
    def test_default_shape(self):
        df = generate_synthetic_ohlcv(days=120)
        assert len(df) == 120
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_seed_reproducible(self):
        a = generate_synthetic_ohlcv(days=50, seed=7)
        b = generate_synthetic_ohlcv(days=50, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_high_geq_low(self):
        df = generate_synthetic_ohlcv(days=200)
        assert (df["high"] >= df["low"]).all()


class TestLoadOrGenerate:
    def test_no_path_returns_synthetic(self):
        df = load_or_generate(None, days=30)
        assert len(df) == 30

    def test_missing_path_falls_back(self, tmp_path):
        df = load_or_generate(tmp_path / "missing.csv", days=40)
        assert len(df) == 40

    def test_existing_path_loads_csv(self, tmp_path, sample_ohlcv):
        path = tmp_path / "ohlcv.csv"
        sample_ohlcv.to_csv(path, index_label="timestamp")
        df = load_or_generate(path)
        assert len(df) == len(sample_ohlcv)
