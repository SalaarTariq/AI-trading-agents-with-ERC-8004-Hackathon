#!/usr/bin/env python3
"""
Generate realistic OHLCV crypto datasets using geometric Brownian motion (GBM)
with regime-switching for realistic crypto market behavior.

Datasets:
  d1_new.csv - ETH/USDC 4-hour bars, ~1200 bars, 2023-2024 timeframe
  d2_new.csv - SOL-like mid-cap altcoin daily bars, ~300 rows, trending + choppy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_ohlcv_gbm(
    start_price: float,
    n_bars: int,
    bar_duration_hours: float,
    daily_vol: float,
    daily_drift: float,
    start_date: datetime,
    volume_base: float,
    volume_noise: float = 0.4,
    regimes: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data using geometric Brownian motion.

    Within each bar the intra-bar path is simulated with 20 sub-steps so that
    open/high/low/close relationships are physically consistent.

    Args:
        start_price: Initial price.
        n_bars: Number of bars to generate.
        bar_duration_hours: Duration of each bar in hours.
        daily_vol: Annualized-ish daily volatility (sigma of daily log-return).
        daily_drift: Daily drift (mu of daily log-return).
        start_date: Timestamp of first bar.
        volume_base: Base volume per bar (in quote currency).
        volume_noise: Relative noise for volume (0-1 scale).
        regimes: Optional list of regime dicts with keys:
            start_frac, end_frac, vol_mult, drift_mult
    """
    hours_per_day = 24.0
    dt = bar_duration_hours / hours_per_day  # fraction of a day per bar
    sub_steps = 20  # intra-bar resolution
    dt_sub = dt / sub_steps

    rows = []
    price = start_price

    for i in range(n_bars):
        # Determine regime multipliers
        frac = i / n_bars
        vol_mult = 1.0
        drift_mult = 1.0
        if regimes:
            for r in regimes:
                if r["start_frac"] <= frac < r["end_frac"]:
                    vol_mult = r.get("vol_mult", 1.0)
                    drift_mult = r.get("drift_mult", 1.0)
                    break

        sigma = daily_vol * vol_mult
        mu = daily_drift * drift_mult

        # Simulate intra-bar path
        intra = [price]
        for _ in range(sub_steps):
            z = np.random.standard_normal()
            log_ret = (mu - 0.5 * sigma**2) * dt_sub + sigma * np.sqrt(dt_sub) * z
            price = price * np.exp(log_ret)
            intra.append(price)

        o = intra[0]
        c = intra[-1]
        h = max(intra)
        l = min(intra)

        # Add a tiny wick extension for realism (sometimes wicks poke beyond path)
        wick_up = abs(np.random.normal(0, 0.001 * h))
        wick_dn = abs(np.random.normal(0, 0.001 * l))
        h += wick_up
        l -= wick_dn

        # Enforce OHLC constraints
        h = max(h, o, c)
        l = min(l, o, c)
        l = max(l, 0.01)  # price floor

        # Volume: correlated with absolute return and volatility
        ret_mag = abs(c / o - 1.0) if o > 0 else 0
        vol_factor = 1.0 + 5.0 * ret_mag  # bigger moves -> more volume
        vol = volume_base * vol_factor * vol_mult
        vol *= np.exp(np.random.normal(0, volume_noise))
        vol = max(vol, 1.0)

        ts = start_date + timedelta(hours=bar_duration_hours * i)

        rows.append({
            "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "volume": round(vol, 2),
        })

        price = c  # carry forward

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset 1: ETH/USDC 4-hour bars, ~1200 bars, Jan 2023 – Oct 2024
# ---------------------------------------------------------------------------
# ETH traded roughly 1200-1800 in early 2023, rallied to ~4000 by Mar 2024,
# then consolidated. We model this with regime-switching drift.

eth_regimes = [
    # Jan-Jun 2023: choppy recovery from bear market
    {"start_frac": 0.00, "end_frac": 0.25, "vol_mult": 1.0, "drift_mult": 0.3},
    # Jul-Oct 2023: low-vol sideways
    {"start_frac": 0.25, "end_frac": 0.42, "vol_mult": 0.7, "drift_mult": 0.0},
    # Nov 2023 - Mar 2024: strong rally
    {"start_frac": 0.42, "end_frac": 0.65, "vol_mult": 1.3, "drift_mult": 3.0},
    # Apr-Jul 2024: high-vol correction / consolidation
    {"start_frac": 0.65, "end_frac": 0.82, "vol_mult": 1.5, "drift_mult": -1.0},
    # Aug-Oct 2024: moderate recovery
    {"start_frac": 0.82, "end_frac": 1.00, "vol_mult": 1.0, "drift_mult": 1.0},
]

n_eth_bars = 1200  # 4h bars -> 1200 bars ≈ 200 days * 6 bars/day => ~200 days
# Actually 1200 * 4h = 4800h = 200 days. We want ~Jan2023-Oct2024 = ~22 months = 660 days
# So let's do 660*6 = 3960 bars. That's a lot. Let's target ~1300 bars for a good dataset.
# 1300 * 4 / 24 ≈ 216 days. To cover Jan 2023 - Oct 2024 (21 months) we need ~3800 bars.
# Let's go with 2000 bars to cover a good chunk (about 333 days ≈ 11 months).
# Actually the user asked for "at least 1000 bars covering 2023-2024". Let's do 2500 bars
# covering ~417 days (~14 months), say Jan 2023 - Mar 2024.

n_eth_bars = 2500

df_eth = generate_ohlcv_gbm(
    start_price=1200.0,    # ETH ~$1200 at start of Jan 2023
    n_bars=n_eth_bars,
    bar_duration_hours=4.0,
    daily_vol=0.035,       # ~3.5% daily vol, typical for ETH
    daily_drift=0.002,     # slight positive drift
    start_date=datetime(2023, 1, 1, 0, 0, 0),
    volume_base=15_000_000,  # ~$15M per 4h bar (reasonable for ETH/USDC)
    volume_noise=0.5,
    regimes=eth_regimes,
)

# ---------------------------------------------------------------------------
# Dataset 2: SOL-like mid-cap altcoin, daily bars, ~300 rows
# ---------------------------------------------------------------------------
# SOL went from ~$10 in Jan 2023 to ~$250 by Dec 2023, then corrected.
# Higher vol than ETH, more dramatic regime changes.

sol_regimes = [
    # Jan-Apr 2023: slow recovery from FTX bottom
    {"start_frac": 0.00, "end_frac": 0.20, "vol_mult": 1.2, "drift_mult": 0.5},
    # May-Jul 2023: choppy sideways
    {"start_frac": 0.20, "end_frac": 0.40, "vol_mult": 0.8, "drift_mult": -0.2},
    # Aug-Oct 2023: start of rally
    {"start_frac": 0.40, "end_frac": 0.55, "vol_mult": 1.0, "drift_mult": 2.5},
    # Nov 2023 - Jan 2024: parabolic rally
    {"start_frac": 0.55, "end_frac": 0.70, "vol_mult": 1.8, "drift_mult": 4.0},
    # Feb-May 2024: volatile correction
    {"start_frac": 0.70, "end_frac": 0.85, "vol_mult": 1.6, "drift_mult": -2.0},
    # Jun-Oct 2024: gradual recovery
    {"start_frac": 0.85, "end_frac": 1.00, "vol_mult": 1.0, "drift_mult": 1.0},
]

n_sol_bars = 300

df_sol = generate_ohlcv_gbm(
    start_price=12.0,      # SOL ~$12 at start of Jan 2023
    n_bars=n_sol_bars,
    bar_duration_hours=24.0,  # daily bars
    daily_vol=0.05,         # ~5% daily vol, typical for mid-cap alt
    daily_drift=0.003,      # slight positive drift
    start_date=datetime(2023, 1, 1, 0, 0, 0),
    volume_base=200_000_000,  # ~$200M daily volume
    volume_noise=0.5,
    regimes=sol_regimes,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
d1_path = os.path.join(DATA_DIR, "d1_new.csv")
d2_path = os.path.join(DATA_DIR, "d2_new.csv")

df_eth.to_csv(d1_path, index=False)
df_sol.to_csv(d2_path, index=False)

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
print("=" * 60)
print("Dataset 1: ETH/USDC 4-hour bars")
print(f"  Rows: {len(df_eth)}")
print(f"  Date range: {df_eth['datetime'].iloc[0]} to {df_eth['datetime'].iloc[-1]}")
print(f"  Price range: ${df_eth['low'].min():.2f} - ${df_eth['high'].max():.2f}")
print(f"  Start: ${df_eth['open'].iloc[0]:.2f}  End: ${df_eth['close'].iloc[-1]:.2f}")
print(f"  Saved to: {d1_path}")
print()
print("Dataset 2: SOL-like daily bars")
print(f"  Rows: {len(df_sol)}")
print(f"  Date range: {df_sol['datetime'].iloc[0]} to {df_sol['datetime'].iloc[-1]}")
print(f"  Price range: ${df_sol['low'].min():.2f} - ${df_sol['high'].max():.2f}")
print(f"  Start: ${df_sol['open'].iloc[0]:.2f}  End: ${df_sol['close'].iloc[-1]:.2f}")
print(f"  Saved to: {d2_path}")
print()

# Validation checks
for name, df in [("d1_new", df_eth), ("d2_new", df_sol)]:
    violations = 0
    for _, row in df.iterrows():
        if row["high"] < max(row["open"], row["close"]):
            violations += 1
        if row["low"] > min(row["open"], row["close"]):
            violations += 1
    print(f"{name}: OHLC constraint violations = {violations}")

print("=" * 60)
print("Done!")
