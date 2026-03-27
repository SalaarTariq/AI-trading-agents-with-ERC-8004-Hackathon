"""
test_agent_on_datasets.py — Test trading agent on real datasets.

Generates signals using the actual momentum and mean-reversion modules,
then backtests trades to measure accuracy and returns.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from modules.momentum import generate_signal as momentum_signal
from modules.mean_reversion import generate_signal as mean_reversion_signal
from modules.confidence_scoring import compute_confidence, get_execution_threshold
from utils.indicators import compute_indicators

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class BacktestResults:
    """Results from backtest."""
    dataset_name: str
    total_bars: int
    signals_generated: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_return_pct: float
    avg_trade_return_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_bars_per_trade: int
    sharpe_ratio: float
    max_drawdown_pct: float


_LAST_TRADES: dict[str, list[dict]] = {}
TRADE_COOLDOWN_BARS = 4


def _detect_regime(ind) -> str:
    """Simple regime detector using trend strength and ATR."""
    if ind is None:
        return "unclear"

    spread = abs(float(ind.ema_spread_norm))
    atr_norm = float(ind.atr_norm_14) if ind.atr_norm_14 is not None else 0.0
    if spread >= 0.003 and atr_norm <= 0.03:
        return "trending_up" if ind.ema_spread_norm > 0 else "trending_down"
    if spread <= 0.0018 and atr_norm <= 0.035:
        return "choppy"
    return "unclear"


def get_last_trades(dataset_name: str) -> list[dict]:
    """Access per-dataset trades captured in the latest run."""
    return _LAST_TRADES.get(dataset_name, [])


def backtest_on_dataset(csv_file: Path, dataset_name: str) -> BacktestResults:
    """Backtest agent signals on dataset."""
    
    logger.info(f"\nBacktesting: {dataset_name}")
    logger.info(f"File: {csv_file}")
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to load {csv_file}: {e}")
        empty = BacktestResults(
            dataset_name=dataset_name,
            total_bars=0,
            signals_generated=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            total_return_pct=0.0,
            avg_trade_return_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            avg_bars_per_trade=0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
        )
        _LAST_TRADES[dataset_name] = []
        return empty
    
    # Normalize columns
    df.columns = df.columns.str.lower().str.strip()
    if "timestamp" in df.columns and "datetime" not in df.columns:
        df.rename(columns={"timestamp": "datetime"}, inplace=True)
    
    required_cols = {"datetime", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        logger.warning(f"Missing columns. Has: {df.columns.tolist()}")
        empty = BacktestResults(
            dataset_name=dataset_name,
            total_bars=len(df),
            signals_generated=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            total_return_pct=0.0,
            avg_trade_return_pct=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            avg_bars_per_trade=0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
        )
        _LAST_TRADES[dataset_name] = []
        return empty
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    logger.info(f"  Loaded {len(df)} candles")
    
    # Generate signals and execute trades
    trades = []  # List of (entry_idx, exit_idx, entry_price, exit_price, profit)
    signals_count = 0
    entry_idx = None
    entry_price = None
    entry_side = 0
    entry_sl_pct = 0.0
    entry_tp_pct = 0.0
    next_entry_idx_allowed = 50
    last_closed_side = 0
    
    for idx in range(50, len(df)):
        # Get data up to this bar
        df_slice = df.iloc[:idx+1].copy()
        
        # Generate indicators and regime once per bar.
        ind = compute_indicators(
            df_slice,
            ema_fast_period=9,
            ema_slow_period=21,
            rsi_period=14,
            atr_period=14,
        )
        atr_norm = float(ind.atr_norm_14) if ind and ind.atr_norm_14 is not None else 0.015
        regime = _detect_regime(ind)

        # Generate strategy signals.
        try:
            mom_result = momentum_signal(df_slice)
            mom_signal_val = mom_result.get("signal", 0) if mom_result else 0
        except Exception as e:
            logger.debug(f"Momentum signal error at {idx}: {e}")
            mom_result = {"signal": 0, "confidence": 0.0}
            mom_signal_val = 0
        
        try:
            mr_result = mean_reversion_signal(df_slice)
            mr_signal_val = mr_result.get("signal", 0) if mr_result else 0
        except Exception as e:
            logger.debug(f"MR signal error at {idx}: {e}")
            mr_result = {"signal": 0, "confidence": 0.0}
            mr_signal_val = 0

        raw_mom_signal = mom_signal_val
        raw_mr_signal = mr_signal_val
        raw_mom_conf = float(mom_result.get("confidence", 0.0) or 0.0)
        raw_mr_conf = float(mr_result.get("confidence", 0.0) or 0.0)

        # Regime routing: avoid momentum overtrading in choppy market.
        if regime in ("trending_up", "trending_down"):
            mr_result = {"signal": 0, "confidence": 0.0, "raw_strength": 0.0}
            mr_signal_val = 0
        elif regime == "choppy":
            mom_result = {"signal": 0, "confidence": 0.0, "raw_strength": 0.0}
            mom_signal_val = 0
        else:
            mom_result = {"signal": 0, "confidence": 0.0, "raw_strength": 0.0}
            mr_result = {"signal": 0, "confidence": 0.0, "raw_strength": 0.0}
            mom_signal_val = 0
            mr_signal_val = 0

        signal_conf, action = compute_confidence(
            mom_result, mr_result,
            current_atr_norm=atr_norm,
            conf_threshold=0.65,
            regime=regime,
        )

        exec_threshold = get_execution_threshold(atr_norm, base_threshold=0.65)
        combined_signal = 1 if action == "BUY" else -1 if action == "SELL" else 0
        trend_strength = abs(float(ind.ema_spread_norm)) if ind is not None else 0.0

        # Strict no-trade zones for conflicting/weak signals and sideways conditions.
        strong_conflict = (
            raw_mom_signal != 0
            and raw_mr_signal != 0
            and raw_mom_signal != raw_mr_signal
            and raw_mom_conf >= 0.55
            and raw_mr_conf >= 0.55
        )
        sideways_zone = atr_norm < 0.010 or trend_strength < 0.0015
        weak_signal_zone = combined_signal != 0 and signal_conf < (exec_threshold + 0.03)
        no_trade_zone = strong_conflict or sideways_zone or weak_signal_zone

        if mom_signal_val != 0 or mr_signal_val != 0:
            signals_count += 1

        logger.debug(
            "bar=%d regime=%s mom=%d mr=%d action=%s conf=%.3f thr=%.3f",
            idx,
            regime,
            mom_signal_val,
            mr_signal_val,
            action,
            signal_conf,
            exec_threshold,
        )

        close = float(df["close"].iloc[idx])

        # Exit logic
        if entry_idx is not None:
            price_move_pct = (close - entry_price) / entry_price
            signed_return = price_move_pct * entry_side

            # Check exit conditions: TP, SL, or signal reversal
            should_exit = False
            if signed_return >= entry_tp_pct:
                should_exit = True
            elif signed_return <= -entry_sl_pct:
                should_exit = True
            elif combined_signal == -entry_side and combined_signal != 0:
                should_exit = True

            if should_exit:
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": idx,
                    "side": entry_side,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "profit": signed_return,
                    "profit_pct": signed_return * 100,
                    "bars": idx - entry_idx,
                })
                entry_idx = None
                entry_price = None
                entry_side = 0
                last_closed_side = trades[-1]["side"]
                next_entry_idx_allowed = idx + TRADE_COOLDOWN_BARS

        # Entry logic with explicit confidence filter.
        same_side_without_fresh_confirmation = (
            combined_signal != 0
            and combined_signal == last_closed_side
            and signal_conf < (exec_threshold + 0.07)
            and trend_strength < 0.0042
        )

        if (
            entry_idx is None
            and combined_signal != 0
            and signal_conf >= exec_threshold
            and idx >= next_entry_idx_allowed
            and not no_trade_zone
            and not same_side_without_fresh_confirmation
        ):
            entry_idx = idx
            entry_price = close
            entry_side = combined_signal
            entry_sl_pct = float(np.clip(max(atr_norm * 1.7, 0.015), 0.015, 0.045))
            entry_tp_pct = float(np.clip(max(atr_norm * 2.8, 0.025), 0.025, 0.080))
            logger.info(
                "TRADE_TAKEN dataset=%s idx=%d side=%s conf=%.3f threshold=%.3f",
                dataset_name,
                idx,
                "BUY" if combined_signal == 1 else "SELL",
                signal_conf,
                exec_threshold,
            )
        elif combined_signal != 0:
            logger.info(
                "TRADE_SKIPPED dataset=%s idx=%d action=%s conf=%.3f threshold=%.3f no_trade=%s",
                dataset_name,
                idx,
                action,
                signal_conf,
                exec_threshold,
                no_trade_zone or same_side_without_fresh_confirmation or idx < next_entry_idx_allowed,
            )
    
    # Close final position
    if entry_idx is not None:
        close_price = float(df["close"].iloc[-1])
        signed_return = ((close_price - entry_price) / entry_price) * entry_side
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": len(df) - 1,
            "side": entry_side,
            "entry_price": entry_price,
            "exit_price": close_price,
            "profit": signed_return,
            "profit_pct": signed_return * 100,
            "bars": len(df) - 1 - entry_idx,
        })
    
    # Calculate metrics
    winning = sum(1 for t in trades if t["profit"] > 0)
    losing = sum(1 for t in trades if t["profit"] < 0)
    total = len(trades)
    
    returns = np.array([t["profit_pct"] for t in trades]) if trades else np.array([])
    
    win_rate = (winning / total * 100) if total > 0 else 0
    total_return = 0.0
    avg_return = np.mean(returns) if len(returns) > 0 else 0
    best_trade = np.max(returns) if len(returns) > 0 else 0
    worst_trade = np.min(returns) if len(returns) > 0 else 0
    avg_bars = int(np.mean([t["bars"] for t in trades])) if trades else 0
    
    # Sharpe (annualized, 4h bars = ~6 bars/day)
    sharpe = 0.0
    if len(returns) > 1:
        trade_returns = returns / 100.0
        if np.std(trade_returns) > 0:
            avg_bars_float = float(np.mean([max(t["bars"], 1) for t in trades]))
            trades_per_year = (252.0 * 6.0) / max(avg_bars_float, 1.0)
            sharpe = float(np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(trades_per_year))
    
    # Max drawdown
    equity_curve = np.array([100.0])
    for r in returns:
        equity_curve = np.append(equity_curve, equity_curve[-1] * (1 + r/100))
    
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak * 100
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    
    total_return = float(equity_curve[-1] - 100.0) if len(equity_curve) > 0 else 0.0

    results = BacktestResults(
        dataset_name=dataset_name,
        total_bars=len(df),
        signals_generated=signals_count,
        total_trades=total,
        winning_trades=winning,
        losing_trades=losing,
        win_rate_pct=win_rate,
        total_return_pct=total_return,
        avg_trade_return_pct=avg_return,
        best_trade_pct=best_trade,
        worst_trade_pct=worst_trade,
        avg_bars_per_trade=avg_bars,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
    )

    _LAST_TRADES[dataset_name] = trades
    return results


def main():
    """Run backtest on all datasets."""
    
    test_files = [
        ("data/btc_live_4h.csv", "BTC (4H)"),
        ("data/eth_live_4h.csv", "ETH (4H)"),
        ("data/sol_live_4h.csv", "SOL (4H)"),
        ("data/d1.csv", "D1 Historical"),
        ("data/d2.csv", "D2 Historical"),
    ]
    
    all_results = []
    all_trades = {}
    
    logger.info("\n" + "="*80)
    logger.info("AGENT BACKTEST ON DATASETS")
    logger.info("="*80)
    
    for csv_path, name in test_files:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.warning(f"Skipping {csv_path} (not found)")
            continue
        
        results = backtest_on_dataset(csv_file, name)
        trades = get_last_trades(name)
        if results:
            all_results.append(results)
            all_trades[name] = trades
    
    # Print summary
    print("\n" + "="*130)
    print("BACKTEST SUMMARY")
    print("="*130)
    print(
        f"{'Dataset':<20} {'Bars':<8} {'Signals':<10} {'Trades':<8} {'Win%':<10} "
        f"{'Total Return':<15} {'Avg Return':<15} {'Sharpe':<10} {'Max DD%':<10}"
    )
    print("-"*130)
    
    for r in all_results:
        print(
            f"{r.dataset_name:<20} {r.total_bars:<8} {r.signals_generated:<10} "
            f"{r.total_trades:<8} {r.win_rate_pct:<10.2f} {r.total_return_pct:<15.2f} "
            f"{r.avg_trade_return_pct:<15.2f} {r.sharpe_ratio:<10.4f} {r.max_drawdown_pct:<10.2f}"
        )
    
    print("="*130)
    
    # Save results
    results_dict = {
        "backtest_results": [asdict(r) for r in all_results],
        "trades": all_trades,
        "summary": {
            "total_datasets": len(all_results),
            "avg_win_rate": np.mean([r.win_rate_pct for r in all_results]),
            "avg_sharpe": np.mean([r.sharpe_ratio for r in all_results]),
            "avg_return": np.mean([r.total_return_pct for r in all_results]),
        }
    }
    
    with open("data/backtest_results_detailed.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    logger.info(f"\n✓ Results saved to data/backtest_results_detailed.json")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    if all_results:
        avg_win_rate = np.mean([r.win_rate_pct for r in all_results])
        avg_return = np.mean([r.total_return_pct for r in all_results])
        total_trades = sum(r.total_trades for r in all_results)
        
        print(f"Average Win Rate:     {avg_win_rate:.2f}%")
        print(f"Average Total Return: {avg_return:.2f}%")
        print(f"Total Trades Executed: {total_trades}")
        print(f"Average Sharpe Ratio: {np.mean([r.sharpe_ratio for r in all_results]):.4f}")
        print(f"Average Max Drawdown: {np.mean([r.max_drawdown_pct for r in all_results]):.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
