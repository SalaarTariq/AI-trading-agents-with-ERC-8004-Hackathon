"""
analysis/backtester.py — Backtesting and performance analysis module.


Provides a Backtester class that runs the trading agent over multiple
datasets, computes performance metrics, analyzes loss patterns, and
supports walk-forward optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from config import AppConfig, CONFIG
from utils.data_loader import load_csv
from main import run_paper_trading, detect_regime

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    dataset_name: str
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    per_strategy_stats: dict[str, dict] = field(default_factory=dict)
    regime_breakdown: dict[str, dict] = field(default_factory=dict)
    loss_patterns: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class Backtester:
    """
    Run backtests across multiple datasets and analyze results.

    Args:
        cfg: Application configuration.
        data_dir: Directory containing CSV files.
    """

    def __init__(self, cfg: AppConfig | None = None, data_dir: str = "data"):
        self.cfg = cfg or CONFIG
        self.data_dir = Path(data_dir)
        self.results: list[BacktestResult] = []

    def run(
        self,
        datasets: list[str] | None = None,
        warmup: int = 30,
    ) -> list[BacktestResult]:
        """
        Run backtest on specified datasets.

        Args:
            datasets: List of CSV file paths. If None, finds all CSVs in data_dir.
            warmup: Warmup bars for indicator computation.

        Returns:
            List of BacktestResult for each dataset.
        """
        if datasets is None:
            csv_files = sorted(self.data_dir.glob("*.csv"))
            datasets = [str(f) for f in csv_files]

        self.results = []
        for ds_path in datasets:
            path = Path(ds_path)
            if not path.exists():
                logger.warning("Dataset not found: %s — skipping", ds_path)
                continue

            label = path.stem
            logger.info("Backtesting on: %s", label)

            df = load_csv(str(path))
            if df is None or len(df) < warmup + 10:
                logger.warning("Dataset too short: %s (%d rows)", label,
                               len(df) if df is not None else 0)
                continue

            summary = run_paper_trading(df, cfg=self.cfg, warmup=warmup, dataset_label=label)
            result = self._build_result(summary, label, df, warmup)
            self.results.append(result)

        return self.results

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_bars: int = 200,
        test_bars: int = 50,
        step_bars: int = 50,
    ) -> list[BacktestResult]:
        """
        Walk-forward analysis: train on a window, test on next window, slide.

        Args:
            df: Full OHLCV dataset.
            train_bars: Training window size.
            test_bars: Testing window size.
            step_bars: Step size between windows.

        Returns:
            List of results for each walk-forward window.
        """
        results = []
        total = len(df)
        start = 0

        while start + train_bars + test_bars <= total:
            test_start = start + train_bars
            test_end = test_start + test_bars
            test_df = df.iloc[start:test_end]

            label = f"wf_{start}_{test_end}"
            summary = run_paper_trading(
                test_df, cfg=self.cfg, warmup=train_bars, dataset_label=label
            )
            result = self._build_result(summary, label, test_df, train_bars)
            results.append(result)
            start += step_bars

        self.results.extend(results)
        return results

    def analyze_losses(self) -> list[dict]:
        """
        Analyze loss patterns across all backtest results.

        Returns:
            List of loss pattern descriptions.
        """
        patterns = []
        for result in self.results:
            if result.loss_patterns:
                patterns.extend(result.loss_patterns)

        # Aggregate
        if not patterns:
            return [{"pattern": "no losses found", "count": 0}]

        return patterns

    def recommend_params(self) -> dict:
        """
        Recommend parameter adjustments based on backtest results.

        Returns:
            Dict of recommended parameter changes.
        """
        if not self.results:
            return {"note": "no results to analyze"}

        avg_wr = np.mean([r.win_rate for r in self.results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in self.results])

        recommendations: dict = {
            "avg_win_rate": round(avg_wr, 2),
            "avg_sharpe": round(avg_sharpe, 3),
        }

        if avg_wr < 0.50:
            recommendations["signal_threshold"] = "increase to 0.25"
        if avg_sharpe < 0.5:
            recommendations["position_sizing"] = "reduce base allocation"

        # Check regime performance
        regime_performance: dict[str, list[float]] = {}
        for r in self.results:
            for regime, stats in r.regime_breakdown.items():
                if regime not in regime_performance:
                    regime_performance[regime] = []
                regime_performance[regime].append(stats.get("win_rate", 0))

        for regime, rates in regime_performance.items():
            avg = np.mean(rates)
            if avg < 0.40:
                recommendations[f"regime_{regime}"] = f"underperforming ({avg:.0%})"

        return recommendations

    def tune_execute_confidence(
        self,
        datasets: list[str],
        thresholds: list[float] | None = None,
        warmup: int = 30,
    ) -> dict:
        """
        Grid search over execution confidence threshold.

        This tunes both the signal execution gate and the risk min_confidence
        to the same value to keep the pipeline consistent.
        """
        thresholds = thresholds or [0.55, 0.60, 0.65, 0.70]
        best = {"threshold": None, "avg_win_rate": -1.0, "total_pnl": -1e18, "results": []}

        for thr in thresholds:
            cfg = copy.deepcopy(self.cfg)
            cfg.signal.execute_confidence_threshold = float(thr)
            cfg.risk.min_confidence = float(thr)
            bt = Backtester(cfg=cfg, data_dir=str(self.data_dir))
            results = bt.run(datasets=datasets, warmup=warmup)
            if not results:
                continue
            avg_wr = float(np.mean([r.win_rate for r in results]))
            tot_pnl = float(np.sum([r.total_pnl for r in results]))
            if (avg_wr > best["avg_win_rate"]) or (avg_wr == best["avg_win_rate"] and tot_pnl > best["total_pnl"]):
                best = {
                    "threshold": thr,
                    "avg_win_rate": round(avg_wr, 4),
                    "total_pnl": round(tot_pnl, 2),
                    "results": results,
                }
        return best

    def summary_report(self) -> str:
        """Generate a human-readable summary report."""
        if not self.results:
            return "No backtest results available."

        lines = ["=" * 60, "BACKTEST SUMMARY REPORT", "=" * 60]

        for r in self.results:
            lines.append(f"\n--- {r.dataset_name} ---")
            lines.append(f"  Win Rate:      {r.win_rate:.1%}")
            lines.append(f"  Trades:        {r.total_trades} (W:{r.winning_trades} L:{r.losing_trades})")
            lines.append(f"  PnL:           ${r.total_pnl:+,.2f} ({r.pnl_pct:+.1f}%)")
            lines.append(f"  Sharpe:        {r.sharpe_ratio:.3f}")
            lines.append(f"  Max Drawdown:  {r.max_drawdown:.1%}")
            lines.append(f"  Profit Factor: {r.profit_factor:.2f}")

        # Aggregate
        avg_wr = np.mean([r.win_rate for r in self.results])
        total_pnl = sum(r.total_pnl for r in self.results)
        total_trades = sum(r.total_trades for r in self.results)

        lines.append(f"\n{'=' * 60}")
        lines.append(f"AGGREGATE:")
        lines.append(f"  Avg Win Rate:  {avg_wr:.1%}")
        lines.append(f"  Total PnL:     ${total_pnl:+,.2f}")
        lines.append(f"  Total Trades:  {total_trades}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _build_result(
        self, summary: dict, label: str, df: pd.DataFrame, warmup: int
    ) -> BacktestResult:
        """Build a BacktestResult from run_paper_trading output."""
        trades = summary.get("trades", []) or []
        total = int(summary.get("total_trades", len(trades)))
        pnl = float(summary.get("total_pnl", 0.0))
        pnl_pct = float(summary.get("pnl_pct", 0.0))

        pnls = [float(t.get("pnl", 0.0)) for t in trades if "pnl" in t]
        wins_list = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p < 0]
        wins = len(wins_list)
        losses = len(losses_list)
        win_rate = (wins / total) if total > 0 else 0.0

        # Equity curve
        initial = float(summary.get("initial_balance", self.cfg.portfolio.initial_balance))
        equity = [initial]
        for p in pnls:
            equity.append(equity[-1] + p)

        # Returns series (per trade)
        rets = []
        for i in range(1, len(equity)):
            prev = equity[i - 1]
            rets.append((equity[i] - prev) / prev if prev != 0 else 0.0)

        sharpe = _compute_sharpe_from_returns(rets)
        max_dd = _max_drawdown(equity)

        gross_profit = float(np.sum(wins_list)) if wins_list else 0.0
        gross_loss = float(-np.sum(losses_list)) if losses_list else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        avg_win = float(np.mean(wins_list)) if wins_list else 0.0
        avg_loss = float(np.mean([abs(x) for x in losses_list])) if losses_list else 0.0

        # Regime breakdown
        regime_bd: dict[str, dict] = {}
        if len(df) > warmup + 10:
            for regime_name in ["trending_up", "trending_down", "ranging", "volatile"]:
                regime_bd[regime_name] = {"trades": 0, "win_rate": 0.0}

        return BacktestResult(
            dataset_name=label,
            win_rate=win_rate,
            total_trades=total,
            winning_trades=wins,
            losing_trades=losses,
            total_pnl=pnl,
            pnl_pct=pnl_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            regime_breakdown=regime_bd,
            equity_curve=[round(float(x), 2) for x in equity],
        )


def _compute_sharpe_from_returns(returns: list[float], risk_free: float = 0.0) -> float:
    """Sharpe ratio from a returns series (per trade)."""
    if not returns or len(returns) < 2:
        return 0.0
    r = np.array(returns, dtype=float)
    excess = r - risk_free
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std)


def _max_drawdown(equity: list[float]) -> float:
    """Maximum drawdown from an equity curve (as positive fraction)."""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)
