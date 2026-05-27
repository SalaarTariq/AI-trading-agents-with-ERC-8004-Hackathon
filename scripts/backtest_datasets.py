"""Backtest the agent across multiple CSV datasets.

Runs the actual ``simulation.run_paper_trading`` pipeline on each
configured dataset and writes a summary JSON. Replaces the previous
``tests/test_agent_on_datasets.py`` script, which lived under tests/
but defined no pytest tests and reimplemented the trade loop instead
of calling it.

Usage:

    python -m scripts.backtest_datasets
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from simulation import run_paper_trading
from utils.data_loader import load_csv

logger = logging.getLogger(__name__)


DEFAULT_DATASETS: list[tuple[str, str]] = [
    ("data/btc_live_4h.csv", "BTC (4H)"),
    ("data/eth_live_4h.csv", "ETH (4H)"),
    ("data/sol_live_4h.csv", "SOL (4H)"),
    ("data/d1.csv", "D1 Historical"),
    ("data/d2.csv", "D2 Historical"),
]


@dataclass
class BacktestSummary:
    """One row per dataset in the summary report."""

    dataset: str
    total_bars: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_return_pct: float
    avg_trade_return_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    proof_hashes: int


def _trade_pct_returns(trades: list[dict], initial_balance: float) -> np.ndarray:
    """Convert trade PnL ($) into per-trade percentage returns."""
    if not trades:
        return np.array([])
    return np.array([(t["pnl"] / initial_balance) * 100.0 for t in trades])


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns / 100.0)])
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / peak) * 100.0)


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    # Treat each closed trade as one period. The annualization factor is
    # intentionally simple — this is a hackathon-grade reporting metric,
    # not a portfolio risk model.
    return float(np.mean(returns) / np.std(returns) * np.sqrt(len(returns)))


def backtest_one(csv_path: Path, label: str) -> BacktestSummary | None:
    """Run one dataset through ``run_paper_trading`` and summarize."""
    if not csv_path.exists():
        logger.warning("Skipping %s (not found)", csv_path)
        return None

    df = load_csv(str(csv_path))
    # Throwaway trade-log so the canonical data/trade_history.jsonl is
    # not polluted by repeated backtests.
    throwaway = Path(".cache/backtest_trades.jsonl")
    throwaway.parent.mkdir(parents=True, exist_ok=True)
    if throwaway.exists():
        throwaway.unlink()

    result = run_paper_trading(df, dataset_label=label, trade_log_path=throwaway)

    trades = result["trades"]
    initial = result["initial_balance"]
    returns = _trade_pct_returns(trades, initial)

    return BacktestSummary(
        dataset=label,
        total_bars=len(df),
        total_trades=result["total_trades"],
        winning_trades=result["winning_trades"],
        losing_trades=result["losing_trades"],
        win_rate_pct=result["win_rate"],
        total_return_pct=result["pnl_pct"],
        avg_trade_return_pct=float(np.mean(returns)) if returns.size else 0.0,
        best_trade_pct=float(np.max(returns)) if returns.size else 0.0,
        worst_trade_pct=float(np.min(returns)) if returns.size else 0.0,
        max_drawdown_pct=_max_drawdown_from_returns(returns),
        sharpe_ratio=_sharpe(returns),
        proof_hashes=result["proof_hashes_generated"],
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    rows: list[BacktestSummary] = []
    for path_str, label in DEFAULT_DATASETS:
        summary = backtest_one(Path(path_str), label)
        if summary is not None:
            rows.append(summary)

    print("\n" + "=" * 110)
    print("BACKTEST SUMMARY")
    print("=" * 110)
    print(
        f"{'Dataset':<18} {'Bars':<6} {'Trades':<7} {'Win%':<7} "
        f"{'Return%':<9} {'AvgTr%':<8} {'MaxDD%':<8} {'Sharpe':<8} {'Proofs':<7}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"{r.dataset:<18} {r.total_bars:<6} {r.total_trades:<7} "
            f"{r.win_rate_pct:<7.2f} {r.total_return_pct:<9.2f} {r.avg_trade_return_pct:<8.2f} "
            f"{r.max_drawdown_pct:<8.2f} {r.sharpe_ratio:<8.3f} {r.proof_hashes:<7}"
        )
    print("=" * 110)

    Path("data").mkdir(exist_ok=True)
    out_path = Path("data/backtest_results_detailed.json")
    out_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    logger.warning("Wrote %s", out_path)


if __name__ == "__main__":
    main()
