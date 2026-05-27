"""CLI entry point for the ERC-8004 hybrid trading agent.

The trading loop lives in ``simulation/paper_trader.py``; this module
only wires arguments → data load → backtest → summary print.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from simulation import run_paper_trading
from utils.data_loader import load_or_generate
from utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ERC-8004 Hybrid Trading Agent")
    parser.add_argument("--data", type=str, default=None, help="OHLCV CSV path")
    parser.add_argument(
        "--days", type=int, default=365, help="Synthetic days if --data omitted"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    df = load_or_generate(args.data, days=args.days)
    dataset_label = Path(args.data).stem if args.data else "synthetic"
    summary = run_paper_trading(df, dataset_label=dataset_label)

    print("\n" + "=" * 60)
    print("PAPER TRADING SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        if key != "trades":
            print(f"  {key:.<30} {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
