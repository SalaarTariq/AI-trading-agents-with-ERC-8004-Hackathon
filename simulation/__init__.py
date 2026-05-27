"""Paper-trading simulation package."""

from simulation.paper_trader import (
    PaperTrader,
    Position,
    TradeRecord,
    run_paper_trading,
    required_warmup_bars,
)

__all__ = [
    "PaperTrader",
    "Position",
    "TradeRecord",
    "run_paper_trading",
    "required_warmup_bars",
]
