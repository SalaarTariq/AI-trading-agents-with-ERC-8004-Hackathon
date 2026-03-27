import json
import logging
from pathlib import Path
import numpy as np
from tests.test_agent_on_datasets import backtest_on_dataset, get_last_trades

for name in ["tests.test_agent_on_datasets", "modules.momentum", "modules.mean_reversion", "root"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

rows = []
for csv_file, name in [
    ("data/btc_live_4h.csv", "BTC (4H)"),
    ("data/eth_live_4h.csv", "ETH (4H)"),
    ("data/sol_live_4h.csv", "SOL (4H)"),
]:
    r = backtest_on_dataset(Path(csv_file), name)
    trades = get_last_trades(name)

    bars = max(r.total_bars, 1)
    trades_per_100 = (r.total_trades / bars) * 100.0

    rets = np.array([t["profit_pct"] for t in trades], dtype=float) if trades else np.array([])
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    avg_return_per_trade = float(np.mean(rets)) if rets.size else 0.0
    rr_ratio = float(np.mean(wins) / abs(np.mean(losses))) if wins.size and losses.size else 0.0
    filtered_pct = ((r.signals_generated - r.total_trades) / r.signals_generated * 100.0) if r.signals_generated else 0.0

    rows.append(
        {
            "dataset": name,
            "total_bars": r.total_bars,
            "signals_generated": r.signals_generated,
            "total_trades": r.total_trades,
            "win_rate_pct": r.win_rate_pct,
            "total_return_pct": r.total_return_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown_pct": r.max_drawdown_pct,
            "avg_trades_per_100_bars": trades_per_100,
            "avg_return_per_trade_pct": avg_return_per_trade,
            "realized_rr_ratio": rr_ratio,
            "filtered_signals_pct_proxy": filtered_pct,
            "avg_bars_per_trade": r.avg_bars_per_trade,
            "best_trade_pct": r.best_trade_pct,
            "worst_trade_pct": r.worst_trade_pct,
        }
    )

print(json.dumps(rows, indent=2))
