"""
dashboard/dashboard.py — Streamlit dashboard for the trading agent.

Displays portfolio value, PnL, trade history, AI prediction confidence,
risk warnings, and validation proof hashes. Auto-refreshes periodically.

Run with:
    streamlit run dashboard/dashboard.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `config` can be imported
# regardless of which directory streamlit is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

logger = logging.getLogger(__name__)

# Proof log and trade log paths (relative to project root)
PROOF_LOG = _PROJECT_ROOT / "validation" / "proof_log.jsonl"
TRADE_LOG = _PROJECT_ROOT / "data" / "trade_history.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _build_trade_df(trades: list[dict]) -> pd.DataFrame:
    """Convert trade log entries to a DataFrame for display."""
    if not trades:
        return pd.DataFrame(columns=[
            "dataset", "timestamp", "action", "pair", "price", "size", "pnl", "exit_reason", "proof_hash"
        ])
    rows = []
    for t in trades:
        rows.append({
            "dataset": t.get("dataset", ""),
            "timestamp": t.get("timestamp", ""),
            "action": t.get("action", ""),
            "pair": t.get("pair", ""),
            "price": t.get("entry_price", 0),
            "size": t.get("size", 0),
            "pnl": t.get("pnl", 0),
            "exit_reason": t.get("exit_reason", ""),
            "proof_hash": t.get("proof_hash", "")[:16] + "...",
        })
    return pd.DataFrame(rows)


def _build_portfolio_series(trades: list[dict], initial: float = 100_000) -> pd.Series:
    """Compute cumulative portfolio value from trade PnL."""
    if not trades:
        return pd.Series([initial], index=["start"])
    values = [initial]
    for t in trades:
        values.append(values[-1] + t.get("pnl", 0))
    return pd.Series(values)


def run_dashboard():
    """Launch the Streamlit dashboard."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Run: pip install streamlit")
        print("Then: streamlit run dashboard/dashboard.py")
        return

    st.set_page_config(page_title="Hybrid AI Trading Agent", layout="wide")
    st.title("Hybrid AI Trading Agent Dashboard")

    # Auto-refresh
    from config import CONFIG
    refresh = CONFIG.dashboard_refresh_seconds

    # --- Load data ---
    trades = _load_jsonl(TRADE_LOG)
    proofs = _load_jsonl(PROOF_LOG)

    # --- Portfolio Overview ---
    st.header("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)

    initial = CONFIG.portfolio.initial_balance
    portfolio_series = _build_portfolio_series(trades, initial)
    current_value = portfolio_series.iloc[-1]
    total_pnl = current_value - initial
    pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0
    num_trades = len(trades)

    col1.metric("Portfolio Value", f"${current_value:,.2f}")
    col2.metric("Total PnL", f"${total_pnl:+,.2f}", f"{pnl_pct:+.1f}%")
    col3.metric("Total Trades", str(num_trades))
    col4.metric("Win Rate", _calc_win_rate(trades))

    # --- PnL Chart ---
    st.subheader("Portfolio Value Over Time")
    if len(portfolio_series) > 1:
        st.line_chart(portfolio_series)
    else:
        st.info("No trades yet. Run the agent to see portfolio changes.")

    # --- Per-Dataset Breakdown ---
    trade_df = _build_trade_df(trades)
    datasets = sorted(trade_df["dataset"].unique()) if not trade_df.empty else []

    if datasets and len(datasets) > 1:
        st.header("Per-Dataset Breakdown")
        ds_cols = st.columns(len(datasets))
        for col, ds_name in zip(ds_cols, datasets):
            ds_trades = [t for t in trades if t.get("dataset") == ds_name]
            ds_pnl = sum(t.get("pnl", 0) for t in ds_trades)
            ds_wins = sum(1 for t in ds_trades if t.get("pnl", 0) > 0)
            with col:
                st.markdown(f"**{ds_name or 'unnamed'}**")
                st.metric("Trades", str(len(ds_trades)))
                st.metric("PnL", f"${ds_pnl:+,.2f}")
                st.metric("Win Rate", f"{ds_wins}/{len(ds_trades)}" if ds_trades else "N/A")

    # --- Two-column layout ---
    left, right = st.columns(2)

    # --- Trade History ---
    with left:
        st.subheader("Trade History (All Datasets)")
        if not trade_df.empty:
            # Filter by dataset
            if datasets and len(datasets) > 1:
                selected_ds = st.selectbox("Filter by dataset:", ["All"] + datasets)
                if selected_ds != "All":
                    trade_df = trade_df[trade_df["dataset"] == selected_ds]
            st.dataframe(trade_df, width="stretch", hide_index=True)
        else:
            st.info("No trades recorded yet.")

    # --- Strategy Signals & Risk ---
    with right:
        st.subheader("Latest Signals & Risk Status")
        if proofs:
            latest = proofs[-1]
            record = latest.get("full_record", {})
            ds_label = record.get("dataset", "")
            if ds_label:
                st.caption(f"From dataset: **{ds_label}**")

            # Strategy signals
            signals = record.get("strategy_signals", {})
            if signals:
                for name, sig in signals.items():
                    direction = sig.get("signal", "?")
                    conf = sig.get("confidence", 0)
                    emoji = {"BUY": ":green_circle:", "SELL": ":red_circle:"}.get(direction, ":white_circle:")
                    st.markdown(f"{emoji} **{name}**: {direction} (confidence: {conf:.1%})")

            # Risk result
            risk = record.get("risk_result", {})
            if risk:
                st.markdown("---")
                approved = risk.get("approved", False)
                status = ":white_check_mark: Approved" if approved else ":x: Rejected"
                st.markdown(f"**Risk Status**: {status}")
                for reason in risk.get("reasons", []):
                    st.warning(reason)
                for warning in risk.get("warnings", []):
                    st.info(warning)
        else:
            st.info("No decisions recorded yet.")

    # --- Proof Log ---
    st.subheader("Validation Proof Log")
    if proofs:
        proof_rows = []
        for p in proofs[-20:]:  # Show last 20
            rec = p.get("full_record", {})
            proof_rows.append({
                "dataset": rec.get("dataset", ""),
                "hash": p.get("proof_hash", "")[:24] + "...",
                "timestamp": p.get("timestamp", ""),
                "summary": p.get("decision_summary", ""),
            })
        st.dataframe(pd.DataFrame(proof_rows), width="stretch", hide_index=True)
    else:
        st.info("No proof entries yet.")

    # --- Auto-refresh note ---
    st.caption(f"Dashboard refreshes every {refresh}s. Last loaded: {len(trades)} trades, {len(proofs)} proofs.")


def _calc_win_rate(trades: list[dict]) -> str:
    """Calculate win rate from trade history."""
    if not trades:
        return "N/A"
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    total = len(trades)
    return f"{wins}/{total} ({wins / total:.0%})"


# Run the dashboard when executed via streamlit
if __name__ == "__main__":
    run_dashboard()
else:
    # When streamlit imports this module, run_dashboard() is called
    try:
        import streamlit
        run_dashboard()
    except ImportError:
        pass
