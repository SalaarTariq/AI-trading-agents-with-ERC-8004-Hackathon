"""
dashboard/dashboard.py — Streamlit dashboard for the Hybrid AI Trading Agent.

Displays portfolio value, PnL, win rate, Sharpe ratio, max drawdown,
regime distribution, trade history, risk warnings, and ERC-8004
validation proof hashes.

Run with:
    streamlit run dashboard/dashboard.py
"""

from __future__ import annotations

import json
import logging
import math
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


def _calc_win_rate(trades: list[dict]) -> str:
    """Calculate win rate from trade history."""
    if not trades:
        return "N/A"
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    total = len(trades)
    return f"{wins}/{total} ({wins / total:.0%})"


def _calc_sharpe(trades: list[dict]) -> float | None:
    """Annualized Sharpe ratio from trade PnLs (assumes ~6 trades/day on 4h)."""
    pnls = [t.get("pnl", 0) for t in trades]
    if len(pnls) < 2:
        return None
    mean_r = sum(pnls) / len(pnls)
    var = sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 0
    if std == 0:
        return None
    # Annualize: assume ~250 trading days
    return (mean_r / std) * math.sqrt(250)


def _calc_max_drawdown(trades: list[dict], initial: float = 100_000) -> float:
    """Max drawdown percentage from peak."""
    if not trades:
        return 0.0
    peak = initial
    max_dd = 0.0
    value = initial
    for t in trades:
        value += t.get("pnl", 0)
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _regime_distribution(proofs: list[dict]) -> dict[str, int]:
    """Count regime occurrences from proof log records."""
    counts: dict[str, int] = {}
    for p in proofs:
        rec = p.get("full_record", {})
        decision = rec.get("decision", {})
        regime_detail = decision.get("regime_detail", {})
        regime = regime_detail.get("regime", "unknown")
        counts[regime] = counts.get(regime, 0) + 1
    return counts


def run_dashboard():
    """Launch the Streamlit dashboard."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Run: pip install streamlit")
        print("Then: streamlit run dashboard/dashboard.py")
        return

    st.set_page_config(page_title="Hybrid AI Trading Agent — ERC-8004", layout="wide")
    st.title("Hybrid AI Trading Agent Dashboard")
    st.caption("ERC-8004 Verifiable Trust | Hackathon Submission")

    # Auto-refresh
    from config import CONFIG
    refresh = CONFIG.dashboard_refresh_seconds

    # --- Load data ---
    trades = _load_jsonl(TRADE_LOG)
    proofs = _load_jsonl(PROOF_LOG)

    # --- Top-level KPIs ---
    st.header("Performance Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    initial = CONFIG.portfolio.initial_balance
    portfolio_series = _build_portfolio_series(trades, initial)
    current_value = portfolio_series.iloc[-1]
    total_pnl = current_value - initial
    pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0
    sharpe = _calc_sharpe(trades)
    max_dd = _calc_max_drawdown(trades, initial)

    c1.metric("Portfolio Value", f"${current_value:,.2f}")
    c2.metric("Total PnL", f"${total_pnl:+,.2f}", f"{pnl_pct:+.1f}%")
    c3.metric("Total Trades", str(len(trades)))
    c4.metric("Win Rate", _calc_win_rate(trades))
    c5.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A")
    c6.metric("Max Drawdown", f"{max_dd:.1%}")

    # --- PnL Chart ---
    st.subheader("Portfolio Value Over Time")
    if len(portfolio_series) > 1:
        st.line_chart(portfolio_series)
    else:
        st.info("No trades yet. Run the agent to see portfolio changes.")

    # --- Regime Distribution + Proof Count ---
    left_stats, right_stats = st.columns(2)

    with left_stats:
        st.subheader("Market Regime Distribution")
        regime_dist = _regime_distribution(proofs)
        if regime_dist:
            regime_df = pd.DataFrame(
                [{"Regime": k, "Count": v} for k, v in sorted(regime_dist.items())]
            )
            st.bar_chart(regime_df.set_index("Regime"))
        else:
            st.info("No regime data yet.")

    with right_stats:
        st.subheader("ERC-8004 Validation Proofs")
        st.metric("Total Proofs Generated", str(len(proofs)))
        if proofs:
            valid = sum(
                1 for p in proofs
                if p.get("proof_hash") and len(p.get("proof_hash", "")) == 64
            )
            st.metric("Valid SHA256 Hashes", f"{valid}/{len(proofs)}")
        st.markdown(
            "Each trade decision produces a **SHA256 proof hash** logged to "
            "`validation/proof_log.jsonl`. These hashes map to the ERC-8004 "
            "**Validation Registry** for tamper-proof auditability."
        )

    # --- Per-Dataset Breakdown ---
    trade_df = _build_trade_df(trades)
    datasets = sorted(trade_df["dataset"].unique()) if not trade_df.empty else []

    if datasets and len(datasets) > 1:
        st.header("Per-Dataset Breakdown")
        ds_cols = st.columns(min(len(datasets), 4))
        for col, ds_name in zip(ds_cols, datasets):
            ds_trades = [t for t in trades if t.get("dataset") == ds_name]
            ds_pnl = sum(t.get("pnl", 0) for t in ds_trades)
            ds_wins = sum(1 for t in ds_trades if t.get("pnl", 0) > 0)
            with col:
                st.markdown(f"**{ds_name or 'unnamed'}**")
                st.metric("Trades", str(len(ds_trades)))
                st.metric("PnL", f"${ds_pnl:+,.2f}")
                st.metric("Win Rate", f"{ds_wins}/{len(ds_trades)}" if ds_trades else "N/A")

    # --- Two-column layout: Trades + Signals ---
    left, right = st.columns(2)

    with left:
        st.subheader("Trade History")
        if not trade_df.empty:
            if datasets and len(datasets) > 1:
                selected_ds = st.selectbox("Filter by dataset:", ["All"] + datasets)
                if selected_ds != "All":
                    trade_df = trade_df[trade_df["dataset"] == selected_ds]
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")

    with right:
        st.subheader("Latest Signals & Risk Status")
        if proofs:
            latest = proofs[-1]
            record = latest.get("full_record", {})
            decision = record.get("decision", {})
            ds_label = decision.get("dataset", "")
            if ds_label:
                st.caption(f"From dataset: **{ds_label}**")

            signals = decision.get("strategy_signals", {})
            if signals:
                for name, sig in signals.items():
                    direction = sig.get("signal", "?")
                    conf = sig.get("confidence", sig.get("raw_strength", 0))
                    st.markdown(f"**{name}**: {direction} (strength: {conf:.3f})")

            risk = decision.get("risk_result", {})
            if risk:
                st.markdown("---")
                approved = risk.get("approved", False)
                status = "Approved" if approved else "Rejected"
                st.markdown(f"**Risk Status**: {status}")
                for reason in risk.get("reasons", []):
                    st.warning(reason)
                for warning in risk.get("warnings", []):
                    st.info(warning)
        else:
            st.info("No decisions recorded yet.")

    # --- Proof Log Table ---
    st.subheader("Validation Proof Log (Last 20)")
    if proofs:
        proof_rows = []
        for p in proofs[-20:]:
            rec = p.get("full_record", {})
            decision = rec.get("decision", {})
            proof_rows.append({
                "dataset": decision.get("dataset", ""),
                "hash": p.get("proof_hash", "")[:24] + "...",
                "timestamp": p.get("timestamp", ""),
                "summary": p.get("decision_summary", ""),
            })
        st.dataframe(pd.DataFrame(proof_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No proof entries yet.")

    st.caption(f"Dashboard refreshes every {refresh}s. Loaded: {len(trades)} trades, {len(proofs)} proofs.")


# Run the dashboard when executed via streamlit
if __name__ == "__main__":
    run_dashboard()
else:
    try:
        import streamlit
        run_dashboard()
    except ImportError:
        pass
