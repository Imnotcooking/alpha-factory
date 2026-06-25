"""Native options desk page for the money dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.options import (  # noqa: E402
    choose_expiration,
    format_scanner_frame,
    historical_holding_returns,
    historical_odds,
    scan_backspreads,
    scan_cash_secured_puts,
    scan_call_butterflies,
    scan_calendar_spreads,
    scan_iron_condors,
    scan_long_options,
    scan_ratio_spreads,
    scan_vertical_spreads,
    score_option_strategies,
    simulate_single_option,
    volatility_snapshot,
)
from oqp.execution import write_option_trade_proposal_from_candidate  # noqa: E402
from oqp.portfolio import default_portfolio_ledger_path, load_latest_live_positions  # noqa: E402


st.set_page_config(page_title="Options Desk", layout="wide", page_icon="O")


def money(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return f"${parsed:,.2f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return f"{parsed * 100:.{digits}f}%"


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    try:
        history = yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()
    return history if isinstance(history, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_expirations(symbol: str) -> list[str]:
    import yfinance as yf

    try:
        return list(yf.Ticker(symbol).options)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chain(symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    try:
        chain = yf.Ticker(symbol).option_chain(expiry)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return chain.calls, chain.puts


def latest_close(history: pd.DataFrame) -> float:
    if history.empty or "Close" not in history.columns:
        return 0.0
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    return float(close.iloc[-1]) if not close.empty else 0.0


def latest_rolling_value(series: pd.Series, window: int, fallback: float = 0.0, op: str = "mean") -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return fallback
    if op == "std":
        rolled = numeric.rolling(window).std().dropna()
    else:
        rolled = numeric.rolling(window).mean().dropna()
    return float(rolled.iloc[-1]) if not rolled.empty else fallback


def estimate_atm_market_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, fallback: float) -> float:
    candidates: list[float] = []
    for chain in (calls, puts):
        if chain.empty or "strike" not in chain.columns or "impliedVolatility" not in chain.columns:
            continue
        option_chain = chain.copy()
        option_chain["strike_distance"] = (pd.to_numeric(option_chain["strike"], errors="coerce") - spot).abs()
        option_chain["impliedVolatility"] = pd.to_numeric(option_chain["impliedVolatility"], errors="coerce")
        option_chain = option_chain.dropna(subset=["strike_distance", "impliedVolatility"])
        if not option_chain.empty:
            candidates.append(float(option_chain.sort_values("strike_distance").iloc[0]["impliedVolatility"]))
    candidates = [value for value in candidates if value > 0]
    return float(sum(candidates) / len(candidates)) if candidates else fallback


def far_calendar_expiry(expirations: list[str], near_expiry: str, min_gap_days: int = 28) -> str | None:
    near_date = pd.to_datetime(near_expiry).date()
    for expiry in expirations:
        if (pd.to_datetime(expiry).date() - near_date).days >= min_gap_days:
            return expiry
    return None


def render_volatility(symbol: str, history: pd.DataFrame, hold_days: int) -> None:
    snapshot = volatility_snapshot(history)
    cols = st.columns(5)
    cols[0].metric("Spot", money(latest_close(history)))
    cols[1].metric("Forecast Vol", pct(snapshot.forecast_vol))
    cols[2].metric("21D Hist Vol", pct(snapshot.historical_vol_21d))
    cols[3].metric("EWMA Vol", pct(snapshot.ewma_vol))
    cols[4].metric("Trend", snapshot.trend.title())

    returns = historical_holding_returns(history, hold_days)
    odds = pd.DataFrame(
        [
            {"Scenario": "+5% move", "Historical Odds": historical_odds(returns, 0.05, "up")},
            {"Scenario": "-5% move", "Historical Odds": historical_odds(returns, -0.05, "down")},
            {"Scenario": "Inside +/-5%", "Historical Odds": historical_odds(returns, 0.05, "inside")},
            {"Scenario": "+10% move", "Historical Odds": historical_odds(returns, 0.10, "up")},
            {"Scenario": "-10% move", "Historical Odds": historical_odds(returns, -0.10, "down")},
        ]
    )
    odds["Historical Odds"] = odds["Historical Odds"] * 100
    st.dataframe(
        odds,
        use_container_width=True,
        hide_index=True,
        column_config={"Historical Odds": st.column_config.NumberColumn(format="%.1f%%")},
    )

    if not history.empty:
        chart = history.copy()
        chart["SMA_20"] = chart["Close"].rolling(20).mean()
        chart["SMA_60"] = chart["Close"].rolling(60).mean()
        fig = go.Figure()
        fig.add_scatter(x=chart.index, y=chart["Close"], name=symbol)
        fig.add_scatter(x=chart.index, y=chart["SMA_20"], name="SMA 20")
        fig.add_scatter(x=chart.index, y=chart["SMA_60"], name="SMA 60")
        fig.update_layout(height=380, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


def render_strategy_fit(
    symbol: str,
    history: pd.DataFrame,
    expirations: list[str],
    spot: float,
    forecast_vol: float,
    rsi_14: float,
    hold_days: int,
) -> None:
    if not expirations:
        st.warning("No listed options expirations found.")
        return

    expiry = choose_expiration(expirations, hold_days)
    calls, puts = fetch_chain(symbol, expiry) if expiry else (pd.DataFrame(), pd.DataFrame())
    market_iv = estimate_atm_market_iv(calls, puts, spot, forecast_vol or 0.20)

    close = history["Close"] if "Close" in history.columns else pd.Series(dtype=float)
    ma20 = latest_rolling_value(close, 20, fallback=spot)
    rolling_std_20 = latest_rolling_value(close, 20, fallback=0.0, op="std")

    fit = score_option_strategies(
        spot=spot,
        moving_average_20=ma20,
        rolling_std_20=rolling_std_20,
        rsi_14=rsi_14,
        market_iv=market_iv,
        forecast_vol=forecast_vol or 0.20,
        target_beta=0.5,
    )

    cols = st.columns(5)
    cols[0].metric("ATM IV", pct(market_iv))
    cols[1].metric("Forecast Vol", pct(forecast_vol or 0.20))
    cols[2].metric("VRP", pct(market_iv - (forecast_vol or 0.20)))
    cols[3].metric("20D Momentum Z", f"{fit['Momentum Z'].iloc[0]:.2f}" if not fit.empty else "0.00")
    cols[4].metric("Reference Expiry", expiry or "n/a")

    if fit.empty:
        st.info("No strategy scores available.")
        return

    display = fit.head(10).drop(columns=["Scanner", "Raw Score", "VRP", "Momentum Z"], errors="ignore")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
        },
    )


def scanner_display(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No candidates passed the current filters.")
        return
    formatted = format_scanner_frame(frame)
    for column in ("PoP", "IV Edge", "Market IV"):
        if column in formatted.columns:
            formatted[column] = formatted[column] * 100
    display = formatted.drop(columns=["Legs"], errors="ignore")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Debit/Credit": st.column_config.NumberColumn(format="$%.2f"),
            "Max Profit": st.column_config.NumberColumn(format="$%.2f"),
            "Max Loss": st.column_config.NumberColumn(format="$%.2f"),
            "PoP": st.column_config.NumberColumn(format="%.1f%%"),
            "EV": st.column_config.NumberColumn(format="$%.2f"),
            "VaR 95": st.column_config.NumberColumn(format="$%.2f"),
            "Edge": st.column_config.NumberColumn(format="$%.2f"),
            "IV Edge": st.column_config.NumberColumn(format="%.1f%%"),
            "Market IV": st.column_config.NumberColumn(format="%.1f%%"),
            "Mid": st.column_config.NumberColumn(format="$%.2f"),
            "Strike": st.column_config.NumberColumn(format="$%.2f"),
            "Width": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_proposal_draft_controls(frame: pd.DataFrame, *, underlying: str, key_prefix: str) -> None:
    if frame.empty or "Legs" not in frame.columns:
        return

    with st.expander("Draft paper proposal", expanded=False):
        st.caption("Writes a proposal artifact only. Paper safety review remains the execution gate.")
        contracts = st.number_input(
            "Contracts per leg unit",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key=f"{key_prefix}_contracts",
        )
        proposal_frame = frame.reset_index(drop=True)
        labels = [
            f"{index + 1}. {_candidate_label(row)}"
            for index, (_, row) in enumerate(proposal_frame.iterrows())
        ]
        selected = st.selectbox(
            "Candidate",
            options=labels,
            index=0,
            key=f"{key_prefix}_candidate",
        )
        selected_index = labels.index(selected)
        if st.button("Write Draft Proposal", key=f"{key_prefix}_write", type="secondary"):
            try:
                result = write_option_trade_proposal_from_candidate(
                    proposal_frame.iloc[selected_index].to_dict(),
                    underlying=underlying,
                    contracts=float(contracts),
                )
            except Exception as exc:
                st.error(f"Could not write proposal: {exc}")
                return
            st.success(f"Draft proposal written: {display_path(result.written_path)}")
            st.caption(f"Proposal ID: `{result.proposal.proposal_id}`")


def _candidate_label(row: pd.Series) -> str:
    strategy = str(row.get("Strategy", "Candidate"))
    expiry = str(row.get("Expiry", ""))
    structure = str(row.get("Structure", row.get("Strike", "")))
    try:
        debit_credit = money(float(row.get("Debit/Credit")))
    except (TypeError, ValueError):
        debit_credit = "n/a"
    return f"{strategy} | {expiry} | {structure} | {debit_credit}"


def render_scanner(
    symbol: str,
    spot: float,
    expirations: list[str],
    budget: float,
    hold_days: int,
    forecast_vol: float,
) -> None:
    if not expirations:
        st.warning("No listed options expirations found.")
        return
    default_expiry = choose_expiration(expirations, hold_days)
    default_index = expirations.index(default_expiry) if default_expiry in expirations else 0
    expiry = st.selectbox("Expiration", expirations, index=default_index)
    calls, puts = fetch_chain(symbol, expiry)

    tab_calls, tab_puts, tab_income, tab_verticals, tab_calendars, tab_condors, tab_butterflies, tab_ratios, tab_backspreads = st.tabs(
        [
            "Long Calls",
            "Long Puts",
            "Cash-Secured Puts",
            "Verticals",
            "Calendars",
            "Condors",
            "Butterflies",
            "Ratios",
            "Backspreads",
        ]
    )
    with tab_calls:
        frame = scan_long_options(
            calls,
            spot=spot,
            expiry=expiry,
            option_type="call",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="long_calls")
    with tab_puts:
        frame = scan_long_options(
            puts,
            spot=spot,
            expiry=expiry,
            option_type="put",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="long_puts")
    with tab_income:
        frame = scan_cash_secured_puts(
            puts,
            spot=spot,
            expiry=expiry,
            max_collateral=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="short_puts")
    with tab_verticals:
        st.subheader("Bull Call Spreads")
        frame = scan_vertical_spreads(
            calls,
            spot=spot,
            expiry=expiry,
            spread_type="bull_call",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="bull_call_spreads")
        st.subheader("Bear Put Spreads")
        frame = scan_vertical_spreads(
            puts,
            spot=spot,
            expiry=expiry,
            spread_type="bear_put",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="bear_put_spreads")
    with tab_calendars:
        far_expiry = far_calendar_expiry(expirations, expiry)
        if far_expiry is None:
            st.info("No farther expiration found for a calendar spread.")
        else:
            far_calls, _ = fetch_chain(symbol, far_expiry)
            frame = scan_calendar_spreads(
                calls,
                far_calls,
                spot=spot,
                near_expiry=expiry,
                far_expiry=far_expiry,
                budget=budget,
                forecast_vol=forecast_vol,
            )
            scanner_display(frame)
            render_proposal_draft_controls(frame, underlying=symbol, key_prefix="calendar_spreads")
    with tab_condors:
        frame = scan_iron_condors(
            calls,
            puts,
            spot=spot,
            expiry=expiry,
            max_risk=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="iron_condors")
    with tab_butterflies:
        frame = scan_call_butterflies(
            calls,
            spot=spot,
            expiry=expiry,
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="call_butterflies")
    with tab_ratios:
        st.subheader("Call Ratio Spreads")
        frame = scan_ratio_spreads(
            calls,
            spot=spot,
            expiry=expiry,
            option_type="call",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="call_ratio_spreads")
        st.subheader("Put Ratio Spreads")
        frame = scan_ratio_spreads(
            puts,
            spot=spot,
            expiry=expiry,
            option_type="put",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="put_ratio_spreads")
    with tab_backspreads:
        st.subheader("Call Backspreads")
        frame = scan_backspreads(
            calls,
            spot=spot,
            expiry=expiry,
            option_type="call",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="call_backspreads")
        st.subheader("Put Backspreads")
        frame = scan_backspreads(
            puts,
            spot=spot,
            expiry=expiry,
            option_type="put",
            budget=budget,
            days_to_hold=hold_days,
            forecast_vol=forecast_vol,
        )
        scanner_display(frame)
        render_proposal_draft_controls(frame, underlying=symbol, key_prefix="put_backspreads")


def render_strategy_lab(spot: float, forecast_vol: float, hold_days: int) -> None:
    st.subheader("Strategy Lab")
    cols = st.columns(5)
    option_type = cols[0].selectbox("Option", ["call", "put"])
    side = cols[1].selectbox("Side", ["long", "short"])
    strike = cols[2].number_input("Strike", min_value=0.01, value=float(round(spot or 100, 2)), step=1.0)
    premium = cols[3].number_input("Premium", min_value=0.0, value=2.50, step=0.05)
    simulations = cols[4].number_input("Simulations", min_value=500, max_value=20000, value=5000, step=500)

    sim = simulate_single_option(
        spot=spot,
        strike=strike,
        premium=premium,
        option_type=option_type,
        side=side,
        days_to_hold=hold_days,
        volatility=forecast_vol,
        simulations=int(simulations),
    )

    metrics = st.columns(5)
    metrics[0].metric("PoP", pct(sim.probability_of_profit))
    metrics[1].metric("EV / Share", money(sim.expected_value))
    metrics[2].metric("VaR 95 / Share", money(sim.value_at_risk_95))
    metrics[3].metric("Worst / Share", money(sim.worst_case))
    metrics[4].metric("Best / Share", money(sim.best_case))

    fig = go.Figure()
    fig.add_histogram(x=sim.profits * 100, nbinsx=60, marker_color="#2563EB")
    fig.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis_title="PnL per contract",
        yaxis_title="Simulations",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_options() -> None:
    positions = load_latest_live_positions(default_portfolio_ledger_path())
    if positions.empty:
        st.info("No live position rows found yet.")
        return
    options = positions[positions["asset_type"].astype(str).str.lower().eq("option")].copy()
    if options.empty:
        st.info("No options positions found in the runtime ledger.")
        return
    for column in ("shares", "avg_cost", "current_price", "unrealized_pnl", "delta", "gamma"):
        options[column] = pd.to_numeric(options[column], errors="coerce").fillna(0.0)
    options["market_value"] = options["shares"] * options["current_price"] * 100
    options["delta_exposure"] = options["market_value"] * options["delta"]
    display = options[
        [
            "date",
            "broker",
            "ticker",
            "shares",
            "avg_cost",
            "current_price",
            "market_value",
            "unrealized_pnl",
            "delta",
            "gamma",
            "delta_exposure",
            "currency",
        ]
    ].rename(
        columns={
            "date": "Date",
            "broker": "Broker",
            "ticker": "Ticker",
            "shares": "Contracts",
            "avg_cost": "Avg Cost",
            "current_price": "Current Price",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L",
            "delta": "Delta",
            "gamma": "Gamma",
            "delta_exposure": "Delta Exposure",
            "currency": "Currency",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


st.title("Options Desk")
st.caption("Options chain scanning, volatility, Monte Carlo payoff simulation, and runtime option exposure.")

with st.sidebar:
    st.header("Scanner")
    target_ticker = st.text_input("Target ticker", value="QQQ").upper().strip()
    max_budget = st.number_input("Max budget / collateral", min_value=100.0, value=1000.0, step=100.0)
    hold_days = st.number_input("Target hold days", min_value=1, max_value=365, value=30, step=1)
    run_scan = st.button("Load Options Desk", type="primary", use_container_width=True)

if run_scan and target_ticker:
    st.session_state["options_loaded_ticker"] = target_ticker
    st.session_state["options_hold_days"] = int(hold_days)
    st.session_state["options_budget"] = float(max_budget)

loaded_ticker = st.session_state.get("options_loaded_ticker")
if not loaded_ticker:
    tab_portfolio, tab_empty = st.tabs(["Portfolio Options", "Desk"])
    with tab_portfolio:
        render_portfolio_options()
    with tab_empty:
        st.info("Load a ticker from the sidebar to open the options desk.")
    st.stop()

hold_days = int(st.session_state.get("options_hold_days", hold_days))
max_budget = float(st.session_state.get("options_budget", max_budget))
history = fetch_market_history(loaded_ticker)
spot = latest_close(history)
expirations = fetch_expirations(loaded_ticker)
snapshot = volatility_snapshot(history)

if spot <= 0:
    st.warning("Could not resolve a usable market price for this ticker.")
    st.stop()

metric_cols = st.columns(5)
metric_cols[0].metric("Ticker", loaded_ticker)
metric_cols[1].metric("Spot", money(spot))
metric_cols[2].metric("Forecast Vol", pct(snapshot.forecast_vol))
metric_cols[3].metric("RSI 14", f"{snapshot.rsi_14:.1f}")
metric_cols[4].metric("ATR 14", money(snapshot.atr_14))

tab_vol, tab_fit, tab_scan, tab_lab, tab_portfolio = st.tabs(
    ["Volatility", "Strategy Fit", "Scanner", "Strategy Lab", "Portfolio Options"]
)

with tab_vol:
    render_volatility(loaded_ticker, history, hold_days)

with tab_fit:
    render_strategy_fit(
        loaded_ticker,
        history,
        expirations,
        spot,
        snapshot.forecast_vol or snapshot.historical_vol_21d or 0.20,
        snapshot.rsi_14,
        hold_days,
    )

with tab_scan:
    render_scanner(
        loaded_ticker,
        spot,
        expirations,
        max_budget,
        hold_days,
        snapshot.forecast_vol or snapshot.historical_vol_21d or 0.20,
    )

with tab_lab:
    render_strategy_lab(spot, snapshot.forecast_vol or snapshot.historical_vol_21d or 0.20, hold_days)

with tab_portfolio:
    render_portfolio_options()
