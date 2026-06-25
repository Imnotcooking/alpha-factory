"""Portfolio valuation and NAV analytics shared by dashboards and jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ManualPortfolioInputs:
    t212_cash_eur: float = 0.0
    futu_cash_usd: float = 0.0
    ibkr_cash_usd: float = 0.0
    cny_mutual_fund: float = 0.0
    cny_mutual_fund_pnl: float = 0.0
    cny_gold_grams: float = 0.0
    cny_gold_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class PortfolioValuationResult:
    position_valuation: pd.DataFrame
    broker_summary: pd.DataFrame
    asset_summary: pd.DataFrame
    usd_history: pd.DataFrame
    total_net_worth: float
    total_pnl: float
    total_cash: float
    portfolio_beta: float
    fx_rates: dict[str, float] = field(default_factory=dict)


def value_portfolio_snapshot(
    positions: pd.DataFrame,
    market_history: pd.DataFrame,
    *,
    benchmark: str = "QQQ",
    manual_inputs: ManualPortfolioInputs | None = None,
    asset_preferences: dict[str, dict[str, Any]] | None = None,
) -> PortfolioValuationResult:
    """Value a live-position snapshot in USD and calculate portfolio risk fields."""

    manual = manual_inputs or ManualPortfolioInputs()
    preferences = asset_preferences or {}
    live_prices = market_history.iloc[-1] if not market_history.empty else pd.Series()
    fx = _fx_rates(live_prices)

    position_rows = _value_position_rows(positions, live_prices, fx)
    position_rows.extend(_manual_position_rows(manual, fx))
    position_valuation = pd.DataFrame(
        position_rows,
        columns=[
            "Broker",
            "Category",
            "Ticker",
            "Exposure_USD",
            "Cost_USD",
            "Current_USD",
            "PnL_USD",
        ],
    )

    if position_valuation.empty:
        return PortfolioValuationResult(
            position_valuation=position_valuation,
            broker_summary=pd.DataFrame(columns=["Broker", "Current_USD", "PnL_USD"]),
            asset_summary=_empty_asset_summary(benchmark),
            usd_history=pd.DataFrame(index=market_history.index),
            total_net_worth=0.0,
            total_pnl=0.0,
            total_cash=_total_cash_usd(manual, fx),
            portfolio_beta=0.0,
            fx_rates=fx,
        )

    broker_summary = (
        position_valuation.groupby("Broker")
        .agg({"Current_USD": "sum", "PnL_USD": "sum"})
        .reset_index()
    )
    asset_summary = (
        position_valuation.groupby("Ticker")
        .agg(
            {
                "Category": "first",
                "Cost_USD": "sum",
                "Current_USD": "sum",
                "Exposure_USD": "sum",
                "PnL_USD": "sum",
            }
        )
        .reset_index()
    )

    total_net_worth = float(asset_summary["Current_USD"].sum())
    total_pnl = float(asset_summary["PnL_USD"].sum())
    asset_summary["PnL_Pct"] = np.where(
        asset_summary["Cost_USD"] != 0,
        (asset_summary["PnL_USD"] / asset_summary["Cost_USD"].abs()) * 100,
        0.0,
    )

    total_delta_exposure = float(asset_summary["Exposure_USD"].sum())
    if total_delta_exposure > 0:
        asset_summary["Actual_Weight_%"] = (
            asset_summary["Exposure_USD"] / total_delta_exposure
        ) * 100
    else:
        asset_summary["Actual_Weight_%"] = 0.0

    asset_summary["Category"] = asset_summary.apply(
        lambda row: preferences.get(row["Ticker"], {}).get("Category", row["Category"]),
        axis=1,
    )
    asset_summary["Target_Weight_%"] = asset_summary.apply(
        lambda row: preferences.get(row["Ticker"], {}).get(
            "Target_Weight_%",
            row["Actual_Weight_%"],
        ),
        axis=1,
    )

    usd_history = _build_usd_history(market_history, fx)
    _add_correlation_and_beta(asset_summary, usd_history, benchmark)
    beta_col = f"Beta_to_{benchmark}"
    if total_net_worth != 0:
        portfolio_beta = float(
            ((asset_summary["Current_USD"] / total_net_worth) * asset_summary[beta_col]).sum()
        )
    else:
        portfolio_beta = 0.0

    return PortfolioValuationResult(
        position_valuation=position_valuation,
        broker_summary=broker_summary,
        asset_summary=asset_summary,
        usd_history=usd_history,
        total_net_worth=total_net_worth,
        total_pnl=total_pnl,
        total_cash=_total_cash_usd(manual, fx),
        portfolio_beta=portfolio_beta,
        fx_rates=fx,
    )


def _value_position_rows(
    positions: pd.DataFrame,
    live_prices: pd.Series,
    fx: dict[str, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for _, row in positions.iterrows():
        ticker = str(row["ticker"])
        shares = _float(row["shares"])
        avg_price = _float(row["avg_cost"])
        broker_name = str(row["broker"])
        asset_type = str(row["asset_type"])
        broker_mark_price = _float(row["current_price"])
        broker_currency = str(row.get("currency", "USD")).strip().upper()
        raw_pnl = _float(row.get("unrealized_pnl", 0.0))
        delta = _float(row.get("delta", 1.0), default=1.0)
        multiplier = 100.0 if asset_type == "Option" else 1.0

        local_cost = shares * avg_price * multiplier
        cost_usd = local_cost * _currency_rate(broker_currency, fx)
        yahoo_ticker = ticker.replace("BRK.B", "BRK-B")

        if broker_name == "IBKR Live" and raw_pnl != 0.0:
            pnl_usd = raw_pnl * _currency_rate(broker_currency, fx)
            current_usd = cost_usd + pnl_usd
        else:
            raw_market_price = _price(live_prices, yahoo_ticker, broker_mark_price)
            if ticker.endswith(".L"):
                current_usd = shares * (raw_market_price / 100.0) * multiplier * fx["GBP"]
            elif ticker.endswith(".DE") or ticker.endswith(".AS") or ticker == "VWCE":
                current_usd = shares * raw_market_price * multiplier * fx["EUR"]
            elif broker_currency == "HKD":
                current_usd = shares * raw_market_price * multiplier * fx["HKD"]
            else:
                current_usd = shares * raw_market_price * multiplier
            pnl_usd = current_usd - cost_usd

        rows.append(
            {
                "Broker": broker_name,
                "Category": _default_category(ticker),
                "Ticker": ticker,
                "Exposure_USD": current_usd * abs(delta),
                "Cost_USD": cost_usd,
                "Current_USD": current_usd,
                "PnL_USD": pnl_usd,
            }
        )
    return rows


def _manual_position_rows(
    manual: ManualPortfolioInputs,
    fx: dict[str, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if manual.t212_cash_eur > 0:
        value = manual.t212_cash_eur * fx["EUR"]
        rows.append(_manual_row("Trading212", "Cash", "EUR Cash", value, 0.0))
    if manual.futu_cash_usd > 0:
        rows.append(_manual_row("Futubull", "Cash", "USD Cash", manual.futu_cash_usd, 0.0))
    if manual.ibkr_cash_usd > 0:
        rows.append(_manual_row("IBKR Live", "Cash", "IBKR Cash", manual.ibkr_cash_usd, 0.0))

    if manual.cny_mutual_fund > 0:
        cost = manual.cny_mutual_fund * fx["CNY"]
        pnl = manual.cny_mutual_fund_pnl * fx["CNY"]
        rows.append(
            _manual_row(
                "Chinese Fund",
                "Core Compounding",
                "CNY Mutual Fund",
                cost + pnl,
                pnl,
                exposure=abs(cost + pnl),
                cost=cost,
            )
        )

    if manual.cny_gold_grams > 0:
        cost = manual.cny_gold_grams * manual.cny_gold_cost * fx["CNY"]
        current = manual.cny_gold_grams * fx["GOLD_GRAM_USD"]
        rows.append(
            _manual_row(
                "Chinese Gold",
                "Defensive",
                "Physical Gold",
                current,
                current - cost,
                exposure=current,
                cost=cost,
            )
        )
    return rows


def _manual_row(
    broker: str,
    category: str,
    ticker: str,
    current: float,
    pnl: float,
    *,
    exposure: float = 0.0,
    cost: float | None = None,
) -> dict[str, float | str]:
    return {
        "Broker": broker,
        "Category": category,
        "Ticker": ticker,
        "Exposure_USD": exposure,
        "Cost_USD": current if cost is None else cost,
        "Current_USD": current,
        "PnL_USD": pnl,
    }


def _build_usd_history(market_history: pd.DataFrame, fx: dict[str, float]) -> pd.DataFrame:
    usd_history = pd.DataFrame(index=market_history.index)
    for column in market_history.columns:
        if column in {"EURUSD=X", "GBPUSD=X", "CNYUSD=X"}:
            continue
        if str(column).endswith(".L"):
            usd_history[column] = (market_history[column] / 100.0) * fx["GBP"]
        elif str(column).endswith(".DE") or str(column).endswith(".AS"):
            usd_history[column] = market_history[column] * fx["EUR"]
        else:
            usd_history[column] = market_history[column]
    return usd_history


def _add_correlation_and_beta(
    asset_summary: pd.DataFrame,
    usd_history: pd.DataFrame,
    benchmark: str,
) -> None:
    corr_col = f"Corr_to_{benchmark}"
    beta_col = f"Beta_to_{benchmark}"
    correlations: dict[str, float] = {}
    betas: dict[str, float] = {}

    log_returns = np.log(usd_history / usd_history.shift(1)).dropna()
    if benchmark in log_returns.columns:
        benchmark_variance = log_returns[benchmark].var()
        for ticker in asset_summary["Ticker"].unique():
            yahoo_ticker = "GC=F" if ticker == "Physical Gold" else str(ticker).replace("BRK.B", "BRK-B")
            if yahoo_ticker in log_returns.columns and benchmark_variance > 0:
                correlations[ticker] = log_returns[yahoo_ticker].corr(log_returns[benchmark])
                betas[ticker] = log_returns[yahoo_ticker].cov(log_returns[benchmark]) / benchmark_variance
            else:
                correlations[ticker], betas[ticker] = 0.0, 0.0

    asset_summary[corr_col] = asset_summary["Ticker"].map(correlations).fillna(0.0)
    asset_summary[beta_col] = asset_summary["Ticker"].map(betas).fillna(0.0)
    asset_summary.loc[asset_summary["Category"] == "Cash", [corr_col, beta_col]] = 0.0


def _fx_rates(live_prices: pd.Series) -> dict[str, float]:
    gold_oz_usd = _price(live_prices, "GC=F", 2000.0)
    return {
        "EUR": _price(live_prices, "EURUSD=X", 1.0),
        "GBP": _price(live_prices, "GBPUSD=X", 1.0),
        "CNY": _price(live_prices, "CNYUSD=X", 0.14),
        "HKD": _price(live_prices, "HKDUSD=X", 0.128),
        "GOLD_GRAM_USD": gold_oz_usd / 31.1035,
    }


def _currency_rate(currency: str, fx: dict[str, float]) -> float:
    return fx.get(currency.upper(), 1.0)


def _total_cash_usd(manual: ManualPortfolioInputs, fx: dict[str, float]) -> float:
    return (
        manual.t212_cash_eur * fx["EUR"]
        + manual.futu_cash_usd
        + manual.ibkr_cash_usd
    )


def _default_category(ticker: str) -> str:
    if ticker in {"SHY", "TLT"}:
        return "Defensive"
    if ticker in {"VWCE", "VUAA", "EQQQ"}:
        return "Core Foundation"
    return "Aggressive" if len(ticker) > 5 else "Core Compounding"


def _price(prices: pd.Series, key: str, default: float) -> float:
    try:
        value = prices.get(key, default)
    except AttributeError:
        return float(default)
    return _float(value, default=default)


def _float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empty_asset_summary(benchmark: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Ticker",
            "Category",
            "Cost_USD",
            "Current_USD",
            "Exposure_USD",
            "PnL_USD",
            "PnL_Pct",
            "Actual_Weight_%",
            "Target_Weight_%",
            f"Corr_to_{benchmark}",
            f"Beta_to_{benchmark}",
        ]
    )
