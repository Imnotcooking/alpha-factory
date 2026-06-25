"""Portfolio risk analytics and hedge sizing helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class PortfolioRiskSummary:
    latest_nav: float = 0.0
    latest_cash: float = 0.0
    portfolio_beta: float = 0.0
    total_market_value: float = 0.0
    gross_exposure: float = 0.0
    net_delta_exposure: float = 0.0
    gross_delta_exposure: float = 0.0
    beta_adjusted_exposure: float = 0.0
    option_gross_exposure: float = 0.0
    concentration_top1_pct: float = 0.0
    concentration_top5_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    annualized_nav_volatility: float = 0.0
    one_day_var_95: float = 0.0
    one_day_cvar_95: float = 0.0
    latest_daily_pnl: float = 0.0
    nav_observations: int = 0
    position_rows: int = 0


@dataclass(frozen=True, slots=True)
class InverseHedgePlan:
    hedge_asset: str
    hedge_price: float
    leverage: float
    budget: float
    shares_to_buy: float
    current_delta: float
    hedge_delta: float
    net_delta_after: float
    beta_after: float
    effective_hedge_pct: float
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass(frozen=True, slots=True)
class HedgeDiagnosis:
    net_contract_delta: float
    total_protection: float
    coverage_pct: float
    unhedged_exposure: float
    additional_contracts_needed: int
    verdict: str


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def position_multiplier(asset_type: Any) -> float:
    return 100.0 if str(asset_type).strip().lower() == "option" else 1.0


def enrich_position_risk(positions: pd.DataFrame) -> pd.DataFrame:
    """Add risk fields to a normalized ``live_positions`` frame."""

    columns = [
        "date",
        "broker",
        "ticker",
        "asset_type",
        "shares",
        "avg_cost",
        "current_price",
        "unrealized_pnl",
        "currency",
        "delta",
        "gamma",
    ]
    if positions.empty:
        return pd.DataFrame(
            columns=columns
            + [
                "multiplier",
                "effective_delta",
                "market_value",
                "gross_exposure",
                "signed_delta_exposure",
                "gross_delta_exposure",
                "cost_basis",
                "pnl_pct",
            ]
        )

    out = positions.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0 if column in {"shares", "avg_cost", "current_price", "unrealized_pnl", "delta", "gamma"} else ""

    for column in ("shares", "avg_cost", "current_price", "unrealized_pnl", "delta", "gamma"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    out["multiplier"] = out["asset_type"].map(position_multiplier)
    non_options = out["asset_type"].astype(str).str.lower().ne("option")
    out["effective_delta"] = out["delta"]
    out.loc[non_options & out["effective_delta"].eq(0.0), "effective_delta"] = 1.0
    out["market_value"] = out["shares"] * out["current_price"] * out["multiplier"]
    out["gross_exposure"] = out["market_value"].abs()
    out["signed_delta_exposure"] = out["market_value"] * out["effective_delta"]
    out["gross_delta_exposure"] = out["signed_delta_exposure"].abs()
    out["cost_basis"] = out["shares"] * out["avg_cost"] * out["multiplier"]
    fallback_pnl = out["market_value"] - out["cost_basis"]
    out["unrealized_pnl"] = out["unrealized_pnl"].where(out["unrealized_pnl"].ne(0.0), fallback_pnl)
    out["pnl_pct"] = out["unrealized_pnl"] / out["cost_basis"].abs().replace({0.0: pd.NA})
    out["pnl_pct"] = out["pnl_pct"].fillna(0.0)
    return out


def concentration_table(enriched_positions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if enriched_positions.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Asset Type",
                "Gross Exposure",
                "Delta Exposure",
                "Weight %",
                "Unrealized P&L",
            ]
        )

    grouped = (
        enriched_positions.groupby(["ticker", "asset_type"], dropna=False)
        .agg(
            gross_exposure=("gross_exposure", "sum"),
            signed_delta_exposure=("signed_delta_exposure", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
    )
    total_gross = safe_float(grouped["gross_exposure"].sum())
    grouped["weight_pct"] = grouped["gross_exposure"] / total_gross if total_gross else 0.0
    grouped = grouped.sort_values("gross_exposure", ascending=False).head(top_n)
    return grouped.rename(
        columns={
            "ticker": "Ticker",
            "asset_type": "Asset Type",
            "gross_exposure": "Gross Exposure",
            "signed_delta_exposure": "Delta Exposure",
            "weight_pct": "Weight %",
            "unrealized_pnl": "Unrealized P&L",
        }
    )


def broker_risk_table(enriched_positions: pd.DataFrame) -> pd.DataFrame:
    if enriched_positions.empty:
        return pd.DataFrame(
            columns=["Broker", "Rows", "Gross Exposure", "Delta Exposure", "Unrealized P&L"]
        )
    return (
        enriched_positions.groupby("broker")
        .agg(
            rows=("ticker", "count"),
            gross_exposure=("gross_exposure", "sum"),
            signed_delta_exposure=("signed_delta_exposure", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "broker": "Broker",
                "rows": "Rows",
                "gross_exposure": "Gross Exposure",
                "signed_delta_exposure": "Delta Exposure",
                "unrealized_pnl": "Unrealized P&L",
            }
        )
    )


def summarize_portfolio_risk(
    enriched_positions: pd.DataFrame,
    nav_history: pd.DataFrame,
) -> PortfolioRiskSummary:
    latest_nav = 0.0
    latest_cash = 0.0
    portfolio_beta = 0.0
    latest_daily_pnl = 0.0
    current_drawdown = 0.0
    current_drawdown_pct = 0.0
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    annualized_vol = 0.0
    var_95 = 0.0
    cvar_95 = 0.0

    if not nav_history.empty:
        nav = nav_history.copy()
        nav["total_net_worth"] = pd.to_numeric(nav["total_net_worth"], errors="coerce").fillna(0.0)
        nav["total_cash"] = pd.to_numeric(nav["total_cash"], errors="coerce").fillna(0.0)
        nav["portfolio_beta"] = pd.to_numeric(nav["portfolio_beta"], errors="coerce").fillna(0.0)
        nav["daily_pnl"] = pd.to_numeric(nav["daily_pnl"], errors="coerce").fillna(0.0)
        latest = nav.iloc[-1]
        latest_nav = safe_float(latest["total_net_worth"])
        latest_cash = safe_float(latest["total_cash"])
        portfolio_beta = safe_float(latest["portfolio_beta"])
        latest_daily_pnl = safe_float(latest["daily_pnl"])

        if "drawdown" in nav.columns and "drawdown_pct" in nav.columns:
            current_drawdown = safe_float(nav["drawdown"].iloc[-1])
            current_drawdown_pct = safe_float(nav["drawdown_pct"].iloc[-1])
            max_drawdown = safe_float(nav["drawdown"].min())
            max_drawdown_pct = safe_float(nav["drawdown_pct"].min())

        returns = nav["total_net_worth"].pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
        if not returns.empty:
            annualized_vol = safe_float(returns.std()) * math.sqrt(TRADING_DAYS)
            threshold = returns.quantile(0.05)
            var_95 = abs(safe_float(threshold) * latest_nav)
            tail = returns[returns <= threshold]
            cvar_95 = abs(safe_float(tail.mean()) * latest_nav) if not tail.empty else var_95

    total_market_value = safe_float(enriched_positions["market_value"].sum()) if not enriched_positions.empty else 0.0
    gross_exposure = safe_float(enriched_positions["gross_exposure"].sum()) if not enriched_positions.empty else 0.0
    net_delta_exposure = safe_float(enriched_positions["signed_delta_exposure"].sum()) if not enriched_positions.empty else 0.0
    gross_delta_exposure = safe_float(enriched_positions["gross_delta_exposure"].sum()) if not enriched_positions.empty else 0.0
    option_mask = enriched_positions["asset_type"].astype(str).str.lower().eq("option") if not enriched_positions.empty else pd.Series(dtype=bool)
    option_gross_exposure = (
        safe_float(enriched_positions.loc[option_mask, "gross_exposure"].sum())
        if not enriched_positions.empty
        else 0.0
    )
    concentration = concentration_table(enriched_positions, top_n=5)
    concentration_top1_pct = safe_float(concentration["Weight %"].iloc[0]) if not concentration.empty else 0.0
    concentration_top5_pct = safe_float(concentration["Weight %"].sum()) if not concentration.empty else 0.0
    beta_adjusted_exposure = latest_nav * portfolio_beta if latest_nav else net_delta_exposure * portfolio_beta

    return PortfolioRiskSummary(
        latest_nav=latest_nav,
        latest_cash=latest_cash,
        portfolio_beta=portfolio_beta,
        total_market_value=total_market_value,
        gross_exposure=gross_exposure,
        net_delta_exposure=net_delta_exposure,
        gross_delta_exposure=gross_delta_exposure,
        beta_adjusted_exposure=beta_adjusted_exposure,
        option_gross_exposure=option_gross_exposure,
        concentration_top1_pct=concentration_top1_pct,
        concentration_top5_pct=concentration_top5_pct,
        current_drawdown=current_drawdown,
        current_drawdown_pct=current_drawdown_pct,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        annualized_nav_volatility=annualized_vol,
        one_day_var_95=var_95,
        one_day_cvar_95=cvar_95,
        latest_daily_pnl=latest_daily_pnl,
        nav_observations=len(nav_history),
        position_rows=len(enriched_positions),
    )


def average_true_range(price_history: pd.DataFrame, window: int = 14) -> float:
    if price_history.empty or not {"High", "Low", "Close"}.issubset(price_history.columns):
        return 0.0
    hist = price_history.copy()
    high_low = hist["High"] - hist["Low"]
    high_prev_close = (hist["High"] - hist["Close"].shift(1)).abs()
    low_prev_close = (hist["Low"] - hist["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean().dropna()
    return safe_float(atr.iloc[-1]) if not atr.empty else 0.0


def inverse_hedge_plan(
    *,
    portfolio_value: float,
    portfolio_beta: float,
    hedge_asset: str,
    hedge_price: float,
    budget: float,
    leverage: float,
    atr: float | None = None,
) -> InverseHedgePlan:
    current_delta = portfolio_value * portfolio_beta
    shares_to_buy = budget / hedge_price if hedge_price > 0 else 0.0
    hedge_delta = budget * leverage
    net_delta_after = current_delta + hedge_delta
    beta_after = net_delta_after / portfolio_value if portfolio_value else 0.0
    effective_hedge_pct = abs(hedge_delta) / abs(current_delta) if current_delta else 0.0
    stop_loss = hedge_price - (1.5 * atr) if atr and hedge_price else None
    take_profit = hedge_price + (3.0 * atr) if atr and hedge_price else None
    return InverseHedgePlan(
        hedge_asset=hedge_asset,
        hedge_price=hedge_price,
        leverage=leverage,
        budget=budget,
        shares_to_buy=shares_to_buy,
        current_delta=current_delta,
        hedge_delta=hedge_delta,
        net_delta_after=net_delta_after,
        beta_after=beta_after,
        effective_hedge_pct=effective_hedge_pct,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
) -> dict[str, float]:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        return {"Price": 0.0, "Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0}

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    pdf_d1 = normal_pdf(d1)

    if option_type.lower() == "call":
        price = spot * normal_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * normal_cdf(d2)
        delta = normal_cdf(d1)
    else:
        price = strike * math.exp(-rate * time_to_expiry) * normal_cdf(-d2) - spot * normal_cdf(-d1)
        delta = normal_cdf(d1) - 1

    gamma = pdf_d1 / (spot * volatility * sqrt_t)
    theta = -((spot * pdf_d1 * volatility) / (2 * sqrt_t)) / 365
    vega = spot * pdf_d1 * sqrt_t / 100
    return {
        "Price": float(price),
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Theta": float(theta),
        "Vega": float(vega),
    }


def hedge_diagnosis(
    *,
    beta_adjusted_exposure: float,
    net_contract_delta: float,
    underlying_price: float,
    contracts: int,
    contract_size: float = 100.0,
) -> HedgeDiagnosis:
    total_protection = abs(net_contract_delta) * underlying_price * contract_size * max(contracts, 0)
    coverage_pct = total_protection / beta_adjusted_exposure if beta_adjusted_exposure > 0 else 1.0
    unhedged = max(beta_adjusted_exposure - total_protection, 0.0)
    per_contract_protection = abs(net_contract_delta) * underlying_price * contract_size
    additional = math.ceil(unhedged / per_contract_protection) if per_contract_protection > 0 else 0
    if contracts == 0:
        verdict = "unhedged"
    elif coverage_pct < 0.30:
        verdict = "severely_under_hedged"
    elif coverage_pct < 0.80:
        verdict = "partially_hedged"
    else:
        verdict = "properly_hedged"
    return HedgeDiagnosis(
        net_contract_delta=net_contract_delta,
        total_protection=total_protection,
        coverage_pct=coverage_pct,
        unhedged_exposure=unhedged,
        additional_contracts_needed=additional,
        verdict=verdict,
    )


def micro_future_multiplier(symbol: str) -> float:
    upper = symbol.upper()
    if "QQQ" in upper or "NDX" in upper:
        return 80.0
    if "SPY" in upper or "SPX" in upper:
        return 50.0
    return 50.0
