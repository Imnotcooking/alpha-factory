from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical


DEFAULT_ASSET_CLASS = "FUTURES_CN"
CORE_DASHBOARD_ASSET_CLASSES = (
    "EQUITY_US",
    "OPTIONS_US",
    "EQUITY_CN",
    "OPTIONS_CN",
    "FUTURES_CN",
)

FALLBACK_ASSET_TAXONOMY: dict[str, dict[str, Any]] = {
    key: dict(value) for key, value in ASSET_TAXONOMY.items()
}

LANE_METADATA: dict[str, dict[str, str]] = {
    "FUTURES_CN": {
        "label": "Chinese Futures",
        "label_zh": "中国期货",
        "role": "Live research lane; QMT execution lane next",
        "data_mode": "Local daily parquet + tick cache; QMT/Wind ready later",
        "provider": "Local runtime files; Wind/QMT later",
        "broker": "华源证券 via QMT",
        "execution": "Planned QMT execution",
        "dashboard_scope": "Research, risk, future CN trading",
        "status": "active_data_planned_execution",
    },
    "FUTURES_US": {
        "label": "US Futures",
        "label_zh": "美国期货",
        "role": "Configured US futures lane",
        "data_mode": "Vendor/API-backed futures bars",
        "provider": "TBD",
        "broker": "TBD",
        "execution": "Not wired",
        "dashboard_scope": "Future extension",
        "status": "planned",
    },
    "EQUITY_US": {
        "label": "US Equities",
        "label_zh": "美国股票",
        "role": "Active discretionary/live portfolio lane",
        "data_mode": "API-backed quotes, fundamentals, watchlists, ledgers",
        "provider": "FMP + Yahoo; Massive where available",
        "broker": "IBKR",
        "execution": "IBKR paper/live guardrails",
        "dashboard_scope": "Ops, discretionary, paper/live accounts",
        "status": "active",
    },
    "EQUITY_CN": {
        "label": "Chinese Equities",
        "label_zh": "中国股票",
        "role": "QMT broker lane for A-share discretionary/systematic flow",
        "data_mode": "QMT/Wind-backed A-share bars and positions",
        "provider": "Wind/QMT planned",
        "broker": "华源证券 via QMT",
        "execution": "Planned QMT execution",
        "dashboard_scope": "Ops, discretionary, future CN portfolios",
        "status": "planned_qmt",
    },
    "EQUITY_HK": {
        "label": "Hong Kong Equities",
        "label_zh": "港股",
        "role": "Configured Hong Kong equities lane",
        "data_mode": "API-backed HK equity bars",
        "provider": "Yahoo/FMP/Wind as available",
        "broker": "IBKR or external broker",
        "execution": "Manual/external for now",
        "dashboard_scope": "Live portfolio holdings bridge",
        "status": "bridge",
    },
    "OPTIONS_US": {
        "label": "US Options",
        "label_zh": "美国期权",
        "role": "Active discretionary options lane",
        "data_mode": "Option chains, marks, IV/Greeks, payoff labs",
        "provider": "Massive primary; Yahoo fallback",
        "broker": "IBKR",
        "execution": "IBKR paper/live guardrails",
        "dashboard_scope": "Ops options hub, workbench, risk",
        "status": "active",
    },
    "OPTIONS_CN": {
        "label": "Chinese Options",
        "label_zh": "中国期权",
        "role": "QMT lane for ETF/index/commodity options",
        "data_mode": "QMT/Wind option chains, marks, Greeks",
        "provider": "Wind/QMT planned",
        "broker": "华源证券 via QMT",
        "execution": "Planned QMT execution",
        "dashboard_scope": "Ops options hub, risk, future CN trading",
        "status": "planned_qmt",
    },
    "FX_SPOT": {
        "label": "FX Spot",
        "label_zh": "外汇现货",
        "role": "Configured FX spot lane",
        "data_mode": "Vendor/API-backed spot bars",
        "provider": "TBD",
        "broker": "TBD",
        "execution": "Not wired",
        "dashboard_scope": "Future extension",
        "status": "planned",
    },
    "CRYPTO_PERP": {
        "label": "Crypto Perpetuals",
        "label_zh": "加密永续",
        "role": "Configured crypto perpetual lane",
        "data_mode": "Exchange/API-backed perpetual bars",
        "provider": "TBD",
        "broker": "TBD",
        "execution": "Not wired",
        "dashboard_scope": "Future extension",
        "status": "planned",
    },
}


def load_asset_taxonomy(base_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the shared market taxonomy.

    ``base_dir`` is accepted for compatibility with the old alpha-lab helper.
    The taxonomy source of truth now lives in ``oqp.contracts.market_vertical``.
    """
    return {key: dict(value) for key, value in FALLBACK_ASSET_TAXONOMY.items()}


def asset_class_label(asset_class: str, taxonomy: dict[str, dict[str, Any]]) -> str:
    asset_class = normalize_market_vertical(asset_class)
    meta = taxonomy.get(asset_class, {})
    description = meta.get("description")
    return f"{asset_class} - {description}" if description else asset_class


def taxonomy_options(
    taxonomy: dict[str, dict[str, Any]],
    observed: set[str] | None = None,
    default: str = DEFAULT_ASSET_CLASS,
) -> list[str]:
    observed = {normalize_market_vertical(value) for value in (observed or set())}
    default = normalize_market_vertical(default)
    ordered: list[str] = []
    for key in [default, *sorted(observed), *sorted(taxonomy)]:
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def core_dashboard_asset_classes() -> list[str]:
    """Return the market lanes that should be visible across dashboards."""

    return [normalize_market_vertical(value) for value in CORE_DASHBOARD_ASSET_CLASSES]


def is_options_asset_class(asset_class: str) -> bool:
    meta = FALLBACK_ASSET_TAXONOMY.get(normalize_market_vertical(asset_class), {})
    return meta.get("instrument_family") == "option"


def is_vectorizable_asset_class(asset_class: str) -> bool:
    meta = FALLBACK_ASSET_TAXONOMY.get(normalize_market_vertical(asset_class), {})
    return bool(meta.get("vectorizable", False))


def backtest_route(asset_class: str) -> str:
    meta = FALLBACK_ASSET_TAXONOMY.get(normalize_market_vertical(asset_class), {})
    return str(meta.get("backtest_route") or "vectorized")


def attach_asset_class(
    df: pd.DataFrame,
    *,
    default: str = DEFAULT_ASSET_CLASS,
    ticker_to_asset_class: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a copy with a normalized asset_class column."""
    out = df.copy()
    if out.empty:
        if "asset_class" not in out.columns:
            out["asset_class"] = pd.Series(dtype="object")
        return out

    if "asset_class" in out.columns:
        out["asset_class"] = out["asset_class"].map(normalize_market_vertical)
    elif "market_vertical" in out.columns:
        out["asset_class"] = out["market_vertical"].map(normalize_market_vertical)
    elif ticker_to_asset_class and "ticker" in out.columns:
        out["asset_class"] = out["ticker"].astype(str).map(ticker_to_asset_class).fillna(default)
    else:
        out["asset_class"] = default

    blank = out["asset_class"].isin(("", "NAN", "UNKNOWN"))
    if blank.any():
        out.loc[blank, "asset_class"] = normalize_market_vertical(default)
    return out


def ticker_asset_class_map(df: pd.DataFrame, default: str = DEFAULT_ASSET_CLASS) -> dict[str, str]:
    if df.empty or "ticker" not in df.columns:
        return {}
    scoped = attach_asset_class(df, default=default)
    subset = scoped.dropna(subset=["ticker"]).copy()
    mapping = (
        subset.groupby(subset["ticker"].astype(str))["asset_class"]
        .agg(lambda values: values.mode().iat[0] if not values.mode().empty else default)
    )
    return mapping.to_dict()


def taxonomy_row(
    asset_class: str,
    taxonomy: dict[str, dict[str, Any]],
    *,
    local_rows: int = 0,
    local_assets: int = 0,
) -> dict[str, Any]:
    asset_class = normalize_market_vertical(asset_class)
    meta = taxonomy.get(asset_class, {})
    lane = LANE_METADATA.get(asset_class, {})
    return {
        "asset_class": asset_class,
        "description": meta.get("description", ""),
        "region": meta.get("region", ""),
        "instrument_family": meta.get("instrument_family", ""),
        "default_currency": meta.get("default_currency", ""),
        "backtest_route": meta.get("backtest_route", ""),
        "t_settlement": meta.get("t_settlement", ""),
        "price_limit": bool(meta.get("price_limit", False)),
        "vectorizable": bool(meta.get("vectorizable", False)),
        "label": lane.get("label", asset_class),
        "label_zh": lane.get("label_zh", lane.get("label", asset_class)),
        "role": lane.get("role", "Configured taxonomy lane"),
        "data_mode": lane.get("data_mode", "No lane metadata configured"),
        "provider": lane.get("provider", "TBD"),
        "broker": lane.get("broker", "TBD"),
        "execution": lane.get("execution", "TBD"),
        "dashboard_scope": lane.get("dashboard_scope", "TBD"),
        "lane_status": lane.get("status", "configured"),
        "local_rows": int(local_rows),
        "local_assets": int(local_assets),
        "has_local_regime_data": bool(local_rows and local_assets),
    }


def taxonomy_frame(
    *,
    asset_classes: list[str] | tuple[str, ...] | None = None,
    taxonomy: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return dashboard-friendly taxonomy rows for the requested lanes."""

    taxonomy = taxonomy or load_asset_taxonomy()
    classes = list(asset_classes or CORE_DASHBOARD_ASSET_CLASSES)
    rows = [taxonomy_row(asset_class, taxonomy) for asset_class in classes]
    return pd.DataFrame(rows)
