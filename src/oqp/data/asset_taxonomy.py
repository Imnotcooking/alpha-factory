from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical


DEFAULT_ASSET_CLASS = "FUTURES_CN"

FALLBACK_ASSET_TAXONOMY: dict[str, dict[str, Any]] = {
    key: dict(value) for key, value in ASSET_TAXONOMY.items()
}

LANE_METADATA: dict[str, dict[str, str]] = {
    "FUTURES_CN": {
        "role": "Current local/static research dataset",
        "data_mode": "Local parquet matrices + tick cache",
        "provider": "Bundled/static files",
    },
    "EQUITY_US": {
        "role": "Next-phase US equities lane",
        "data_mode": "API-backed; no public data bundled",
        "provider": "FMP",
    },
    "OPTIONS_US": {
        "role": "Next-phase US options lane",
        "data_mode": "API-backed option chains; event-driven/non-vectorized",
        "provider": "Massive",
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
        "t_settlement": meta.get("t_settlement", ""),
        "price_limit": bool(meta.get("price_limit", False)),
        "vectorizable": bool(meta.get("vectorizable", False)),
        "role": lane.get("role", "Configured taxonomy lane"),
        "data_mode": lane.get("data_mode", "No lane metadata configured"),
        "provider": lane.get("provider", "TBD"),
        "local_rows": int(local_rows),
        "local_assets": int(local_assets),
        "has_local_regime_data": bool(local_rows and local_assets),
    }
