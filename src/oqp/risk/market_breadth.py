"""Directional, concentration, and volatility views of market breadth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from oqp.data.instruments import InstrumentMaster
from oqp.risk.factor_breadth import extract_base_symbol


__all__ = [
    "MarketBreadthConfig",
    "build_research_window_table",
    "compute_concentration_breadth",
    "compute_directional_breadth",
    "compute_market_structure",
    "compute_volatility_map",
]


@dataclass(frozen=True)
class MarketBreadthConfig:
    """Configuration shared by the non-PCA market-breadth lenses."""

    volatility_lookback: int = 252
    annualization: float = 252.0
    minimum_observations: int = 20
    concentration_weight: str = "auto"


def compute_directional_breadth(
    frame: pd.DataFrame,
    *,
    sector_map: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate advance/decline breadth from observed close-to-close returns.

    Missing assets are excluded from that date's active universe. No synthetic
    prices are introduced into this participation measure.
    """

    returns = _return_observations(frame, sector_map=sector_map)
    if returns.empty:
        empty = pd.DataFrame(
            columns=[
                "date",
                "active_assets",
                "advancers",
                "decliners",
                "unchanged",
                "directional_breadth",
                "advance_ratio",
                "decline_ratio",
                "equal_weight_return",
                "return_dispersion",
            ]
        )
        return {"daily": empty, "by_sector": empty.assign(sector=pd.Series(dtype=str))}

    return {
        "daily": _aggregate_directional(returns, ["date"]),
        "by_sector": _aggregate_directional(returns, ["date", "sector"]),
    }


def compute_concentration_breadth(
    frame: pd.DataFrame,
    *,
    asset_class: str = "",
    sector_map: dict[str, str] | None = None,
    weight_mode: str = "auto",
) -> dict[str, Any]:
    """Calculate HHI and effective asset counts using an explicit weight source."""

    work = _normalize_market_frame(frame, sector_map=sector_map)
    if work.empty:
        return {
            "daily": pd.DataFrame(),
            "sector_weights": pd.DataFrame(),
            "latest_assets": pd.DataFrame(),
            "weight_source": "unavailable",
        }

    weight_source, values = _concentration_weight_values(
        work,
        asset_class=asset_class,
        weight_mode=weight_mode,
    )
    work = work.assign(weight_value=pd.to_numeric(values, errors="coerce"))
    work = work[np.isfinite(work["weight_value"]) & work["weight_value"].gt(0)].copy()
    if work.empty:
        return {
            "daily": pd.DataFrame(),
            "sector_weights": pd.DataFrame(),
            "latest_assets": pd.DataFrame(),
            "weight_source": weight_source,
        }

    work["weight"] = work["weight_value"] / work.groupby("date")["weight_value"].transform("sum")
    work = work[np.isfinite(work["weight"]) & work["weight"].gt(0)].copy()
    work["weight_sq"] = work["weight"].pow(2)

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("date", sort=True):
        weights = group["weight"].sort_values(ascending=False).to_numpy(dtype=float)
        hhi = float(np.square(weights).sum())
        sector_weights = group.groupby("sector")["weight"].sum().to_numpy(dtype=float)
        sector_hhi = float(np.square(sector_weights).sum())
        rows.append(
            {
                "date": pd.Timestamp(date),
                "assets": int(len(weights)),
                "hhi": hhi,
                "effective_assets": float(1.0 / hhi) if hhi > 0 else np.nan,
                "top_5_share": float(weights[:5].sum()),
                "top_10_share": float(weights[:10].sum()),
                "largest_weight": float(weights[0]),
                "effective_sectors": float(1.0 / sector_hhi) if sector_hhi > 0 else np.nan,
            }
        )
    daily = pd.DataFrame(rows)

    sector_weights = (
        work.groupby(["date", "sector"], as_index=False)
        .agg(weight=("weight", "sum"), assets=("ticker", "nunique"))
        .sort_values(["date", "weight"], ascending=[True, False])
    )
    latest_date = work["date"].max()
    latest_assets = (
        work[work["date"].eq(latest_date)][
            ["ticker", "name", "sector", "weight", "weight_value"]
        ]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "daily": daily,
        "sector_weights": sector_weights,
        "latest_assets": latest_assets,
        "weight_source": weight_source,
    }


def compute_volatility_map(
    frame: pd.DataFrame,
    *,
    sector_map: dict[str, str] | None = None,
    lookback: int = 252,
    annualization: float = 252.0,
    minimum_observations: int = 20,
) -> dict[str, pd.DataFrame]:
    """Summarize realized volatility by asset, sector, and date."""

    returns = _return_observations(frame, sector_map=sector_map)
    if returns.empty:
        return {
            "by_asset": pd.DataFrame(),
            "by_sector": pd.DataFrame(),
            "sector_timeline": pd.DataFrame(),
            "market_timeline": pd.DataFrame(),
        }

    lookback = max(2, int(lookback))
    minimum_observations = max(2, min(int(minimum_observations), lookback))
    annualization_scale = float(np.sqrt(annualization))
    recent = returns.groupby("ticker", group_keys=False).tail(lookback).copy()

    asset_rows: list[dict[str, Any]] = []
    for ticker, group in recent.groupby("ticker", sort=False):
        values = pd.to_numeric(group["log_return"], errors="coerce").dropna()
        if len(values) < minimum_observations:
            continue
        latest = group.sort_values("date").iloc[-1]
        asset_rows.append(
            {
                "ticker": ticker,
                "name": latest.get("name", ""),
                "sector": latest.get("sector", "Unknown"),
                "observations": int(len(values)),
                "annualized_vol": float(values.std(ddof=1) * annualization_scale),
                "latest_return": float(values.iloc[-1]),
                "last_date": pd.Timestamp(latest["date"]),
            }
        )
    by_asset = pd.DataFrame(asset_rows)
    if by_asset.empty:
        by_sector = pd.DataFrame()
    else:
        by_asset["vol_percentile"] = by_asset["annualized_vol"].rank(pct=True)
        high_cutoff = float(by_asset["annualized_vol"].quantile(0.75))
        by_asset["high_vol"] = by_asset["annualized_vol"].ge(high_cutoff)
        by_sector = (
            by_asset.groupby("sector", as_index=False)
            .agg(
                assets=("ticker", "nunique"),
                median_vol=("annualized_vol", "median"),
                mean_vol=("annualized_vol", "mean"),
                high_vol_share=("high_vol", "mean"),
            )
            .sort_values("median_vol", ascending=False)
        )

    timeline = returns.copy()
    timeline["annualized_vol"] = timeline.groupby("ticker")["log_return"].transform(
        lambda values: values.rolling(
            window=lookback,
            min_periods=minimum_observations,
        ).std(ddof=1)
        * annualization_scale
    )
    timeline = timeline.dropna(subset=["annualized_vol"])
    sector_timeline = (
        timeline.groupby(["date", "sector"], as_index=False)
        .agg(median_vol=("annualized_vol", "median"), assets=("ticker", "nunique"))
        .sort_values(["date", "sector"])
    )
    market_timeline = (
        timeline.groupby("date", as_index=False)
        .agg(
            median_vol=("annualized_vol", "median"),
            p75_vol=("annualized_vol", lambda values: values.quantile(0.75)),
            assets=("ticker", "nunique"),
        )
        .sort_values("date")
    )
    return {
        "by_asset": by_asset.sort_values("annualized_vol", ascending=False),
        "by_sector": by_sector,
        "sector_timeline": sector_timeline,
        "market_timeline": market_timeline,
    }


def compute_market_structure(
    frame: pd.DataFrame,
    *,
    asset_class: str = "",
    sector_map: dict[str, str] | None = None,
    config: MarketBreadthConfig | None = None,
) -> dict[str, Any]:
    """Compute all non-PCA market-structure lenses from one normalized panel."""

    cfg = config or MarketBreadthConfig()
    if str(asset_class).strip().upper() == "FUTURES_CN":
        sector_map = InstrumentMaster("FUTURES_CN").get_sector_map()
    directional = compute_directional_breadth(frame, sector_map=sector_map)
    concentration = compute_concentration_breadth(
        frame,
        asset_class=asset_class,
        sector_map=sector_map,
        weight_mode=cfg.concentration_weight,
    )
    volatility = compute_volatility_map(
        frame,
        sector_map=sector_map,
        lookback=cfg.volatility_lookback,
        annualization=cfg.annualization,
        minimum_observations=cfg.minimum_observations,
    )
    return {
        "directional": directional,
        "concentration": concentration,
        "volatility": volatility,
    }


def build_research_window_table(
    directional: pd.DataFrame,
    concentration: pd.DataFrame,
    market_volatility: pd.DataFrame,
    risk_breadth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build monthly market-state rows for train/validation window selection."""

    pieces: list[pd.DataFrame] = []
    if not directional.empty:
        direction = directional.set_index("date").resample("ME").agg(
            directional_breadth=("directional_breadth", "mean"),
            positive_day_share=("directional_breadth", lambda values: values.gt(0).mean()),
        )
        pieces.append(direction)
    if not concentration.empty:
        conc = concentration.set_index("date").resample("ME").agg(
            effective_assets=("effective_assets", "mean"),
            top_5_share=("top_5_share", "mean"),
        )
        pieces.append(conc)
    if not market_volatility.empty:
        vol = market_volatility.set_index("date").resample("ME").agg(
            median_vol=("median_vol", "mean"),
        )
        pieces.append(vol)
    if not pieces:
        return pd.DataFrame()

    windows = pd.concat(pieces, axis=1).dropna(how="all").reset_index()
    windows["direction_state"] = windows.get(
        "directional_breadth", pd.Series(np.nan, index=windows.index)
    ).map(_direction_state)
    windows["concentration_state"] = _quantile_state(
        windows.get("effective_assets"),
        low_label="High concentration",
        middle_label="Normal concentration",
        high_label="Low concentration",
    )
    windows["volatility_state"] = _quantile_state(
        windows.get("median_vol"),
        low_label="Low volatility",
        middle_label="Normal volatility",
        high_label="High volatility",
    )

    windows["risk_state"] = "Unavailable"
    if risk_breadth is not None and not risk_breadth.empty:
        risk = risk_breadth.copy()
        risk["date"] = pd.to_datetime(risk["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        risk = risk.dropna(subset=["date"]).sort_values("date").groupby("date", as_index=False).tail(1)
        risk_col = "breadth_regime" if "breadth_regime" in risk.columns else None
        if risk_col:
            windows = windows.merge(risk[["date", risk_col]], on="date", how="left")
            windows["risk_state"] = windows[risk_col].fillna("Unavailable")
            windows = windows.drop(columns=[risk_col])

    guidance = windows.apply(_window_guidance, axis=1, result_type="expand")
    guidance.columns = ["research_use_en", "research_use_zh"]
    return pd.concat([windows, guidance], axis=1)


def _normalize_market_frame(
    frame: pd.DataFrame,
    *,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Market breadth input missing required columns: {missing}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ticker"] = work["ticker"].astype(str).str.strip()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna(subset=["date", "ticker", "close"])
    work = work[work["ticker"].ne("") & work["close"].gt(0)]
    work = work.sort_values(["ticker", "date"]).groupby(["date", "ticker"], as_index=False).last()
    if "sector" not in work.columns:
        work["sector"] = work["ticker"].map(
            lambda ticker: _lookup_sector(ticker, sector_map or {})
        )
    else:
        work["sector"] = work["sector"].fillna("").astype(str).str.strip()
        if sector_map:
            missing_sector = work["sector"].isin(["", "nan", "None", "Unknown"])
            work.loc[missing_sector, "sector"] = work.loc[missing_sector, "ticker"].map(
                lambda ticker: _lookup_sector(ticker, sector_map)
            )
        work["sector"] = work["sector"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"}).fillna("Unknown")
    if "name" not in work.columns:
        work["name"] = ""
    else:
        work["name"] = work["name"].fillna("").astype(str)
    return work.sort_values(["ticker", "date"]).reset_index(drop=True)


def _lookup_sector(ticker: str, sector_map: dict[str, str]) -> str:
    text = str(ticker).strip()
    symbol = extract_base_symbol(text)
    return str(
        sector_map.get(
            text,
            sector_map.get(
                symbol,
                sector_map.get(symbol.lower(), sector_map.get(symbol.upper(), "Unknown")),
            ),
        )
    )


def _return_observations(
    frame: pd.DataFrame,
    *,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    work = _normalize_market_frame(frame, sector_map=sector_map)
    if work.empty:
        return work.assign(log_return=pd.Series(dtype=float))
    work["log_return"] = work.groupby("ticker")["close"].transform(
        lambda values: np.log(values).diff()
    )
    return work.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return"])


def _aggregate_directional(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = frame.copy()
    work["is_up"] = work["log_return"].gt(0).astype(int)
    work["is_down"] = work["log_return"].lt(0).astype(int)
    work["is_flat"] = work["log_return"].eq(0).astype(int)
    grouped = (
        work.groupby(group_cols, as_index=False)
        .agg(
            active_assets=("ticker", "nunique"),
            advancers=("is_up", "sum"),
            decliners=("is_down", "sum"),
            unchanged=("is_flat", "sum"),
            equal_weight_return=("log_return", "mean"),
            return_dispersion=("log_return", "std"),
        )
        .sort_values(group_cols)
    )
    denominator = grouped["active_assets"].replace(0, np.nan)
    grouped["directional_breadth"] = (grouped["advancers"] - grouped["decliners"]) / denominator
    grouped["advance_ratio"] = grouped["advancers"] / denominator
    grouped["decline_ratio"] = grouped["decliners"] / denominator
    return grouped


def _concentration_weight_values(
    frame: pd.DataFrame,
    *,
    asset_class: str,
    weight_mode: str,
) -> tuple[str, pd.Series]:
    mode = str(weight_mode).strip().lower()
    asset_class = str(asset_class).strip().upper()

    def usable(column: str) -> bool:
        if column not in frame.columns:
            return False
        values = pd.to_numeric(frame[column], errors="coerce")
        return bool(values.gt(0).mean() >= 0.50)

    if mode in {"auto", "market_cap"} and usable("market_cap"):
        return "market_cap", pd.to_numeric(frame["market_cap"], errors="coerce")
    if mode in {"auto", "open_interest"} and asset_class.startswith("FUTURES") and usable("open_interest"):
        return "open_interest_notional_proxy", (
            pd.to_numeric(frame["open_interest"], errors="coerce") * frame["close"]
        )
    if mode in {"auto", "turnover"} and usable("turnover"):
        return "traded_value", pd.to_numeric(frame["turnover"], errors="coerce")
    if mode in {"auto", "volume"} and usable("volume"):
        return "price_x_volume_proxy", (
            pd.to_numeric(frame["volume"], errors="coerce") * frame["close"]
        )
    return "equal_weight_fallback", pd.Series(1.0, index=frame.index)


def _direction_state(value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    if value >= 0.20:
        return "Broad advance"
    if value <= -0.20:
        return "Broad decline"
    return "Mixed"


def _quantile_state(
    values: pd.Series | None,
    *,
    low_label: str,
    middle_label: str,
    high_label: str,
) -> pd.Series:
    if values is None:
        return pd.Series(dtype=str)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series("Unavailable", index=numeric.index)
    low, high = valid.quantile([1 / 3, 2 / 3]).tolist()
    if np.isclose(low, high, equal_nan=False):
        return pd.Series(middle_label, index=numeric.index).where(
            numeric.notna(), "Unavailable"
        )
    return pd.Series(
        np.where(numeric.le(low), low_label, np.where(numeric.ge(high), high_label, middle_label)),
        index=numeric.index,
    ).where(numeric.notna(), "Unavailable")


def _window_guidance(row: pd.Series) -> tuple[str, str]:
    risk = str(row.get("risk_state", ""))
    concentration = str(row.get("concentration_state", ""))
    volatility = str(row.get("volatility_state", ""))
    direction = str(row.get("direction_state", ""))
    if risk == "Low" or concentration == "High concentration":
        return (
            "Structural stress window: reserve for stress or out-of-sample validation; do not use alone for model selection.",
            "结构性压力窗口：适合作为压力或样本外验证，不应单独用于模型筛选。",
        )
    if volatility == "High volatility":
        return (
            "High-volatility validation window: test execution costs, sizing, and drawdown controls.",
            "高波动验证窗口：重点检验交易成本、仓位规模和回撤控制。",
        )
    if direction in {"Broad advance", "Broad decline"}:
        return (
            "Broad directional window: test whether performance is dependent on market beta.",
            "广泛单边窗口：检验策略表现是否依赖市场 Beta。",
        )
    return (
        "Balanced window: suitable as part of a core train or validation segment.",
        "均衡窗口：适合作为核心训练或验证区间的一部分。",
    )
