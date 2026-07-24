"""Shared market-data preparation for factor-portfolio CLI and dashboard runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oqp.config import REPO_ROOT
from oqp.data import DataEngineFactory
from oqp.data.instruments import InstrumentMaster
from oqp.research.api_datasets import load_materialized_api_dataset
from oqp.research.artifacts import slugify
from oqp.research.backtesting.return_horizons import attach_return_horizon
from oqp.research.datasets import (
    attach_dataset_tradability_attrs,
    infer_dataset_tradability,
)
from oqp.research.dataset_fingerprints import (
    DEFAULT_DATASET_MANIFEST_ROOT,
    attach_dataset_manifest_attrs,
    register_dataset_manifest,
)

_PLACEHOLDER_SECTOR_LABELS = frozenset(
    {
        "",
        "macro",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
        "unclassified",
        "unknown",
        "unspecified",
    }
)


@dataclass(frozen=True)
class FactorPortfolioDataBundle:
    frame: pd.DataFrame
    crisis_period: object
    source_path: Path
    data_frequency: str


def load_factor_portfolio_data(
    data_file: str | Path,
    *,
    market_vertical: str,
    return_horizon: str = "auto",
    start_date: str | None = None,
    end_date: str | None = None,
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    adjustment_method: str = "unknown",
    dataset_manifest_root: str | Path = DEFAULT_DATASET_MANIFEST_ROOT,
    workspace_root: str | Path = REPO_ROOT,
) -> FactorPortfolioDataBundle:
    source = Path(data_file).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"data file not found: {source}")
    is_api_materialization = (
        source.name == "materialization.json"
        or (source.is_dir() and (source / "materialization.json").exists())
    )
    if is_api_materialization:
        frame = load_materialized_api_dataset(
            source,
            require_historical_backtest_eligible=True,
            workspace_root=workspace_root,
        )
        lineage_attrs = dict(frame.attrs)
        feed_path = Path(str(frame.attrs["source_path"]))
        feed = DataEngineFactory.create_feed(
            market_vertical.split("_")[0],
            str(feed_path),
        )
        frame = normalize_market_frame(frame)
        frame.attrs.update(lineage_attrs)
    else:
        if source.suffix.lower() in {".csv", ".txt"}:
            feed = None
            frame = normalize_market_frame(pd.read_csv(source))
        else:
            feed = DataEngineFactory.create_feed(
                market_vertical.split("_")[0], str(source)
            )
            frame = normalize_market_frame(feed.load_data())
        manifest, manifest_path = register_dataset_manifest(
            source,
            dataset_id=dataset_id or slugify(source.stem, fallback="dataset"),
            dataset_version=dataset_version,
            market_vertical=market_vertical,
            data_frequency=infer_market_frequency(frame),
            adjustment_method=adjustment_method,
            frame=frame,
            manifest_root=dataset_manifest_root,
            workspace_root=workspace_root,
        )
        frame = attach_dataset_manifest_attrs(
            frame,
            manifest,
            manifest_path,
            verified=True,
            workspace_root=workspace_root,
        )
    frame = attach_instrument_classification(frame, market_vertical)
    if infer_market_frequency(frame) == "daily":
        frame = normalize_daily_session_rows(frame)
    frame = filter_market_dates(frame, start_date, end_date)
    data_frequency = infer_market_frequency(frame)
    frame = attach_return_horizon(
        frame,
        return_horizon=return_horizon,
        data_frequency=data_frequency,
    )
    data_profile = infer_dataset_tradability(
        frame,
        source_path=str(frame.attrs.get("source_path") or source),
        asset_class=market_vertical,
        data_frequency=data_frequency,
    )
    return_horizon_value = frame.attrs.get("return_horizon")
    return_horizon_description = frame.attrs.get("return_horizon_description")
    execution_assumption = frame.attrs.get("execution_assumption")
    benchmark_return_col = frame.attrs.get("benchmark_return_col")
    frame.attrs.update(
        {
            "market_vertical": market_vertical,
            "source_path": str(frame.attrs.get("source_path") or source),
            "data_file": str(frame.attrs.get("data_file") or source),
            "data_vendor": str(frame.attrs.get("data_vendor") or "local_file"),
            "data_frequency": data_frequency,
            "backtest_start": frame["date"].min().isoformat(),
            "backtest_end": frame["date"].max().isoformat(),
            "backtest_rows": int(len(frame)),
            "return_horizon": return_horizon_value,
            "return_horizon_description": return_horizon_description,
            "execution_assumption": execution_assumption,
            "benchmark_return_col": benchmark_return_col,
        }
    )
    frame = attach_dataset_tradability_attrs(frame, data_profile)
    return FactorPortfolioDataBundle(
        frame=frame,
        crisis_period=feed.get_crisis_period() if feed is not None else None,
        source_path=source,
        data_frequency=data_frequency,
    )


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ticker" not in out.columns and "symbol" in out.columns:
        out["ticker"] = out["symbol"].astype(str)
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    if "close" not in out.columns and "last_price" in out.columns:
        out["close"] = pd.to_numeric(out["last_price"], errors="coerce")
    open_interest_source = next(
        (
            column
            for column in ("open_interest", "open_oi", "oi")
            if column in out.columns
        ),
        None,
    )
    if open_interest_source is not None:
        open_interest = pd.to_numeric(
            out[open_interest_source],
            errors="coerce",
        )
        for alias in ("open_interest", "open_oi", "oi"):
            if alias not in out.columns:
                out[alias] = open_interest
    required = {"date", "ticker", "close"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"data file is missing required columns: {', '.join(missing)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "close"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def normalize_daily_session_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate intraday-stamped snapshots into one daily bar.

    Some daily feature snapshots append a zero-volume end-of-day mark beside
    the completed OHLCV bar.  Daily sleeves key on session date, so retaining
    both would create duplicate date/ticker rows.  Prefer the row carrying
    actual volume and a non-degenerate range, then the most complete row.
    """

    out = frame.copy()
    attrs = dict(frame.attrs)
    out["_daily_source_timestamp"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    )
    out["_daily_session"] = out["_daily_source_timestamp"].dt.normalize()
    out["_daily_row_order"] = range(len(out))
    volume_col = next(
        (column for column in ("volume", "vol") if column in out.columns),
        None,
    )
    if volume_col is None:
        out["_daily_positive_volume"] = 0
    else:
        out["_daily_positive_volume"] = (
            pd.to_numeric(out[volume_col], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .astype(int)
        )
    if {"high", "low"}.issubset(out.columns):
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        out["_daily_positive_range"] = high.sub(low).gt(0.0).astype(int)
    else:
        out["_daily_positive_range"] = 0
    source_columns = [
        column
        for column in frame.columns
        if column not in {"date", "ticker"}
    ]
    out["_daily_observed_fields"] = (
        out[source_columns].notna().sum(axis=1)
        if source_columns
        else 0
    )
    duplicate_mask = out.duplicated(
        ["_daily_session", "ticker"],
        keep=False,
    )
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_groups = int(
        out.loc[duplicate_mask, ["_daily_session", "ticker"]]
        .drop_duplicates()
        .shape[0]
    )
    out = (
        out.sort_values(
            [
                "_daily_session",
                "ticker",
                "_daily_positive_volume",
                "_daily_positive_range",
                "_daily_observed_fields",
                "_daily_source_timestamp",
                "_daily_row_order",
            ],
            ascending=[True, True, False, False, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(["_daily_session", "ticker"], keep="first")
        .sort_values(["ticker", "_daily_session"], kind="mergesort")
        .reset_index(drop=True)
    )
    out["date"] = out["_daily_session"]
    out = out.drop(
        columns=[
            "_daily_source_timestamp",
            "_daily_session",
            "_daily_row_order",
            "_daily_positive_volume",
            "_daily_positive_range",
            "_daily_observed_fields",
        ]
    )
    out.attrs.update(attrs)
    out.attrs["daily_session_normalization"] = {
        "schema_version": 1,
        "selection_rule": (
            "positive_volume_then_positive_range_then_field_completeness_"
            "then_earliest_timestamp"
        ),
        "duplicate_source_rows": duplicate_rows,
        "collapsed_session_ticker_groups": duplicate_groups,
    }
    return out


def attach_instrument_classification(
    frame: pd.DataFrame,
    market_vertical: str,
) -> pd.DataFrame:
    """Replace an absent or wholly-placeholder sector with canonical taxonomy.

    A dataset that contains at least one genuine sector label remains
    authoritative and is returned unchanged.  InstrumentMaster enrichment is
    therefore limited to datasets whose sector column is absent or consists
    entirely of generic placeholders such as ``Macro`` or ``Unknown``.

    Instrument classification is static contract metadata keyed only by the
    ticker/root symbol.  It does not use future prices or observations.
    """

    out = frame.copy()
    attrs = dict(frame.attrs)
    sector_was_present = "sector" in out.columns
    if sector_was_present:
        normalized_sector = (
            out["sector"].astype("string").fillna("").str.strip().str.casefold()
        )
        observed_labels = set(normalized_sector.unique())
        if not observed_labels.issubset(_PLACEHOLDER_SECTOR_LABELS):
            out.attrs.update(attrs)
            out.attrs["instrument_classification"] = {
                "schema_version": 1,
                "source": "dataset_sector",
                "market_vertical": str(market_vertical),
                "enriched": False,
                "reason": "genuine_dataset_sector_preserved",
                "time_semantics": "dataset_provided",
            }
            return out

    input_state = "absent" if not sector_was_present else "placeholder_only"
    master = InstrumentMaster(market_vertical)
    unique_tickers = out["ticker"].dropna().astype(str).unique()
    sector_map: dict[str, str] = {}
    unresolved_tickers: list[str] = []
    for ticker in unique_tickers:
        profile = master.get_profile(ticker)
        exchange = str(profile.exchange or "").strip().casefold()
        sector = str(profile.sector or "").strip()
        if (
            not sector
            or sector.casefold() in _PLACEHOLDER_SECTOR_LABELS
            or exchange in {"", "unknown"}
        ):
            sector_map[ticker] = "Unknown"
            unresolved_tickers.append(ticker)
        else:
            sector_map[ticker] = sector

    ticker_series = out["ticker"].astype(str)
    out["sector"] = ticker_series.map(sector_map).fillna("Unknown")
    unresolved_mask = out["sector"].eq("Unknown")
    out.attrs.update(attrs)
    out.attrs["instrument_classification"] = {
        "schema_version": 1,
        "source": "instrument_master",
        "market_vertical": str(market_vertical),
        "enriched": True,
        "reason": f"{input_state}_sector_enriched",
        "time_semantics": "static_instrument_taxonomy_no_market_observations",
        "resolved_tickers": int(len(unique_tickers) - len(unresolved_tickers)),
        "unresolved_tickers": sorted(unresolved_tickers),
        "resolved_rows": int((~unresolved_mask).sum()),
        "unresolved_rows": int(unresolved_mask.sum()),
    }
    return out


def filter_market_dates(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return frame
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = dates.notna()
    if start_date:
        mask &= dates.ge(pd.Timestamp(start_date))
    if end_date:
        end = pd.Timestamp(end_date)
        if len(str(end_date)) <= 10:
            end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= dates.le(end)
    out = frame.loc[mask].copy()
    if out.empty:
        raise ValueError("date filter removed every row")
    return out.reset_index(drop=True)


def infer_market_frequency(frame: pd.DataFrame) -> str:
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values()
    if dates.empty:
        return "daily"
    unique = dates.drop_duplicates()
    if len(unique) < 2:
        return "daily"
    median_seconds = unique.diff().dropna().dt.total_seconds().median()
    return "intraday" if median_seconds < 20 * 60 * 60 else "daily"


def load_router_state_data(path: str | Path) -> pd.DataFrame:
    """Load a reproducible router-state table without interpreting its schema."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"router state file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source)
    raise ValueError("router state file must be CSV or parquet")


__all__ = [
    "FactorPortfolioDataBundle",
    "attach_instrument_classification",
    "filter_market_dates",
    "infer_market_frequency",
    "load_factor_portfolio_data",
    "load_router_state_data",
    "normalize_daily_session_rows",
    "normalize_market_frame",
]
