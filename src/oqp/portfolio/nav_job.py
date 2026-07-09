"""Scheduled portfolio NAV update job.

This module keeps the server-side NAV refresh out of Streamlit. It loads the
latest live-position snapshot, fetches market history, values the portfolio,
and writes one daily NAV row back into the shared ledger.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.config.paths import REPO_ROOT
from oqp.portfolio.ledger import (
    default_portfolio_ledger_path,
    load_latest_live_positions,
    write_historical_nav,
)
from oqp.portfolio.symbols import to_yahoo_ticker
from oqp.portfolio.valuation import ManualPortfolioInputs, value_portfolio_snapshot


DEFAULT_PORTFOLIO_STATE_DIR = REPO_ROOT / "runtime" / "state" / "portfolio"
DEFAULT_DEFAULTS_PATH = DEFAULT_PORTFOLIO_STATE_DIR / "manual_inputs.json"
DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH = DEFAULT_DEFAULTS_PATH
DEFAULT_IBKR_METRICS_PATH = DEFAULT_PORTFOLIO_STATE_DIR / "ibkr_metrics.json"

DEFAULT_FX_TICKERS = ("EURUSD=X", "GBPUSD=X", "CNYUSD=X", "HKDUSD=X")
DEFAULT_MACRO_TICKERS = (
    "QQQ",
    "SPY",
    "TLT",
    "^HSI",
    "000300.SS",
    "CBON",
    "VGK",
    "EWJ",
    "EWY",
    "GC=F",
    "CL=F",
    "BTC-USD",
)

MarketDataProvider = Callable[[Sequence[str], str], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class PortfolioNavJobSettings:
    manual_inputs: ManualPortfolioInputs
    asset_preferences: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PortfolioNavUpdateResult:
    status: str
    db_path: Path
    snapshot_date: str
    position_rows: int
    total_net_worth: float = 0.0
    total_pnl: float = 0.0
    total_cash: float = 0.0
    portfolio_beta: float = 0.0
    tickers_requested: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "db_path": str(self.db_path),
            "snapshot_date": self.snapshot_date,
            "position_rows": self.position_rows,
            "total_net_worth": self.total_net_worth,
            "total_pnl": self.total_pnl,
            "total_cash": self.total_cash,
            "portfolio_beta": self.portfolio_beta,
            "tickers_requested": list(self.tickers_requested),
            "message": self.message,
        }


def update_portfolio_nav(
    *,
    db_path: str | Path | None = None,
    snapshot_date: str | date | datetime | None = None,
    period: str = "6mo",
    benchmark: str = "QQQ",
    defaults_path: str | Path | None = None,
    ibkr_metrics_path: str | Path | None = None,
    market_data_provider: MarketDataProvider | None = None,
    dry_run: bool = False,
) -> PortfolioNavUpdateResult:
    """Refresh and optionally persist one daily portfolio NAV observation."""

    ledger_path = Path(db_path) if db_path is not None else default_portfolio_ledger_path()
    date_value = _date_text(snapshot_date or date.today())
    positions = load_latest_live_positions(ledger_path)
    settings = load_portfolio_nav_job_settings(
        defaults_path=defaults_path,
        ibkr_metrics_path=ibkr_metrics_path,
    )
    manual_only = positions.empty
    if manual_only and not _manual_inputs_have_value(settings.manual_inputs):
        return PortfolioNavUpdateResult(
            status="no_positions",
            db_path=ledger_path,
            snapshot_date=date_value,
            position_rows=0,
            message=(
                "No live_positions snapshot or manual portfolio inputs are "
                "available in the portfolio ledger."
            ),
        )

    tickers = market_tickers_for_positions(positions, extra_tickers=(benchmark,))
    provider = market_data_provider or fetch_yahoo_market_history
    market_history = provider(tickers, period)
    if market_history.empty:
        raise RuntimeError("Market data provider returned no price history.")

    valuation = value_portfolio_snapshot(
        positions,
        market_history,
        benchmark=benchmark,
        manual_inputs=settings.manual_inputs,
        asset_preferences=settings.asset_preferences,
    )

    if not dry_run:
        write_historical_nav(
            ledger_path,
            snapshot_date=date_value,
            total_net_worth=valuation.total_net_worth,
            total_cash=valuation.total_cash,
            portfolio_beta=valuation.portfolio_beta,
        )

    return PortfolioNavUpdateResult(
        status="dry_run" if dry_run else "updated",
        db_path=ledger_path,
        snapshot_date=date_value,
        position_rows=int(len(positions)),
        total_net_worth=float(valuation.total_net_worth),
        total_pnl=float(valuation.total_pnl),
        total_cash=float(valuation.total_cash),
        portfolio_beta=float(valuation.portfolio_beta),
        tickers_requested=tuple(tickers),
        message=(
            "Valued manual portfolio inputs only; no live position rows were present."
            if manual_only
            else ""
        ),
    )


def load_portfolio_nav_job_settings(
    *,
    defaults_path: str | Path | None = None,
    ibkr_metrics_path: str | Path | None = None,
) -> PortfolioNavJobSettings:
    """Load non-secret manual portfolio inputs for server-side valuation."""

    defaults = _read_json_object(
        Path(defaults_path) if defaults_path is not None else DEFAULT_DEFAULTS_PATH,
    )
    ibkr_metrics = _read_json_object(
        Path(ibkr_metrics_path)
        if ibkr_metrics_path is not None
        else DEFAULT_IBKR_METRICS_PATH,
    )

    raw_preferences = defaults.get("asset_preferences", {})
    asset_preferences = raw_preferences if isinstance(raw_preferences, dict) else {}

    return PortfolioNavJobSettings(
        manual_inputs=ManualPortfolioInputs(
            t212_cash_eur=_float(defaults.get("t212_cash_eur", 0.0)),
            futu_cash_usd=_float(defaults.get("futu_cash_usd", 0.0)),
            ibkr_cash_usd=_float(ibkr_metrics.get("Available_Cash_USD", 0.0)),
            cny_mutual_fund=_float(defaults.get("cny_mutual_fund", 0.0)),
            cny_mutual_fund_pnl=_float(defaults.get("cny_mutual_fund_pnl", 0.0)),
            cny_gold_grams=_float(defaults.get("cny_gold_grams", 0.0)),
            cny_gold_cost=_float(defaults.get("cny_gold_cost", 0.0)),
        ),
        asset_preferences=asset_preferences,
    )


def market_tickers_for_positions(
    positions: pd.DataFrame,
    *,
    extra_tickers: Iterable[str] = (),
) -> list[str]:
    """Build the Yahoo ticker list needed by the valuation engine."""

    position_tickers: list[str] = []
    if "ticker" in positions.columns:
        for value in positions["ticker"].dropna().astype(str):
            ticker = value.strip()
            if not ticker or len(ticker) >= 10:
                continue
            position_tickers.append(to_yahoo_ticker(ticker))

    all_tickers = [
        *position_tickers,
        *DEFAULT_FX_TICKERS,
        *DEFAULT_MACRO_TICKERS,
        *extra_tickers,
    ]
    return sorted({ticker for ticker in all_tickers if str(ticker).strip()})


def fetch_yahoo_market_history(tickers: Sequence[str], period: str) -> pd.DataFrame:
    """Fetch close-price history from Yahoo Finance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for the default portfolio NAV market-data provider."
        ) from exc

    data = yf.download(
        list(tickers),
        period=period,
        progress=False,
        auto_adjust=False,
    )
    close = _close_prices_from_yahoo_download(data, tickers)
    close = close.dropna(axis=1, how="all")
    return close.ffill()


def _close_prices_from_yahoo_download(
    data: pd.DataFrame | pd.Series,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        name = tickers[0] if tickers else "Close"
        return data.to_frame(name=name)

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        elif "Close" in data.columns.get_level_values(-1):
            close = data.xs("Close", axis=1, level=-1)
        else:
            close = data
    elif "Close" in data.columns:
        close = data[["Close"]].rename(columns={"Close": tickers[0] if tickers else "Close"})
    else:
        close = data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0] if tickers else "Close")
    return close


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _manual_inputs_have_value(manual: ManualPortfolioInputs) -> bool:
    return any(
        abs(value) > 0
        for value in (
            manual.t212_cash_eur,
            manual.futu_cash_usd,
            manual.ibkr_cash_usd,
            manual.cny_mutual_fund,
            manual.cny_mutual_fund_pnl,
            manual.cny_gold_grams,
        )
    )


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
