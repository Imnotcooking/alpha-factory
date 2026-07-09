"""Unified live portfolio ingestion job."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.accounts import (
    account_snapshot_from_live_positions_frame,
    write_account_snapshot,
)
from oqp.brokers import (
    fetch_ibkr_readonly_portfolio_snapshot,
    get_broker_adapter,
    get_broker_profile_config,
)
from oqp.config import REPO_ROOT, load_settings
from oqp.data import PolygonOptionsSnapshotAdapter
from oqp.portfolio.broker_imports import (
    futubull_option_to_occ,
    parse_futubull_csv,
    parse_trading212_csv,
)
from oqp.portfolio.ledger import default_portfolio_ledger_path, write_live_positions_frame


DEFAULT_BROKER_EXPORTS_DIR = REPO_ROOT / "runtime" / "imports" / "broker_exports"
DEFAULT_PORTFOLIO_STATE_DIR = REPO_ROOT / "runtime" / "state" / "portfolio"
DEFAULT_PORTFOLIO_EXPORTS_DIR = REPO_ROOT / "runtime" / "exports" / "portfolio_snapshots"
DEFAULT_IBKR_METRICS_PATH = DEFAULT_PORTFOLIO_STATE_DIR / "ibkr_metrics.json"
DEFAULT_BANKED_PROFITS_PATH = DEFAULT_PORTFOLIO_STATE_DIR / "banked_profits.json"


@dataclass(frozen=True, slots=True)
class PortfolioIngestionResult:
    status: str
    db_path: Path
    snapshot_date: str
    raw_dir: Path
    source_raw_dir: Path | None
    position_rows: int
    ibkr_position_rows: int
    futubull_position_rows: int
    trading212_position_rows: int
    ibkr_metrics_path: Path | None = None
    banked_profits_path: Path | None = None
    backup_csv_path: Path | None = None
    account_ledger_path: Path | None = None
    account_snapshot_id: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "db_path": str(self.db_path),
            "snapshot_date": self.snapshot_date,
            "raw_dir": str(self.raw_dir),
            "source_raw_dir": None if self.source_raw_dir is None else str(self.source_raw_dir),
            "position_rows": self.position_rows,
            "ibkr_position_rows": self.ibkr_position_rows,
            "futubull_position_rows": self.futubull_position_rows,
            "trading212_position_rows": self.trading212_position_rows,
            "ibkr_metrics_path": None
            if self.ibkr_metrics_path is None
            else str(self.ibkr_metrics_path),
            "banked_profits_path": None
            if self.banked_profits_path is None
            else str(self.banked_profits_path),
            "backup_csv_path": None if self.backup_csv_path is None else str(self.backup_csv_path),
            "account_ledger_path": None
            if self.account_ledger_path is None
            else str(self.account_ledger_path),
            "account_snapshot_id": self.account_snapshot_id,
            "message": self.message,
        }


def futu_to_occ(futu_ticker: Any) -> tuple[str | None, str | None]:
    return futubull_option_to_occ(futu_ticker)


def fetch_polygon_greeks(occ_ticker: str, underlying: str) -> tuple[float, float]:
    adapter = _build_polygon_options_adapter()
    if adapter is None or not adapter.healthcheck().ok:
        return 1.0, 0.0

    try:
        return adapter.get_option_greeks(underlying, occ_ticker)
    except Exception:
        return 1.0, 0.0


def process_futu_csv(file_path: str | Path) -> pd.DataFrame:
    return parse_futubull_csv(file_path, greeks_provider=fetch_polygon_greeks)


def process_t212_csv(file_path: str | Path) -> tuple[pd.DataFrame, float]:
    result = parse_trading212_csv(file_path)
    return result.positions, result.banked_profit


def fetch_live_ibkr_portfolio(
    *,
    profile: str = "ibkr_live_readonly",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fetch IBKR live positions through the shared read-only broker adapter."""

    try:
        settings = load_settings()
        broker_config = get_broker_profile_config(profile, settings=settings)
        broker = get_broker_adapter("ibkr", settings=settings)
        snapshot = fetch_ibkr_readonly_portfolio_snapshot(
            broker_config,
            adapter=broker,
        )
    except Exception as exc:
        print(f"IBKR connection failed: {exc}")
        return pd.DataFrame(), {}

    if snapshot.error:
        print(f"IBKR connection failed: {snapshot.error}")
        return pd.DataFrame(), {}

    return pd.DataFrame(list(snapshot.position_rows)), dict(snapshot.metrics)


def save_ibkr_metrics(
    metrics: dict[str, Any],
    *,
    metrics_path: str | Path | None = None,
    clean_dir: str | Path | None = None,
) -> Path | None:
    """Persist IBKR account metrics even when there are no open positions."""

    if not metrics:
        return None

    path = (
        Path(metrics_path)
        if metrics_path is not None
        else Path(clean_dir) / "ibkr_metrics.json"
        if clean_dir is not None
        else DEFAULT_IBKR_METRICS_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_portfolio_ingestion(
    *,
    db_path: str | Path | None = None,
    snapshot_date: str | date | datetime | None = None,
    raw_dir: str | Path | None = None,
    state_dir: str | Path | None = None,
    backup_csv_dir: str | Path | None = None,
    account_ledger_path: str | Path | None = None,
) -> PortfolioIngestionResult:
    date_value = _date_text(snapshot_date or date.today())
    ledger_path = Path(db_path) if db_path is not None else default_portfolio_ledger_path()
    import_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_BROKER_EXPORTS_DIR
    state_path = Path(state_dir) if state_dir is not None else DEFAULT_PORTFOLIO_STATE_DIR
    backup_dir = (
        Path(backup_csv_dir)
        if backup_csv_dir is not None
        else DEFAULT_PORTFOLIO_EXPORTS_DIR
    )
    account_ledger = Path(account_ledger_path) if account_ledger_path is not None else None

    import_dir.mkdir(parents=True, exist_ok=True)
    state_path.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    raw_files, source_raw_dir = _raw_files_for_ingestion(import_dir)
    t212_file = _latest_matching_file(raw_files, "t212")
    futu_file = _latest_matching_file(raw_files, "futu")

    frames: list[pd.DataFrame] = []
    banked_profits = {"Trading212_EUR": 0.0, "Futubull_USD": 0.0}

    df_ibkr, ibkr_metrics = fetch_live_ibkr_portfolio()
    ibkr_metrics_path = save_ibkr_metrics(
        ibkr_metrics,
        metrics_path=state_path / "ibkr_metrics.json",
    )
    account_snapshot_id = None
    account_ledger_written_path = None
    if account_ledger is not None and (not df_ibkr.empty or ibkr_metrics):
        account_write = write_account_snapshot(
            account_ledger,
            account_snapshot_from_live_positions_frame(
                df_ibkr,
                metrics=ibkr_metrics,
                environment="live",
                profile="ibkr_live_readonly",
                broker="ibkr",
                broker_label="IBKR Live",
                snapshot_date=date_value,
            ),
            snapshot_date=date_value,
        )
        account_snapshot_id = account_write.snapshot_id
        account_ledger_written_path = account_write.db_path
    if not df_ibkr.empty:
        frames.append(df_ibkr)

    df_t212 = pd.DataFrame()
    if t212_file is not None:
        try:
            df_t212, t212_banked = process_t212_csv(t212_file)
            banked_profits["Trading212_EUR"] = float(t212_banked)
            if not df_t212.empty:
                frames.append(df_t212)
        except Exception as exc:
            print(f"Trading212 import failed: {exc}")

    df_futu = pd.DataFrame()
    if futu_file is not None:
        try:
            df_futu = process_futu_csv(futu_file)
            if not df_futu.empty:
                frames.append(df_futu)
        except Exception as exc:
            print(f"Futubull import failed: {exc}")

    banked_profits_path = _write_json(
        state_path / "banked_profits.json",
        banked_profits,
    )

    if not frames:
        return PortfolioIngestionResult(
            status="no_positions",
            db_path=ledger_path,
            snapshot_date=date_value,
            raw_dir=import_dir,
            source_raw_dir=source_raw_dir,
            position_rows=0,
            ibkr_position_rows=0,
            futubull_position_rows=0,
            trading212_position_rows=0,
            ibkr_metrics_path=ibkr_metrics_path,
            banked_profits_path=banked_profits_path,
            account_ledger_path=account_ledger_written_path,
            account_snapshot_id=account_snapshot_id,
            message="No broker position rows were available to write.",
        )

    master_portfolio = pd.concat(frames, ignore_index=True)
    rows_written = write_live_positions_frame(
        ledger_path,
        master_portfolio,
        snapshot_date=date_value,
        replace_date=True,
    )
    backup_csv_path = backup_dir / f"unified_portfolio_{date_value}.csv"
    master_portfolio.to_csv(backup_csv_path, index=False)

    return PortfolioIngestionResult(
        status="updated",
        db_path=ledger_path,
        snapshot_date=date_value,
        raw_dir=import_dir,
        source_raw_dir=source_raw_dir,
        position_rows=rows_written,
        ibkr_position_rows=len(df_ibkr),
        futubull_position_rows=len(df_futu),
        trading212_position_rows=len(df_t212),
        ibkr_metrics_path=ibkr_metrics_path,
        banked_profits_path=banked_profits_path,
        backup_csv_path=backup_csv_path,
        account_ledger_path=account_ledger_written_path,
        account_snapshot_id=account_snapshot_id,
    )


def _build_polygon_options_adapter() -> PolygonOptionsSnapshotAdapter | None:
    api_key = (
        os.getenv("MASSIVE_API_KEY")
        or os.getenv("OPTIONS_API_KEY")
        or os.getenv("POLYGON_API_KEY")
    )
    settings = load_settings()
    api_key = (
        settings.massive_api_key
        or settings.options_api_key
        or settings.polygon_api_key
        or api_key
    )
    return PolygonOptionsSnapshotAdapter(api_key=api_key)


def _raw_files_for_ingestion(raw_dir: Path) -> tuple[list[Path], Path | None]:
    raw_files = _csv_files(raw_dir)
    if raw_files:
        return raw_files, raw_dir
    return [], raw_dir


def _csv_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [path for path in directory.iterdir() if path.is_file() and path.suffix == ".csv"]


def _latest_matching_file(files: list[Path], token: str) -> Path | None:
    matches = [path for path in files if token in path.name.lower()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
