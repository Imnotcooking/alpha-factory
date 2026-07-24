"""Immutable API dataset materialization for reproducible research."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oqp.config import REPO_ROOT
from oqp.contracts.market_vertical import normalize_market_vertical
from oqp.data.runtime_paths import RUNTIME_DATA_ROOT
from oqp.data.vendors import FMPDataAdapter, MassiveOptionsDataAdapter
from oqp.domain import AssetClass, Instrument
from oqp.options.chain_loader import option_quotes_to_frame
from oqp.research.artifacts import normalize_workspace_path, slugify
from oqp.research.dataset_fingerprints import (
    DEFAULT_DATASET_MANIFEST_ROOT,
    DatasetFingerprintError,
    DatasetFrameProfile,
    attach_dataset_manifest_attrs,
    load_dataset_manifest,
    register_dataset_manifest,
    verify_dataset_manifest,
)


API_REQUEST_SCHEMA_VERSION = 1
API_MATERIALIZATION_SCHEMA_VERSION = 1
DEFAULT_API_MATERIALIZATION_ROOT = RUNTIME_DATA_ROOT / "api_materialized"
_REDACTED = "<redacted>"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "access_key",
    "secret",
    "token",
    "password",
    "signature",
)


class ApiDatasetError(ValueError):
    """Raised when an API dataset cannot be materialized or loaded safely."""


class ApiDatasetQualityError(ApiDatasetError):
    """Raised when a vendor response fails publication quality checks."""


class HistoricalBacktestEligibilityError(ApiDatasetError):
    """Raised when current-only data is passed to a historical backtest."""


@dataclass(frozen=True, slots=True)
class ApiDatasetRequest:
    """Credential-free, canonical description of one vendor data request."""

    provider: str
    dataset_id: str
    market_vertical: str
    data_frequency: str
    endpoint: str
    symbols: tuple[str, ...]
    adjustment_method: str = "unknown"
    start_date: str | None = None
    end_date: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_schema_version: int = API_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        provider = slugify(self.provider, fallback="")
        dataset_id = slugify(self.dataset_id, fallback="")
        endpoint = str(self.endpoint or "").strip()
        symbols = tuple(
            sorted(
                {
                    str(symbol).strip().upper()
                    for symbol in self.symbols
                    if str(symbol).strip()
                }
            )
        )
        if not provider:
            raise ApiDatasetError("provider is required")
        if not dataset_id:
            raise ApiDatasetError("dataset_id is required")
        if not endpoint:
            raise ApiDatasetError("endpoint is required")
        if not symbols:
            raise ApiDatasetError("at least one symbol is required")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(
            self,
            "market_vertical",
            normalize_market_vertical(self.market_vertical),
        )
        object.__setattr__(self, "data_frequency", str(self.data_frequency).lower())
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "params", _sanitize_mapping(self.params))
        object.__setattr__(self, "metadata", _sanitize_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return _json_safe(payload)

    @property
    def request_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MaterializedApiDataset:
    descriptor_path: Path
    data_path: Path
    request_path: Path
    raw_response_path: Path
    quality_path: Path
    manifest_path: Path
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    provider: str
    market_vertical: str
    data_frequency: str
    historical_backtest_eligible: bool
    limitations: tuple[str, ...]

    @property
    def root(self) -> Path:
        return self.descriptor_path.parent

    @classmethod
    def load_descriptor(
        cls,
        path: str | Path,
        *,
        workspace_root: str | Path = REPO_ROOT,
    ) -> "MaterializedApiDataset":
        workspace = Path(workspace_root).resolve()
        descriptor = Path(path).expanduser()
        if descriptor.is_dir():
            descriptor = descriptor / "materialization.json"
        if not descriptor.is_absolute():
            descriptor = workspace / descriptor
        if not descriptor.exists():
            raise FileNotFoundError(f"API dataset descriptor not found: {descriptor}")
        with descriptor.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("materialization_schema_version", 0)) != API_MATERIALIZATION_SCHEMA_VERSION:
            raise ApiDatasetError("Unsupported API materialization schema version")

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else workspace / candidate

        return cls(
            descriptor_path=descriptor.resolve(),
            data_path=resolve(payload["data_path"]).resolve(),
            request_path=resolve(payload["request_path"]).resolve(),
            raw_response_path=resolve(payload["raw_response_path"]).resolve(),
            quality_path=resolve(payload["quality_path"]).resolve(),
            manifest_path=resolve(payload["manifest_path"]).resolve(),
            dataset_id=str(payload["dataset_id"]),
            dataset_version=str(payload["dataset_version"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            provider=str(payload["provider"]),
            market_vertical=str(payload["market_vertical"]),
            data_frequency=str(payload["data_frequency"]),
            historical_backtest_eligible=bool(
                payload.get("historical_backtest_eligible", False)
            ),
            limitations=tuple(str(item) for item in payload.get("limitations") or ()),
        )


def materialize_fmp_us_equity_daily(
    symbols: Iterable[str],
    *,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    adjustment_method: str = "provider_default",
    dataset_id: str = "fmp_us_equity_daily",
    adapter: FMPDataAdapter | None = None,
    storage_root: str | Path = DEFAULT_API_MATERIALIZATION_ROOT,
    manifest_root: str | Path = DEFAULT_DATASET_MANIFEST_ROOT,
    workspace_root: str | Path = REPO_ROOT,
    retrieved_at: datetime | None = None,
    point_in_time_universe: bool = False,
    universe_as_of: str | date | None = None,
    strict_quality: bool = True,
) -> MaterializedApiDataset:
    """Fetch FMP daily bars and publish a frozen, fingerprinted research bundle."""

    endpoint_map = {
        "provider_default": "historical-price-eod/full",
        "non_split_adjusted": "historical-price-eod/non-split-adjusted",
        "dividend_adjusted": "historical-price-eod/dividend-adjusted",
    }
    adjustment = str(adjustment_method or "provider_default").strip().lower()
    if adjustment not in endpoint_map:
        raise ApiDatasetError(
            f"Unsupported FMP adjustment_method={adjustment!r}; "
            f"choose one of {sorted(endpoint_map)}"
        )
    normalized_symbols = _normalize_symbols(symbols)
    params: dict[str, Any] = {}
    if start_date is not None:
        params["from"] = _iso_date(start_date)
    if end_date is not None:
        params["to"] = _iso_date(end_date)
    limitations = []
    if not point_in_time_universe:
        limitations.append("point_in_time_universe_not_supplied")
    request = ApiDatasetRequest(
        provider="fmp",
        dataset_id=dataset_id,
        market_vertical="EQUITY_US",
        data_frequency="daily",
        endpoint=endpoint_map[adjustment],
        symbols=normalized_symbols,
        adjustment_method=adjustment,
        start_date=_optional_iso_date(start_date),
        end_date=_optional_iso_date(end_date),
        params=params,
        metadata={
            "historical_backtest_eligible": True,
            "cross_sectional_universe_eligible": bool(point_in_time_universe),
            "point_in_time_universe": bool(point_in_time_universe),
            "universe_as_of": _optional_iso_date(universe_as_of),
            "limitations": limitations,
        },
    )
    vendor = adapter or FMPDataAdapter()
    retrieved = _utc_timestamp(retrieved_at)

    def batches() -> Iterator[tuple[str, Any, pd.DataFrame]]:
        for symbol in request.symbols:
            query = {"symbol": symbol, **params}
            payload = vendor.get_json(request.endpoint, stable=True, params=query)
            rows = _extract_fmp_rows(payload)
            yield (
                symbol,
                payload,
                _normalize_fmp_daily_rows(
                    rows,
                    symbol=symbol,
                    adjustment_method=adjustment,
                    retrieved_at=retrieved,
                ),
            )

    return _materialize_api_batches(
        request,
        batches(),
        quality=_QualityAccumulator("equity_daily", request.symbols),
        historical_backtest_eligible=True,
        limitations=tuple(limitations),
        storage_root=storage_root,
        manifest_root=manifest_root,
        workspace_root=workspace_root,
        retrieved_at=retrieved,
        strict_quality=strict_quality,
    )


def materialize_massive_us_option_snapshot(
    underlyings: Iterable[str],
    *,
    expiration: str | date | None = None,
    min_strike: float | None = None,
    max_strike: float | None = None,
    dataset_id: str = "massive_us_option_snapshot",
    adapter: MassiveOptionsDataAdapter | None = None,
    storage_root: str | Path = DEFAULT_API_MATERIALIZATION_ROOT,
    manifest_root: str | Path = DEFAULT_DATASET_MANIFEST_ROOT,
    workspace_root: str | Path = REPO_ROOT,
    retrieved_at: datetime | None = None,
    strict_quality: bool = True,
) -> MaterializedApiDataset:
    """Freeze current Massive chains without claiming historical eligibility."""

    normalized_underlyings = _normalize_symbols(underlyings)
    params: dict[str, Any] = {
        "expiration_date": _optional_iso_date(expiration),
        "strike_price.gte": min_strike,
        "strike_price.lte": max_strike,
    }
    params = {key: value for key, value in params.items() if value is not None}
    limitations = (
        "current_snapshot_only",
        "not_a_historical_option_panel",
        "historical_fills_require_time_aligned_quotes_or_a_declared_proxy",
    )
    request = ApiDatasetRequest(
        provider="massive",
        dataset_id=dataset_id,
        market_vertical="OPTIONS_US",
        data_frequency="snapshot",
        endpoint="/v3/snapshot/options/{underlying}",
        symbols=normalized_underlyings,
        adjustment_method="raw_option_snapshot",
        params=params,
        metadata={
            "historical_backtest_eligible": False,
            "prospective_collection_eligible": True,
            "limitations": limitations,
        },
    )
    vendor = adapter or MassiveOptionsDataAdapter()
    retrieved = _utc_timestamp(retrieved_at)

    def batches() -> Iterator[tuple[str, Any, pd.DataFrame]]:
        for symbol in request.symbols:
            underlying = Instrument(symbol=symbol, asset_class=AssetClass.EQUITY)
            rows = vendor.get_option_snapshot_rows(
                symbol,
                expiration=expiration,
                min_strike=min_strike,
                max_strike=max_strike,
            )
            quotes = vendor.option_quotes_from_snapshot_rows(underlying, rows)
            frame = option_quotes_to_frame(quotes, market_vertical="OPTIONS_US")
            if not frame.empty:
                frame["retrieved_at_utc"] = retrieved
                frame["snapshot_kind"] = "current_only"
            yield symbol, rows, frame

    return _materialize_api_batches(
        request,
        batches(),
        quality=_QualityAccumulator("option_snapshot", request.symbols),
        historical_backtest_eligible=False,
        limitations=limitations,
        storage_root=storage_root,
        manifest_root=manifest_root,
        workspace_root=workspace_root,
        retrieved_at=retrieved,
        strict_quality=strict_quality,
    )


def load_materialized_api_dataset(
    path: str | Path,
    *,
    require_historical_backtest_eligible: bool = True,
    verify: bool = True,
    workspace_root: str | Path = REPO_ROOT,
) -> pd.DataFrame:
    """Load and verify a published API bundle, rejecting current-only snapshots."""

    bundle = MaterializedApiDataset.load_descriptor(
        path,
        workspace_root=workspace_root,
    )
    if require_historical_backtest_eligible and not bundle.historical_backtest_eligible:
        raise HistoricalBacktestEligibilityError(
            f"Dataset {bundle.dataset_id} is current-only and cannot be used for a "
            "historical backtest: {', '.join(bundle.limitations)}"
        )
    manifest = load_dataset_manifest(bundle.manifest_path)
    if manifest.aggregate_sha256 != bundle.dataset_fingerprint:
        raise DatasetFingerprintError(
            "API materialization descriptor does not match its dataset manifest"
        )
    if verify:
        verify_dataset_manifest(
            manifest,
            workspace_root=workspace_root,
            strict=True,
        )
    frame = pd.read_parquet(bundle.data_path)
    if manifest.row_count is not None and len(frame) != manifest.row_count:
        raise DatasetFingerprintError(
            f"Materialized row count changed: expected {manifest.row_count}, got {len(frame)}"
        )
    attach_dataset_manifest_attrs(
        frame,
        manifest,
        bundle.manifest_path,
        verified=verify,
        workspace_root=workspace_root,
    )
    frame.attrs.update(
        {
            "source_path": str(bundle.data_path),
            "data_file": str(bundle.data_path),
            "materialization_descriptor": str(bundle.descriptor_path),
            "data_vendor": bundle.provider,
            "market_vertical": bundle.market_vertical,
            "data_frequency": bundle.data_frequency,
            "dataset_role": "immutable_api_materialization",
            "historical_backtest_eligible": bundle.historical_backtest_eligible,
            "dataset_limitations": list(bundle.limitations),
        }
    )
    return frame


class _QualityAccumulator:
    def __init__(self, kind: str, expected_symbols: tuple[str, ...]):
        self.kind = kind
        self.expected_symbols = expected_symbols
        self.row_count = 0
        self.instruments: set[str] = set()
        self.empty_requests: list[str] = []
        self.duplicate_keys = 0
        self.missing_primary_value = 0
        self.invalid_ohlc = 0
        self.crossed_quotes = 0
        self.expired_contracts = 0
        self.date_start: pd.Timestamp | None = None
        self.date_end: pd.Timestamp | None = None
        self.schema: tuple[tuple[str, str], ...] = ()

    def update(self, request_key: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            self.empty_requests.append(request_key)
            return
        schema = tuple((str(column), str(dtype)) for column, dtype in frame.dtypes.items())
        if not self.schema:
            self.schema = schema
        elif schema != self.schema:
            raise ApiDatasetQualityError(
                f"Normalized schema changed within one materialization at {request_key}"
            )
        self.row_count += int(len(frame))
        date_values = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
        if not date_values.empty:
            batch_start = date_values.min()
            batch_end = date_values.max()
            self.date_start = batch_start if self.date_start is None else min(self.date_start, batch_start)
            self.date_end = batch_end if self.date_end is None else max(self.date_end, batch_end)

        if self.kind == "equity_daily":
            self.instruments.update(frame["ticker"].dropna().astype(str).unique())
            self.duplicate_keys += int(frame.duplicated(["ticker", "date"]).sum())
            close = pd.to_numeric(frame["close"], errors="coerce")
            self.missing_primary_value += int(close.isna().sum())
            open_ = pd.to_numeric(frame["open"], errors="coerce")
            high = pd.to_numeric(frame["high"], errors="coerce")
            low = pd.to_numeric(frame["low"], errors="coerce")
            invalid = high.lt(pd.concat([open_, close, low], axis=1).max(axis=1))
            invalid |= low.gt(pd.concat([open_, close, high], axis=1).min(axis=1))
            self.invalid_ohlc += int(invalid.fillna(False).sum())
        elif self.kind == "option_snapshot":
            self.instruments.update(frame["option_symbol"].dropna().astype(str).unique())
            self.duplicate_keys += int(frame.duplicated(["date", "option_symbol"]).sum())
            mark = pd.to_numeric(frame["mark"], errors="coerce")
            self.missing_primary_value += int(mark.isna().sum())
            bid = pd.to_numeric(frame["bid"], errors="coerce")
            ask = pd.to_numeric(frame["ask"], errors="coerce")
            self.crossed_quotes += int((bid.notna() & ask.notna() & bid.gt(ask)).sum())
            expiry = pd.to_datetime(frame["expiry"], errors="coerce")
            dates = pd.to_datetime(frame["date"], errors="coerce")
            self.expired_contracts += int((expiry < dates).fillna(False).sum())

    def report(self) -> dict[str, Any]:
        errors = []
        warnings = []
        if self.row_count == 0:
            errors.append("no_normalized_rows")
        if self.empty_requests:
            errors.append("one_or_more_requested_symbols_returned_no_rows")
        if self.duplicate_keys:
            errors.append("duplicate_primary_keys")
        if self.invalid_ohlc:
            errors.append("invalid_ohlc_relationship")
        if self.crossed_quotes:
            errors.append("crossed_option_quotes")
        if self.expired_contracts:
            errors.append("expired_contracts_in_snapshot")
        if self.missing_primary_value:
            warnings.append("missing_close_or_mark_values")
        return {
            "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
            "errors": errors,
            "warnings": warnings,
            "expected_request_keys": list(self.expected_symbols),
            "empty_request_keys": self.empty_requests,
            "row_count": self.row_count,
            "instrument_count": len(self.instruments),
            "date_start": _timestamp_text(self.date_start),
            "date_end": _timestamp_text(self.date_end),
            "duplicate_key_count": self.duplicate_keys,
            "missing_primary_value_count": self.missing_primary_value,
            "invalid_ohlc_count": self.invalid_ohlc,
            "crossed_quote_count": self.crossed_quotes,
            "expired_contract_count": self.expired_contracts,
        }

    def frame_profile(self) -> DatasetFrameProfile:
        return DatasetFrameProfile(
            schema=self.schema,
            row_count=self.row_count,
            instrument_count=len(self.instruments),
            date_start=_timestamp_text(self.date_start),
            date_end=_timestamp_text(self.date_end),
        )


def _materialize_api_batches(
    request: ApiDatasetRequest,
    batches: Iterable[tuple[str, Any, pd.DataFrame]],
    *,
    quality: _QualityAccumulator,
    historical_backtest_eligible: bool,
    limitations: tuple[str, ...],
    storage_root: str | Path,
    manifest_root: str | Path,
    workspace_root: str | Path,
    retrieved_at: datetime,
    strict_quality: bool,
) -> MaterializedApiDataset:
    workspace = Path(workspace_root).resolve()
    root = Path(storage_root)
    if not root.is_absolute():
        root = workspace / root
    parent = (
        root
        / request.market_vertical.lower()
        / request.provider
        / request.dataset_id
    )
    parent.mkdir(parents=True, exist_ok=True)
    snapshot_id = (
        f"{retrieved_at.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{request.request_sha256[:12]}"
    )
    destination = parent / snapshot_id
    if destination.exists():
        raise ApiDatasetError(
            f"Immutable API materialization already exists: {destination}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".materializing_", dir=parent))
    writer: pq.ParquetWriter | None = None
    try:
        data_path = temporary / "data.parquet"
        request_path = temporary / "request.json"
        raw_path = temporary / "raw_response.jsonl.gz"
        quality_path = temporary / "quality.json"
        _atomic_json_write(
            request_path,
            {
                **request.to_dict(),
                "request_sha256": request.request_sha256,
                "retrieved_at_utc": retrieved_at.isoformat(),
            },
        )
        with raw_path.open("wb") as raw_binary:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_binary,
                mtime=int(retrieved_at.timestamp()),
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as raw_handle:
                    for request_key, raw_payload, frame in batches:
                        raw_handle.write(
                            json.dumps(
                                {
                                    "request_key": request_key,
                                    "payload": _json_safe(raw_payload),
                                },
                                sort_keys=True,
                                ensure_ascii=True,
                            )
                        )
                        raw_handle.write("\n")
                        quality.update(request_key, frame)
                        if frame.empty:
                            continue
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(
                                data_path,
                                table.schema,
                                compression="zstd",
                            )
                        writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
        report = quality.report()
        _atomic_json_write(quality_path, report)
        if strict_quality and report["errors"]:
            raise ApiDatasetQualityError(
                "API materialization failed quality checks: "
                + ", ".join(report["errors"])
            )
        if not data_path.exists():
            raise ApiDatasetQualityError("API materialization produced no Parquet data")

        os.replace(temporary, destination)
        data_path = destination / data_path.name
        request_path = destination / request_path.name
        raw_path = destination / raw_path.name
        quality_path = destination / quality_path.name
        manifest_metadata = {
            "provider": request.provider,
            "endpoint": request.endpoint,
            "request_sha256": request.request_sha256,
            "retrieved_at_utc": retrieved_at.isoformat(),
            "historical_backtest_eligible": historical_backtest_eligible,
            "limitations": list(limitations),
            "quality_status": report["status"],
            "materialization_schema_version": API_MATERIALIZATION_SCHEMA_VERSION,
        }
        manifest, manifest_path = register_dataset_manifest(
            (data_path, request_path, raw_path, quality_path),
            dataset_id=request.dataset_id,
            market_vertical=request.market_vertical,
            data_frequency=request.data_frequency,
            adjustment_method=request.adjustment_method,
            frame_profile=quality.frame_profile(),
            metadata=manifest_metadata,
            manifest_root=manifest_root,
            workspace_root=workspace,
        )
        descriptor_path = destination / "materialization.json"
        descriptor = {
            "materialization_schema_version": API_MATERIALIZATION_SCHEMA_VERSION,
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "dataset_fingerprint": manifest.aggregate_sha256,
            "provider": request.provider,
            "market_vertical": request.market_vertical,
            "data_frequency": request.data_frequency,
            "historical_backtest_eligible": historical_backtest_eligible,
            "limitations": list(limitations),
            "data_path": normalize_workspace_path(data_path, workspace),
            "request_path": normalize_workspace_path(request_path, workspace),
            "raw_response_path": normalize_workspace_path(raw_path, workspace),
            "quality_path": normalize_workspace_path(quality_path, workspace),
            "manifest_path": normalize_workspace_path(manifest_path, workspace),
        }
        _atomic_json_write(descriptor_path, descriptor)
        return MaterializedApiDataset.load_descriptor(
            descriptor_path,
            workspace_root=workspace,
        )
    except Exception:
        if writer is not None:
            writer.close()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _extract_fmp_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("historical") or payload.get("data") or []
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _normalize_fmp_daily_rows(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    adjustment_method: str,
    retrieved_at: datetime,
) -> pd.DataFrame:
    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "vwap",
        "source",
        "adjustment_method",
        "retrieved_at_utc",
    ]
    normalized = []
    for row in rows:
        normalized.append(
            {
                "date": row.get("date"),
                "ticker": symbol,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "adjusted_close": (
                    row.get("adjClose")
                    if row.get("adjClose") is not None
                    else row.get("adjustedClose", row.get("close"))
                ),
                "volume": row.get("volume"),
                "vwap": row.get("vwap"),
                "source": "fmp",
                "adjustment_method": adjustment_method,
                "retrieved_at_utc": retrieved_at,
            }
        )
    frame = pd.DataFrame(normalized, columns=columns)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype("string")
    for column in ("open", "high", "low", "close", "adjusted_close", "volume", "vwap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["source"] = frame["source"].astype("string")
    frame["adjustment_method"] = frame["adjustment_method"].astype("string")
    frame["retrieved_at_utc"] = pd.to_datetime(frame["retrieved_at_utc"], utc=True)
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _normalize_symbols(values: Iterable[str]) -> tuple[str, ...]:
    symbols = tuple(
        sorted({str(value).strip().upper() for value in values if str(value).strip()})
    )
    if not symbols:
        raise ApiDatasetError("at least one symbol is required")
    return symbols


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for raw_key, raw_value in dict(value or {}).items():
        key = str(raw_key)
        lowered = key.lower()
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            sanitized[key] = _REDACTED
        elif isinstance(raw_value, Mapping):
            sanitized[key] = _sanitize_mapping(raw_value)
        elif isinstance(raw_value, (list, tuple)):
            sanitized[key] = [
                _sanitize_mapping(item) if isinstance(item, Mapping) else _json_safe(item)
                for item in raw_value
            ]
        else:
            sanitized[key] = _json_safe(raw_value)
    return sanitized


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _utc_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).replace(microsecond=0)


def _iso_date(value: str | date) -> str:
    return pd.Timestamp(value).date().isoformat()


def _optional_iso_date(value: str | date | None) -> str | None:
    return None if value is None else _iso_date(value)


def _timestamp_text(value: pd.Timestamp | None) -> str | None:
    return None if value is None or pd.isna(value) else value.isoformat()


__all__ = [
    "API_MATERIALIZATION_SCHEMA_VERSION",
    "API_REQUEST_SCHEMA_VERSION",
    "DEFAULT_API_MATERIALIZATION_ROOT",
    "ApiDatasetError",
    "ApiDatasetQualityError",
    "ApiDatasetRequest",
    "HistoricalBacktestEligibilityError",
    "MaterializedApiDataset",
    "load_materialized_api_dataset",
    "materialize_fmp_us_equity_daily",
    "materialize_massive_us_option_snapshot",
]
