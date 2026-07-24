"""Versioned, memory-safe dataset identity manifests for research runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from oqp.contracts.market_vertical import normalize_market_vertical
from oqp.research.artifacts import normalize_workspace_path, sha256_file, slugify


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_MANIFEST_ROOT = (
    _REPO_ROOT / "runtime" / "artifacts" / "research" / "dataset_manifests"
)
DATASET_MANIFEST_SCHEMA_VERSION = 1
_DATA_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".jsonl",
    ".parquet",
    ".pkl",
    ".pickle",
    ".txt",
}


class DatasetFingerprintError(ValueError):
    """Raised when a dataset cannot be registered or verified safely."""


@dataclass(frozen=True, slots=True)
class DatasetSourceFingerprint:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetSourceFingerprint":
        return cls(
            path=str(payload["path"]),
            size_bytes=int(payload["size_bytes"]),
            mtime_ns=int(payload["mtime_ns"]),
            sha256=str(payload["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    dataset_version: str
    market_vertical: str
    data_frequency: str
    adjustment_method: str
    row_count: int | None
    instrument_count: int | None
    date_start: str | None
    date_end: str | None
    schema: tuple[tuple[str, str], ...]
    schema_sha256: str
    source_files: tuple[DatasetSourceFingerprint, ...]
    content_sha256: str
    aggregate_sha256: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest_schema_version: int = DATASET_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = [
            {"name": name, "dtype": dtype} for name, dtype in self.schema
        ]
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        raw_schema = payload.get("schema") or ()
        schema = tuple(
            (str(item["name"]), str(item["dtype"]))
            if isinstance(item, Mapping)
            else (str(item[0]), str(item[1]))
            for item in raw_schema
        )
        return cls(
            dataset_id=str(payload["dataset_id"]),
            dataset_version=str(payload["dataset_version"]),
            market_vertical=str(payload["market_vertical"]),
            data_frequency=str(payload["data_frequency"]),
            adjustment_method=str(payload.get("adjustment_method") or "unknown"),
            row_count=_optional_int(payload.get("row_count")),
            instrument_count=_optional_int(payload.get("instrument_count")),
            date_start=_optional_text(payload.get("date_start")),
            date_end=_optional_text(payload.get("date_end")),
            schema=schema,
            schema_sha256=str(payload["schema_sha256"]),
            source_files=tuple(
                DatasetSourceFingerprint.from_mapping(item)
                for item in payload.get("source_files") or ()
            ),
            content_sha256=str(payload["content_sha256"]),
            aggregate_sha256=str(payload["aggregate_sha256"]),
            created_at=str(payload["created_at"]),
            metadata=dict(payload.get("metadata") or {}),
            manifest_schema_version=int(
                payload.get("manifest_schema_version", DATASET_MANIFEST_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetVerificationResult:
    verified: bool
    aggregate_sha256: str
    checked_files: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetFrameProfile:
    """Schema and coverage summary produced without retaining a full dataset."""

    schema: tuple[tuple[str, str], ...]
    row_count: int
    instrument_count: int | None = None
    date_start: str | None = None
    date_end: str | None = None


def register_dataset_manifest(
    source: str | Path | Iterable[str | Path],
    *,
    dataset_id: str,
    market_vertical: str,
    data_frequency: str,
    frame: pd.DataFrame | None = None,
    frame_profile: DatasetFrameProfile | None = None,
    dataset_version: str | None = None,
    adjustment_method: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
    manifest_root: str | Path = DEFAULT_DATASET_MANIFEST_ROOT,
    workspace_root: str | Path | None = None,
) -> tuple[DatasetManifest, Path]:
    """Fingerprint source files, persist an immutable manifest, and return it."""

    workspace = Path(workspace_root or _REPO_ROOT).resolve()
    root = Path(manifest_root)
    if not root.is_absolute():
        root = workspace / root
    root.mkdir(parents=True, exist_ok=True)

    files = discover_dataset_files(source)
    fingerprints = _fingerprint_sources(
        files,
        cache_path=root / ".file_hash_cache.json",
        workspace_root=workspace,
    )
    content_payload = [
        {
            "path": item.path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in fingerprints
    ]
    content_sha256 = _canonical_sha256(content_payload)
    if frame is not None and frame_profile is not None:
        raise DatasetFingerprintError(
            "Provide either frame or frame_profile when registering a dataset, not both"
        )
    if frame_profile is not None:
        schema = frame_profile.schema
        profile = {
            "row_count": int(frame_profile.row_count),
            "instrument_count": frame_profile.instrument_count,
            "date_start": frame_profile.date_start,
            "date_end": frame_profile.date_end,
        }
    else:
        schema, profile = _frame_profile(frame)
    schema_sha256 = _canonical_sha256(
        [{"name": name, "dtype": dtype} for name, dtype in schema]
    )
    resolved_version = str(dataset_version or f"sha256:{content_sha256[:12]}")
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest_without_hash = {
        "manifest_schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_id": _validate_dataset_id(dataset_id),
        "dataset_version": resolved_version,
        "market_vertical": normalize_market_vertical(market_vertical),
        "data_frequency": str(data_frequency or "unknown").strip().lower(),
        "adjustment_method": str(adjustment_method or "unknown").strip(),
        "row_count": profile["row_count"],
        "instrument_count": profile["instrument_count"],
        "date_start": profile["date_start"],
        "date_end": profile["date_end"],
        "schema": schema,
        "schema_sha256": schema_sha256,
        "source_files": fingerprints,
        "content_sha256": content_sha256,
        "metadata": dict(metadata or {}),
    }
    aggregate_sha256 = _canonical_sha256(
        _identity_payload_from_values(manifest_without_hash)
    )
    manifest = DatasetManifest(
        **manifest_without_hash,
        aggregate_sha256=aggregate_sha256,
        created_at=created_at,
    )
    output_dir = root / slugify(manifest.dataset_id, fallback="dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{aggregate_sha256}.json"
    if output_path.exists():
        existing = load_dataset_manifest(output_path)
        if existing.aggregate_sha256 != manifest.aggregate_sha256:
            raise DatasetFingerprintError(
                f"Existing dataset manifest conflicts with {output_path}"
            )
        return existing, output_path
    _atomic_json_write(output_path, manifest.to_dict())
    return manifest, output_path


def discover_dataset_files(
    source: str | Path | Iterable[str | Path],
) -> tuple[Path, ...]:
    """Return a deterministic list of physical files belonging to a dataset."""

    raw_sources: Iterable[str | Path]
    if isinstance(source, (str, Path)):
        raw_sources = (source,)
    else:
        raw_sources = source
    discovered: set[Path] = set()
    for raw in raw_sources:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            discovered.add(path)
        elif path.is_dir():
            discovered.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and not candidate.name.startswith(".")
                and candidate.suffix.lower() in _DATA_SUFFIXES
            )
        else:
            raise FileNotFoundError(f"Dataset source does not exist: {path}")
    if not discovered:
        raise DatasetFingerprintError("Dataset source contains no supported data files")
    return tuple(sorted(discovered, key=lambda item: item.as_posix()))


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise DatasetFingerprintError("Dataset manifest must contain a JSON object")
    manifest = DatasetManifest.from_mapping(payload)
    expected = _canonical_sha256(_identity_payload(manifest))
    if expected != manifest.aggregate_sha256:
        raise DatasetFingerprintError("Dataset manifest identity hash is invalid")
    return manifest


def verify_dataset_manifest(
    manifest_or_path: DatasetManifest | str | Path,
    *,
    workspace_root: str | Path | None = None,
    strict: bool = True,
) -> DatasetVerificationResult:
    """Verify every source byte against a previously persisted manifest."""

    manifest = (
        manifest_or_path
        if isinstance(manifest_or_path, DatasetManifest)
        else load_dataset_manifest(manifest_or_path)
    )
    workspace = Path(workspace_root or _REPO_ROOT).resolve()
    errors: list[str] = []
    for expected in manifest.source_files:
        source = Path(expected.path)
        if not source.is_absolute():
            source = workspace / source
        if not source.exists():
            errors.append(f"missing:{expected.path}")
            continue
        stat = source.stat()
        if int(stat.st_size) != expected.size_bytes:
            errors.append(f"size_changed:{expected.path}")
            continue
        if sha256_file(source) != expected.sha256:
            errors.append(f"content_changed:{expected.path}")
    result = DatasetVerificationResult(
        verified=not errors,
        aggregate_sha256=manifest.aggregate_sha256,
        checked_files=len(manifest.source_files),
        errors=tuple(errors),
    )
    if strict and errors:
        raise DatasetFingerprintError(
            "Dataset verification failed: " + ", ".join(errors)
        )
    return result


def attach_dataset_manifest_attrs(
    frame: pd.DataFrame,
    manifest: DatasetManifest,
    manifest_path: str | Path,
    *,
    verified: bool,
    workspace_root: str | Path | None = None,
) -> pd.DataFrame:
    """Attach compact dataset lineage to a frame without copying its rows."""

    frame.attrs.update(
        {
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "dataset_fingerprint": manifest.aggregate_sha256,
            "dataset_content_sha256": manifest.content_sha256,
            "dataset_schema_sha256": manifest.schema_sha256,
            "dataset_manifest_path": normalize_workspace_path(
                manifest_path, workspace_root or _REPO_ROOT
            ),
            "dataset_verified": bool(verified),
        }
    )
    return frame


def ensure_dataset_manifest_attrs(
    frame: pd.DataFrame,
    *,
    market_vertical: str,
    strict: bool = False,
    manifest_root: str | Path = DEFAULT_DATASET_MANIFEST_ROOT,
    workspace_root: str | Path | None = None,
) -> pd.DataFrame:
    """Register source-backed lineage when a caller has not done so already."""

    if frame.attrs.get("dataset_fingerprint"):
        return frame
    source = next(
        (
            frame.attrs.get(name)
            for name in ("source_path", "data_file", "dataset_path")
            if frame.attrs.get(name)
        ),
        None,
    )
    if source is None or not Path(str(source)).expanduser().exists():
        frame.attrs["dataset_verified"] = False
        frame.attrs["dataset_fingerprint_status"] = "source_path_unavailable"
        if strict:
            raise DatasetFingerprintError(
                "Strict dataset identity requires an existing source_path"
            )
        return frame
    dataset_id = str(
        frame.attrs.get("dataset_id")
        or slugify(Path(str(source)).stem, fallback="dataset")
    )
    manifest, manifest_path = register_dataset_manifest(
        source,
        dataset_id=dataset_id,
        dataset_version=_optional_text(frame.attrs.get("dataset_version")),
        market_vertical=market_vertical,
        data_frequency=str(frame.attrs.get("data_frequency") or "unknown"),
        adjustment_method=str(frame.attrs.get("adjustment_method") or "unknown"),
        frame=frame,
        metadata={"registration_source": "shared_research_evaluator"},
        manifest_root=manifest_root,
        workspace_root=workspace_root,
    )
    return attach_dataset_manifest_attrs(
        frame,
        manifest,
        manifest_path,
        verified=True,
        workspace_root=workspace_root,
    )


def _fingerprint_sources(
    files: tuple[Path, ...],
    *,
    cache_path: Path,
    workspace_root: Path,
) -> tuple[DatasetSourceFingerprint, ...]:
    cache = _load_hash_cache(cache_path)
    cache_files = cache.setdefault("files", {})
    fingerprints: list[DatasetSourceFingerprint] = []
    changed = False
    for path in files:
        stat = path.stat()
        cache_key = path.as_posix()
        cached = cache_files.get(cache_key, {})
        if (
            cached.get("size_bytes") == int(stat.st_size)
            and cached.get("mtime_ns") == int(stat.st_mtime_ns)
            and cached.get("sha256")
        ):
            digest = str(cached["sha256"])
        else:
            digest = sha256_file(path)
            cache_files[cache_key] = {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest,
            }
            changed = True
        fingerprints.append(
            DatasetSourceFingerprint(
                path=normalize_workspace_path(path, workspace_root),
                size_bytes=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
                sha256=digest,
            )
        )
    if changed or not cache_path.exists():
        _atomic_json_write(cache_path, cache)
    return tuple(fingerprints)


def _frame_profile(
    frame: pd.DataFrame | None,
) -> tuple[tuple[tuple[str, str], ...], dict[str, Any]]:
    if frame is None:
        return (), {
            "row_count": None,
            "instrument_count": None,
            "date_start": None,
            "date_end": None,
        }
    schema = tuple((str(column), str(dtype)) for column, dtype in frame.dtypes.items())
    date_col = next(
        (name for name in ("date", "datetime", "trading_date") if name in frame.columns),
        None,
    )
    ticker_col = next(
        (name for name in ("ticker", "symbol", "root") if name in frame.columns),
        None,
    )
    dates = (
        pd.to_datetime(frame[date_col], errors="coerce").dropna()
        if date_col
        else pd.Series(dtype="datetime64[ns]")
    )
    return schema, {
        "row_count": int(len(frame)),
        "instrument_count": (
            int(frame[ticker_col].dropna().astype(str).nunique()) if ticker_col else None
        ),
        "date_start": dates.min().isoformat() if not dates.empty else None,
        "date_end": dates.max().isoformat() if not dates.empty else None,
    }


def _identity_payload(manifest: DatasetManifest) -> dict[str, Any]:
    return _identity_payload_from_values(
        {
            "manifest_schema_version": manifest.manifest_schema_version,
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.dataset_version,
            "market_vertical": manifest.market_vertical,
            "data_frequency": manifest.data_frequency,
            "adjustment_method": manifest.adjustment_method,
            "row_count": manifest.row_count,
            "instrument_count": manifest.instrument_count,
            "date_start": manifest.date_start,
            "date_end": manifest.date_end,
            "schema": manifest.schema,
            "schema_sha256": manifest.schema_sha256,
            "source_files": manifest.source_files,
            "content_sha256": manifest.content_sha256,
            "metadata": manifest.metadata,
        }
    )


def _identity_payload_from_values(values: Mapping[str, Any]) -> dict[str, Any]:
    source_files = values.get("source_files") or ()
    schema = values.get("schema") or ()
    return {
        "manifest_schema_version": int(values["manifest_schema_version"]),
        "dataset_id": str(values["dataset_id"]),
        "dataset_version": str(values["dataset_version"]),
        "market_vertical": str(values["market_vertical"]),
        "data_frequency": str(values["data_frequency"]),
        "adjustment_method": str(values["adjustment_method"]),
        "row_count": values.get("row_count"),
        "instrument_count": values.get("instrument_count"),
        "date_start": values.get("date_start"),
        "date_end": values.get("date_end"),
        "schema": [
            {"name": str(item[0]), "dtype": str(item[1])}
            for item in schema
        ],
        "schema_sha256": str(values["schema_sha256"]),
        "source_files": [
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            if isinstance(item, DatasetSourceFingerprint)
            else {
                "path": str(item["path"]),
                "size_bytes": int(item["size_bytes"]),
                "sha256": str(item["sha256"]),
            }
            for item in source_files
        ],
        "content_sha256": str(values["content_sha256"]),
        "metadata": dict(values.get("metadata") or {}),
    }


def _load_hash_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "files": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "files": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        return {"schema_version": 1, "files": {}}
    return payload


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        handle.write("\n")
    temporary.replace(path)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_dataset_id(value: str) -> str:
    dataset_id = str(value or "").strip()
    if not dataset_id:
        raise DatasetFingerprintError("dataset_id cannot be empty")
    if slugify(dataset_id, fallback="") != dataset_id.lower():
        raise DatasetFingerprintError(
            "dataset_id may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return dataset_id


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_DATASET_MANIFEST_ROOT",
    "DatasetFingerprintError",
    "DatasetFrameProfile",
    "DatasetManifest",
    "DatasetSourceFingerprint",
    "DatasetVerificationResult",
    "attach_dataset_manifest_attrs",
    "discover_dataset_files",
    "ensure_dataset_manifest_attrs",
    "load_dataset_manifest",
    "register_dataset_manifest",
    "verify_dataset_manifest",
]
