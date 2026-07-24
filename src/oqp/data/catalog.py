"""Validated data-platform catalog contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from oqp.config import REPO_ROOT


CATALOG_SCHEMA_VERSION = "oqp_data_source_catalog_v1"
DEFAULT_DATA_CATALOG_PATH = (
    REPO_ROOT / "departments" / "data_platform" / "source_catalog.yaml"
)


class DataCatalogError(ValueError):
    """Raised when the data-platform catalog violates its contract."""


@dataclass(frozen=True, slots=True)
class DataCatalogEntry:
    dataset_id: str
    asset_class: str
    timeframe: str
    relative_path: PurePosixPath
    storage_role: str
    providers: tuple[str, ...]
    required: bool
    freshness_max_age_days: int
    update_mode: str
    owner: str
    description: str

    def resolve(self, runtime_data_root: str | Path) -> Path:
        """Resolve the validated relative lane beneath a runtime data root."""

        return Path(runtime_data_root).joinpath(*self.relative_path.parts)


@dataclass(frozen=True, slots=True)
class DataCatalog:
    schema_version: str
    catalog_owner: str
    data_root: PurePosixPath
    datasets: tuple[DataCatalogEntry, ...]
    source_path: Path


def load_data_catalog(path: str | Path | None = None) -> DataCatalog:
    """Load and validate the committed data-platform source catalog."""

    source_path = Path(path) if path is not None else DEFAULT_DATA_CATALOG_PATH
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DataCatalogError(f"Invalid YAML in data catalog {source_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise DataCatalogError("Data catalog root must be a mapping.")

    schema_version = _required_text(payload, "schema_version", context="catalog")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise DataCatalogError(
            f"Unsupported data catalog schema_version: {schema_version!r}."
        )

    data_root = _safe_relative_path(
        _required_text(payload, "data_root", context="catalog"),
        field="data_root",
    )
    if data_root != PurePosixPath("runtime/data"):
        raise DataCatalogError("Catalog data_root must be 'runtime/data'.")

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise DataCatalogError("Catalog datasets must be a non-empty list.")

    entries: list[DataCatalogEntry] = []
    dataset_ids: set[str] = set()
    relative_paths: set[PurePosixPath] = set()
    for index, raw_entry in enumerate(raw_datasets):
        context = f"datasets[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise DataCatalogError(f"{context} must be a mapping.")
        entry = _parse_entry(raw_entry, context=context)
        if entry.dataset_id in dataset_ids:
            raise DataCatalogError(f"Duplicate dataset id: {entry.dataset_id!r}.")
        if entry.relative_path in relative_paths:
            raise DataCatalogError(
                f"Duplicate dataset relative_path: {entry.relative_path.as_posix()!r}."
            )
        dataset_ids.add(entry.dataset_id)
        relative_paths.add(entry.relative_path)
        entries.append(entry)

    return DataCatalog(
        schema_version=schema_version,
        catalog_owner=_required_text(payload, "catalog_owner", context="catalog"),
        data_root=data_root,
        datasets=tuple(entries),
        source_path=source_path,
    )


def _parse_entry(raw: Mapping[str, Any], *, context: str) -> DataCatalogEntry:
    providers = raw.get("providers")
    if not isinstance(providers, list) or not providers:
        raise DataCatalogError(f"{context}.providers must be a non-empty list.")
    provider_names = tuple(_nonempty_scalar(value, f"{context}.providers") for value in providers)

    required = raw.get("required")
    if not isinstance(required, bool):
        raise DataCatalogError(f"{context}.required must be a boolean.")

    freshness = raw.get("freshness_max_age_days")
    if not isinstance(freshness, int) or isinstance(freshness, bool) or freshness < 0:
        raise DataCatalogError(
            f"{context}.freshness_max_age_days must be a non-negative integer."
        )

    asset_class = _required_text(raw, "asset_class", context=context).upper()
    timeframe = _required_text(raw, "timeframe", context=context).lower()
    if not asset_class.replace("_", "").isalnum():
        raise DataCatalogError(f"{context}.asset_class contains unsupported characters.")
    if not timeframe.replace("_", "").isalnum():
        raise DataCatalogError(f"{context}.timeframe contains unsupported characters.")

    return DataCatalogEntry(
        dataset_id=_required_text(raw, "id", context=context),
        asset_class=asset_class,
        timeframe=timeframe,
        relative_path=_safe_relative_path(
            _required_text(raw, "relative_path", context=context),
            field=f"{context}.relative_path",
        ),
        storage_role=_required_text(raw, "storage_role", context=context),
        providers=provider_names,
        required=required,
        freshness_max_age_days=freshness,
        update_mode=_required_text(raw, "update_mode", context=context),
        owner=_required_text(raw, "owner", context=context),
        description=_required_text(raw, "description", context=context),
    )


def _required_text(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    if key not in payload:
        raise DataCatalogError(f"{context}.{key} is required.")
    return _nonempty_scalar(payload[key], f"{context}.{key}")


def _nonempty_scalar(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataCatalogError(f"{field} must be a non-empty string.")
    return value.strip()


def _safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    if "\\" in value:
        raise DataCatalogError(f"{field} must use POSIX separators.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path == PurePosixPath("."):
        raise DataCatalogError(f"{field} must be a non-empty relative path.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DataCatalogError(f"{field} cannot contain empty, '.' or '..' segments.")
    return path
