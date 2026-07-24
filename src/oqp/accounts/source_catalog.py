"""Validated Middle Office account-source catalog contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from oqp.config import REPO_ROOT


ACCOUNT_SOURCE_CATALOG_SCHEMA_VERSION = "oqp_account_source_catalog_v1"
DEFAULT_ACCOUNT_SOURCE_CATALOG_PATH = (
    REPO_ROOT / "departments" / "middle_office" / "account_sources.yaml"
)

_ALLOWED_ENVIRONMENTS = frozenset({"live", "paper", "sim"})
_ALLOWED_BROKER_ACCESS = frozenset({"read_only", "none"})
_ALLOWED_LIFECYCLES = frozenset({"active", "migration", "planned"})
_ALLOWED_SCOPES = frozenset({"positions", "cash", "nav", "trade_events"})
_ALLOWED_AUTHORITIES = frozenset(
    {"broker", "broker_export", "approved_manual", "aggregate_reporting", "legacy_derived"}
)


class AccountSourceCatalogError(ValueError):
    """Raised when the account-source catalog violates its contract."""


@dataclass(frozen=True, slots=True)
class AccountSourceEntry:
    source_id: str
    provider: str
    profile: str
    environment: str
    account_role: str
    authority: str
    writers: tuple[PurePosixPath, ...]
    runtime_paths: tuple[PurePosixPath, ...]
    freshness_max_age_hours: int
    required: bool
    broker_access: str
    lifecycle: str
    reconciliation_scope: tuple[str, ...]
    description: str

    def resolve_runtime_paths(self, repo_root: str | Path) -> tuple[Path, ...]:
        """Resolve catalog paths beneath a repository root without reading them."""

        root = Path(repo_root)
        return tuple(root.joinpath(*path.parts) for path in self.runtime_paths)


@dataclass(frozen=True, slots=True)
class AccountSourceCatalog:
    schema_version: str
    catalog_owner: str
    canonical_ledger: PurePosixPath
    sources: tuple[AccountSourceEntry, ...]
    source_path: Path

    def entry_for_profile(self, profile: str) -> AccountSourceEntry | None:
        """Return the source registered for an exact profile name."""

        profile_key = str(profile).strip()
        return next(
            (entry for entry in self.sources if entry.profile == profile_key),
            None,
        )


def load_account_source_catalog(
    path: str | Path | None = None,
) -> AccountSourceCatalog:
    """Load and validate the committed Middle Office source inventory."""

    source_path = (
        Path(path) if path is not None else DEFAULT_ACCOUNT_SOURCE_CATALOG_PATH
    )
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AccountSourceCatalogError(
            f"Invalid YAML in account source catalog {source_path}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise AccountSourceCatalogError("Account source catalog root must be a mapping.")

    schema_version = _required_text(payload, "schema_version", context="catalog")
    if schema_version != ACCOUNT_SOURCE_CATALOG_SCHEMA_VERSION:
        raise AccountSourceCatalogError(
            f"Unsupported account source catalog schema_version: {schema_version!r}."
        )

    canonical_ledger = _runtime_path(
        _required_text(payload, "canonical_ledger", context="catalog"),
        field="catalog.canonical_ledger",
    )
    if canonical_ledger != PurePosixPath(
        "runtime/db/accounts/account_ledger.db"
    ):
        raise AccountSourceCatalogError(
            "Catalog canonical_ledger must be "
            "'runtime/db/accounts/account_ledger.db'."
        )

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise AccountSourceCatalogError("Catalog sources must be a non-empty list.")

    entries: list[AccountSourceEntry] = []
    source_ids: set[str] = set()
    profiles: set[str] = set()
    for index, raw_entry in enumerate(raw_sources):
        context = f"sources[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise AccountSourceCatalogError(f"{context} must be a mapping.")
        entry = _parse_entry(raw_entry, context=context)
        if entry.source_id in source_ids:
            raise AccountSourceCatalogError(
                f"Duplicate account source id: {entry.source_id!r}."
            )
        if entry.profile in profiles:
            raise AccountSourceCatalogError(
                f"Duplicate account source profile: {entry.profile!r}."
            )
        source_ids.add(entry.source_id)
        profiles.add(entry.profile)
        entries.append(entry)

    return AccountSourceCatalog(
        schema_version=schema_version,
        catalog_owner=_required_text(payload, "catalog_owner", context="catalog"),
        canonical_ledger=canonical_ledger,
        sources=tuple(entries),
        source_path=source_path,
    )


def _parse_entry(raw: Mapping[str, Any], *, context: str) -> AccountSourceEntry:
    environment = _choice(
        raw,
        "environment",
        context=context,
        allowed=_ALLOWED_ENVIRONMENTS,
    )
    broker_access = _choice(
        raw,
        "broker_access",
        context=context,
        allowed=_ALLOWED_BROKER_ACCESS,
    )
    lifecycle = _choice(
        raw,
        "lifecycle",
        context=context,
        allowed=_ALLOWED_LIFECYCLES,
    )
    authority = _choice(
        raw,
        "authority",
        context=context,
        allowed=_ALLOWED_AUTHORITIES,
    )

    required = raw.get("required")
    if not isinstance(required, bool):
        raise AccountSourceCatalogError(f"{context}.required must be a boolean.")

    freshness = raw.get("freshness_max_age_hours")
    if not isinstance(freshness, int) or isinstance(freshness, bool) or freshness < 0:
        raise AccountSourceCatalogError(
            f"{context}.freshness_max_age_hours must be a non-negative integer."
        )

    raw_runtime_paths = raw.get("runtime_paths")
    if not isinstance(raw_runtime_paths, list) or not raw_runtime_paths:
        raise AccountSourceCatalogError(
            f"{context}.runtime_paths must be a non-empty list."
        )
    runtime_paths = tuple(
        _runtime_path(
            _nonempty_text(value, f"{context}.runtime_paths"),
            field=f"{context}.runtime_paths",
        )
        for value in raw_runtime_paths
    )
    if len(set(runtime_paths)) != len(runtime_paths):
        raise AccountSourceCatalogError(
            f"{context}.runtime_paths cannot contain duplicates."
        )

    raw_writers = raw.get("writers")
    if not isinstance(raw_writers, list) or not raw_writers:
        raise AccountSourceCatalogError(f"{context}.writers must be a non-empty list.")
    writers = tuple(
        _safe_relative_path(
            _nonempty_text(value, f"{context}.writers"),
            field=f"{context}.writers",
        )
        for value in raw_writers
    )
    if len(set(writers)) != len(writers):
        raise AccountSourceCatalogError(f"{context}.writers cannot contain duplicates.")

    raw_scope = raw.get("reconciliation_scope")
    if not isinstance(raw_scope, list) or not raw_scope:
        raise AccountSourceCatalogError(
            f"{context}.reconciliation_scope must be a non-empty list."
        )
    scope = tuple(
        _nonempty_text(value, f"{context}.reconciliation_scope").lower()
        for value in raw_scope
    )
    unsupported_scope = set(scope) - _ALLOWED_SCOPES
    if unsupported_scope:
        raise AccountSourceCatalogError(
            f"{context}.reconciliation_scope contains unsupported values: "
            f"{sorted(unsupported_scope)}."
        )
    if len(set(scope)) != len(scope):
        raise AccountSourceCatalogError(
            f"{context}.reconciliation_scope cannot contain duplicates."
        )

    return AccountSourceEntry(
        source_id=_required_text(raw, "id", context=context),
        provider=_required_text(raw, "provider", context=context),
        profile=_required_text(raw, "profile", context=context),
        environment=environment,
        account_role=_required_text(raw, "account_role", context=context),
        authority=authority,
        writers=writers,
        runtime_paths=runtime_paths,
        freshness_max_age_hours=freshness,
        required=required,
        broker_access=broker_access,
        lifecycle=lifecycle,
        reconciliation_scope=scope,
        description=_required_text(raw, "description", context=context),
    )


def _choice(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    allowed: frozenset[str],
) -> str:
    value = _required_text(payload, key, context=context).lower()
    if value not in allowed:
        raise AccountSourceCatalogError(
            f"{context}.{key} must be one of {sorted(allowed)}."
        )
    return value


def _required_text(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    if key not in payload:
        raise AccountSourceCatalogError(f"{context}.{key} is required.")
    return _nonempty_text(payload[key], f"{context}.{key}")


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccountSourceCatalogError(f"{field} must be a non-empty string.")
    return value.strip()


def _runtime_path(value: str, *, field: str) -> PurePosixPath:
    path = _safe_relative_path(value, field=field)
    if not path.parts or path.parts[0] != "runtime":
        raise AccountSourceCatalogError(f"{field} must be beneath 'runtime/'.")
    return path


def _safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    if "\\" in value:
        raise AccountSourceCatalogError(f"{field} must use POSIX separators.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path == PurePosixPath("."):
        raise AccountSourceCatalogError(f"{field} must be a non-empty relative path.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AccountSourceCatalogError(
            f"{field} cannot contain empty, '.' or '..' segments."
        )
    return path
