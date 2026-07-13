"""Hashed, run-scoped artifacts for the daily-regime study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from oqp.research.artifacts import (
    FileFingerprint,
    fingerprint_file,
    normalize_workspace_path,
    sha256_file,
    slugify,
)


MANIFEST_SCHEMA_VERSION = "daily_regime_run_manifest_v1"
VALID_RUN_STATUSES = ("started", "complete", "failed")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    name: str
    kind: str
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    project_id: str
    mode: str
    status: str
    config_hash: str
    seed: int
    created_at: str
    input_fingerprints: tuple[FileFingerprint, ...] = field(default_factory=tuple)
    artifacts: tuple[ArtifactRecord, ...] = field(default_factory=tuple)
    capabilities: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    git_commit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.project_id.strip():
            raise ValueError("run_id and project_id cannot be empty.")
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(f"Unknown run status: {self.status!r}")
        if len(self.config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_hash.lower()
        ):
            raise ValueError("config_hash must be a 64-character SHA-256 digest.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "mode": self.mode,
            "status": self.status,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "input_fingerprints": [asdict(item) for item in self.input_fingerprints],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "capabilities": [_jsonable(item) for item in self.capabilities],
            "metadata": _jsonable(self.metadata),
        }

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WrittenManifest:
    manifest: RunManifest
    record: ArtifactRecord


def build_run_id(
    project_id: str,
    config_hash: str,
    seed: int,
    *,
    label: str | None = None,
) -> str:
    """Build a deterministic run ID; callers add a label for intentional reruns."""

    if len(config_hash) < 12:
        raise ValueError("config_hash must contain at least 12 characters.")
    parts = [slugify(project_id, fallback="daily_regimes"), config_hash[:12], f"s{int(seed)}"]
    if label:
        parts.append(slugify(label, fallback="run"))
    return "_".join(parts)


class RunArtifactWriter:
    """Write immutable-by-default artifacts inside one run directory."""

    __slots__ = (
        "root_dir",
        "run_id",
        "mode",
        "workspace_root",
        "overwrite",
        "_records",
    )

    def __init__(
        self,
        root_dir: str | Path,
        *,
        run_id: str,
        mode: str,
        workspace_root: str | Path | None = None,
        overwrite: bool = False,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.run_id = slugify(run_id, fallback="run")
        normalized_mode = slugify(mode, fallback="synthetic")
        self.mode = "synthetic" if normalized_mode == "synthetic_smoke" else normalized_mode
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.overwrite = bool(overwrite)
        self._records: list[ArtifactRecord] = []

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.mode / self.run_id

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._records)

    def write_json(
        self,
        relative_path: str | Path,
        payload: Any,
        *,
        kind: str = "json",
        register: bool = True,
    ) -> ArtifactRecord:
        destination = self._destination(relative_path)
        content = (
            json.dumps(
                _jsonable(payload),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        self._atomic_write_text(destination, content)
        return self._record(destination, kind=kind, register=register)

    def write_text(
        self,
        relative_path: str | Path,
        content: str,
        *,
        kind: str = "text",
        register: bool = True,
    ) -> ArtifactRecord:
        if not isinstance(content, str):
            raise TypeError("content must be a string.")
        destination = self._destination(relative_path)
        self._atomic_write_text(destination, content)
        return self._record(destination, kind=kind, register=register)

    def write_frame(
        self,
        relative_path: str | Path,
        frame: pd.DataFrame,
        *,
        kind: str = "table",
    ) -> ArtifactRecord:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        destination = self._destination(relative_path)
        suffix = destination.suffix.lower()
        if suffix not in {".parquet", ".csv"}:
            raise ValueError("DataFrame artifacts must use .parquet or .csv.")
        self._prepare_destination(destination)
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
        try:
            if suffix == ".parquet":
                frame.to_parquet(temporary, index=False)
            else:
                frame.to_csv(temporary, index=False)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return self._record(destination, kind=kind, register=True)

    def register_existing(
        self,
        relative_path: str | Path,
        *,
        kind: str,
    ) -> ArtifactRecord:
        destination = self._destination(relative_path)
        if not destination.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {destination}")
        return self._record(destination, kind=kind, register=True)

    def finalize(
        self,
        *,
        project_id: str,
        config_hash: str,
        seed: int,
        status: str = "complete",
        input_paths: Iterable[str | Path] = (),
        capabilities: Iterable[Any] = (),
        git_commit: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        manifest_name: str = "manifest.json",
        created_at: str | None = None,
    ) -> WrittenManifest:
        input_fingerprints = tuple(
            fingerprint_file(
                path,
                include_hash=True,
                workspace_root=self.workspace_root,
            )
            for path in input_paths
        )
        manifest = RunManifest(
            run_id=self.run_id,
            project_id=project_id,
            mode=self.mode,
            status=status,
            config_hash=config_hash,
            seed=int(seed),
            created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            input_fingerprints=input_fingerprints,
            artifacts=self.records,
            capabilities=tuple(_capability_mapping(item) for item in capabilities),
            git_commit=git_commit,
            metadata=dict(metadata or {}),
        )
        record = self.write_json(
            manifest_name,
            manifest.to_dict(),
            kind="manifest",
            register=False,
        )
        return WrittenManifest(manifest=manifest, record=record)

    def _destination(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("Artifact path must be a non-empty relative path without '..'.")
        return self.run_dir / relative

    def _prepare_destination(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not self.overwrite:
            raise FileExistsError(
                f"Artifact already exists and overwrite is disabled: {destination}"
            )

    def _atomic_write_text(self, destination: Path, content: str) -> None:
        self._prepare_destination(destination)
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _record(
        self,
        destination: Path,
        *,
        kind: str,
        register: bool,
    ) -> ArtifactRecord:
        stat = destination.stat()
        record = ArtifactRecord(
            name=destination.name,
            kind=str(kind),
            path=normalize_workspace_path(destination, self.workspace_root),
            size_bytes=int(stat.st_size),
            sha256=sha256_file(destination),
        )
        if register:
            self._records = [item for item in self._records if item.path != record.path]
            self._records.append(record)
        return record


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _capability_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("Each capability must be a mapping or expose to_dict().")


__all__ = [
    "ArtifactRecord",
    "MANIFEST_SCHEMA_VERSION",
    "RunArtifactWriter",
    "RunManifest",
    "VALID_RUN_STATUSES",
    "WrittenManifest",
    "build_run_id",
]
