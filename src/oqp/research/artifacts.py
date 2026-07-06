from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    exists: bool
    size_bytes: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class StoredArtifact:
    artifact_id: str
    path: str
    size_bytes: int
    sha256: str


def slugify(value: str, fallback: str = "model") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-").lower()
    return slug or fallback


def normalize_workspace_path(path: str | Path, workspace_root: str | Path | None = None) -> str:
    raw = Path(path)
    root = Path(workspace_root or Path.cwd()).resolve()
    try:
        return raw.resolve().relative_to(root).as_posix()
    except ValueError:
        return raw.as_posix()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(
    path: str | Path,
    *,
    include_hash: bool = True,
    workspace_root: str | Path | None = None,
) -> FileFingerprint:
    file_path = Path(path)
    if not file_path.exists():
        return FileFingerprint(path=normalize_workspace_path(file_path, workspace_root), exists=False)

    stat = file_path.stat()
    return FileFingerprint(
        path=normalize_workspace_path(file_path, workspace_root),
        exists=True,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        sha256=sha256_file(file_path) if include_hash else None,
    )


class ModelArtifactStore:
    """
    Filesystem store for trained ML artifacts.

    Source code can keep writing legacy paths during migration, while this store
    creates a versioned copy under runtime/artifacts for auditing.
    """

    def __init__(
        self,
        root_dir: str | Path = "runtime/artifacts/research/model_artifacts",
        workspace_root: str | Path | None = None,
    ):
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_absolute():
            self.root_dir = self.workspace_root / self.root_dir

    def reserve_artifact_id(self, model_name: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"{slugify(model_name)}_{timestamp}_{uuid4().hex[:8]}"

    def artifact_dir(self, model_name: str, artifact_id: str | None = None) -> Path:
        artifact_id = artifact_id or self.reserve_artifact_id(model_name)
        return self.root_dir / slugify(model_name) / artifact_id

    def archive_file(
        self,
        source_path: str | Path,
        *,
        model_name: str,
        artifact_id: str | None = None,
        filename: str | None = None,
    ) -> StoredArtifact:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {source}")

        artifact_id = artifact_id or self.reserve_artifact_id(model_name)
        output_dir = self.artifact_dir(model_name, artifact_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        destination = output_dir / (filename or source.name)
        shutil.copy2(source, destination)
        stat = destination.stat()
        return StoredArtifact(
            artifact_id=artifact_id,
            path=normalize_workspace_path(destination, self.workspace_root),
            size_bytes=int(stat.st_size),
            sha256=sha256_file(destination),
        )
