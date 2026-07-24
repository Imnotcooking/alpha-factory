"""Atomic, pickle-free persistence for matrix preprocessing artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .artifact import (
    FittedMatrixPreprocessor,
    PreprocessingError,
    canonical_json_dumps,
)


def dump_preprocessor_json(
    artifact: FittedMatrixPreprocessor,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist authenticated state as inert JSON, never executable pickle."""

    if not isinstance(artifact, FittedMatrixPreprocessor):
        raise TypeError("artifact must be a FittedMatrixPreprocessor")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be boolean")
    artifact.require_integrity()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output,
        canonical_json_dumps(artifact.state_dict()) + "\n",
        overwrite=overwrite,
    )
    return output


def load_preprocessor_json(
    path: str | Path,
    *,
    expected_artifact_id: str,
    expected_artifact_sha256: str,
) -> FittedMatrixPreprocessor:
    """Load only against identity and digest obtained outside the file."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessingError("unable to read preprocessing artifact JSON") from exc
    if not isinstance(payload, Mapping):
        raise PreprocessingError("preprocessing artifact JSON must contain an object")
    return FittedMatrixPreprocessor.from_state_dict(
        payload,
        expected_artifact_id=expected_artifact_id,
        expected_artifact_sha256=expected_artifact_sha256,
    )


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Same-filesystem hard-link promotion is atomic and no-clobber.
            os.link(temporary, path)
            temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["dump_preprocessor_json", "load_preprocessor_json"]
