"""Pickle-free serialization helpers for canonical shared regime models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically and reject non-finite numbers."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON representation of ``value``."""

    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def dump_fitted_hmm_json(
    model: "FittedDiagonalHMM",
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a fitted HMM as auditable JSON rather than executable pickle."""

    from .fitted import FittedDiagonalHMM

    if not isinstance(model, FittedDiagonalHMM):
        raise TypeError("model must be a FittedDiagonalHMM")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output,
        canonical_json_dumps(model.state_dict()) + "\n",
        overwrite=overwrite,
    )
    return output


def load_fitted_hmm_json(
    path: str | Path,
    *,
    expected_model_id: str,
    expected_parameter_sha256: str,
) -> "FittedDiagonalHMM":
    """Load and authenticate a fitted HMM from canonical JSON."""

    from .fitted import FittedDiagonalHMM

    payload = _read_json_mapping(path)
    return FittedDiagonalHMM.from_state_dict(
        payload,
        expected_model_id=expected_model_id,
        expected_parameter_sha256=expected_parameter_sha256,
    )


def dump_filter_checkpoint_json(
    checkpoint: "FilterCheckpoint",
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an authenticated causal continuation state as JSON."""

    from .filtering import FilterCheckpoint

    if not isinstance(checkpoint, FilterCheckpoint):
        raise TypeError("checkpoint must be a FilterCheckpoint")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output,
        canonical_json_dumps(checkpoint.state_dict()) + "\n",
        overwrite=overwrite,
    )
    return output


def load_filter_checkpoint_json(
    path: str | Path,
    *,
    expected_model_id: str,
    expected_parameter_sha256: str,
    expected_entity_id: str,
    expected_checkpoint_sha256: str,
) -> "FilterCheckpoint":
    """Load a checkpoint only for an independently authenticated model."""

    from .filtering import FilterCheckpoint

    payload = _read_json_mapping(path)
    return FilterCheckpoint.from_state_dict(
        payload,
        expected_model_id=expected_model_id,
        expected_parameter_sha256=expected_parameter_sha256,
        expected_entity_id=expected_entity_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )


def _read_json_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("serialized regime artifact must contain a JSON object")
    return payload


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
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
            # Linking is an atomic no-clobber promotion on the same filesystem.
            # A competing writer wins with EEXIST rather than being replaced.
            os.link(temporary, path)
            temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if TYPE_CHECKING:  # pragma: no cover
    from .filtering import FilterCheckpoint
    from .fitted import FittedDiagonalHMM


__all__ = [
    "canonical_json_dumps",
    "dump_filter_checkpoint_json",
    "dump_fitted_hmm_json",
    "load_filter_checkpoint_json",
    "load_fitted_hmm_json",
    "sha256_json",
]
