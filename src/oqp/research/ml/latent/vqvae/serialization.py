"""Canonical non-pickle VQ-VAE bundles with strict integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import VQVAEConfig
from .model import (
    FittedVQVAE,
    VQTrainingEpoch,
    VQVAE_CORE_VERSION,
    VQVAEError,
)
from .network import require_torch


VQVAE_ARTIFACT_VERSION = "oqp_vqvae_bundle_v1"
MANIFEST_FILENAME = "manifest.json"
PARAMETERS_FILENAME = "parameters.npz"


def save_vqvae_bundle(
    fitted: FittedVQVAE,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    if not isinstance(fitted, FittedVQVAE):
        raise TypeError("fitted must be a FittedVQVAE")
    if overwrite:
        raise VQVAEError("VQ-VAE bundles are immutable; save a new versioned directory")
    fitted.require_parameter_integrity()
    root = Path(directory)
    if os.path.lexists(root):
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        _write_bundle(fitted, staging)
        _promote_new_bundle(staging=staging, destination=root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def _write_bundle(fitted: FittedVQVAE, root: Path) -> None:
    manifest_path = root / MANIFEST_FILENAME
    parameters_path = root / PARAMETERS_FILENAME
    arrays = fitted.parameter_arrays()
    records: dict[str, dict[str, Any]] = {}
    stored: dict[str, np.ndarray] = {}
    for index, (name, array) in enumerate(sorted(arrays.items())):
        storage_key = f"tensor_{index:04d}"
        contiguous = np.ascontiguousarray(array)
        stored[storage_key] = contiguous
        records[name] = {
            "storage_key": storage_key,
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
        }
    np.savez_compressed(parameters_path, **stored)
    manifest = {
        "artifact_version": VQVAE_ARTIFACT_VERSION,
        "core_version": fitted.version,
        "model_id": fitted.model_id,
        "parameter_hash": fitted.parameter_hash,
        "training_rows_hash": fitted.training_rows_hash,
        "feature_names": list(fitted.feature_names),
        "config": fitted.config.state_dict(),
        "parameters_file": PARAMETERS_FILENAME,
        "tensors": records,
        "history": [
            {
                "epoch": item.epoch,
                "loss": item.loss,
                "reconstruction_loss": item.reconstruction_loss,
                "codebook_loss": item.codebook_loss,
                "commitment_loss": item.commitment_loss,
                "active_codes": item.active_codes,
            }
            for item in fitted.history
        ],
    }
    manifest["bundle_hash"] = _manifest_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _promote_new_bundle(*, staging: Path, destination: Path) -> None:
    """Atomically publish one immutable, previously absent bundle directory."""

    if os.path.lexists(destination):
        raise FileExistsError(destination)
    try:
        os.rename(staging, destination)
    except OSError as exc:
        if os.path.lexists(destination):
            raise FileExistsError(destination) from exc
        raise


def load_vqvae_bundle(
    directory: str | Path,
    *,
    expected_model_id: str,
    expected_parameter_hash: str,
    expected_bundle_hash: str,
) -> FittedVQVAE:
    """Load a bundle against identity and digests from an independent registry."""

    require_torch()
    if type(expected_model_id) is not str or not expected_model_id:
        raise VQVAEError("expected_model_id must be a non-empty string")
    if not _is_sha256(expected_parameter_hash):
        raise VQVAEError("expected_parameter_hash must be a lowercase SHA-256")
    if not _is_sha256(expected_bundle_hash):
        raise VQVAEError("expected_bundle_hash must be a lowercase SHA-256")
    root = Path(directory)
    manifest_path = root / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VQVAEError("unable to read VQ-VAE manifest") from exc
    if not isinstance(manifest, Mapping):
        raise VQVAEError("VQ-VAE manifest must be an object")
    required = {
        "artifact_version",
        "core_version",
        "model_id",
        "parameter_hash",
        "training_rows_hash",
        "feature_names",
        "config",
        "parameters_file",
        "tensors",
        "history",
        "bundle_hash",
    }
    if set(manifest) != required:
        raise VQVAEError("VQ-VAE manifest has unknown or missing fields")
    supplied_bundle_hash = manifest["bundle_hash"]
    if not _is_sha256(supplied_bundle_hash):
        raise VQVAEError("VQ-VAE manifest bundle hash is invalid")
    if supplied_bundle_hash != expected_bundle_hash:
        raise VQVAEError("VQ-VAE bundle differs from the independently trusted hash")
    hash_payload = dict(manifest)
    del hash_payload["bundle_hash"]
    if _manifest_hash(hash_payload) != supplied_bundle_hash:
        raise VQVAEError("VQ-VAE manifest bundle hash mismatch")
    if manifest["artifact_version"] != VQVAE_ARTIFACT_VERSION:
        raise VQVAEError("unsupported VQ-VAE artifact version")
    if manifest["core_version"] != VQVAE_CORE_VERSION:
        raise VQVAEError("unsupported VQ-VAE core version")
    if manifest["parameters_file"] != PARAMETERS_FILENAME:
        raise VQVAEError("VQ-VAE parameters path is not canonical")
    if manifest["model_id"] != expected_model_id:
        raise VQVAEError("VQ-VAE model_id differs from expected_model_id")
    if manifest["parameter_hash"] != expected_parameter_hash:
        raise VQVAEError("VQ-VAE parameter hash differs from the trusted hash")
    try:
        config = VQVAEConfig.from_state_dict(manifest["config"])
    except (TypeError, ValueError) as exc:
        raise VQVAEError("invalid VQ-VAE configuration manifest") from exc
    feature_names_raw = manifest["feature_names"]
    if not isinstance(feature_names_raw, list) or any(
        type(value) is not str for value in feature_names_raw
    ):
        raise VQVAEError("VQ-VAE feature_names must be a string array")
    feature_names = tuple(feature_names_raw)
    tensors = manifest["tensors"]
    if not isinstance(tensors, Mapping) or not tensors:
        raise VQVAEError("VQ-VAE tensor manifest is missing")
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(root / PARAMETERS_FILENAME, allow_pickle=False) as archive:
            expected_storage: set[str] = set()
            for name, raw_record in tensors.items():
                if type(name) is not str or not isinstance(raw_record, Mapping):
                    raise VQVAEError(f"invalid tensor record for {name}")
                if set(raw_record) != {"storage_key", "shape", "dtype"}:
                    raise VQVAEError(f"invalid tensor fields for {name}")
                key = raw_record["storage_key"]
                shape_raw = raw_record["shape"]
                dtype_raw = raw_record["dtype"]
                if type(key) is not str or type(dtype_raw) is not str:
                    raise VQVAEError(f"invalid tensor metadata for {name}")
                if not isinstance(shape_raw, list) or any(
                    type(value) is not int or value < 0 for value in shape_raw
                ):
                    raise VQVAEError(f"invalid tensor shape for {name}")
                expected_storage.add(key)
                if key not in archive.files:
                    raise VQVAEError(f"missing stored tensor for {name}")
                observed = np.asarray(archive[key])
                shape = tuple(shape_raw)
                if observed.shape != shape or str(observed.dtype) != dtype_raw:
                    raise VQVAEError(f"tensor geometry mismatch for {name}")
                arrays[name] = observed.copy()
            if set(archive.files) != expected_storage:
                raise VQVAEError("VQ-VAE parameter archive has unexpected tensors")
    except (OSError, ValueError) as exc:
        if isinstance(exc, VQVAEError):
            raise
        raise VQVAEError("unable to read VQ-VAE parameter archive") from exc

    history_raw = manifest["history"]
    if not isinstance(history_raw, list):
        raise VQVAEError("VQ-VAE history must be an array")
    try:
        history = tuple(_parse_history_item(item) for item in history_raw)
        return FittedVQVAE(
            config=config,
            feature_names=feature_names,
            _parameter_tensors=tuple(sorted(arrays.items())),
            model_id=manifest["model_id"],
            parameter_hash=manifest["parameter_hash"],
            training_rows_hash=manifest["training_rows_hash"],
            history=history,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, VQVAEError):
            raise
        raise VQVAEError("invalid VQ-VAE lineage or training history") from exc


def _parse_history_item(item: Any) -> VQTrainingEpoch:
    if not isinstance(item, Mapping):
        raise TypeError("history entries must be objects")
    expected = {
        "epoch",
        "loss",
        "reconstruction_loss",
        "codebook_loss",
        "commitment_loss",
        "active_codes",
    }
    if set(item) != expected:
        raise ValueError("history entry has unknown or missing fields")
    return VQTrainingEpoch(
        epoch=item["epoch"],
        loss=item["loss"],
        reconstruction_loss=item["reconstruction_loss"],
        codebook_loss=item["codebook_loss"],
        commitment_loss=item["commitment_loss"],
        active_codes=item["active_codes"],
    )


def _manifest_hash(manifest_without_hash: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest_without_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "MANIFEST_FILENAME",
    "PARAMETERS_FILENAME",
    "VQVAE_ARTIFACT_VERSION",
    "load_vqvae_bundle",
    "save_vqvae_bundle",
]
