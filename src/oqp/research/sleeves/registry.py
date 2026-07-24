"""Discovery and loading for reusable private sleeve recipes."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import ModuleType

import yaml

from oqp.research.factor_purity import factor_implementation_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[4]
PRIVATE_SLEEVE_ROOT = (
    REPO_ROOT / "departments" / "research" / "strategies" / "sleeves"
)
PRIVATE_SLEEVE_STABLE_ID_FILE = PRIVATE_SLEEVE_ROOT / "stable_ids.yaml"


def iter_sleeve_files() -> tuple[Path, ...]:
    if not PRIVATE_SLEEVE_ROOT.exists():
        return ()
    return tuple(sorted(PRIVATE_SLEEVE_ROOT.glob("slv_*.py")))


def resolve_sleeve_path(sleeve_name: str) -> Path:
    stem = Path(sleeve_name).stem
    for path in iter_sleeve_files():
        if path.stem == stem:
            return path
    raise ModuleNotFoundError(
        f"Could not resolve sleeve module {stem!r} in {PRIVATE_SLEEVE_ROOT}"
    )


def load_sleeve_module(sleeve_name: str) -> ModuleType:
    path = resolve_sleeve_path(sleeve_name)
    spec = importlib.util.spec_from_file_location(
        f"oqp_private_sleeve_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sleeve module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_config", None)):
        raise ValueError(f"{path.stem} must expose build_config()")
    sleeve_id = str(getattr(module, "SLEEVE_ID", "")).strip()
    if sleeve_id != path.stem:
        raise ValueError(
            f"{path.name} SLEEVE_ID {sleeve_id!r} does not match its filename"
        )
    registered = load_registered_sleeve_ids()
    if registered and sleeve_id not in registered:
        raise ValueError(f"{sleeve_id} is absent from {PRIVATE_SLEEVE_STABLE_ID_FILE}")
    module.SLEEVE_IMPLEMENTATION_FINGERPRINT = (
        sleeve_implementation_fingerprint(path)
    )
    module.SLEEVE_DEFINITION_FINGERPRINT = sleeve_definition_fingerprint(
        path,
        module,
    )
    return module


def load_registered_sleeve_ids() -> tuple[str, ...]:
    """Return immutable canonical sleeve IDs from the private stable-ID ledger."""

    if not PRIVATE_SLEEVE_STABLE_ID_FILE.exists():
        return ()
    payload = (
        yaml.safe_load(
            PRIVATE_SLEEVE_STABLE_ID_FILE.read_text(encoding="utf-8")
        )
        or {}
    )
    values = tuple(str(value) for value in payload.get("registered_ids", []))
    if len(values) != len(set(values)):
        raise ValueError("sleeve stable-ID ledger contains duplicate IDs")
    return values


def sleeve_implementation_fingerprint(path: Path) -> str:
    """Hash the sleeve definition and directly imported repository helpers."""

    return factor_implementation_fingerprint(path)


def sleeve_definition_fingerprint(
    path: Path,
    module: ModuleType,
) -> str:
    """Hash sleeve declarations together with implementation source."""

    payload = {
        "sleeve_id": str(getattr(module, "SLEEVE_ID", "")),
        "metadata": getattr(module, "SLEEVE_METADATA", {}) or {},
        "contract": getattr(module, "SLEEVE_CONTRACT", {}) or {},
        "parameters": getattr(module, "SLEEVE_PARAMETERS", {}) or {},
        "implementation_fingerprint": sleeve_implementation_fingerprint(path),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PRIVATE_SLEEVE_ROOT",
    "PRIVATE_SLEEVE_STABLE_ID_FILE",
    "REPO_ROOT",
    "iter_sleeve_files",
    "load_sleeve_module",
    "load_registered_sleeve_ids",
    "resolve_sleeve_path",
    "sleeve_definition_fingerprint",
    "sleeve_implementation_fingerprint",
]
