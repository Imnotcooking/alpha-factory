"""Factor discovery utilities for the research strategy registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_FACTOR_ROOT = REPO_ROOT / "departments" / "research" / "factors"
PRIVATE_FACTOR_ALIAS_FILE = PRIVATE_FACTOR_ROOT / "stable_ids.yaml"
RETIRED_FACTOR_ROOT = REPO_ROOT / "departments" / "research" / "retired_factors"
PUBLIC_EXAMPLE_ROOT = RETIRED_FACTOR_ROOT


def factor_search_roots(include_public_examples: bool = True) -> tuple[Path, ...]:
    """Return factor source directories in import precedence order."""

    roots: list[Path] = []
    if PRIVATE_FACTOR_ROOT.exists():
        roots.append(PRIVATE_FACTOR_ROOT)
    if include_public_examples and RETIRED_FACTOR_ROOT.exists():
        roots.append(RETIRED_FACTOR_ROOT)
    return tuple(roots)


def iter_factor_files(include_public_examples: bool = True) -> tuple[Path, ...]:
    """Return unique factor recipe files discovered across registry roots."""

    seen: set[str] = set()
    files: list[Path] = []
    for root in factor_search_roots(include_public_examples=include_public_examples):
        for path in sorted(root.glob("fac_*.py")):
            if path.name in seen:
                continue
            seen.add(path.name)
            files.append(path)
    return tuple(files)


def resolve_factor_path(
    factor_name: str,
    *,
    include_public_examples: bool = True,
) -> Path:
    """Resolve a factor module name or path stem to a source file."""

    stem = canonical_factor_id(factor_name)
    for path in iter_factor_files(include_public_examples=include_public_examples):
        if path.stem == stem:
            return path
    search_paths = ", ".join(
        str(path) for path in factor_search_roots(include_public_examples)
    )
    raise ModuleNotFoundError(
        f"Could not resolve research factor module {stem!r} in: {search_paths}"
    )


def factor_id_aliases() -> dict[str, str]:
    """Return legacy-to-canonical IDs from the private stable-ID ledger."""

    if not PRIVATE_FACTOR_ALIAS_FILE.exists():
        return {}
    payload = yaml.safe_load(PRIVATE_FACTOR_ALIAS_FILE.read_text(encoding="utf-8")) or {}
    aliases = payload.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError("stable_ids.yaml aliases must be a mapping")
    normalized: dict[str, str] = {}
    for legacy_id, canonical_id in aliases.items():
        legacy = Path(str(legacy_id)).stem.strip()
        canonical = Path(str(canonical_id)).stem.strip()
        if not legacy or not canonical:
            raise ValueError("factor aliases cannot contain empty IDs")
        if legacy == canonical:
            raise ValueError(f"factor alias {legacy!r} points to itself")
        normalized[legacy] = canonical
    return normalized


def canonical_factor_id(factor_name: str) -> str:
    stem = Path(factor_name).stem
    aliases = factor_id_aliases()
    seen: set[str] = set()
    while stem in aliases:
        if stem in seen:
            raise ValueError(f"factor alias cycle detected at {stem!r}")
        seen.add(stem)
        stem = aliases[stem]
    return stem


def load_factor_module(
    factor_name: str,
    *,
    module_prefix: str = "oqp_private_factor",
    include_public_examples: bool = True,
) -> ModuleType:
    """Load a factor recipe from the strategy registry without a legacy package."""

    path = resolve_factor_path(
        factor_name, include_public_examples=include_public_examples
    )
    module_name = f"{module_prefix}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load factor module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "FACTOR_PARAMETERS", None) is not None:
        from oqp.research.parameter_schema import resolve_factor_parameter_schema

        resolve_factor_parameter_schema(module)
    return module


__all__ = [
    "PRIVATE_FACTOR_ROOT",
    "PRIVATE_FACTOR_ALIAS_FILE",
    "PUBLIC_EXAMPLE_ROOT",
    "RETIRED_FACTOR_ROOT",
    "REPO_ROOT",
    "factor_search_roots",
    "factor_id_aliases",
    "canonical_factor_id",
    "iter_factor_files",
    "load_factor_module",
    "resolve_factor_path",
]
