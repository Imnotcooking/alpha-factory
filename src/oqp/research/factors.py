"""Factor discovery utilities for the research strategy registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_FACTOR_ROOT = REPO_ROOT / "departments" / "research" / "factors"
RETIRED_FACTOR_ROOT = REPO_ROOT / "departments" / "research" / "retired_factors"
PUBLIC_EXAMPLE_ROOT = RETIRED_FACTOR_ROOT


def factor_search_roots(include_public_examples: bool = True) -> tuple[Path, ...]:
    """Return factor source directories in import precedence order."""

    roots: list[Path] = []
    if PRIVATE_FACTOR_ROOT.exists():
        roots.extend(
            path for path in sorted(PRIVATE_FACTOR_ROOT.iterdir()) if path.is_dir()
        )
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

    stem = Path(factor_name).stem
    for path in iter_factor_files(include_public_examples=include_public_examples):
        if path.stem == stem:
            return path
    search_paths = ", ".join(
        str(path) for path in factor_search_roots(include_public_examples)
    )
    raise ModuleNotFoundError(
        f"Could not resolve research factor module {stem!r} in: {search_paths}"
    )


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
    return module


__all__ = [
    "PRIVATE_FACTOR_ROOT",
    "PUBLIC_EXAMPLE_ROOT",
    "RETIRED_FACTOR_ROOT",
    "REPO_ROOT",
    "factor_search_roots",
    "iter_factor_files",
    "load_factor_module",
    "resolve_factor_path",
]
