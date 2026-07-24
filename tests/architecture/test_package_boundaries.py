from __future__ import annotations

import ast
from pathlib import Path

from oqp.config.paths import resolve_repo_root


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
PACKAGE_ROOT = REPO_ROOT / "src" / "oqp"

ACTIVE_PACKAGES = {
    "accounts",
    "brokers",
    "commands",
    "config",
    "contracts",
    "data",
    "demo",
    "domain",
    "execution",
    "investing",
    "journal",
    "market",
    "native",
    "ops",
    "options",
    "paper_trading",
    "portfolio",
    "qmt_connector",
    "research",
    "risk",
    "ui",
}
RETIRED_PACKAGES = {"artifacts", "intelligence", "storage", "utils"}


def _top_level_packages() -> set[str]:
    return {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _python_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend((REPO_ROOT / root).rglob("*.py"))
    return sorted(files)


def test_top_level_package_inventory_is_explicit() -> None:
    assert _top_level_packages() == ACTIVE_PACKAGES
    assert not (_top_level_packages() & RETIRED_PACKAGES)


def test_retired_packages_have_no_consumers() -> None:
    forbidden = tuple(f"oqp.{name}" for name in RETIRED_PACKAGES)
    offenders: list[str] = []
    for path in _python_files("src", "apps", "scripts", "tests", "departments"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {imported}")
    assert not offenders, "Retired package imports found:\n" + "\n".join(offenders)


def test_foundational_packages_do_not_import_ui() -> None:
    offenders: list[str] = []
    for package in ("config", "domain"):
        for path in (PACKAGE_ROOT / package).rglob("*.py"):
            for imported in _imports(path):
                if imported.startswith("oqp.ui"):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {imported}")
    assert not offenders, "Foundational UI imports found:\n" + "\n".join(offenders)


def test_repo_root_resolution_uses_project_markers_and_explicit_override() -> None:
    assert resolve_repo_root(start=REPO_ROOT / "apps" / "research_dashboard") == REPO_ROOT
    assert resolve_repo_root(configured_root=REPO_ROOT) == REPO_ROOT
