from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDES = (
    "apps/README.md",
    "apps/research_dashboard/README.md",
    "apps/ops_dashboard/README.md",
    "departments/README.md",
    "departments/data_platform/README.md",
    "departments/middle_office/README.md",
    "departments/platform/README.md",
    "departments/research/README.md",
    "departments/risk/README.md",
    "departments/trading/README.md",
    "docs/README.md",
    "notebooks/README.md",
    "requirements/README.md",
    "runtime/README.md",
    "scripts/README.md",
    "src/README.md",
    "src/oqp/README.md",
    "tests/README.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_public_navigation_guides_exist_and_have_start_here() -> None:
    missing: list[str] = []
    incomplete: list[str] = []
    for relative_path in GUIDES:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            missing.append(relative_path)
            continue
        if "## Start Here" not in path.read_text(encoding="utf-8"):
            incomplete.append(relative_path)

    assert not missing, "Missing public navigation guides: " + ", ".join(missing)
    assert not incomplete, "Guides without a Start Here section: " + ", ".join(incomplete)


def test_public_navigation_links_resolve() -> None:
    broken: list[str] = []
    for relative_path in ("README.md", *GUIDES):
        guide = REPO_ROOT / relative_path
        text = guide.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            resolved = (guide.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{relative_path} -> {target}")

    assert not broken, "Broken documentation links:\n" + "\n".join(broken)


def test_architecture_uses_public_project_name() -> None:
    first_line = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "# Oxford Quant Pipeline Architecture"
