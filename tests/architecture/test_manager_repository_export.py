from __future__ import annotations

from pathlib import Path

import pytest

from scripts.platform.export_manager_repository import (
    export_repository,
    is_exportable,
)


def test_manager_boundary_includes_research_and_excludes_operations() -> None:
    assert is_exportable(
        Path("departments/research/factors/fac_001_private.py")
    )
    assert is_exportable(
        Path("apps/research_dashboard/pages/08_Research_Review.py")
    )
    assert not is_exportable(
        Path("departments/middle_office/account_snapshot_contract.md")
    )
    assert not is_exportable(Path("apps/ops_dashboard/Homepage.py"))
    assert not is_exportable(Path("runtime/db/research.db"))
    assert not is_exportable(Path("data/vendor/contracts.parquet"))
    assert not is_exportable(Path(".env"))


def test_manager_export_preserves_git_and_copies_private_research(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "manager"
    (source / "apps/research_dashboard").mkdir(parents=True)
    (source / "departments/research/factors").mkdir(parents=True)
    (source / "departments/middle_office").mkdir(parents=True)
    (source / "runtime").mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n")
    (source / "README.md").write_text("source\n")
    (source / "apps/research_dashboard/app.py").write_text("APP = True\n")
    (source / "departments/research/factors/fac_001.py").write_text(
        "FACTOR_ID = 'fac_001'\n"
    )
    (source / "departments/middle_office/private.py").write_text("SECRET = 1\n")
    (source / "runtime/local.db").write_text("runtime\n")

    (destination / ".git").mkdir(parents=True)
    (destination / "old.txt").write_text("old\n")
    copied, skipped = export_repository(
        destination,
        source=source,
        clean=True,
    )

    assert copied >= 3
    assert skipped == 0
    assert (destination / ".git").is_dir()
    assert not (destination / "old.txt").exists()
    assert (
        destination / "departments/research/factors/fac_001.py"
    ).is_file()
    assert not (destination / "departments/middle_office").exists()
    assert not (destination / "runtime").exists()
    assert (destination / "MANAGER_REPOSITORY_BOUNDARY.md").is_file()


def test_manager_export_refuses_to_clean_non_git_folder(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "manager"
    source.mkdir()
    destination.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n")
    (destination / "unrelated.txt").write_text("keep\n")

    with pytest.raises(ValueError, match="existing Git repository"):
        export_repository(destination, source=source, clean=True)
