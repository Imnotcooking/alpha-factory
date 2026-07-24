from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oqp.accounts.source_catalog import (
    ACCOUNT_SOURCE_CATALOG_SCHEMA_VERSION,
    AccountSourceCatalogError,
    load_account_source_catalog,
)


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
CATALOG_PATH = REPO_ROOT / "departments" / "middle_office" / "account_sources.yaml"


def test_committed_account_source_catalog_covers_active_and_migration_lanes() -> None:
    catalog = load_account_source_catalog(CATALOG_PATH)
    by_id = {entry.source_id: entry for entry in catalog.sources}

    assert catalog.schema_version == ACCOUNT_SOURCE_CATALOG_SCHEMA_VERSION
    assert catalog.canonical_ledger.as_posix() == (
        "runtime/db/accounts/account_ledger.db"
    )
    assert set(by_id) == {
        "ibkr_live_readonly",
        "ibkr_paper_readonly",
        "qmt_live_readonly",
        "qmt_paper_readonly",
        "broker_csv_imports",
        "manual_external",
        "unified_live",
        "legacy_portfolio_ledger",
    }
    assert by_id["ibkr_live_readonly"].broker_access == "read_only"
    assert by_id["qmt_live_readonly"].lifecycle == "planned"
    assert by_id["legacy_portfolio_ledger"].lifecycle == "migration"
    assert len(by_id["legacy_portfolio_ledger"].writers) == 2
    assert all(
        (REPO_ROOT / writer).is_file()
        for entry in catalog.sources
        for writer in entry.writers
    )
    assert catalog.entry_for_profile("unified_live") == by_id["unified_live"]
    assert by_id["manual_external"].resolve_runtime_paths(REPO_ROOT)[0] == (
        REPO_ROOT / "runtime/state/portfolio/manual_external_holdings.json"
    )


def test_account_source_catalog_rejects_parent_traversal(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    payload["sources"][0]["writers"] = ["../unsafe.py"]
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(AccountSourceCatalogError, match="cannot contain"):
        load_account_source_catalog(path)


def test_account_source_catalog_rejects_non_runtime_state_path(
    tmp_path: Path,
) -> None:
    payload = _minimal_catalog()
    payload["sources"][0]["runtime_paths"] = ["private/account.db"]
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(AccountSourceCatalogError, match="beneath 'runtime/'"):
        load_account_source_catalog(path)


def test_account_source_catalog_rejects_duplicate_profiles(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    duplicate = dict(payload["sources"][0])
    duplicate["id"] = "second_source"
    payload["sources"].append(duplicate)
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(AccountSourceCatalogError, match="Duplicate account source profile"):
        load_account_source_catalog(path)


def _minimal_catalog() -> dict:
    return {
        "schema_version": ACCOUNT_SOURCE_CATALOG_SCHEMA_VERSION,
        "catalog_owner": "middle_office",
        "canonical_ledger": "runtime/db/accounts/account_ledger.db",
        "sources": [
            {
                "id": "example",
                "provider": "test",
                "profile": "test_readonly",
                "environment": "sim",
                "account_role": "test_account",
                "authority": "broker",
                "writers": ["scripts/test_writer.py"],
                "runtime_paths": ["runtime/db/accounts/test.db"],
                "freshness_max_age_hours": 1,
                "required": False,
                "broker_access": "none",
                "lifecycle": "planned",
                "reconciliation_scope": ["positions"],
                "description": "Test source.",
            }
        ],
    }


def _write_catalog(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "account_sources.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
