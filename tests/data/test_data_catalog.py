from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oqp.data.catalog import (
    CATALOG_SCHEMA_VERSION,
    DataCatalogError,
    load_data_catalog,
)


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
CATALOG_PATH = REPO_ROOT / "departments" / "data_platform" / "source_catalog.yaml"


def test_committed_data_catalog_covers_canonical_runtime_lanes() -> None:
    catalog = load_data_catalog(CATALOG_PATH)
    by_id = {entry.dataset_id: entry for entry in catalog.datasets}

    assert catalog.schema_version == CATALOG_SCHEMA_VERSION
    assert catalog.data_root.as_posix() == "runtime/data"
    assert set(by_id) == {
        "core_feature_store",
        "core_regime",
        "core_metadata",
        "core_universes",
        "futures_cn_daily",
        "futures_cn_intraday",
        "futures_cn_tick",
        "equity_cn_daily",
        "equity_cn_intraday",
        "equity_cn_tick",
        "options_cn_daily",
        "options_cn_tick",
        "equity_us_api_cache",
        "options_us_api_cache",
        "equity_us_api_materialized",
        "options_us_api_materialized",
    }
    assert by_id["futures_cn_daily"].required is True
    assert by_id["futures_cn_tick"].freshness_max_age_days == 14
    assert by_id["options_us_api_cache"].providers == ("massive",)
    assert (
        by_id["equity_us_api_materialized"].storage_role
        == "immutable_research_dataset"
    )
    assert (
        by_id["options_us_api_materialized"].storage_role
        == "immutable_vendor_snapshot"
    )
    assert by_id["equity_cn_daily"].resolve("/tmp/runtime/data") == Path(
        "/tmp/runtime/data/equity_cn/daily"
    )


def test_data_catalog_rejects_parent_traversal(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    payload["datasets"][0]["relative_path"] = "../secrets"
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(DataCatalogError, match="cannot contain"):
        load_data_catalog(path)


def test_data_catalog_rejects_duplicate_runtime_paths(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    duplicate = dict(payload["datasets"][0])
    duplicate["id"] = "second_dataset"
    payload["datasets"].append(duplicate)
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(DataCatalogError, match="Duplicate dataset relative_path"):
        load_data_catalog(path)


def test_data_catalog_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = _minimal_catalog()
    payload["schema_version"] = "unknown"
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(DataCatalogError, match="Unsupported data catalog"):
        load_data_catalog(path)


def _minimal_catalog() -> dict:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_owner": "data_platform",
        "data_root": "runtime/data",
        "datasets": [
            {
                "id": "example",
                "asset_class": "CORE",
                "timeframe": "metadata",
                "relative_path": "metadata",
                "storage_role": "reference_metadata",
                "providers": ["local_pipeline"],
                "required": False,
                "freshness_max_age_days": 30,
                "update_mode": "generated",
                "owner": "data_platform",
                "description": "Test entry.",
            }
        ],
    }


def _write_catalog(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
