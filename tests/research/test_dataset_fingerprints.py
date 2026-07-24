from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from oqp.research.backtesting.evaluator import AlphaEvaluator
from oqp.research.dataset_fingerprints import (
    DatasetFingerprintError,
    attach_dataset_manifest_attrs,
    ensure_dataset_manifest_attrs,
    load_dataset_manifest,
    register_dataset_manifest,
    verify_dataset_manifest,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "ticker": ["rb", "cu"],
            "close": [3_100.0, 78_000.0],
        }
    )


def test_partition_order_produces_one_stable_manifest(tmp_path: Path) -> None:
    part_a = tmp_path / "part_a.csv"
    part_b = tmp_path / "part_b.csv"
    part_a.write_text("date,ticker,close\n2026-01-02,rb,3100\n", encoding="utf-8")
    part_b.write_text("date,ticker,close\n2026-01-03,cu,78000\n", encoding="utf-8")
    manifest_root = tmp_path / "manifests"

    first, first_path = register_dataset_manifest(
        [part_b, part_a],
        dataset_id="cn_futures_test",
        market_vertical="FUTURES_CN",
        data_frequency="daily",
        frame=_frame(),
        manifest_root=manifest_root,
        workspace_root=tmp_path,
    )
    second, second_path = register_dataset_manifest(
        [part_a, part_b],
        dataset_id="cn_futures_test",
        market_vertical="FUTURES_CN",
        data_frequency="daily",
        frame=_frame(),
        manifest_root=manifest_root,
        workspace_root=tmp_path,
    )

    assert first.aggregate_sha256 == second.aggregate_sha256
    assert first.content_sha256 == second.content_sha256
    assert first_path == second_path
    assert [item.path for item in first.source_files] == ["part_a.csv", "part_b.csv"]
    assert load_dataset_manifest(first_path) == first


def test_verification_detects_source_mutation(tmp_path: Path) -> None:
    source = tmp_path / "daily.csv"
    source.write_text("alpha", encoding="utf-8")
    manifest, path = register_dataset_manifest(
        source,
        dataset_id="mutation_test",
        market_vertical="FUTURES_CN",
        data_frequency="daily",
        manifest_root=tmp_path / "manifests",
        workspace_root=tmp_path,
    )
    assert verify_dataset_manifest(
        path, workspace_root=tmp_path
    ).verified

    source.write_text("bravo", encoding="utf-8")
    result = verify_dataset_manifest(
        manifest,
        workspace_root=tmp_path,
        strict=False,
    )
    assert not result.verified
    assert result.errors == ("content_changed:daily.csv",)
    with pytest.raises(DatasetFingerprintError, match="content_changed"):
        verify_dataset_manifest(manifest, workspace_root=tmp_path, strict=True)


def test_unchanged_source_reuses_cached_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "daily.csv"
    source.write_text("stable", encoding="utf-8")
    kwargs = {
        "dataset_id": "cache_test",
        "market_vertical": "FUTURES_CN",
        "data_frequency": "daily",
        "manifest_root": tmp_path / "manifests",
        "workspace_root": tmp_path,
    }
    register_dataset_manifest(source, **kwargs)

    def fail_if_rehashed(path: str | Path) -> str:
        raise AssertionError(f"Unexpected rehash of {path}")

    monkeypatch.setattr(
        "oqp.research.dataset_fingerprints.sha256_file",
        fail_if_rehashed,
    )
    register_dataset_manifest(source, **kwargs)


def test_frame_attrs_and_run_schema_retain_dataset_identity(tmp_path: Path) -> None:
    source = tmp_path / "daily.csv"
    source.write_text("date,ticker,close\n2026-01-02,rb,3100\n", encoding="utf-8")
    manifest, path = register_dataset_manifest(
        source,
        dataset_id="run_link_test",
        market_vertical="FUTURES_CN",
        data_frequency="daily",
        frame=_frame(),
        manifest_root=tmp_path / "manifests",
        workspace_root=tmp_path,
    )
    frame = attach_dataset_manifest_attrs(
        _frame(),
        manifest,
        path,
        verified=True,
        workspace_root=tmp_path,
    )
    frame.attrs["market_vertical"] = "FUTURES_CN"
    evaluator = AlphaEvaluator(
        db_path=tmp_path / "research.db",
        logs_dir=tmp_path / "artifacts",
        asset_class="FUTURES_CN",
    )
    metadata = evaluator._vertical_trial_metadata(
        frame,
        universe_size=2,
        traded_tickers="ALL",
    )

    assert metadata["dataset_id"] == "run_link_test"
    assert metadata["dataset_fingerprint"] == manifest.aggregate_sha256
    assert metadata["dataset_verified"] == 1
    with sqlite3.connect(tmp_path / "research.db") as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(backtest_runs)").fetchall()
        }
    assert {
        "dataset_version",
        "dataset_fingerprint",
        "dataset_content_sha256",
        "dataset_schema_sha256",
        "dataset_manifest_path",
        "dataset_verified",
    }.issubset(columns)


def test_evaluator_fallback_marks_missing_source_unverified() -> None:
    frame = _frame()
    result = ensure_dataset_manifest_attrs(
        frame,
        market_vertical="FUTURES_CN",
        strict=False,
    )
    assert result.attrs["dataset_verified"] is False
    assert result.attrs["dataset_fingerprint_status"] == "source_path_unavailable"
    with pytest.raises(DatasetFingerprintError, match="source_path"):
        ensure_dataset_manifest_attrs(
            _frame(),
            market_vertical="FUTURES_CN",
            strict=True,
        )
