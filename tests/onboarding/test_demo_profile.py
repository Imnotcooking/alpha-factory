from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from oqp.accounts import load_account_nav_history, load_latest_account_positions
from oqp.demo.profile import DEMO_PROFILE, demo_environment, demo_paths, read_profile_marker
from oqp.demo.seed import seed_demo_profile


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "oqp").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return root


def test_demo_seed_is_complete_and_idempotent(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    first = seed_demo_profile(root, as_of="2026-07-17")
    first_manifest = first.paths.seed_manifest.read_bytes()
    second = seed_demo_profile(root, as_of="2026-07-17")

    assert first_manifest == second.paths.seed_manifest.read_bytes()
    assert first.research_runs == 5
    assert first.account_snapshots == 105
    assert first.option_contracts == 40
    assert read_profile_marker(root)["profile"] == DEMO_PROFILE

    daily = pd.read_parquet(
        first.paths.data_root / "futures_cn" / "daily" / "demo_futures_cn_daily.parquet"
    )
    chain = pd.read_parquet(
        first.paths.data_root / "options_us" / "api_cache" / "demo_options_us_chain.parquet"
    )
    assert daily["ticker"].nunique() == 8
    assert {"date", "ticker", "open", "high", "low", "close", "is_fresh"}.issubset(daily.columns)
    assert chain["underlying_symbol"].nunique() == 2
    assert set(chain["right"]) == {"call", "put"}

    with sqlite3.connect(first.paths.research_db) as conn:
        run_count = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
        factor_count = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    assert run_count == 5
    assert factor_count == 3

    nav = load_account_nav_history(first.paths.account_ledger, environment="live")
    positions = load_latest_account_positions(first.paths.account_ledger, environment="live")
    assert len(nav) == 60
    assert "OPTIONS_US" in set(positions["asset_class"])


def test_demo_environment_points_only_at_demo_runtime(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    paths = demo_paths(root)
    env = demo_environment(root, base={"PATH": "/usr/bin"})

    assert env["OQP_PROFILE"] == DEMO_PROFILE
    assert env["OQP_ACCOUNT_LEDGER_PATH"] == str(paths.account_ledger)
    assert env["ALPHA_RESEARCH_DB_PATH"] == str(paths.research_db)
    assert env["ALPHA_RUNTIME_DATA_ROOT"] == str(paths.data_root)
    assert str(root / "runtime" / "db") not in env["OQP_ACCOUNT_LEDGER_PATH"]


def test_seed_manifest_records_checksums(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    result = seed_demo_profile(root, as_of="2026-07-17")
    payload = json.loads(result.paths.seed_manifest.read_text(encoding="utf-8"))

    assert payload["profile"] == DEMO_PROFILE
    assert payload["seed"] == result.seed
    assert payload["files"]
    assert all(len(item["sha256"]) == 64 for item in payload["files"])
