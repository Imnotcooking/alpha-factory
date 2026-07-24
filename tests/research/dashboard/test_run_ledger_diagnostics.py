from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"
if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from data_manager import DataManager  # noqa: E402


def test_run_ledger_aggregates_multiple_diagnostics_to_one_row(tmp_path) -> None:
    db_path = tmp_path / "research.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE factors (
                factor_id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT
            );
            CREATE TABLE backtest_runs (
                run_id TEXT PRIMARY KEY,
                factor_id TEXT,
                round_number INTEGER,
                holdout_ic REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE diagnostics (
                diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                failure_code TEXT,
                suggested_action TEXT
            );
            INSERT INTO factors VALUES ('str_001', 'Screened strategy', 'Screen');
            INSERT INTO backtest_runs (
                run_id, factor_id, round_number, holdout_ic
            ) VALUES ('run_001', 'str_001', 1, 0.01);
            INSERT INTO diagnostics (
                run_id, failure_code, suggested_action
            ) VALUES
                ('run_001', 'proxy_data', 'Validate tradable contracts.'),
                ('run_001', 'cost_bleed', 'Reduce turnover.');
            """
        )

    runs = DataManager(db_path=str(db_path)).get_all_runs()

    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == "run_001"
    assert runs.iloc[0]["failure_code"] == "proxy_data; cost_bleed"
    assert (
        runs.iloc[0]["suggested_action"]
        == "Validate tradable contracts. | Reduce turnover."
    )


def test_run_ledger_separates_completed_and_preflight_blocked(tmp_path) -> None:
    db_path = tmp_path / "research.db"
    returns_path = tmp_path / "completed_returns.csv"
    returns_path.write_text(
        "date,net_return\n2026-01-05,0.001\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE factors (
                factor_id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT
            );
            CREATE TABLE backtest_runs (
                run_id TEXT PRIMARY KEY,
                factor_id TEXT,
                round_number INTEGER,
                holdout_ic REAL,
                returns_file_path TEXT,
                data_execution_reality TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE diagnostics (
                diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                failure_code TEXT,
                suggested_action TEXT
            );
            INSERT INTO factors VALUES
                ('str_completed', 'Completed strategy', 'Screen'),
                ('str_blocked', 'Blocked strategy', 'Screen');
            INSERT INTO backtest_runs (
                run_id, factor_id, round_number, holdout_ic,
                returns_file_path, data_execution_reality
            ) VALUES
                ('run_completed', 'str_completed', 1, 0.01, '__RETURNS__', 'proxy_data'),
                ('run_blocked', 'str_blocked', 1, NULL, NULL, 'preflight_blocked');
            INSERT INTO diagnostics (
                run_id, failure_code, suggested_action
            ) VALUES
                ('run_completed', 'crisis_failure', 'Review crisis behavior.'),
                ('run_blocked', 'missing_required_fields', 'Load the required data.');
            """.replace("__RETURNS__", str(returns_path))
        )

    runs = DataManager(db_path=str(db_path)).get_all_runs()
    status_by_run = runs.set_index("run_id")["execution_status"].to_dict()

    assert status_by_run == {
        "run_completed": "completed",
        "run_blocked": "blocked",
    }
