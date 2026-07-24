from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"
if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from data_manager import DataManager  # noqa: E402
from views.tearsheet_view import TearSheetView  # noqa: E402


def test_scope_summary_returns_none_when_return_history_is_absent() -> None:
    summary = TearSheetView(None).test_scope_summary(
        pd.DataFrame(),
        pd.Series({"run_id": "run_without_returns"}),
        {},
    )

    assert summary is None


def test_scope_summary_accepts_timestamp_alias() -> None:
    summary = TearSheetView(None).test_scope_summary(
        pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-05 09:00:00",
                    "2026-01-06 09:00:00",
                ]
            }
        ),
        pd.Series({"run_id": "run_with_timestamp"}),
        {},
    )

    assert summary is not None
    assert summary["start"] == "2026-01-05"
    assert summary["end"] == "2026-01-06"


def test_data_manager_normalizes_timestamp_alias(tmp_path: Path) -> None:
    returns_path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-06", "2026-01-05"],
            "strategy_return": [0.02, -0.01],
        }
    ).to_csv(returns_path, index=False)

    returns = DataManager(db_path=str(tmp_path / "unused.db")).get_run_returns(
        "run_alias",
        returns_path=str(returns_path),
    )

    assert "date" in returns
    assert returns["date"].is_monotonic_increasing
    assert returns["net_return"].tolist() == [-0.01, 0.02]


def test_data_manager_rejects_undated_return_file_without_crashing(
    tmp_path: Path,
) -> None:
    returns_path = tmp_path / "returns.csv"
    pd.DataFrame({"strategy_return": [0.01, -0.01]}).to_csv(
        returns_path,
        index=False,
    )

    returns = DataManager(db_path=str(tmp_path / "unused.db")).get_run_returns(
        "run_without_dates",
        returns_path=str(returns_path),
    )

    assert returns.empty
