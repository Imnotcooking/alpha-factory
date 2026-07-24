from __future__ import annotations

import math

from apps.research_dashboard.run_ledger import format_run_ledger_label


def test_strategy_ledger_label_shows_only_requested_fields() -> None:
    label = format_run_ledger_label(
        {
            "name": (
                "Moving Average Crossover (20/60) / \u53cc\u5747\u7ebf\u4ea4\u53c9 "
                "\u00d7 Capped proportional score [screening_batch]"
            ),
            "market_vertical": "FUTURES_CN",
            "asset_class": "FUTURES",
            "round_number": 1,
            "holdout_ic": 0.012345,
        }
    )

    assert label == (
        "Moving Average Crossover (20/60) / \u53cc\u5747\u7ebf\u4ea4\u53c9\n"
        "FUTURES_CN | v1 | IC: 0.0123"
    )
    assert "Capped proportional" not in label
    assert "screening_batch" not in label


def test_strategy_ledger_label_handles_missing_ic() -> None:
    label = format_run_ledger_label(
        {
            "name": "Donchian Channel Breakout",
            "market_vertical": "",
            "asset_class": "FUTURES_CN",
            "round_number": 2.0,
            "holdout_ic": math.nan,
        }
    )

    assert label == "Donchian Channel Breakout\nFUTURES_CN | v2 | IC: N/A"
