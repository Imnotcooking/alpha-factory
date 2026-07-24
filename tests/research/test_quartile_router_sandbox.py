from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oqp.data.futures_cn_names import format_futures_cn_product_zh
from oqp.research.quartile_router_sandbox import (
    PAPER_ASSIGNMENTS,
    QuartileRouterConfig,
    build_quartile_router,
)


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
REAL_ARTIFACT_ROOT = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "cn_futures_daily_volatility_router_replication"
)
REAL_EXTENSION_ROOT = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "cn_futures_product_state_ema_router"
)


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = ["2021-01", "2021-02", "2021-03", "2021-04"]
    momentum = [0.10, -0.10, 0.10, -0.10]
    reversal = [-value for value in momentum]
    monthly = pd.DataFrame(
        {
            "month": months,
            "momentum_return": momentum,
            "reversal_return": reversal,
            "static_50_50_return": [0.0] * 4,
        }
    )
    returns = [(0.10, 0.0), (0.0, 0.10), (0.05, -0.05), (-0.05, 0.05)]
    target_rows = []
    for month, (a_return, b_return) in zip(months, returns):
        for root, holding_return, mom_weight, rev_weight in [
            ("A", a_return, 1.0, -1.0),
            ("B", b_return, -1.0, 1.0),
        ]:
            target_rows.append(
                {
                    "month": month,
                    "root": root,
                    "holding_return": holding_return,
                    "momentum_score": 1.0 if root == "A" else -1.0,
                    "reversal_score": -1.0 if root == "A" else 1.0,
                    "mom_weight": mom_weight,
                    "rev_weight": rev_weight,
                    "static_50_50_weight": 0.0,
                }
            )
    targets = pd.DataFrame(target_rows)
    states = pd.DataFrame(
        {
            "signal_month": ["2020-12", "2021-01", "2021-02", "2021-03"],
            "holding_month": months,
            "proxy": ["lagged_oi_notional"] * 4,
            "current_volatility": [0.01, 0.02, 0.03, 0.04],
            "q25": [0.015] * 4,
            "q50": [0.025] * 4,
            "q75": [0.035] * 4,
            "volatility_state": ["Q1", "Q2", "Q3", "Q4"],
        }
    )
    costs = pd.DataFrame(
        [
            {
                "month": month,
                "root": root,
                "one_way_cost_ratio": 0.001,
                "roll_cost_ratio": 0.0,
            }
            for month in months
            for root in ["A", "B"]
        ]
    )
    return monthly, targets, states, costs


def test_paper_assignment_routes_q4_to_reversal_and_reconciles_holdings() -> None:
    monthly, targets, states, costs = _fixtures()
    result = build_quartile_router(
        monthly,
        targets,
        states,
        costs,
        QuartileRouterConfig("lagged_oi_notional", PAPER_ASSIGNMENTS),
    )

    np.testing.assert_allclose(
        result.monthly["router_gross_return"], [0.10, -0.10, 0.10, 0.10]
    )
    np.testing.assert_allclose(
        result.monthly["router_gross_return"],
        result.monthly["holdings_gross_return"],
    )
    assert result.monthly["selected_strategy"].tolist() == [
        "momentum",
        "momentum",
        "momentum",
        "reversal",
    ]
    assert result.monthly["router_target_turnover"].tolist() == [2.0, 0.0, 0.0, 4.0]
    np.testing.assert_allclose(result.monthly["router_cost_return"], [0.002, 0.0, 0.0, 0.004])


def test_flat_assignment_closes_positions_and_charges_turnover() -> None:
    monthly, targets, states, costs = _fixtures()
    assignments = {**PAPER_ASSIGNMENTS, "Q4": "flat"}
    result = build_quartile_router(
        monthly,
        targets,
        states,
        costs,
        QuartileRouterConfig("lagged_oi_notional", assignments),
    )

    q4 = result.monthly.iloc[-1]
    assert q4["router_gross_return"] == 0.0
    assert q4["router_target_turnover"] == 2.0
    assert q4["router_cost_return"] == pytest.approx(0.002)
    q4_holdings = result.holdings.loc[result.holdings["month"].eq("2021-04")]
    assert q4_holdings["target_weight"].eq(0.0).all()


def test_state_timing_must_route_signal_month_into_next_month() -> None:
    monthly, targets, states, costs = _fixtures()
    states.loc[0, "holding_month"] = "2020-12"

    with pytest.raises(ValueError, match="signal month t"):
        build_quartile_router(
            monthly,
            targets,
            states,
            costs,
            QuartileRouterConfig("lagged_oi_notional", PAPER_ASSIGNMENTS),
        )


def test_run_id_is_stable_to_assignment_dictionary_order() -> None:
    forward = QuartileRouterConfig("lagged_oi_notional", PAPER_ASSIGNMENTS)
    reverse = QuartileRouterConfig(
        "lagged_oi_notional", dict(reversed(list(PAPER_ASSIGNMENTS.items())))
    )
    assert forward.run_id() == reverse.run_id()


def test_product_selector_has_chinese_names_for_all_extension_roots() -> None:
    alignment = (
        REAL_EXTENSION_ROOT / "product_market_alignment.parquet"
    )
    if not alignment.exists():
        pytest.skip("product-state backtest artifacts are unavailable")
    roots = pd.read_parquet(alignment, columns=["root"])["root"].astype(str).unique()
    missing = [root for root in roots if " · " not in format_futures_cn_product_zh(root)]
    assert missing == []
    assert format_futures_cn_product_zh("AP") == "AP · 苹果"


def test_two_speed_ema_artifact_routes_slow_then_fast_by_state() -> None:
    path = REAL_EXTENSION_ROOT / "sleeve_comparison_monthly.csv"
    if not path.exists():
        pytest.skip("two-speed EMA artifacts are unavailable")
    comparison = pd.read_csv(path)
    gross = comparison.pivot(
        index=["month", "volatility_state"],
        columns="strategy",
        values="gross_return",
    ).reset_index()
    expected_dual = np.where(
        gross["volatility_state"].eq("Q4"),
        gross["ema_5_10"],
        gross["ema_20_60"],
    )
    expected_slow_reversal = np.where(
        gross["volatility_state"].eq("Q4"),
        gross["reversal"],
        gross["ema_20_60"],
    )
    np.testing.assert_allclose(gross["dual_ema_router"], expected_dual)
    np.testing.assert_allclose(
        gross["ema_20_60_reversal_router"], expected_slow_reversal
    )
    expected_macd_ema = np.where(
        gross["volatility_state"].eq("Q4"),
        gross["ema_5_10"],
        gross["macd_12_26_9"],
    )
    np.testing.assert_allclose(gross["macd_ema_router"], expected_macd_ema)


def test_real_frozen_paper_router_reconciles_when_artifacts_exist() -> None:
    required = [
        REAL_ARTIFACT_ROOT / "paper_replication_monthly_returns.csv",
        REAL_ARTIFACT_ROOT / "paper_replication_targets.parquet",
        REAL_ARTIFACT_ROOT / "paper_replication_volatility_states.csv",
        REAL_ARTIFACT_ROOT / "paper_replication_cost_ledger.parquet",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("frozen 07_04 runtime artifacts are unavailable")

    result = build_quartile_router(
        pd.read_csv(required[0]),
        pd.read_parquet(required[1]),
        pd.read_csv(required[2]),
        pd.read_parquet(required[3]),
        QuartileRouterConfig("lagged_oi_notional", PAPER_ASSIGNMENTS),
    )
    np.testing.assert_allclose(
        result.monthly["router_gross_return"],
        result.monthly["switch_lagged_oi_notional_return"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.monthly["router_net_return"],
        result.monthly["switch_lagged_oi_notional_net_return"],
        atol=1e-12,
    )
