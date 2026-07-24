from __future__ import annotations

import json
import sqlite3

from oqp.research.backtesting import AlphaEvaluator


def test_record_blocked_run_is_visible_and_idempotent(tmp_path) -> None:
    db_path = tmp_path / "research.db"
    artifact_root = tmp_path / "artifacts"
    evaluator = AlphaEvaluator(
        db_path=db_path,
        logs_dir=artifact_root,
        asset_class="FUTURES_CN",
    )
    assumptions = {
        "capital": 10_000_000.0,
        "capital_currency": "CNY",
        "max_margin_utilization": 0.30,
        "minimum_cash_reserve": 0.70,
        "max_contract_weight": 0.10,
        "risk_overlay_ids": [],
        "risk_overlay_policy": "No optional overlay in baseline screening.",
        "transaction_cost_profile": "cn_futures_broker_v1",
        "slippage_ticks_per_side": 0.5,
    }
    kwargs = {
        "strategy_id": "str_daily_screen_fac_999_slv_001",
        "strategy_name": "Example factor × quintile sleeve [batch_001]",
        "component_factor_id": "fac_999_Example",
        "sleeve_id": "slv_001_Cross_Sectional_Quintile_Long_Short",
        "failure_code": "missing_required_fields",
        "suggested_action": "Provide the missing point-in-time input columns.",
        "strategy_config": {"strategy": {"strategy_id": "str_daily_screen_fac_999_slv_001"}},
        "strategy_config_fingerprint": "fingerprint-001",
        "batch_id": "batch_001",
        "dataset_metadata": {
            "market_vertical": "FUTURES_CN",
            "dataset_id": "fixture",
            "data_frequency": "daily",
            "data_tradability": "research_proxy",
            "evaluation_geometry": "cross_sectional",
        },
        "assumptions": assumptions,
    }

    first_run_id = evaluator.record_blocked_run(**kwargs)
    second_run_id = evaluator.record_blocked_run(**kwargs)

    assert second_run_id == first_run_id
    with sqlite3.connect(db_path) as conn:
        run = conn.execute(
            """
            SELECT factor_id, total_trades, component_type, strategy_core_type,
                   backtest_engine, runner
            FROM backtest_runs
            WHERE run_id = ?
            """,
            (first_run_id,),
        ).fetchone()
        diagnostic = conn.execute(
            """
            SELECT failure_code, suggested_action
            FROM diagnostics
            WHERE run_id = ?
            """,
            (first_run_id,),
        ).fetchall()
        factor = conn.execute(
            "SELECT name, category FROM factors WHERE factor_id = ?",
            ("str_daily_screen_fac_999_slv_001",),
        ).fetchone()
        trial = conn.execute(
            "SELECT metric_name, significance FROM research_trials WHERE run_id = ?",
            (first_run_id,),
        ).fetchone()

    assert run == (
        "str_daily_screen_fac_999_slv_001",
        0,
        "strategy",
        "factor_sleeve",
        "typed_strategy_composition",
        "daily_factor_screen",
    )
    assert diagnostic == [
        (
            "missing_required_fields",
            "Provide the missing point-in-time input columns.",
        )
    ]
    assert factor == (
        "Example factor × quintile sleeve [batch_001]",
        "Daily Single-Factor Screen",
    )
    assert trial == ("blocked_preflight", "blocked")

    manifest_path = (
        artifact_root
        / "assumptions"
        / f"assumptions_{first_run_id}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["screening"]["status"] == "blocked"
    assert manifest["risk_and_allocation"]["minimum_cash_reserve"] == 0.70
    assert manifest["risk_and_allocation"]["risk_overlay_ids"] == []
    assert (
        manifest["costs_and_slippage"]["fixed_slippage_ticks_per_side"]
        == 0.5
    )
