"""Readiness audit for Phase 7 strategy assemblies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.research.strategy_composition.contracts import (
    STRATEGY_COMPOSITION_SCHEMA_VERSION,
    load_strategy_composition_config,
)


def audit_strategy_composition_readiness(
    router_evidence_root: str | Path,
    strategy_recipe_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    router_root = Path(router_evidence_root).expanduser().resolve()
    recipe_root = Path(strategy_recipe_root).expanduser().resolve()
    eligible_routers: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(router_root.glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        router_id = str(summary.get("router_id") or "")
        if not router_id:
            continue
        manifest_path = summary_path.parent / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        status = str(summary.get("router_status") or "unknown")
        row = {
            "component_type": "router_evidence",
            "component_id": router_id,
            "status": status,
            "eligible": status == "eligible_for_strategy_review",
            "reason": "",
            "artifact_path": str(summary_path.parent),
        }
        rows.append(row)
        if row["eligible"]:
            eligible_routers[router_id] = manifest

    declared = 0
    admissible = 0
    for recipe_path in sorted(recipe_root.glob("str_*.yaml")):
        declared += 1
        try:
            config = load_strategy_composition_config(recipe_path)
            manifest = eligible_routers.get(str(config.router))
            if manifest is None:
                reason = "configured router has no eligible Phase 6 evidence"
                eligible = False
            else:
                router_config = manifest.get("config") or {}
                router_sleeves = {
                    str(router_config.get("sleeve_a_id") or ""),
                    str(router_config.get("sleeve_b_id") or ""),
                }
                eligible = router_sleeves == set(config.sleeves)
                reason = "" if eligible else "router and strategy sleeve IDs differ"
            admissible += int(eligible)
            rows.append(
                {
                    "component_type": "strategy_recipe",
                    "component_id": config.strategy_id,
                    "status": "admissible" if eligible else "blocked",
                    "eligible": eligible,
                    "reason": reason,
                    "artifact_path": str(recipe_path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "component_type": "strategy_recipe",
                    "component_id": recipe_path.stem,
                    "status": "invalid",
                    "eligible": False,
                    "reason": str(exc),
                    "artifact_path": str(recipe_path),
                }
            )

    blockers = []
    if not eligible_routers:
        blockers.append(
            "No router has passed both Phase 6 validation and holdout confirmation."
        )
    if declared == 0:
        blockers.append(
            "No final str_*.yaml composition has been frozen; create one only after "
            "its router becomes eligible."
        )
    elif admissible == 0:
        blockers.append(
            "Declared strategy recipes do not match an eligible frozen router."
        )
    ready = admissible > 0
    summary = {
        "schema_version": STRATEGY_COMPOSITION_SCHEMA_VERSION,
        "phase": "Phase 7: Strategy Composition",
        "status": "ready" if ready else "blocked",
        "eligible_phase6_routers": len(eligible_routers),
        "declared_strategy_recipes": declared,
        "admissible_strategy_recipes": admissible,
        "ready_for_composition_backtest": ready,
        "blockers": blockers,
        "operation_order": [
            "sleeve_targets",
            "router_allocation",
            "risk_overlays",
            "allocator",
            "final_position_execution",
            "transaction_costs",
        ],
        "cost_rule": (
            "Ignore sleeve-level hypothetical costs and calculate fees/slippage "
            "once from final executed position changes."
        ),
    }
    return summary, pd.DataFrame(
        rows,
        columns=[
            "component_type",
            "component_id",
            "status",
            "eligible",
            "reason",
            "artifact_path",
        ],
    )


def write_strategy_composition_readiness(
    summary: dict[str, Any], components: pd.DataFrame, output_dir: str | Path
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "readiness.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    components.to_csv(destination / "components.csv", index=False)
    return destination


__all__ = [
    "audit_strategy_composition_readiness",
    "write_strategy_composition_readiness",
]
