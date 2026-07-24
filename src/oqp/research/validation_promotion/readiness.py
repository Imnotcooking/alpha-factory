"""Phase 10 promotion ledger and readiness artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.research.validation_promotion.contracts import (
    DEFAULT_PROMOTION_POLICY_REGISTRY,
    PHASE10_SCHEMA_VERSION,
    PromotionPolicyRegistry,
)


def audit_validation_promotion(
    review_root: str | Path,
    policy_path: str | Path = DEFAULT_PROMOTION_POLICY_REGISTRY,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    root = Path(review_root).expanduser().resolve()
    registry = PromotionPolicyRegistry.load(policy_path)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "review_id": payload.get("review_id", path.parent.name),
                    "router_id": payload.get("router_id", ""),
                    "current_stage": payload.get("current_stage", ""),
                    "decision": payload.get("decision", ""),
                    "next_stage": payload.get("next_stage"),
                    "passed_gate_count": payload.get("passed_gate_count", 0),
                    "gate_count": payload.get("gate_count", 0),
                    "failed_gate_ids": ", ".join(
                        payload.get("failed_gate_ids") or []
                    ),
                    "failure_is_valid_research_result": bool(
                        payload.get("failure_is_valid_research_result", False)
                    ),
                    "review_path": str(path.parent),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "review_id": path.parent.name,
                    "router_id": "",
                    "current_stage": "",
                    "decision": "invalid_artifact",
                    "next_stage": None,
                    "passed_gate_count": 0,
                    "gate_count": 0,
                    "failed_gate_ids": str(exc),
                    "failure_is_valid_research_result": False,
                    "review_path": str(path.parent),
                }
            )
    ledger = pd.DataFrame(
        rows,
        columns=[
            "review_id",
            "router_id",
            "current_stage",
            "decision",
            "next_stage",
            "passed_gate_count",
            "gate_count",
            "failed_gate_ids",
            "failure_is_valid_research_result",
            "review_path",
        ],
    )
    policy_rows = []
    for profile in registry.profiles.values():
        payload = profile.to_dict()
        for key, value in payload.items():
            if key in {"profile_id", "status", "research_object"}:
                continue
            policy_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "research_object": profile.research_object,
                    "parameter": key,
                    "value": value,
                    "policy_fingerprint": profile.fingerprint,
                }
            )
    policies = pd.DataFrame(policy_rows)
    decision_counts = (
        ledger["decision"].value_counts().to_dict() if not ledger.empty else {}
    )
    if ledger.empty:
        status = "awaiting_evidence"
    elif ledger["decision"].eq("invalid_artifact").any():
        status = "blocked"
    elif ledger["decision"].isin(
        ["blocked_governance", "failed_research_result"]
    ).any():
        status = "attention_required"
    else:
        status = "active"
    summary = {
        "schema_version": PHASE10_SCHEMA_VERSION,
        "phase": "Phase 10: Validation and Promotion",
        "status": status,
        "lifecycle": [
            "Discovery",
            "Chronological validation",
            "Frozen holdout",
            "Paper trading",
            "Production review",
        ],
        "declared_policy_profiles": len(registry.profiles),
        "review_count": len(ledger),
        "decision_counts": decision_counts,
        "paper_eligible_count": int(
            ledger["decision"].eq("eligible_for_paper_trading").sum()
            if not ledger.empty
            else 0
        ),
        "production_review_eligible_count": int(
            ledger["decision"].eq("eligible_for_production_review").sum()
            if not ledger.empty
            else 0
        ),
        "failed_research_result_count": int(
            ledger["decision"].eq("failed_research_result").sum()
            if not ledger.empty
            else 0
        ),
        "full_sample_sharpe_is_promotion_gate": False,
        "failure_retention_required": True,
        "boundary": (
            "Promotion uses chronological validation, one frozen holdout, "
            "reproducibility, robustness, concentration, routing-event, and "
            "switching-economics evidence. Paper eligibility is not production approval."
        ),
    }
    return summary, ledger, policies


def write_validation_promotion_readiness(
    summary: dict[str, Any],
    ledger: pd.DataFrame,
    policies: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "readiness.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ledger.to_csv(destination / "promotion_ledger.csv", index=False)
    policies.to_csv(destination / "promotion_policy.csv", index=False)
    return destination


__all__ = [
    "audit_validation_promotion",
    "write_validation_promotion_readiness",
]
