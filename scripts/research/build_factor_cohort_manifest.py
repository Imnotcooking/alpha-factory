#!/usr/bin/env python3
"""Build the normalized factor-cohort manifest without loading market data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from oqp.research.factor_governance import validate_factor_governance
from oqp.research.factor_definitions import inspect_factor_definition
from oqp.research.factor_purity import (
    build_factor_purity_review_index,
    inspect_factor_source_purity,
)
from oqp.research.factors import (
    PRIVATE_FACTOR_ALIAS_FILE,
    resolve_factor_path,
    load_factor_module,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "factor_registry_normalization"
)
PREDICTIVE_EVIDENCE_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "predictive_evidence"
)


def predictive_evidence_status(
    factor_id: str,
    native_market: str,
    current_definition_fingerprint: str,
) -> tuple[str, str]:
    """Compare saved predictive evidence with the current factor definition."""

    manifest_path = (
        PREDICTIVE_EVIDENCE_ROOT
        / factor_id
        / native_market
        / "manifest.json"
    )
    if not manifest_path.is_file():
        return "missing", ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unreadable", ""
    evidence_fingerprint = str(
        manifest.get("factor_definition_fingerprint") or ""
    )
    if (
        current_definition_fingerprint
        and evidence_fingerprint == current_definition_fingerprint
    ):
        return "current", evidence_fingerprint
    return "stale_definition", evidence_fingerprint


def normalized_factor_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    factor_ids: list[str] = []
    batches = payload.get("normalization_batches") or {}
    for batch in batches.values():
        if not isinstance(batch, dict) or batch.get("status") not in {
            "normalized_metadata",
            "empirically_cleaned",
        }:
            continue
        for key, values in batch.items():
            is_member_key = key == "canonical_ids" or (
                key.endswith("_ids") and key != "skipped_ids"
            )
            if not is_member_key:
                continue
            if not isinstance(values, list):
                continue
            factor_ids.extend(str(value) for value in values)
    if len(factor_ids) != len(set(factor_ids)):
        raise ValueError("normalized stable-ID batches contain duplicate factors")
    return tuple(factor_ids)


def phase_1_issue_category(issue: str) -> str:
    text = str(issue).strip()
    if text.startswith("SIGNAL_ORIENTATION"):
        return "missing_or_invalid_signal_orientation"
    if text.startswith("KNOWN_LIMITATIONS"):
        return "missing_known_limitations"
    if text.startswith("missing EXPECTED_HOLDING_HORIZON"):
        return "missing_expected_holding_horizon"
    if text.startswith("missing FACTOR_PARAMETERS"):
        return "missing_parameter_schema"
    if text.startswith("invalid FACTOR_PARAMETERS"):
        return "invalid_parameter_schema"
    if text.startswith("portfolio_layer"):
        return "non_predictive_portfolio_layer"
    if text.startswith("allocation parameters belong outside"):
        return "embedded_allocation_parameters"
    if text.startswith("execution_weight_col differs"):
        return "embedded_execution_target"
    if text.startswith("position/holding output columns remain"):
        return "embedded_position_or_holding_output"
    if text.startswith("position/holding helper functions remain"):
        return "embedded_position_or_holding_helper"
    if text.startswith("TEMPORAL_POLICY owns"):
        return "embedded_holding_policy"
    if text.startswith("negative shift may consume future rows"):
        return "lookahead_risk"
    return text


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = yaml.safe_load(PRIVATE_FACTOR_ALIAS_FILE.read_text(encoding="utf-8")) or {}
    factor_ids = normalized_factor_ids(payload)
    purity_reviews = build_factor_purity_review_index(factor_ids)
    rows: list[dict[str, Any]] = []
    for factor_id in factor_ids:
        path = resolve_factor_path(factor_id, include_public_examples=False)
        module = load_factor_module(factor_id, include_public_examples=False)
        record = validate_factor_governance(path, module)
        contract = module.FACTOR_CONTRACT
        metadata = module.FACTOR_METADATA
        definition = inspect_factor_definition(path, module)
        purity = inspect_factor_source_purity(path, module)
        definition_row = definition.to_manifest_row()
        purity_review = purity_reviews[factor_id]
        evidence_status, evidence_definition_fingerprint = (
            predictive_evidence_status(
                record.factor_id,
                record.native_market,
                str(definition_row["definition_fingerprint"]),
            )
        )
        rows.append(
            {
                "factor_id": record.factor_id,
                "phase_1_ready": definition_row["phase_1_ready"],
                "factor_family": record.factor_family,
                "factor_subfamily": record.factor_subfamily,
                "native_market": record.native_market,
                "data_frequency": record.data_frequency,
                "signal_frequency": metadata["signal_frequency"],
                "signal_orientation": definition_row["signal_orientation"],
                "evaluation_geometry": contract["evaluation_geometry"],
                "expected_holding_horizon": definition_row["holding_horizon"],
                "parameter_count": definition_row["parameter_count"],
                "known_limitation_count": definition_row[
                    "known_limitation_count"
                ],
                "definition_fingerprint": definition_row[
                    "definition_fingerprint"
                ],
                "implementation_fingerprint": purity.implementation_fingerprint,
                "predictive_evidence_status": evidence_status,
                "predictive_evidence_current": evidence_status == "current",
                "predictive_evidence_definition_fingerprint": (
                    evidence_definition_fingerprint
                ),
                "execution_mode": contract["execution_mode"],
                "execution_lag": contract["execution_lag"],
                "return_assumption": contract["return_assumption"],
                "alpha_signal_col": contract["alpha_signal_col"],
                "execution_weight_col": contract["execution_weight_col"],
                "portfolio_layer": record.portfolio_layer,
                "content_pure": purity.pure,
                "purity_issues": "; ".join(purity.issues),
                "has_target_logic": bool(
                    purity.forbidden_output_columns
                    or (
                        purity.alpha_signal_col
                        and purity.execution_weight_col
                        and purity.alpha_signal_col != purity.execution_weight_col
                    )
                ),
                "has_holding_logic": bool(
                    purity.lifecycle_helpers
                    or {
                        "hold_days",
                        "hold_ticks",
                        "max_hold_days",
                        "max_holding_days",
                        "cooldown_ticks",
                    }.intersection(purity.allocation_parameters)
                ),
                "has_risk_or_sizing_logic": bool(
                    purity.allocation_parameters
                ),
                "has_lookahead_risk": bool(purity.negative_shift_lines),
                "purity_review_group": purity_review["review_group"],
                "purity_review_conclusion": purity_review[
                    "review_conclusion"
                ],
                "lookahead_fix": purity_review["lookahead_fix"],
                "extracted_component_ids": ";".join(
                    purity_review["extracted_component_ids"]
                    or tuple(
                        str(value)
                        for value in metadata.get("extracted_component_ids", [])
                    )
                ),
                "deduplication_cohort": record.deduplication_cohort,
                "cost_model": metadata["cost_model"],
                "required_fields": ";".join(metadata["required_fields"]),
                "collection": str(metadata.get("collection") or ""),
                "source_alpha_number": metadata.get("source_alpha_number", ""),
                "implementation_status": str(
                    metadata.get("implementation_status") or ""
                ),
                "block_reason": str(metadata.get("block_reason") or ""),
                "futures_cn_adaptation_status": str(
                    metadata.get("futures_cn_adaptation_status") or ""
                ),
                "futures_cn_adaptation_geometry": str(
                    metadata.get("futures_cn_adaptation_geometry") or ""
                ),
                "futures_cn_adaptation_reason": str(
                    metadata.get("futures_cn_adaptation_reason") or ""
                ),
                "reference_license": str(
                    metadata.get("reference_license") or ""
                ),
                "phase_1_issues": definition_row["issues"],
                "source": str(path.relative_to(REPO_ROOT)),
            }
        )

    reclassification_ids: list[str] = []
    for batch in (payload.get("normalization_batches") or {}).values():
        if not isinstance(batch, dict) or batch.get("status") != "reclassification_required":
            continue
        reclassification_ids.extend(str(value) for value in batch.get("canonical_ids", []))
    summary = {
        "normalized_factor_count": len(rows),
        "phase_1_ready_count": sum(bool(row["phase_1_ready"]) for row in rows),
        "phase_1_incomplete_count": sum(
            not bool(row["phase_1_ready"]) for row in rows
        ),
        "content_pure_count": sum(bool(row["content_pure"]) for row in rows),
        "content_boundary_violation_count": sum(
            not bool(row["content_pure"]) for row in rows
        ),
        "current_predictive_evidence_count": sum(
            bool(row["predictive_evidence_current"]) for row in rows
        ),
        "stale_predictive_evidence_count": sum(
            row["predictive_evidence_status"] == "stale_definition"
            for row in rows
        ),
        "phase_1_gap_counts": dict(
            sorted(
                Counter(
                    phase_1_issue_category(issue)
                    for row in rows
                    for issue in str(row["phase_1_issues"]).split(";")
                    if issue.strip()
                ).items()
            )
        ),
        "cohort_counts": dict(
            sorted(Counter(row["deduplication_cohort"] for row in rows).items())
        ),
        "reclassification_required": reclassification_ids,
    }
    return rows, summary


def write_manifest(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cohort_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "phase_1_factor_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, summary = build_manifest()
    write_manifest(rows, summary, args.output_dir)
    print(f"Wrote {len(rows)} normalized factors to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
