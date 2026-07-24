#!/usr/bin/env python3
"""Audit private factor, router, and strategy-overlay registries statically."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from oqp.research.factor_purity import (
    factor_implementation_fingerprint,
    inspect_factor_source_purity,
)
from oqp.research.factors import load_factor_module


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTOR_ROOT = REPO_ROOT / "departments" / "research" / "factors"
ROUTER_ROOT = REPO_ROOT / "departments" / "research" / "routers"
OVERLAY_ROOT = REPO_ROOT / "departments" / "research" / "strategy_overlays"
POSITION_POLICY_ROOT = REPO_ROOT / "departments" / "research" / "position_policies"
DIAGNOSTIC_ROOT = REPO_ROOT / "departments" / "research" / "diagnostics"
ROUTER_STATE_ROOT = ROUTER_ROOT / "states"
SLEEVE_ROOT = (
    REPO_ROOT / "departments" / "research" / "strategies" / "sleeves"
)
CATALOG_PATH = FACTOR_ROOT / "catalog.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runtime" / "artifacts" / "research" / "component_registry_audit"
NORMALIZED_METADATA_KEYS = {
    "metadata_schema_version",
    "component_type",
    "status",
    "factor_family",
    "factor_subfamily",
    "native_market",
    "supported_markets",
    "data_frequency",
    "signal_frequency",
    "rebalance_frequency",
    "signal_horizon",
    "execution_style",
    "portfolio_layer",
    "deduplication_cohort",
    "cost_model",
    "required_fields",
    "legacy_ids",
}


# These are governance decisions, not filename guesses. Empirical promotion or
# retirement remains a later phase after correlation and IC evidence is joined.
LEGACY_DISPOSITIONS: dict[str, tuple[str, str, str]] = {
    "cnf_monthly_daily_state_screen.py": (
        "router_state_family",
        "move_to_router_support",
        "Exploratory second-state screen; not a production factor.",
    ),
    "cnf_monthly_ema_trend.py": (
        "legacy_sleeve_adapter",
        "retire_after_dependency_migration",
        "Overlaps fac_064 and should not become another factor ID.",
    ),
    "cnf_monthly_macd_crossover.py": (
        "legacy_sleeve_adapter",
        "retire_after_dependency_migration",
        "Wraps fac_065 and should not become another factor ID.",
    ),
    "cnf_monthly_methodology_audit.py": (
        "experiment_diagnostic",
        "move_to_owner_project",
        "Replication audit machinery, not reusable alpha or routing logic.",
    ),
    "cnf_monthly_paper_sleeves.py": (
        "frozen_replication_sleeves",
        "move_to_owner_project",
        "Paper-specific frozen definitions; exclude from active factor inventory.",
    ),
    "cnf_monthly_positioning_state.py": (
        "router_state",
        "move_to_router_support",
        "Rejected diagnostic state retained for reproducibility.",
    ),
    "cnf_monthly_product_volatility.py": (
        "router_state",
        "move_to_router_support",
        "Product-level state, not an alpha signal.",
    ),
    "cnf_monthly_shock_breadth.py": (
        "router_state",
        "move_to_router_support",
        "Prospective second routing dimension.",
    ),
    "cnf_monthly_shock_stage.py": (
        "router_state",
        "move_to_router_support",
        "Rejected diagnostic state retained for reproducibility.",
    ),
    "cnf_monthly_universe_quality.py": (
        "portfolio_policy",
        "move_to_strategy_support",
        "Eligibility and weighting policy, not alpha.",
    ),
    "cnf_monthly_volatility_state.py": (
        "router_state",
        "move_to_router_support",
        "Primary causal state input for volatility routers.",
    ),
}

FACTOR_DISPOSITIONS: dict[str, tuple[str, str, str]] = {
    "fac_038_Regime_Router.py": (
        "router_embedded_in_factor",
        "split_into_sleeves_and_router",
        "Blends SMA and Bollinger targets by a volatility state inside compute().",
    ),
    "fac_053_Conditional_Regime_Test.py": (
        "diagnostic_embedded_in_factor",
        "move_to_diagnostics",
        "Returns a regime PnL comparison table rather than an alpha score panel.",
    ),
    "fac_055_Ultimate_ML_Router.py": (
        "hybrid_factor_and_policy",
        "split_alpha_from_risk_policy",
        "Combines ML alpha, a cash gate, volatility sizing, and execution smoothing.",
    ),
}


def _literal_assignments(tree: ast.Module) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _declared_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.rsplit(".", 1)[-1])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _declared_dict_keys(tree: ast.Module, assignment_name: str) -> set[str]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            return set()
        return {
            str(key.value)
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def _imports(tree: ast.Module) -> list[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            modules.add(prefix + (node.module or ""))
    return sorted(module for module in modules if module)


def _functions(tree: ast.Module) -> list[str]:
    return sorted(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )


def _catalog_by_source() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    if not CATALOG_PATH.exists():
        return {}
    payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    by_source: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for component_id, details in (payload.get("components") or {}).items():
        if not isinstance(details, dict):
            continue
        source = str(details.get("canonical_source", ""))
        by_source[source].append((str(component_id), details))
    return dict(by_source)


def _metadata_gaps(kind: str, declared_names: set[str]) -> list[str]:
    if kind in {"factor", "router_embedded_in_factor", "hybrid_factor_and_policy"}:
        required = ("FACTOR_ID", "FACTOR_METADATA", "FACTOR_CONTRACT")
    elif kind == "router":
        required = ("ROUTER_ID", "ROUTER_METADATA", "ROUTER_CONTRACT")
    elif kind == "strategy_risk_overlay":
        required = ("OVERLAY_ID", "OVERLAY_METADATA", "OVERLAY_CONTRACT")
    elif kind == "position_policy":
        required = (
            "POSITION_POLICY_ID",
            "POSITION_POLICY_METADATA",
            "POSITION_POLICY_CONTRACT",
        )
    elif kind == "diagnostic":
        required = ("DIAGNOSTIC_ID", "DIAGNOSTIC_METADATA", "DIAGNOSTIC_CONTRACT")
    elif kind == "router_state":
        required = (
            "ROUTER_STATE_ID",
            "ROUTER_STATE_METADATA",
            "ROUTER_STATE_CONTRACT",
        )
    elif kind == "strategy_sleeve":
        required = (
            "SLEEVE_ID",
            "SLEEVE_METADATA",
            "SLEEVE_CONTRACT",
            "build_config",
        )
    else:
        return []
    return [field for field in required if field not in declared_names]


def _row(path: Path, catalog: dict[str, list[tuple[str, dict[str, Any]]]]) -> dict[str, Any]:
    source = path.relative_to(REPO_ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _literal_assignments(tree)
    declared_names = _declared_names(tree)
    metadata_keys = _declared_dict_keys(tree, "FACTOR_METADATA")
    catalog_entries = catalog.get(source, [])
    declared_id = str(
        values.get("FACTOR_ID")
        or values.get("ROUTER_ID")
        or values.get("OVERLAY_ID")
        or values.get("POSITION_POLICY_ID")
        or values.get("DIAGNOSTIC_ID")
        or values.get("ROUTER_STATE_ID")
        or values.get("SLEEVE_ID")
        or values.get("COMPONENT_ID")
        or values.get("MOMENTUM_COMPONENT_ID")
        or path.stem
    )
    if path.name.startswith("fac_"):
        kind, action, note = FACTOR_DISPOSITIONS.get(
            path.name,
            (
                "factor",
                "retain_pending_empirical_deduplication",
                "Active factor naming is compliant.",
            ),
        )
    elif path.name.startswith("rtr_"):
        kind = "router"
        action = "retain_router_registry"
        note = "Router naming is compliant."
    elif path.name.startswith("ovl_"):
        kind = "strategy_risk_overlay"
        action = "retain_strategy_overlay_registry"
        note = "Strategy risk-overlay naming is compliant."
    elif path.name.startswith("pos_"):
        kind = "position_policy"
        action = "retain_position_policy_registry"
        note = "Position-policy naming is compliant."
    elif path.name.startswith("dgn_"):
        kind = "diagnostic"
        action = "retain_diagnostic_registry"
        note = "Diagnostic naming is compliant."
    elif path.name.startswith("rst_"):
        kind = "router_state"
        action = "retain_router_state_registry"
        note = "Router-state naming is compliant."
    elif path.name.startswith("slv_"):
        kind = "strategy_sleeve"
        action = "retain_sleeve_registry"
        note = "Sleeve naming is compliant."
    else:
        kind, action, note = LEGACY_DISPOSITIONS.get(
            path.name,
            ("unclassified_legacy", "manual_review", "No disposition recorded."),
        )
    catalog_types = sorted(
        {str(details.get("type", "")) for _, details in catalog_entries if details.get("type")}
    )
    catalog_statuses = sorted(
        {str(details.get("status", "")) for _, details in catalog_entries if details.get("status")}
    )
    gaps = _metadata_gaps(kind, declared_names)
    purity = None
    if path.name.startswith("fac_"):
        module = load_factor_module(path.stem, include_public_examples=False)
        purity = inspect_factor_source_purity(path, module)
    content_pure = bool(purity.pure) if purity is not None else True
    if path.name.startswith("fac_") and not content_pure and purity is not None:
        action = "split_embedded_components"
        note = "; ".join(purity.issues)
    return {
        "source": source,
        "filename": path.name,
        "declared_id": declared_id,
        "declared_id_matches_filename": declared_id == path.stem,
        "registry_kind": kind,
        "registry_compliant": kind in {
            "factor",
            "router",
            "strategy_risk_overlay",
            "position_policy",
            "diagnostic",
            "router_state",
            "strategy_sleeve",
        }
        and content_pure,
        "content_pure": content_pure,
        "content_issues": "; ".join(purity.issues) if purity is not None else "",
        "implementation_fingerprint": factor_implementation_fingerprint(path),
        "metadata_complete": not gaps,
        "metadata_gaps": ";".join(gaps),
        "normalized_metadata": bool(
            path.name.startswith("fac_")
            and kind == "factor"
            and NORMALIZED_METADATA_KEYS.issubset(metadata_keys)
        ),
        "normalized_metadata_gaps": ";".join(
            sorted(NORMALIZED_METADATA_KEYS.difference(metadata_keys))
            if path.name.startswith("fac_")
            else []
        ),
        "catalog_ids": ";".join(component_id for component_id, _ in catalog_entries),
        "catalog_types": ";".join(catalog_types),
        "catalog_statuses": ";".join(catalog_statuses),
        "public_functions": ";".join(_functions(tree)),
        "internal_imports": ";".join(
            module
            for module in _imports(tree)
            if module.startswith(("departments.", "oqp.", "."))
        ),
        "recommended_action": action,
        "decision_note": note,
    }


def build_audit() -> list[dict[str, Any]]:
    catalog = _catalog_by_source()
    paths = sorted(FACTOR_ROOT.glob("fac_*.py"))
    paths += sorted(FACTOR_ROOT.glob("cnf_*.py"))
    paths += sorted(ROUTER_ROOT.glob("rtr_*.py"))
    paths += sorted(OVERLAY_ROOT.glob("ovl_*.py"))
    paths += sorted(POSITION_POLICY_ROOT.glob("pos_*.py"))
    paths += sorted(DIAGNOSTIC_ROOT.glob("dgn_*.py"))
    paths += sorted(ROUTER_STATE_ROOT.glob("rst_*.py"))
    paths += sorted(SLEEVE_ROOT.glob("slv_*.py"))
    return [_row(path, catalog) for path in paths]


def _duplicate_numeric_factor_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        name = str(row["filename"])
        if not name.startswith("fac_"):
            continue
        token = name.split("_", 2)[1]
        if token.isdigit():
            grouped[token].append(name)
    return {token: names for token, names in grouped.items() if len(names) > 1}


def write_audit(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (output_dir / "component_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["registry_kind"]) for row in rows)
    actions = Counter(str(row["recommended_action"]) for row in rows)
    duplicates = _duplicate_numeric_factor_ids(rows)
    summary = {
        "component_count": len(rows),
        "kind_counts": dict(sorted(counts.items())),
        "action_counts": dict(sorted(actions.items())),
        "registry_boundary_violations": sum(
            not bool(row["registry_compliant"]) for row in rows
        ),
        "metadata_incomplete": sum(not bool(row["metadata_complete"]) for row in rows),
        "normalized_factor_metadata": sum(
            bool(row["normalized_metadata"]) for row in rows
        ),
        "declared_id_mismatches": sum(
            not bool(row["declared_id_matches_filename"])
            for row in rows
            if str(row["filename"]).startswith(
                ("fac_", "rtr_", "ovl_", "pos_", "dgn_", "rst_")
                + ("slv_",)
            )
        ),
        "duplicate_numeric_factor_ids": duplicates,
        "content_pure_factor_count": sum(
            bool(row["content_pure"])
            for row in rows
            if str(row["filename"]).startswith("fac_")
        ),
        "content_boundary_violation_count": sum(
            not bool(row["content_pure"])
            for row in rows
            if str(row["filename"]).startswith("fac_")
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    legacy = [row for row in rows if str(row["filename"]).startswith("cnf_")]
    embedded = [
        row
        for row in rows
        if str(row["filename"]).startswith("fac_")
        and (
            str(row["registry_kind"]) != "factor"
            or not bool(row["content_pure"])
        )
    ]
    lines = [
        "# Component Registry Audit",
        "",
        f"- Components scanned: {len(rows)}",
        f"- Legacy `cnf_*` names: {len(legacy)}",
        f"- Non-factor implementations using `fac_*`: {len(embedded)}",
        f"- Incomplete declared component metadata: {summary['metadata_incomplete']}",
        f"- Factors on normalized metadata schema: {summary['normalized_factor_metadata']}",
        f"- Declared ID/filename mismatches: {summary['declared_id_mismatches']}",
        f"- Duplicate numeric factor IDs: {len(duplicates)}",
        "",
        "## Legacy Dispositions",
        "",
        "| File | Classification | Next action | Reason |",
        "|---|---|---|---|",
    ]
    for row in legacy:
        lines.append(
            "| `{filename}` | {registry_kind} | `{recommended_action}` | {decision_note} |".format(
                **row
            )
        )
    lines.extend(["", "## Embedded Boundary Violations", ""])
    lines.extend(
        [
            "| File | Actual role | Next action | Reason |",
            "|---|---|---|---|",
        ]
    )
    for row in embedded:
        lines.append(
            "| `{filename}` | {registry_kind} | `{recommended_action}` | {decision_note} |".format(
                **row
            )
        )
    lines.extend(["", "## Duplicate Numeric Factor IDs", ""])
    if duplicates:
        for token, names in sorted(duplicates.items()):
            lines.append(f"- `{token}`: " + ", ".join(f"`{name}`" for name in names))
    else:
        lines.append("None.")
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_audit()
    write_audit(rows, args.output_dir)
    print(f"Wrote {len(rows)} component rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
