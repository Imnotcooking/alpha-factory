#!/usr/bin/env python3
"""Audit every active factor's implementation against the pure-signal boundary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from oqp.research.factor_purity import (
    PURITY_REVIEW_PATH,
    build_factor_purity_review_index,
    inspect_factor_source_purity,
)
from oqp.research.factors import (
    PRIVATE_FACTOR_ALIAS_FILE,
    load_factor_module,
    resolve_factor_path,
)

from scripts.research.build_factor_cohort_manifest import normalized_factor_ids


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "factor_purity_audit"
)


def build_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stable_ids = (
        yaml.safe_load(PRIVATE_FACTOR_ALIAS_FILE.read_text(encoding="utf-8"))
        or {}
    )
    factor_ids = normalized_factor_ids(stable_ids)
    reviews = build_factor_purity_review_index(
        factor_ids,
        review_path=PURITY_REVIEW_PATH,
    )
    rows: list[dict[str, Any]] = []
    for factor_id in factor_ids:
        path = resolve_factor_path(factor_id, include_public_examples=False)
        module = load_factor_module(factor_id, include_public_examples=False)
        inspection = inspect_factor_source_purity(path, module)
        review = reviews[factor_id]
        rows.append(
            {
                **inspection.to_dict(),
                "review_group": review["review_group"],
                "review_conclusion": review["review_conclusion"],
                "extracted_component_ids": ";".join(
                    review["extracted_component_ids"]
                ),
                "lookahead_fix": review["lookahead_fix"],
            }
        )
    summary = {
        "active_factor_count": len(rows),
        "reviewed_factor_count": len(reviews),
        "content_pure_count": sum(bool(row["content_pure"]) for row in rows),
        "boundary_violation_count": sum(
            not bool(row["content_pure"]) for row in rows
        ),
        "individually_extracted_factor_count": sum(
            row["review_group"] == "factor_boundary_extraction"
            for row in rows
        ),
        "factor_with_extracted_component_count": sum(
            bool(row["extracted_component_ids"]) for row in rows
        ),
        "extracted_component_count": len(
            {
                component_id
                for row in rows
                for component_id in str(
                    row["extracted_component_ids"]
                ).split(";")
                if component_id
            }
        ),
        "implementation_fingerprint_count": len(
            {
                str(row["implementation_fingerprint"])
                for row in rows
                if row["implementation_fingerprint"]
            }
        ),
        "lookahead_risk_count": sum(
            bool(row["negative_shift_lines"]) for row in rows
        ),
    }
    return rows, summary


def write_audit(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "factor_purity.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    violations = [row for row in rows if not bool(row["content_pure"])]
    lines = [
        "# Factor Purity Audit",
        "",
        f"- Active normalized factors: {summary['active_factor_count']}",
        f"- Content-reviewed factors: {summary['reviewed_factor_count']}",
        f"- Pure factor implementations: {summary['content_pure_count']}",
        f"- Remaining boundary violations: {summary['boundary_violation_count']}",
        (
            "- Factors with an extracted owner-specific component: "
            f"{summary['factor_with_extracted_component_count']}"
        ),
        (
            "- Individually refactored hybrid factors: "
            f"{summary['individually_extracted_factor_count']}"
        ),
        f"- Distinct extracted components: {summary['extracted_component_count']}",
        f"- Remaining future-shift risks: {summary['lookahead_risk_count']}",
        "",
        "## Remaining Violations",
        "",
    ]
    if not violations:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Factor | Issues | Source |",
                "|---|---|---|",
            ]
        )
        for row in violations:
            lines.append(
                f"| `{row['factor_id']}` | {row['purity_issues']} | "
                f"`{row['source']}` |"
            )
    (output_dir / "audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-violations",
        action="store_true",
        help="Write the audit but do not fail when hard violations remain.",
    )
    args = parser.parse_args()
    rows, summary = build_audit()
    write_audit(rows, summary, args.output_dir)
    print(f"Wrote {len(rows)} reviewed factors to {args.output_dir}")
    if summary["boundary_violation_count"] and not args.allow_violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
