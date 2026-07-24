"""Plan and optionally execute governed daily single-factor screens."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from oqp.config import REPO_ROOT
from oqp.research.daily_factor_screening import (
    DEFAULT_BATCH_ID,
    DEFAULT_MARKET_VERTICAL,
    build_daily_factor_screening_plan,
    execute_screening_plan,
    execute_screening_plan_subprocess,
    write_execution_results,
    write_screening_manifest,
    write_strategy_configs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic factor-to-sleeve manifest and typed YAML "
            "strategies; execution is opt-in."
        )
    )
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--market-vertical", default=DEFAULT_MARKET_VERTICAL)
    parser.add_argument(
        "--factor-id",
        action="append",
        default=[],
        help="Full factor ID; repeat the option or provide comma-separated IDs.",
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-dir", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests and YAMLs only (the default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute each planned strategy and persist every result or block.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=("in_process", "subprocess"),
        default="in_process",
        help="In-process shares one prepared dataset across the batch.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=3_600)
    return parser


def _factor_ids(values: Sequence[str]) -> tuple[str, ...] | None:
    resolved = tuple(
        dict.fromkeys(
            item.strip()
            for value in values
            for item in str(value).split(",")
            if item.strip()
        )
    )
    return resolved or None


def _default_paths(batch_id: str) -> tuple[Path, Path]:
    batch_slug = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in batch_id.strip()
    ).strip("_")
    batch_slug = batch_slug or DEFAULT_BATCH_ID
    output = (
        Path("runtime")
        / "artifacts"
        / "research"
        / "daily_factor_screening"
        / batch_slug
    )
    configs = (
        Path("runtime")
        / "configs"
        / "research"
        / "strategy_builder"
        / "daily_factor_screening"
        / batch_slug
    )
    return output, configs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    default_output, default_configs = _default_paths(args.batch_id)
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    config_dir = Path(args.config_dir) if args.config_dir else default_configs

    plan = build_daily_factor_screening_plan(
        batch_id=args.batch_id,
        market_vertical=args.market_vertical,
        factor_ids=_factor_ids(args.factor_id),
        data_file=args.data_file,
        dataset_id=args.dataset_id,
        workspace_root=REPO_ROOT,
    )
    if args.limit is not None:
        plan = replace(plan, items=plan.items[: args.limit])
    plan = write_strategy_configs(
        plan,
        config_dir,
        workspace_root=REPO_ROOT,
    )
    manifest_json, manifest_csv = write_screening_manifest(
        plan,
        output_dir,
        workspace_root=REPO_ROOT,
    )

    ready_count = sum(item.status == "ready" for item in plan.items)
    blocked_count = len(plan.items) - ready_count
    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "batch_id": plan.batch_id,
        "plan_fingerprint": plan.fingerprint,
        "factor_count": len(plan.items),
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "config_dir": str(
            config_dir if config_dir.is_absolute() else Path(REPO_ROOT) / config_dir
        ),
    }
    if args.execute:
        if args.execution_backend == "subprocess":
            results = execute_screening_plan_subprocess(
                plan,
                workspace_root=REPO_ROOT,
                timeout_seconds=args.timeout_seconds,
                start_date=args.start_date,
                end_date=args.end_date,
                split_date=args.split_date,
            )
        else:
            results = execute_screening_plan(
                plan,
                workspace_root=REPO_ROOT,
                start_date=args.start_date,
                end_date=args.end_date,
                split_date=args.split_date,
            )
        results_path = write_execution_results(
            results,
            output_dir,
            workspace_root=REPO_ROOT,
        )
        summary["execution_results"] = str(results_path)
        summary["completed_count"] = sum(
            result.status == "completed" for result in results
        )
        summary["recorded_block_count"] = sum(
            result.status == "blocked_recorded" for result in results
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]
