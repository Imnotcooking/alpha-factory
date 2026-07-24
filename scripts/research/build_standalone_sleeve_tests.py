#!/usr/bin/env python3
"""Build Phase 4 standalone tests from saved Phase 3 sleeve evidence."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from oqp.research.artifacts import sha256_file
from oqp.research.sleeves import (
    StandaloneSleeveTestConfig,
    build_standalone_sleeve_test,
    load_sleeve_evidence_bundle,
    load_sleeve_module,
    write_standalone_sleeve_test_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE3_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "sleeve_construction"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "standalone_sleeve_tests"
)
DEFAULT_SLEEVE_ID = "slv_001_Cross_Sectional_Quintile_Long_Short"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--market-vertical", default="FUTURES_CN")
    parser.add_argument("--sleeve-id", default=DEFAULT_SLEEVE_ID)
    parser.add_argument("--factor-id", action="append", dest="factor_ids")
    parser.add_argument("--extreme-quantile", type=float, default=0.99)
    parser.add_argument("--event-pre-periods", type=int, default=5)
    parser.add_argument("--event-post-periods", type=int, default=5)
    return parser.parse_args()


def build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    phase3_root = args.phase3_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    phase3_index_path = phase3_root / "evidence_index.csv"
    if not phase3_index_path.exists():
        raise FileNotFoundError(phase3_index_path)
    phase3_index = pd.read_csv(phase3_index_path)
    eligible = phase3_index.loc[
        phase3_index["market_vertical"].eq(args.market_vertical)
        & phase3_index["sleeve_id"].eq(args.sleeve_id)
    ].copy()
    factor_ids = args.factor_ids or eligible["factor_id"].astype(str).tolist()
    unknown = sorted(set(factor_ids).difference(eligible["factor_id"].astype(str)))
    if unknown:
        raise ValueError(f"factor IDs are absent from Phase 3 evidence: {unknown}")

    config = StandaloneSleeveTestConfig(
        extreme_event_quantile=args.extreme_quantile,
        extreme_event_pre_periods=args.event_pre_periods,
        extreme_event_post_periods=args.event_post_periods,
    )
    sleeve_module = load_sleeve_module(args.sleeve_id)
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        source = (
            phase3_root
            / factor_id
            / args.market_vertical
            / args.sleeve_id
        )
        phase3_manifest_path = source / "manifest.json"
        phase3 = load_sleeve_evidence_bundle(source)
        if phase3.config.factor_id != factor_id:
            raise ValueError(f"Phase 3 factor identity mismatch for {factor_id}")
        if (
            phase3.manifest.get("sleeve_definition_fingerprint")
            != sleeve_module.SLEEVE_DEFINITION_FINGERPRINT
        ):
            raise ValueError(
                f"Phase 3 sleeve evidence is stale for {args.sleeve_id}"
            )
        bundle = build_standalone_sleeve_test(phase3, config)
        bundle.manifest["phase3_artifact_path"] = str(source.relative_to(REPO_ROOT))
        bundle.manifest["phase3_manifest_sha256"] = sha256_file(phase3_manifest_path)
        destination = write_standalone_sleeve_test_bundle(
            bundle,
            output_root / factor_id / args.market_vertical / args.sleeve_id,
        )
        split = bundle.split_metrics.set_index("research_split")
        validation = split.loc[config.validation_label]
        holdout = split.loc[config.holdout_label]
        rows.append(
            {
                "factor_id": factor_id,
                "market_vertical": args.market_vertical,
                "sleeve_id": args.sleeve_id,
                "standalone_status": bundle.summary["standalone_status"],
                "validation_decision": bundle.summary["validation_decision"],
                "holdout_confirmation_passed": bundle.summary[
                    "holdout_confirmation_passed"
                ],
                "router_eligible": bundle.summary["router_eligible"],
                "validation_net_annualized_mean": validation[
                    "net_annualized_mean"
                ],
                "validation_net_sharpe": validation["net_sharpe"],
                "validation_break_even_cost_multiple": validation[
                    "break_even_cost_multiple"
                ],
                "validation_active_days": int(validation["active_date_count"]),
                "holdout_net_annualized_mean": holdout["net_annualized_mean"],
                "holdout_net_sharpe": holdout["net_sharpe"],
                "full_net_hit_rate": split.loc["full", "active_net_hit_rate"],
                "full_maximum_drawdown": split.loc["full", "maximum_drawdown"],
                "full_annualized_turnover": split.loc["full", "annualized_turnover"],
                "extreme_event_count": int(
                    bundle.summary["extreme_event"]["full_event_count"]
                ),
                "artifact_path": str(destination.relative_to(REPO_ROOT)),
            }
        )
        print(
            f"built {factor_id}: validation={bundle.summary['validation_decision']}, "
            f"holdout={bundle.summary['holdout_confirmation_passed']}, "
            f"status={bundle.summary['standalone_status']}",
            flush=True,
        )
        del phase3, bundle
        gc.collect()

    index = pd.DataFrame(rows).sort_values("factor_id").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_root / "evidence_index.csv", index=False)
    run_manifest = {
        "schema_version": 1,
        "phase": "Phase 4: Standalone Sleeve Test",
        "phase3_root": str(phase3_root.relative_to(REPO_ROOT)),
        "phase3_index_sha256": sha256_file(phase3_index_path),
        "market_vertical": args.market_vertical,
        "sleeve_id": args.sleeve_id,
        "factor_ids": factor_ids,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "optimization_permitted": False,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    print(build_cohort(parse_args()).to_string(index=False))


if __name__ == "__main__":
    main()
