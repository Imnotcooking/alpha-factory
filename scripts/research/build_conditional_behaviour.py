#!/usr/bin/env python3
"""Build Phase 5 conditional behaviour for frozen daily futures sleeves."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from oqp.research.artifacts import sha256_file
from oqp.research.sleeves import (
    ConditionalBehaviourConfig,
    build_conditional_behaviour,
    build_observable_conditions,
    load_sleeve_evidence_bundle,
    load_standalone_sleeve_test_bundle,
    write_conditional_behaviour_bundle,
    write_observable_conditions_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "phase1b_daily_mean_reversion"
)
DEFAULT_PHASE3_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "sleeve_construction"
)
DEFAULT_PHASE4_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "standalone_sleeve_tests"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "conditional_behaviour"
)
DEFAULT_SLEEVE_ID = "slv_001_Cross_Sectional_Quintile_Long_Short"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--market-vertical", default="FUTURES_CN")
    parser.add_argument("--sleeve-id", default=DEFAULT_SLEEVE_ID)
    parser.add_argument("--factor-id", action="append", dest="factor_ids")
    return parser.parse_args()


def build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    cohort_root = args.cohort_root.expanduser().resolve()
    phase3_root = args.phase3_root.expanduser().resolve()
    phase4_root = args.phase4_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    protocol_path = cohort_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    factor_ids = args.factor_ids or list(protocol.get("factor_ids") or [])
    unknown = sorted(set(factor_ids).difference(protocol.get("factor_ids") or []))
    if unknown:
        raise ValueError(f"factor IDs are outside the frozen cohort: {unknown}")

    source = cohort_root / "roll_clean_signal_panel.parquet"
    source_sha256 = sha256_file(source)
    print(f"Building common observable conditions from {source}", flush=True)
    market_panel = pd.read_parquet(
        source,
        columns=["date", "ticker", "symbol", "close", "volume", "open_interest"],
    )
    config = ConditionalBehaviourConfig()
    observables = build_observable_conditions(
        market_panel,
        config,
        source_fingerprint=source_sha256,
    )
    common_destination = write_observable_conditions_bundle(
        observables,
        output_root / "common",
    )
    del market_panel
    gc.collect()

    rows: list[dict[str, object]] = []
    for index, factor_id in enumerate(factor_ids, start=1):
        print(f"[{index}/{len(factor_ids)}] {factor_id}", flush=True)
        phase3_source = (
            phase3_root / factor_id / args.market_vertical / args.sleeve_id
        )
        phase4_source = (
            phase4_root / factor_id / args.market_vertical / args.sleeve_id
        )
        phase3 = load_sleeve_evidence_bundle(phase3_source)
        phase4 = load_standalone_sleeve_test_bundle(phase4_source)
        if phase3.manifest.get("input_data_fingerprint") != protocol.get(
            "source_sha256"
        ):
            raise ValueError(f"Phase 3 source fingerprint mismatch for {factor_id}")
        bundle = build_conditional_behaviour(phase3, phase4, observables)
        bundle.manifest["protocol_path"] = str(protocol_path.relative_to(REPO_ROOT))
        bundle.manifest["protocol_sha256"] = sha256_file(protocol_path)
        bundle.manifest["observable_artifact_path"] = str(
            common_destination.relative_to(REPO_ROOT)
        )
        bundle.manifest["phase3_artifact_path"] = str(
            phase3_source.relative_to(REPO_ROOT)
        )
        bundle.manifest["phase4_artifact_path"] = str(
            phase4_source.relative_to(REPO_ROOT)
        )
        destination = write_conditional_behaviour_bundle(
            bundle,
            output_root / factor_id / args.market_vertical / args.sleeve_id,
        )
        validation = bundle.bucket_metrics.loc[
            bundle.bucket_metrics["research_split"].eq("validation")
        ]
        rows.append(
            {
                "factor_id": factor_id,
                "market_vertical": args.market_vertical,
                "sleeve_id": args.sleeve_id,
                "standalone_status": bundle.summary["standalone_status"],
                "condition_count": bundle.summary["condition_count"],
                "validation_bucket_rows": int(len(validation)),
                "validation_observations": int(validation["date_count"].sum()),
                "router_backtest_permitted": False,
                "artifact_path": str(destination.relative_to(REPO_ROOT)),
            }
        )
        del phase3, phase4, bundle
        gc.collect()

    index_frame = pd.DataFrame(rows).sort_values("factor_id").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    index_frame.to_csv(output_root / "evidence_index.csv", index=False)
    run_manifest = {
        "schema_version": 1,
        "phase": "Phase 5: Conditional Behaviour",
        "cohort_root": str(cohort_root.relative_to(REPO_ROOT)),
        "phase3_root": str(phase3_root.relative_to(REPO_ROOT)),
        "phase4_root": str(phase4_root.relative_to(REPO_ROOT)),
        "market_vertical": args.market_vertical,
        "sleeve_id": args.sleeve_id,
        "factor_ids": factor_ids,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "condition_source_sha256": source_sha256,
        "optimization_permitted": False,
        "router_backtest_permitted": False,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_frame


def main() -> None:
    print(build_cohort(parse_args()).to_string(index=False))


if __name__ == "__main__":
    main()
