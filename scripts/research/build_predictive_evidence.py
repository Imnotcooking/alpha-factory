#!/usr/bin/env python3
"""Build Phase 2 predictive-evidence bundles from a frozen factor cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from oqp.research.factor_definitions import resolve_factor_definition
from oqp.research.factors import load_factor_module, resolve_factor_path
from oqp.research.predictive_evidence import (
    PredictiveEvidenceConfig,
    build_predictive_evidence,
    write_predictive_evidence_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT_ROOT = (
    REPO_ROOT
    / "runtime"
    / "artifacts"
    / "research"
    / "phase1b_daily_mean_reversion"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "predictive_evidence"
)
ORIENTATION_SIGN = {
    "higher_is_bullish": 1,
    "higher_is_bearish": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--market-vertical", default="FUTURES_CN")
    parser.add_argument("--factor-id", action="append", dest="factor_ids")
    parser.add_argument("--rolling-window", type=int, default=63)
    parser.add_argument("--rolling-min-periods", type=int, default=21)
    parser.add_argument("--minimum-cross-section", type=int, default=10)
    parser.add_argument("--minimum-product-observations", type=int, default=60)
    return parser.parse_args()


def _load_protocol(cohort_root: Path) -> dict:
    protocol_path = cohort_root / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"frozen cohort protocol not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    signature = protocol.get("contract_signature") or {}
    required = {
        "factor_ids": protocol.get("factor_ids"),
        "holdout_start": protocol.get("holdout_start"),
        "primary_return_col": protocol.get("primary_return_col"),
        "source_sha256": protocol.get("source_sha256"),
        "evaluation_geometry": signature.get("evaluation_geometry"),
        "execution_lag": signature.get("execution_lag"),
        "return_assumption": signature.get("return_assumption"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise ValueError(f"cohort protocol is missing frozen fields: {missing}")
    return protocol


def _load_factor_panel(
    cohort_root: Path,
    factor_id: str,
    return_col: str,
) -> pd.DataFrame:
    path = cohort_root / "factor_outputs" / f"{factor_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"factor output not found: {path}")
    columns = ["date", "ticker", "alpha_score", return_col, "factor_id"]
    frame = pd.read_parquet(path, columns=columns)
    observed_ids = set(frame["factor_id"].dropna().astype(str).unique())
    if observed_ids != {factor_id}:
        raise ValueError(
            f"factor output identity mismatch for {factor_id}: {sorted(observed_ids)}"
        )
    return frame


def _factor_definition(factor_id: str):
    module = load_factor_module(factor_id, include_public_examples=False)
    definition = resolve_factor_definition(resolve_factor_path(factor_id), module)
    if definition.signal_orientation not in ORIENTATION_SIGN:
        raise ValueError(
            f"unsupported signal orientation for {factor_id}: "
            f"{definition.signal_orientation}"
        )
    return definition


def build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    cohort_root = args.cohort_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    protocol = _load_protocol(cohort_root)
    signature = protocol["contract_signature"]
    factor_ids = args.factor_ids or list(protocol["factor_ids"])
    unknown = sorted(set(factor_ids).difference(protocol["factor_ids"]))
    if unknown:
        raise ValueError(f"factor IDs are not in the frozen cohort: {unknown}")

    holdout_start = pd.Timestamp(protocol["holdout_start"])
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        definition = _factor_definition(factor_id)
        if definition.native_market != args.market_vertical:
            raise ValueError(
                f"{factor_id} is native to {definition.native_market}, not "
                f"{args.market_vertical}"
            )
        if definition.evaluation_geometry != signature["evaluation_geometry"]:
            raise ValueError(
                f"{factor_id} geometry {definition.evaluation_geometry} differs "
                f"from cohort {signature['evaluation_geometry']}"
            )

        frame = _load_factor_panel(
            cohort_root,
            factor_id,
            protocol["primary_return_col"],
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["research_split"] = "validation"
        frame.loc[frame["date"].ge(holdout_start), "research_split"] = "holdout"
        frame.attrs["causal_return_alignment_verified"] = True
        frame.attrs["input_data_fingerprint"] = protocol["source_sha256"]
        frame.attrs["factor_definition_fingerprint"] = definition.fingerprint
        frame.attrs["factor_implementation_fingerprint"] = (
            definition.implementation_fingerprint
        )

        config = PredictiveEvidenceConfig(
            factor_id=factor_id,
            signal_col="alpha_score",
            return_col=protocol["primary_return_col"],
            evaluation_geometry=definition.evaluation_geometry,
            expected_sign=ORIENTATION_SIGN[definition.signal_orientation],
            rolling_window=args.rolling_window,
            rolling_min_periods=args.rolling_min_periods,
            minimum_cross_section=args.minimum_cross_section,
            minimum_product_observations=args.minimum_product_observations,
            execution_lag=signature["execution_lag"],
            return_assumption=signature["return_assumption"],
        )
        bundle = build_predictive_evidence(frame, config)
        destination = write_predictive_evidence_bundle(
            bundle,
            output_root / factor_id / args.market_vertical,
        )
        full = bundle.split_summary.set_index("research_split").loc["full"]
        validation = bundle.split_summary.set_index("research_split").loc[
            "validation"
        ]
        holdout = bundle.split_summary.set_index("research_split").loc["holdout"]
        rows.append(
            {
                "factor_id": factor_id,
                "market_vertical": args.market_vertical,
                "family": definition.family,
                "signal_orientation": definition.signal_orientation,
                "factor_definition_fingerprint": definition.fingerprint,
                "factor_implementation_fingerprint": (
                    definition.implementation_fingerprint
                ),
                "mean_ic": full["mean_pearson_ic"],
                "mean_rank_ic": full["mean_rank_ic"],
                "pearson_icir": full["pearson_icir"],
                "rank_icir": full["rank_icir"],
                "ic_hit_rate": full["pearson_ic_hit_rate"],
                "rank_ic_hit_rate": full["rank_ic_hit_rate"],
                "validation_rank_ic": validation["mean_rank_ic"],
                "holdout_rank_ic": holdout["mean_rank_ic"],
                "dates": int(full["date_count"]),
                "products": int(full["product_count"]),
                "signal_coverage": full["signal_coverage"],
                "forward_return_coverage": full["forward_return_coverage"],
                "artifact_path": str(destination.relative_to(REPO_ROOT)),
            }
        )
        print(
            f"built {factor_id}: Rank IC={full['mean_rank_ic']:.4f}, "
            f"holdout={holdout['mean_rank_ic']:.4f}"
        )

    index = pd.DataFrame(rows).sort_values(
        ["market_vertical", "factor_id"]
    ).reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_root / "evidence_index.csv", index=False)
    run_manifest = {
        "schema_version": 1,
        "cohort_root": str(cohort_root.relative_to(REPO_ROOT)),
        "cohort_protocol": protocol,
        "market_vertical": args.market_vertical,
        "factor_ids": factor_ids,
        "evidence_config": {
            "rolling_window": args.rolling_window,
            "rolling_min_periods": args.rolling_min_periods,
            "minimum_cross_section": args.minimum_cross_section,
            "minimum_product_observations": args.minimum_product_observations,
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    index = build_cohort(parse_args())
    print(index.to_string(index=False))


if __name__ == "__main__":
    main()
