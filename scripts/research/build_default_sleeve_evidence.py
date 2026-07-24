#!/usr/bin/env python3
"""Build the frozen Phase 3 default sleeve for the Phase 1B factor cohort."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from oqp.research.factor_definitions import resolve_factor_definition
from oqp.research.factor_portfolios.data import attach_instrument_classification
from oqp.research.factors import load_factor_module, resolve_factor_path
from oqp.research.sleeves import (
    build_sleeve_evidence,
    build_sleeve_targets,
    load_sleeve_module,
    write_sleeve_evidence_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "phase1b_daily_mean_reversion"
)
DEFAULT_PREDICTIVE_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "predictive_evidence"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "artifacts" / "research" / "sleeve_construction"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--predictive-root", type=Path, default=DEFAULT_PREDICTIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--market-vertical", default="FUTURES_CN")
    parser.add_argument("--factor-id", action="append", dest="factor_ids")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_upstream(
    protocol: dict,
    predictive_root: Path,
    factor_id: str,
    market_vertical: str,
) -> dict:
    manifest = _read_json(
        predictive_root / factor_id / market_vertical / "manifest.json"
    )
    if manifest.get("factor_id") != factor_id:
        raise ValueError(f"Phase 2 identity mismatch for {factor_id}")
    if manifest.get("input_data_fingerprint") != protocol.get("source_sha256"):
        raise ValueError(f"Phase 2 input fingerprint mismatch for {factor_id}")
    alignment = manifest.get("causal_alignment") or {}
    if not alignment.get("verified"):
        raise ValueError(f"Phase 2 causal alignment is not verified for {factor_id}")
    config = manifest.get("config") or {}
    if config.get("return_col") != protocol.get("primary_return_col"):
        raise ValueError(f"Phase 2 return column mismatch for {factor_id}")
    return manifest


def _load_panel(cohort_root: Path, factor_id: str) -> pd.DataFrame:
    path = cohort_root / "factor_outputs" / f"{factor_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    cohort_root = args.cohort_root.expanduser().resolve()
    predictive_root = args.predictive_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    protocol = _read_json(cohort_root / "protocol.json")
    factor_ids = args.factor_ids or list(protocol.get("factor_ids") or [])
    if not factor_ids:
        raise ValueError("the frozen cohort has no factor IDs")
    unknown = sorted(set(factor_ids).difference(protocol["factor_ids"]))
    if unknown:
        raise ValueError(f"factor IDs are outside the frozen cohort: {unknown}")
    signature = protocol.get("contract_signature") or {}
    if signature.get("evaluation_geometry") != "cross_sectional":
        raise ValueError("the default Phase 3 sleeve requires cross-sectional factors")
    if signature.get("return_assumption") != "close_signal_next_open_to_close":
        raise ValueError("the current Phase 3 runner requires next-open-to-close returns")

    holdout_start = pd.Timestamp(protocol["holdout_start"])
    sleeve_module = load_sleeve_module(
        "slv_001_Cross_Sectional_Quintile_Long_Short"
    )
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        phase2_manifest = _verify_upstream(
            protocol, predictive_root, factor_id, args.market_vertical
        )
        module = load_factor_module(factor_id, include_public_examples=False)
        definition = resolve_factor_definition(resolve_factor_path(factor_id), module)
        if (
            phase2_manifest.get("factor_definition_fingerprint")
            != definition.fingerprint
        ):
            raise ValueError(
                f"Phase 2 evidence is stale for the current {factor_id} definition"
            )
        if definition.native_market != args.market_vertical:
            raise ValueError(f"{factor_id} is native to {definition.native_market}")
        if definition.evaluation_geometry != "cross_sectional":
            raise ValueError(f"{factor_id} is not a cross-sectional factor")

        frame = _load_panel(cohort_root, factor_id)
        observed = set(frame["factor_id"].dropna().astype(str).unique())
        if observed != {factor_id}:
            raise ValueError(f"factor output identity mismatch: {sorted(observed)}")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["research_split"] = "validation"
        frame.loc[frame["date"].ge(holdout_start), "research_split"] = "holdout"
        frame = attach_instrument_classification(frame, args.market_vertical)
        frame.attrs["causal_return_alignment_verified"] = True
        frame.attrs["input_data_fingerprint"] = protocol["source_sha256"]
        frame.attrs["factor_definition_fingerprint"] = definition.fingerprint
        frame.attrs["factor_implementation_fingerprint"] = (
            definition.implementation_fingerprint
        )
        frame.attrs["sleeve_implementation_fingerprint"] = str(
            sleeve_module.SLEEVE_IMPLEMENTATION_FINGERPRINT
        )
        frame.attrs["sleeve_definition_fingerprint"] = str(
            sleeve_module.SLEEVE_DEFINITION_FINGERPRINT
        )
        frame.attrs["predictive_evidence_config_fingerprint"] = phase2_manifest[
            "config_fingerprint"
        ]

        config = sleeve_module.build_config(
            factor_id,
            market_vertical=args.market_vertical,
            signal_orientation=definition.signal_orientation,
        )
        construction = build_sleeve_targets(frame, config)
        bundle = build_sleeve_evidence(
            construction,
            capital=float(protocol["capital_cny"]),
            slippage_ticks_per_side=float(protocol["slippage_ticks_per_side"]),
        )
        destination = write_sleeve_evidence_bundle(
            bundle,
            output_root / factor_id / args.market_vertical / config.sleeve_id,
        )
        summary = bundle.split_summary.set_index("research_split")
        full = summary.loc["full"]
        validation = summary.loc["validation"]
        holdout = summary.loc["holdout"]
        rows.append(
            {
                "factor_id": factor_id,
                "market_vertical": args.market_vertical,
                "sleeve_id": config.sleeve_id,
                "net_annualized_mean": full["net_annualized_mean"],
                "net_sharpe": full["net_sharpe"],
                "maximum_drawdown": full["maximum_drawdown"],
                "annualized_turnover": full["annualized_turnover"],
                "annualized_cost": full["annualized_cost"],
                "validation_net_sharpe": validation["net_sharpe"],
                "holdout_net_sharpe": holdout["net_sharpe"],
                "mean_executed_gross": full["mean_executed_gross"],
                "artifact_path": str(destination.relative_to(REPO_ROOT)),
            }
        )
        print(
            f"built {factor_id}: net Sharpe={full['net_sharpe']:.3f}, "
            f"holdout={holdout['net_sharpe']:.3f}, "
            f"turnover={full['annualized_turnover']:.1f}x",
            flush=True,
        )
        del frame, construction, bundle
        gc.collect()

    index = pd.DataFrame(rows).sort_values("factor_id").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_root / "evidence_index.csv", index=False)
    run_manifest = {
        "schema_version": 1,
        "phase": "Phase 3: Sleeve Construction",
        "cohort_root": str(cohort_root.relative_to(REPO_ROOT)),
        "predictive_evidence_root": str(predictive_root.relative_to(REPO_ROOT)),
        "market_vertical": args.market_vertical,
        "factor_ids": factor_ids,
        "sleeve_id": "slv_001_Cross_Sectional_Quintile_Long_Short",
        "optimization_permitted": False,
        "capital": float(protocol["capital_cny"]),
        "slippage_ticks_per_side": float(protocol["slippage_ticks_per_side"]),
        "source_fingerprint": protocol["source_sha256"],
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
