from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from oqp.research.tick_pulse.constants import RESEARCH_SWEEP_HORIZONS
from oqp.research.tick_pulse.engine import (
    _evaluate_horizon_summary,
    build_adaptive_thresholds,
)
from oqp.research.tick_pulse.feature_bridge import build_pulse_features_fast
from oqp.research.tick_pulse.features import contract_summary, load_tick_scope, load_ticks


def effective_thresholds(
    thresholds: dict[str, float] | None,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Overlay adaptive threshold values on a caller-provided default set."""
    thresholds = thresholds or {}
    return {key: thresholds.get(key, value) for key, value in defaults.items()}


def compute_main_contract_file_sweep(
    path: str | Path,
    *,
    hypothesis: str,
    window: int,
    min_success_ticks: float,
    threshold_mode: str,
    default_thresholds: dict[str, float],
    source_base_dir: str | Path | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run a horizon sweep on each product's most active contract in one tick file.

    This is intentionally Streamlit-free so dashboards, scripts, and tests can
    reuse the same event-study contract.
    """
    path = Path(path)
    raw = load_ticks(str(path))
    summary = contract_summary(raw)
    if summary.empty:
        return pd.DataFrame()

    rows = []
    horizon_values = horizons or RESEARCH_SWEEP_HORIZONS
    source_file = _relpath(path, source_base_dir)
    for product in sorted(summary["product"].dropna().unique()):
        product_summary = summary[summary["product"] == product]
        if product_summary.empty:
            continue

        main_contract = product_summary.sort_values("positive_volume_delta", ascending=False).iloc[0]
        symbol = str(main_contract["symbol"])
        scoped = load_tick_scope(str(path), product=product, symbol=symbol)
        if scoped.empty:
            continue

        features = build_pulse_features_fast(scoped, window=window)
        thresholds = build_adaptive_thresholds(features, hypothesis) if threshold_mode == "adaptive" else {}
        resolved_thresholds = effective_thresholds(thresholds, default_thresholds)
        threshold_payload = {
            f"threshold_{key}": value
            for key, value in resolved_thresholds.items()
        }
        threshold_mode_detail_code = (
            "adaptive_per_asset" if threshold_mode == "adaptive" else "fixed_defaults"
        )
        threshold_rule_code = f"{threshold_mode}_{hypothesis}"

        for horizon in horizon_values:
            result = _evaluate_horizon_summary(
                features,
                horizon,
                hypothesis,
                min_success_ticks,
                thresholds,
            )
            rows.append(
                {
                    "asset": product,
                    "main_contract": symbol,
                    "horizon": horizon,
                    "events": result["events"],
                    "successes": result["successes"],
                    "accuracy": result["accuracy"],
                    "base_rate": result["base_rate"],
                    "lift": result["lift"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "avg_move": result["avg_move"],
                    "expected_avg": result["expected_avg"],
                    "backend": result.get("backend", "python"),
                    "source_file": source_file,
                    "threshold_mode": threshold_mode,
                    "threshold_mode_detail_code": threshold_mode_detail_code,
                    "threshold_rule_code": threshold_rule_code,
                    "feature_rows": len(features),
                    **threshold_payload,
                }
            )

    return pd.DataFrame(rows)


def _relpath(path: Path, source_base_dir: str | Path | None) -> str:
    if source_base_dir is None:
        return str(path)
    try:
        return os.path.relpath(path, source_base_dir)
    except ValueError:
        return str(path)
