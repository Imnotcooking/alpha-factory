"""Diagnostics for factor overlap, coverage and composite attribution."""

from __future__ import annotations

import pandas as pd

from oqp.research.factor_portfolios.composer import CompositionResult


def factor_correlation(result: CompositionResult) -> pd.DataFrame:
    frame = result.frame[list(result.normalized_columns.values())].rename(
        columns={value: key for key, value in result.normalized_columns.items()}
    )
    return frame.corr(min_periods=3)


def factor_coverage(result: CompositionResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_rows = max(len(result.frame), 1)
    for factor_id, column in result.normalized_columns.items():
        valid_rows = int(result.frame[column].notna().sum())
        rows.append(
            {
                "factor_id": factor_id,
                "configured_weight": result.configured_weights[factor_id],
                "valid_rows": valid_rows,
                "coverage": valid_rows / total_rows,
            }
        )
    return pd.DataFrame(rows)


def contribution_summary(result: CompositionResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    composite = pd.to_numeric(result.frame["composite_score"], errors="coerce")
    for factor_id, column in result.contribution_columns.items():
        contribution = pd.to_numeric(result.frame[column], errors="coerce")
        rows.append(
            {
                "factor_id": factor_id,
                "configured_weight": result.configured_weights[factor_id],
                "mean_abs_contribution": float(contribution.abs().mean()),
                "correlation_to_composite": float(contribution.corr(composite)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_summary(result: CompositionResult) -> pd.DataFrame:
    """Measure how much the composite changes when each factor is removed."""

    original = pd.to_numeric(result.frame["composite_score"], errors="coerce")
    rows: list[dict[str, object]] = []
    for omitted_factor in result.normalized_columns:
        remaining = [
            factor_id
            for factor_id in result.normalized_columns
            if factor_id != omitted_factor
        ]
        remaining_weight = sum(result.configured_weights[factor_id] for factor_id in remaining)
        numerator = pd.Series(0.0, index=result.frame.index)
        denominator = pd.Series(0.0, index=result.frame.index)
        for factor_id in remaining:
            values = pd.to_numeric(
                result.frame[result.normalized_columns[factor_id]],
                errors="coerce",
            )
            weight = result.configured_weights[factor_id]
            numerator = numerator.add(values.fillna(0.0) * weight, fill_value=0.0)
            denominator = denominator.add(values.notna().astype(float) * weight, fill_value=0.0)
        leave_one_out = numerator / denominator.where(denominator.ne(0.0))
        rows.append(
            {
                "omitted_factor": omitted_factor,
                "remaining_configured_weight": remaining_weight,
                "correlation_to_full": float(leave_one_out.corr(original)),
                "mean_abs_signal_change": float((leave_one_out - original).abs().mean()),
                "valid_rows": int(leave_one_out.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "contribution_summary",
    "factor_correlation",
    "factor_coverage",
    "leave_one_out_summary",
]
