from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultipleTestingAdjustment:
    trial_count: int
    bonferroni_p_value: float
    holm_p_value: float
    fdr_q_value: float
    significance: str


def stable_trial_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def bonferroni_p_value(raw_p_value: float, trial_count: int) -> float:
    if not np.isfinite(raw_p_value) or trial_count <= 0:
        return np.nan
    return float(min(max(raw_p_value, 0.0) * trial_count, 1.0))


def holm_bonferroni_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce")
    valid = values.dropna().clip(lower=0.0, upper=1.0)
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return adjusted

    ordered = valid.sort_values(kind="mergesort")
    m = len(ordered)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(ordered.items(), start=1):
        candidate = min((m - rank + 1) * float(p_value), 1.0)
        running_max = max(running_max, candidate)
        adjusted.loc[idx] = running_max
    return adjusted.clip(upper=1.0)


def benjamini_hochberg_q_values(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce")
    valid = values.dropna().clip(lower=0.0, upper=1.0)
    q_values = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return q_values

    ordered = valid.sort_values(ascending=False, kind="mergesort")
    m = len(ordered)
    running_min = 1.0
    for reverse_rank, (idx, p_value) in enumerate(ordered.items(), start=1):
        rank = m - reverse_rank + 1
        candidate = min(float(p_value) * m / rank, 1.0)
        running_min = min(running_min, candidate)
        q_values.loc[idx] = running_min
    return q_values.clip(upper=1.0)


def significance_label(raw_p_value: float, adjusted_p_value: float, alpha: float = 0.05) -> str:
    if not np.isfinite(raw_p_value):
        return "missing"
    if raw_p_value > alpha:
        return "raw_not_significant"
    if not np.isfinite(adjusted_p_value):
        return "uncorrected"
    if adjusted_p_value <= alpha:
        return "survives_multiple_testing"
    return "fails_multiple_testing"
