from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.research.tick_pulse.constants import (
    BREAKDOWN_MAX_BOOK_IMBALANCE,
    BREAKDOWN_MAX_ROLLING_MID_MOVE,
    BREAKDOWN_MIN_PRICE_SHOCK,
    BREAKDOWN_MIN_VOLUME_INTENSITY,
    EVENT_CLUSTER_GAP_TICKS,
    MIN_BOOK_IMBALANCE,
    MIN_FLOW_IMBALANCE,
    MIN_RESILIENCE_TICKS,
    MIN_VOLUME_INTENSITY,
    RTV_FAST_WINDOW_TICKS,
    RTV_MIN_FAST_MOVE_TICKS,
    RTV_PERCENTILE,
    RTV_SLOW_WINDOW_TICKS,
)
from oqp.research.tick_pulse.native_bridge import load_tick_pulse_core


def _feature_group_keys(df: pd.DataFrame) -> list[str]:
    return ["symbol", "_session_id"] if "_session_id" in df.columns else ["symbol"]


def _direction_labels(codes: np.ndarray) -> np.ndarray:
    return np.select(
        [codes > 0, codes < 0],
        ["Up", "Down"],
        default="Flat",
    )


def _expected_direction_label(code: int) -> str:
    if code > 0:
        return "Up"
    if code < 0:
        return "Down"
    return "Flat"


def _encode_int_codes(values: pd.Series) -> np.ndarray:
    return pd.Categorical(values).codes.astype(np.int32, copy=False)


def _assign_python_cluster_ids(candidates: pd.DataFrame, gap_ticks: int) -> pd.DataFrame:
    if candidates.empty or "_event_row_pos" not in candidates.columns:
        return candidates

    keys = _feature_group_keys(candidates)
    pieces = []
    cluster_offset = 0
    for _, group in candidates.sort_values([*keys, "_event_row_pos"]).groupby(keys, sort=False):
        group = group.copy()
        new_episode = group["_event_row_pos"].diff().gt(gap_ticks).fillna(True)
        local_cluster = new_episode.cumsum().astype(int) - 1
        group["_event_cluster_id"] = local_cluster + cluster_offset
        pieces.append(group)
        cluster_offset += int(local_cluster.max()) + 1

    if not pieces:
        return candidates
    return pd.concat(pieces).sort_index()


def compute_rtv_frame(
    features: pd.DataFrame,
    *,
    horizon_ticks: int,
    fast_window: int,
    slow_window: int,
    percentile: float,
    min_periods: int,
    min_fast_move_ticks: float,
    min_success_ticks: float,
    fade: bool,
    gap_ticks: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc = load_tick_pulse_core(required_features=("compute_tick_rtv_pipeline",))

    if not hasattr(qc, "compute_tick_rtv_pipeline"):
        raise AttributeError("native quant core is missing compute_tick_rtv_pipeline.")

    required = {"symbol", "datetime", "mid_price", "tick_size_est"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"C++ RTV bridge missing required columns: {sorted(missing)}")

    out = features.sort_values(["symbol", "datetime"]).copy()
    out["_row_pos"] = out.groupby("symbol", sort=False).cumcount()
    group_keys = _feature_group_keys(out)
    grouped = out.groupby(group_keys, sort=False)
    out["_event_row_pos"] = grouped.cumcount()

    tick_size = out["tick_size_est"].replace(0, np.nan)
    session_values = out["_session_id"] if "_session_id" in out.columns else pd.Series(0, index=out.index)

    result = qc.compute_tick_rtv_pipeline(
        np.ascontiguousarray(out["mid_price"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(tick_size.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(_encode_int_codes(out["symbol"])),
        np.ascontiguousarray(_encode_int_codes(session_values)),
        int(horizon_ticks),
        int(fast_window),
        int(slow_window),
        float(percentile),
        int(min_periods),
        float(min_fast_move_ticks),
        float(min_success_ticks),
        bool(fade),
        int(gap_ticks),
    )

    direction_codes = np.asarray(result["rtv_direction_code"])
    expected_codes = np.asarray(result["expected_direction_code"])
    out["future_datetime"] = grouped["datetime"].shift(-horizon_ticks)
    out["future_mid_price"] = np.asarray(result["future_mid_price"], dtype=float)
    out["future_move_ticks"] = np.asarray(result["future_move_ticks"], dtype=float)
    out["future_return_bps"] = (out["future_mid_price"] / out["mid_price"] - 1.0) * 10000.0
    out["rtv_fast_move_ticks"] = np.asarray(result["rtv_fast_move_ticks"], dtype=float)
    out["rtv_abs_move_ticks"] = np.asarray(result["rtv_abs_move_ticks"], dtype=float)
    out["rtv_threshold_ticks"] = np.asarray(result["rtv_threshold_ticks"], dtype=float)
    out["rtv_threshold_ratio"] = np.asarray(result["rtv_threshold_ratio"], dtype=float)
    out["rtv_direction"] = _direction_labels(direction_codes)
    out["criterion_velocity_percentile"] = out["rtv_abs_move_ticks"] >= out["rtv_threshold_ticks"]
    out["criterion_min_fast_move"] = out["rtv_abs_move_ticks"] >= float(min_fast_move_ticks)
    out["expected_direction"] = _direction_labels(expected_codes)
    out["hypothesis_signal"] = np.asarray(result["hypothesis_signal"], dtype=bool)
    out["is_correct"] = np.asarray(result["is_correct"], dtype=bool)

    candidate_indices = np.asarray(result["candidate_indices"], dtype=np.int64)
    if len(candidate_indices):
        candidates = out.iloc[candidate_indices].copy()
        candidates["is_correct"] = np.asarray(result["is_correct"], dtype=bool)[candidate_indices]
        candidates["outcome"] = np.where(candidates["is_correct"], "Correct", "Failed")
        candidates["_event_cluster_id"] = np.asarray(result["event_cluster_id"], dtype=np.int64)
        candidates["_event_cluster_size"] = np.asarray(result["event_cluster_size"], dtype=np.int64)
        candidates = _assign_python_cluster_ids(candidates, gap_ticks)
        candidates = candidates.sort_values("datetime").copy()
    else:
        candidates = out.iloc[0:0].copy()
        candidates["is_correct"] = pd.Series(dtype=bool)
        candidates["outcome"] = pd.Series(dtype=object)
        candidates["_event_cluster_id"] = pd.Series(dtype=np.int64)
        candidates["_event_cluster_size"] = pd.Series(dtype=np.int64)

    out.attrs["tick_pulse_backend"] = "cpp"
    candidates.attrs["tick_pulse_backend"] = "cpp"
    return out, candidates


def compute_heuristic_frame(
    features: pd.DataFrame,
    *,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    thresholds: dict[str, float],
    gap_ticks: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc = load_tick_pulse_core(required_features=("compute_tick_heuristic_pipeline",))

    if not hasattr(qc, "compute_tick_heuristic_pipeline"):
        raise AttributeError("native quant core is missing compute_tick_heuristic_pipeline.")

    hypothesis_codes = {
        "bullish": 0,
        "bearish": 1,
        "bearish_breakdown": 2,
    }
    if hypothesis not in hypothesis_codes:
        raise ValueError(f"C++ heuristic bridge does not support hypothesis: {hypothesis}")

    required = {
        "symbol",
        "datetime",
        "mid_price",
        "tick_size_est",
        "flow_imbalance",
        "book_imbalance",
        "volume_intensity",
        "rolling_mid_move_ticks",
        "price_shock",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"C++ heuristic bridge missing required columns: {sorted(missing)}")

    out = features.sort_values(["symbol", "datetime"]).copy()
    out["_row_pos"] = out.groupby("symbol", sort=False).cumcount()
    group_keys = _feature_group_keys(out)
    grouped = out.groupby(group_keys, sort=False)
    out["_event_row_pos"] = grouped.cumcount()

    tick_size = out["tick_size_est"].replace(0, np.nan)
    session_values = out["_session_id"] if "_session_id" in out.columns else pd.Series(0, index=out.index)

    result = qc.compute_tick_heuristic_pipeline(
        np.ascontiguousarray(out["mid_price"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(tick_size.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["flow_imbalance"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["book_imbalance"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["volume_intensity"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["rolling_mid_move_ticks"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["price_shock"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(_encode_int_codes(out["symbol"])),
        np.ascontiguousarray(_encode_int_codes(session_values)),
        int(horizon_ticks),
        int(hypothesis_codes[hypothesis]),
        float(min_success_ticks),
        float(thresholds["flow_sell_max"]),
        float(thresholds["book_buy_min"]),
        float(thresholds["book_sell_max"]),
        float(thresholds["breakdown_book_max"]),
        float(thresholds["volume_burst_min"]),
        float(thresholds["breakdown_volume_burst_min"]),
        float(thresholds["rolling_mid_up_min"]),
        float(thresholds["rolling_mid_down_max"]),
        float(thresholds["breakdown_rolling_mid_max"]),
        float(thresholds["price_shock_min"]),
        int(gap_ticks),
    )

    expected_codes = np.asarray(result["expected_direction_code"])
    out["future_datetime"] = grouped["datetime"].shift(-horizon_ticks)
    out["future_mid_price"] = np.asarray(result["future_mid_price"], dtype=float)
    out["future_move_ticks"] = np.asarray(result["future_move_ticks"], dtype=float)
    out["future_return_bps"] = (out["future_mid_price"] / out["mid_price"] - 1.0) * 10000.0
    out["criterion_flow"] = np.asarray(result["criterion_flow"], dtype=bool)
    out["criterion_book"] = np.asarray(result["criterion_book"], dtype=bool)
    out["criterion_price_resilience"] = np.asarray(result["criterion_price_resilience"], dtype=bool)
    out["criterion_volume_burst"] = np.asarray(result["criterion_volume_burst"], dtype=bool)
    if hypothesis == "bearish_breakdown":
        out["criterion_price_shock"] = np.asarray(result["criterion_price_shock"], dtype=bool)
    out["expected_direction"] = np.array([_expected_direction_label(code) for code in expected_codes])
    out["hypothesis_signal"] = np.asarray(result["hypothesis_signal"], dtype=bool)
    out["is_correct"] = np.asarray(result["is_correct"], dtype=bool)

    candidate_indices = np.asarray(result["candidate_indices"], dtype=np.int64)
    if len(candidate_indices):
        candidates = out.iloc[candidate_indices].copy()
        candidates["is_correct"] = out["is_correct"].to_numpy()[candidate_indices]
        candidates["outcome"] = np.where(candidates["is_correct"], "Correct", "Failed")
        candidates["_event_cluster_id"] = np.asarray(result["event_cluster_id"], dtype=np.int64)
        candidates["_event_cluster_size"] = np.asarray(result["event_cluster_size"], dtype=np.int64)
        candidates = candidates.sort_values("datetime").copy()
    else:
        candidates = out.iloc[0:0].copy()
        candidates["is_correct"] = pd.Series(dtype=bool)
        candidates["outcome"] = pd.Series(dtype=object)
        candidates["_event_cluster_id"] = pd.Series(dtype=np.int64)
        candidates["_event_cluster_size"] = pd.Series(dtype=np.int64)

    out.attrs["tick_pulse_backend"] = "cpp"
    candidates.attrs["tick_pulse_backend"] = "cpp"
    return out, candidates


def compute_relative_velocity_cpp(
    features: pd.DataFrame,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    min_fast_move_ticks: float = RTV_MIN_FAST_MOVE_TICKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the relative-velocity hypothesis through the native bridge."""
    min_periods = min(500, max(50, RTV_SLOW_WINDOW_TICKS // 10))
    return compute_rtv_frame(
        features,
        horizon_ticks=horizon_ticks,
        fast_window=RTV_FAST_WINDOW_TICKS,
        slow_window=RTV_SLOW_WINDOW_TICKS,
        percentile=RTV_PERCENTILE,
        min_periods=min_periods,
        min_fast_move_ticks=min_fast_move_ticks,
        min_success_ticks=min_success_ticks,
        fade=hypothesis == "relative_velocity_fade",
        gap_ticks=EVENT_CLUSTER_GAP_TICKS,
    )


def compute_heuristic_cpp(
    features: pd.DataFrame,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate deterministic tick-pulse heuristics through the native bridge."""
    thresholds = thresholds or {}
    normalized_thresholds = {
        "flow_sell_max": thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE),
        "book_buy_min": thresholds.get("book_buy_min", MIN_BOOK_IMBALANCE),
        "book_sell_max": thresholds.get("book_sell_max", -MIN_BOOK_IMBALANCE),
        "breakdown_book_max": thresholds.get("breakdown_book_max", BREAKDOWN_MAX_BOOK_IMBALANCE),
        "volume_burst_min": thresholds.get("volume_burst_min", MIN_VOLUME_INTENSITY),
        "breakdown_volume_burst_min": thresholds.get(
            "breakdown_volume_burst_min",
            BREAKDOWN_MIN_VOLUME_INTENSITY,
        ),
        "rolling_mid_up_min": thresholds.get("rolling_mid_up_min", MIN_RESILIENCE_TICKS),
        "rolling_mid_down_max": thresholds.get("rolling_mid_down_max", -MIN_RESILIENCE_TICKS),
        "breakdown_rolling_mid_max": thresholds.get(
            "breakdown_rolling_mid_max",
            BREAKDOWN_MAX_ROLLING_MID_MOVE,
        ),
        "price_shock_min": thresholds.get("price_shock_min", BREAKDOWN_MIN_PRICE_SHOCK),
    }
    return compute_heuristic_frame(
        features,
        horizon_ticks=horizon_ticks,
        hypothesis=hypothesis,
        min_success_ticks=min_success_ticks,
        thresholds=normalized_thresholds,
        gap_ticks=EVENT_CLUSTER_GAP_TICKS,
    )
