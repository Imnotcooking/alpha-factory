from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    BREAKDOWN_MAX_BOOK_IMBALANCE,
    BREAKDOWN_MAX_ROLLING_MID_MOVE,
    BREAKDOWN_MIN_PRICE_SHOCK,
    BREAKDOWN_MIN_VOLUME_INTENSITY,
    EVENT_CLUSTER_GAP_TICKS,
    HYPOTHESIS_KEYS,
    MIN_BOOK_IMBALANCE,
    MIN_FLOW_IMBALANCE,
    MIN_RESILIENCE_TICKS,
    MIN_VOLUME_INTENSITY,
    RESEARCH_SWEEP_HORIZONS,
    RTV_FAST_WINDOW_TICKS,
    RTV_MIN_FAST_MOVE_TICKS,
    RTV_PERCENTILE,
    RTV_SLOW_WINDOW_TICKS,
)

def _feature_group_keys(df: pd.DataFrame) -> list[str]:
    return ["symbol", "_session_id"] if "_session_id" in df.columns else ["symbol"]


def _collapse_event_episodes(candidates: pd.DataFrame, gap_ticks: int = EVENT_CLUSTER_GAP_TICKS) -> pd.DataFrame:
    row_pos_column = "_event_row_pos" if "_event_row_pos" in candidates.columns else "_row_pos"
    if candidates.empty or row_pos_column not in candidates.columns:
        return candidates

    keys = ["symbol", "_session_id"] if "_session_id" in candidates.columns else ["symbol"]
    pieces = []
    cluster_offset = 0
    for _, group in candidates.sort_values([*keys, row_pos_column]).groupby(keys, sort=False):
        group = group.copy()
        new_episode = group[row_pos_column].diff().gt(gap_ticks).fillna(True)
        local_cluster = new_episode.cumsum().astype(int) - 1
        group["_event_cluster_id"] = local_cluster + cluster_offset
        group["_event_cluster_size"] = group.groupby("_event_cluster_id")["_event_cluster_id"].transform("size")
        pieces.append(group[group["_event_cluster_id"].ne(group["_event_cluster_id"].shift(1))])
        cluster_offset += int(local_cluster.max()) + 1

    if not pieces:
        return candidates.iloc[0:0].copy()
    return pd.concat(pieces).sort_values("datetime").copy()


def _is_bearish_hypothesis(hypothesis: str) -> bool:
    return hypothesis in {"bearish", "bearish_breakdown"}


def _is_relative_velocity_hypothesis(hypothesis: str) -> bool:
    return hypothesis in {"relative_velocity", "relative_velocity_fade"}


def _is_relative_velocity_fade_hypothesis(hypothesis: str) -> bool:
    return hypothesis == "relative_velocity_fade"


def _velocity_expected_direction(breakout_direction: pd.Series, hypothesis: str) -> np.ndarray:
    if _is_relative_velocity_fade_hypothesis(hypothesis):
        return np.select(
            [breakout_direction.eq("Up"), breakout_direction.eq("Down")],
            ["Down", "Up"],
            default="Flat",
        )
    return breakout_direction.to_numpy()


def _clean_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    return values[np.isfinite(values)]


def _safe_quantile(series: pd.Series, q: float, fallback: float) -> float:
    values = _clean_numeric(series)
    if len(values) < 100:
        return float(fallback)
    value = np.quantile(values, q)
    return float(value) if pd.notna(value) else float(fallback)


def build_adaptive_thresholds(features: pd.DataFrame, hypothesis: str) -> dict[str, float]:
    """Probability-match hypothesis thresholds to the selected contract."""
    work = features.copy()
    if _is_relative_velocity_hypothesis(hypothesis):
        grouped = work.groupby(_feature_group_keys(work), sort=False)
        tick_size = work["tick_size_est"].replace(0, np.nan)
        work["rtv_abs_move_ticks"] = (
            (work["mid_price"] - grouped["mid_price"].shift(RTV_FAST_WINDOW_TICKS)) / tick_size
        ).abs()

    return {
        "flow_sell_max": _safe_quantile(work["flow_imbalance"], 0.10, -MIN_FLOW_IMBALANCE),
        "book_buy_min": _safe_quantile(work["book_imbalance"], 0.70, MIN_BOOK_IMBALANCE),
        "book_sell_max": _safe_quantile(work["book_imbalance"], 0.30, -MIN_BOOK_IMBALANCE),
        "breakdown_book_max": _safe_quantile(work["book_imbalance"], 0.45, BREAKDOWN_MAX_BOOK_IMBALANCE),
        "volume_burst_min": _safe_quantile(work["volume_intensity"], 0.90, MIN_VOLUME_INTENSITY),
        "breakdown_volume_burst_min": _safe_quantile(
            work["volume_intensity"],
            0.90,
            BREAKDOWN_MIN_VOLUME_INTENSITY,
        ),
        "rolling_mid_up_min": _safe_quantile(work["rolling_mid_move_ticks"], 0.55, MIN_RESILIENCE_TICKS),
        "rolling_mid_down_max": _safe_quantile(work["rolling_mid_move_ticks"], 0.45, -MIN_RESILIENCE_TICKS),
        "breakdown_rolling_mid_max": _safe_quantile(
            work["rolling_mid_move_ticks"],
            0.10,
            BREAKDOWN_MAX_ROLLING_MID_MOVE,
        ),
        "price_shock_min": _safe_quantile(work["price_shock"], 0.95, BREAKDOWN_MIN_PRICE_SHOCK),
        "rtv_min_fast_move_ticks": max(
            1.0,
            _safe_quantile(
                work["rtv_abs_move_ticks"] if "rtv_abs_move_ticks" in work.columns else pd.Series(dtype=float),
                0.90,
                RTV_MIN_FAST_MOVE_TICKS,
            ),
        ),
    }


def _add_relative_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    if {
        "rtv_fast_move_ticks",
        "rtv_abs_move_ticks",
        "rtv_threshold_ticks",
        "rtv_threshold_ratio",
        "rtv_direction",
    }.issubset(df.columns):
        return df

    out = df.copy()
    group_keys = _feature_group_keys(out)
    grouped = out.groupby(group_keys, sort=False)
    tick_size = out["tick_size_est"].replace(0, np.nan)

    out["rtv_fast_move_ticks"] = (
        out["mid_price"] - grouped["mid_price"].shift(RTV_FAST_WINDOW_TICKS)
    ) / tick_size
    out["rtv_abs_move_ticks"] = out["rtv_fast_move_ticks"].abs()

    min_periods = min(500, max(50, RTV_SLOW_WINDOW_TICKS // 10))
    out["rtv_threshold_ticks"] = grouped["rtv_abs_move_ticks"].transform(
        lambda s: s.shift(1).rolling(
            RTV_SLOW_WINDOW_TICKS,
            min_periods=min_periods,
        ).quantile(RTV_PERCENTILE)
    )
    out["rtv_threshold_ratio"] = (
        out["rtv_abs_move_ticks"]
        / out["rtv_threshold_ticks"].replace(0, np.nan)
    )
    out["rtv_direction"] = np.select(
        [out["rtv_fast_move_ticks"] > 0, out["rtv_fast_move_ticks"] < 0],
        ["Up", "Down"],
        default="Flat",
    )
    return out


def _heuristic_signal_mask(
    df: pd.DataFrame,
    hypothesis: str,
    thresholds: dict[str, float] | None = None,
) -> pd.Series:
    thresholds = thresholds or {}
    if _is_relative_velocity_hypothesis(hypothesis):
        out = _add_relative_velocity_features(df)
        min_fast_move = thresholds.get("rtv_min_fast_move_ticks", RTV_MIN_FAST_MOVE_TICKS)
        return (
            out["rtv_threshold_ticks"].notna()
            & out["rtv_fast_move_ticks"].notna()
            & out["rtv_direction"].ne("Flat")
            & (out["rtv_abs_move_ticks"] >= min_fast_move)
            & (out["rtv_abs_move_ticks"] >= out["rtv_threshold_ticks"])
        )

    if hypothesis == "bearish_breakdown":
        return (
            (df["flow_imbalance"] <= thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE))
            & (df["book_imbalance"] <= thresholds.get("breakdown_book_max", BREAKDOWN_MAX_BOOK_IMBALANCE))
            & (df["volume_intensity"] >= thresholds.get("breakdown_volume_burst_min", BREAKDOWN_MIN_VOLUME_INTENSITY))
            & (df["rolling_mid_move_ticks"] <= thresholds.get("breakdown_rolling_mid_max", BREAKDOWN_MAX_ROLLING_MID_MOVE))
            & (df["price_shock"] >= thresholds.get("price_shock_min", BREAKDOWN_MIN_PRICE_SHOCK))
        )
    if hypothesis == "bearish":
        return (
            (df["flow_imbalance"] <= thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE))
            & (df["book_imbalance"] <= thresholds.get("book_sell_max", -MIN_BOOK_IMBALANCE))
            & (df["volume_intensity"] >= thresholds.get("volume_burst_min", MIN_VOLUME_INTENSITY))
            & (df["rolling_mid_move_ticks"] <= thresholds.get("rolling_mid_down_max", -MIN_RESILIENCE_TICKS))
        )
    return (
        (df["flow_imbalance"] <= thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE))
        & (df["book_imbalance"] >= thresholds.get("book_buy_min", MIN_BOOK_IMBALANCE))
        & (df["volume_intensity"] >= thresholds.get("volume_burst_min", MIN_VOLUME_INTENSITY))
        & (df["rolling_mid_move_ticks"] >= thresholds.get("rolling_mid_up_min", MIN_RESILIENCE_TICKS))
    )


def evaluate_microstructure_hypothesis(
    features: pd.DataFrame,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = thresholds or {}
    fallback_reason = None
    if _is_relative_velocity_hypothesis(hypothesis):
        try:
            from .cpp_bridge import compute_relative_velocity_cpp

            out, candidates = compute_relative_velocity_cpp(
                features,
                horizon_ticks,
                hypothesis,
                min_success_ticks,
                min_fast_move_ticks=thresholds.get("rtv_min_fast_move_ticks", RTV_MIN_FAST_MOVE_TICKS),
            )
            out.attrs["tick_pulse_thresholds"] = thresholds
            candidates.attrs["tick_pulse_thresholds"] = thresholds
            return out, candidates
        except Exception as exc:
            fallback_reason = str(exc)
    else:
        try:
            from .cpp_bridge import compute_heuristic_cpp

            out, candidates = compute_heuristic_cpp(
                features,
                horizon_ticks,
                hypothesis,
                min_success_ticks,
                thresholds,
            )
            out.attrs["tick_pulse_thresholds"] = thresholds
            candidates.attrs["tick_pulse_thresholds"] = thresholds
            return out, candidates
        except Exception as exc:
            fallback_reason = str(exc)

    out = features.sort_values(["symbol", "datetime"]).copy()
    out["_row_pos"] = out.groupby("symbol", sort=False).cumcount()
    grouped = out.groupby(_feature_group_keys(out), sort=False)
    out["_event_row_pos"] = grouped.cumcount()
    tick_size = out["tick_size_est"].replace(0, np.nan)

    out["future_datetime"] = grouped["datetime"].shift(-horizon_ticks)
    out["future_mid_price"] = grouped["mid_price"].shift(-horizon_ticks)
    out["future_move_ticks"] = (out["future_mid_price"] - out["mid_price"]) / tick_size
    out["future_return_bps"] = (out["future_mid_price"] / out["mid_price"] - 1.0) * 10000.0

    if _is_relative_velocity_hypothesis(hypothesis):
        out = _add_relative_velocity_features(out)
        out["criterion_velocity_percentile"] = (
            out["rtv_abs_move_ticks"] >= out["rtv_threshold_ticks"]
        )
        out["criterion_min_fast_move"] = out["rtv_abs_move_ticks"] >= thresholds.get(
            "rtv_min_fast_move_ticks",
            RTV_MIN_FAST_MOVE_TICKS,
        )
        out["expected_direction"] = _velocity_expected_direction(out["rtv_direction"], hypothesis)
        correct_mask = (
            ((out["expected_direction"] == "Up") & (out["future_move_ticks"] >= min_success_ticks))
            | ((out["expected_direction"] == "Down") & (out["future_move_ticks"] <= -min_success_ticks))
        )
        signal_mask = (
            out["criterion_velocity_percentile"]
            & out["criterion_min_fast_move"]
            & out["expected_direction"].isin(["Up", "Down"])
        )
    elif hypothesis == "bearish_breakdown":
        out["criterion_flow"] = out["flow_imbalance"] <= thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE)
        out["criterion_book"] = out["book_imbalance"] <= thresholds.get("breakdown_book_max", BREAKDOWN_MAX_BOOK_IMBALANCE)
        out["criterion_price_resilience"] = out["rolling_mid_move_ticks"] <= thresholds.get("breakdown_rolling_mid_max", BREAKDOWN_MAX_ROLLING_MID_MOVE)
        out["criterion_volume_burst"] = out["volume_intensity"] >= thresholds.get("breakdown_volume_burst_min", BREAKDOWN_MIN_VOLUME_INTENSITY)
        out["criterion_price_shock"] = out["price_shock"] >= thresholds.get("price_shock_min", BREAKDOWN_MIN_PRICE_SHOCK)
        out["expected_direction"] = "Down"
        correct_mask = out["future_move_ticks"] <= -min_success_ticks
        signal_mask = (
            out["criterion_flow"]
            & out["criterion_book"]
            & out["criterion_volume_burst"]
            & out["criterion_price_resilience"]
            & out["criterion_price_shock"]
        )
    elif hypothesis == "bearish":
        out["criterion_flow"] = out["flow_imbalance"] <= thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE)
        out["criterion_book"] = out["book_imbalance"] <= thresholds.get("book_sell_max", -MIN_BOOK_IMBALANCE)
        out["criterion_price_resilience"] = out["rolling_mid_move_ticks"] <= thresholds.get("rolling_mid_down_max", -MIN_RESILIENCE_TICKS)
        out["criterion_volume_burst"] = out["volume_intensity"] >= thresholds.get("volume_burst_min", MIN_VOLUME_INTENSITY)
        out["expected_direction"] = "Down"
        correct_mask = out["future_move_ticks"] <= -min_success_ticks
        signal_mask = (
            out["criterion_flow"]
            & out["criterion_book"]
            & out["criterion_volume_burst"]
            & out["criterion_price_resilience"]
        )
    else:
        out["criterion_flow"] = out["flow_imbalance"] <= thresholds.get("flow_sell_max", -MIN_FLOW_IMBALANCE)
        out["criterion_book"] = out["book_imbalance"] >= thresholds.get("book_buy_min", MIN_BOOK_IMBALANCE)
        out["criterion_price_resilience"] = out["rolling_mid_move_ticks"] >= thresholds.get("rolling_mid_up_min", MIN_RESILIENCE_TICKS)
        out["criterion_volume_burst"] = out["volume_intensity"] >= thresholds.get("volume_burst_min", MIN_VOLUME_INTENSITY)
        out["expected_direction"] = "Up"
        correct_mask = out["future_move_ticks"] >= min_success_ticks
        signal_mask = (
            out["criterion_flow"]
            & out["criterion_book"]
            & out["criterion_volume_burst"]
            & out["criterion_price_resilience"]
        )

    out["hypothesis_signal"] = signal_mask
    valid = out["hypothesis_signal"] & out["future_move_ticks"].notna()
    candidates = out.loc[valid].copy()
    candidates["is_correct"] = correct_mask.loc[valid]
    candidates["outcome"] = np.where(candidates["is_correct"], "Correct", "Failed")
    candidates = _collapse_event_episodes(candidates)
    backend = "python_fallback" if fallback_reason else "python"
    out.attrs["tick_pulse_backend"] = backend
    out.attrs["tick_pulse_thresholds"] = thresholds
    candidates.attrs["tick_pulse_backend"] = backend
    candidates.attrs["tick_pulse_thresholds"] = thresholds
    if fallback_reason:
        out.attrs["tick_pulse_backend_error"] = fallback_reason
        candidates.attrs["tick_pulse_backend_error"] = fallback_reason
    return out, candidates


def evaluate_seed_hypothesis(
    features: pd.DataFrame,
    horizon_ticks: int,
    seed: dict,
    min_success_ticks: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a saved discovery seed as a simple deterministic hypothesis."""
    rule = seed.get("rule", {}) if isinstance(seed, dict) else {}
    behavior = str(rule.get("behavior") or seed.get("behavior") or "continuation")
    direction_filter = str(rule.get("direction_filter") or seed.get("pulse_direction") or "Any")
    min_abs_move = float(rule.get("min_abs_move_ticks", 1.0) or 1.0)
    min_volume_intensity = float(rule.get("min_volume_intensity", 0.0) or 0.0)
    min_session_age = float(rule.get("min_session_age_seconds", 0.0) or 0.0)
    spread_mode = str(rule.get("spread_filter_mode") or "any")
    max_spread_ticks = float(rule.get("max_spread_ticks", np.inf) or np.inf)
    min_spread_ticks = float(rule.get("min_spread_ticks", 0.0) or 0.0)

    out = features.sort_values(["symbol", "datetime"]).copy()
    out["_row_pos"] = out.groupby("symbol", sort=False).cumcount()
    grouped = out.groupby(_feature_group_keys(out), sort=False)
    out["_event_row_pos"] = grouped.cumcount()
    tick_size = out["tick_size_est"].replace(0, np.nan)

    out["future_datetime"] = grouped["datetime"].shift(-horizon_ticks)
    out["future_mid_price"] = grouped["mid_price"].shift(-horizon_ticks)
    out["future_move_ticks"] = (out["future_mid_price"] - out["mid_price"]) / tick_size
    out["future_return_bps"] = (out["future_mid_price"] / out["mid_price"] - 1.0) * 10000.0

    raw_move = out["rolling_mid_move_ticks"] if "rolling_mid_move_ticks" in out.columns else pd.Series(np.nan, index=out.index)
    move = pd.to_numeric(raw_move, errors="coerce")
    out["seed_abs_move_ticks"] = move.abs()
    out["seed_direction"] = np.select(
        [move >= min_abs_move, move <= -min_abs_move],
        ["Up", "Down"],
        default="Flat",
    )
    out["criterion_seed_move"] = out["seed_abs_move_ticks"] >= min_abs_move
    if direction_filter in {"Up", "Down"}:
        out["criterion_seed_direction"] = out["seed_direction"].eq(direction_filter)
    else:
        out["criterion_seed_direction"] = out["seed_direction"].isin(["Up", "Down"])

    if "spread" in out.columns:
        out["seed_spread_ticks"] = pd.to_numeric(out["spread"], errors="coerce") / tick_size
    else:
        out["seed_spread_ticks"] = np.nan
    if spread_mode == "controlled":
        out["criterion_seed_spread"] = out["seed_spread_ticks"].le(max_spread_ticks)
    elif spread_mode == "wide":
        out["criterion_seed_spread"] = out["seed_spread_ticks"].ge(min_spread_ticks)
    else:
        out["criterion_seed_spread"] = True

    raw_volume = out["volume_intensity"] if "volume_intensity" in out.columns else pd.Series(0.0, index=out.index)
    volume_intensity = pd.to_numeric(raw_volume, errors="coerce").fillna(0.0)
    out["criterion_seed_volume"] = volume_intensity >= min_volume_intensity

    session_start = grouped["datetime"].transform("min")
    out["seed_session_age_seconds"] = (out["datetime"] - session_start).dt.total_seconds()
    out["criterion_seed_session_age"] = out["seed_session_age_seconds"] >= min_session_age

    if behavior == "fade":
        out["expected_direction"] = np.select(
            [out["seed_direction"].eq("Up"), out["seed_direction"].eq("Down")],
            ["Down", "Up"],
            default="Flat",
        )
    else:
        out["expected_direction"] = out["seed_direction"]

    correct_mask = (
        ((out["expected_direction"] == "Up") & (out["future_move_ticks"] >= min_success_ticks))
        | ((out["expected_direction"] == "Down") & (out["future_move_ticks"] <= -min_success_ticks))
    )
    signal_mask = (
        out["criterion_seed_move"]
        & out["criterion_seed_direction"]
        & out["criterion_seed_spread"].fillna(False)
        & out["criterion_seed_volume"]
        & out["criterion_seed_session_age"]
        & out["expected_direction"].isin(["Up", "Down"])
    )

    out["hypothesis_signal"] = signal_mask
    out["is_correct"] = correct_mask
    valid = out["hypothesis_signal"] & out["future_move_ticks"].notna()
    candidates = out.loc[valid].copy()
    candidates["outcome"] = np.where(candidates["is_correct"], "Correct", "Failed")
    candidates = _collapse_event_episodes(candidates)

    attrs = {
        "tick_pulse_backend": "python_seed",
        "tick_pulse_thresholds": rule,
        "tick_pulse_seed": seed,
    }
    out.attrs.update(attrs)
    candidates.attrs.update(attrs)
    return out, candidates


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = p_hat + z * z / (2.0 * total)
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
    return (center - margin) / denom, (center + margin) / denom


def _pct_text(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1%}"


def _signed_pct_text(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:+.1%}"


def _tick_text(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def _summarize_labeled_frame(
    out: pd.DataFrame,
    horizon_ticks: int,
    hypothesis: str,
    backend: str,
) -> dict:
    valid = out["future_move_ticks"].notna()
    signal = out["hypothesis_signal"].fillna(False).astype(bool)
    target = out["is_correct"].fillna(False).astype(bool)
    event_mask = valid & signal
    event_count = int(event_mask.sum())
    success_count = int(target[event_mask].sum()) if event_count else 0
    accuracy = success_count / event_count if event_count else np.nan
    base_rate = float(target[valid].mean()) if valid.any() else np.nan
    avg_move = float(out.loc[event_mask, "future_move_ticks"].mean()) if event_count else np.nan
    expected_move_values = np.where(
        out["expected_direction"].eq("Down"),
        -out["future_move_ticks"],
        out["future_move_ticks"],
    )
    ci_low, ci_high = _wilson_interval(success_count, event_count)
    return {
        "hypothesis": hypothesis,
        "horizon": int(horizon_ticks),
        "events": event_count,
        "successes": success_count,
        "accuracy": accuracy,
        "base_rate": base_rate,
        "lift": accuracy - base_rate if event_count else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "avg_move": avg_move,
        "expected_avg": (
            float(pd.Series(expected_move_values, index=out.index).loc[event_mask].mean())
            if event_count
            else np.nan
        ),
        "backend": backend,
    }


def _evaluate_horizon_summary(
    features: pd.DataFrame,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    thresholds: dict[str, float] | None = None,
) -> dict:
    thresholds = thresholds or {}
    cpp_fallback = False
    if _is_relative_velocity_hypothesis(hypothesis):
        try:
            from .cpp_bridge import compute_relative_velocity_cpp

            out, _ = compute_relative_velocity_cpp(
                features,
                horizon_ticks,
                hypothesis,
                min_success_ticks,
                min_fast_move_ticks=thresholds.get("rtv_min_fast_move_ticks", RTV_MIN_FAST_MOVE_TICKS),
            )
            valid = out["future_move_ticks"].notna()
            signal = out["hypothesis_signal"].fillna(False).astype(bool)
            if "is_correct" in out.columns:
                target = out["is_correct"].fillna(False).astype(bool)
            else:
                target = (
                    ((out["expected_direction"] == "Up") & (out["future_move_ticks"] >= min_success_ticks))
                    | ((out["expected_direction"] == "Down") & (out["future_move_ticks"] <= -min_success_ticks))
                )
            event_mask = valid & signal
            event_count = int(event_mask.sum())
            success_count = int(target[event_mask].sum()) if event_count else 0
            accuracy = success_count / event_count if event_count else np.nan
            base_rate = float(target[valid].mean()) if valid.any() else np.nan
            avg_move = float(out.loc[event_mask, "future_move_ticks"].mean()) if event_count else np.nan
            expected_move_values = np.where(
                out["expected_direction"].eq("Down"),
                -out["future_move_ticks"],
                out["future_move_ticks"],
            )
            ci_low, ci_high = _wilson_interval(success_count, event_count)
            return {
                "hypothesis": hypothesis,
                "horizon": int(horizon_ticks),
                "events": event_count,
                "successes": success_count,
                "accuracy": accuracy,
                "base_rate": base_rate,
                "lift": accuracy - base_rate if event_count else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "avg_move": avg_move,
                "expected_avg": (
                    float(pd.Series(expected_move_values, index=out.index).loc[event_mask].mean())
                    if event_count
                    else np.nan
                ),
                "backend": out.attrs.get("tick_pulse_backend", "cpp"),
            }
        except Exception:
            cpp_fallback = True
    else:
        try:
            from .cpp_bridge import compute_heuristic_cpp

            out, _ = compute_heuristic_cpp(
                features,
                horizon_ticks,
                hypothesis,
                min_success_ticks,
                thresholds,
            )
            return _summarize_labeled_frame(
                out,
                horizon_ticks,
                hypothesis,
                out.attrs.get("tick_pulse_backend", "cpp"),
            )
        except Exception:
            cpp_fallback = True

    out = features.sort_values(["symbol", "datetime"]).copy()
    grouped = out.groupby(_feature_group_keys(out), sort=False)
    tick_size = out["tick_size_est"].replace(0, np.nan)
    out["future_mid_price"] = grouped["mid_price"].shift(-horizon_ticks)
    out["future_move_ticks"] = (out["future_mid_price"] - out["mid_price"]) / tick_size

    if _is_relative_velocity_hypothesis(hypothesis):
        out = _add_relative_velocity_features(out)
        signal = _heuristic_signal_mask(out, hypothesis, thresholds)
        expected_direction = pd.Series(
            _velocity_expected_direction(out["rtv_direction"], hypothesis),
            index=out.index,
        )
        target = (
            ((expected_direction == "Up") & (out["future_move_ticks"] >= min_success_ticks))
            | ((expected_direction == "Down") & (out["future_move_ticks"] <= -min_success_ticks))
        )
        expected_move_values = np.where(
            expected_direction.eq("Down"),
            -out["future_move_ticks"],
            out["future_move_ticks"],
        )
        direction_multiplier = np.nan
    elif _is_bearish_hypothesis(hypothesis):
        signal = _heuristic_signal_mask(out, hypothesis, thresholds)
        target = out["future_move_ticks"] <= -min_success_ticks
        expected_move_values = None
        direction_multiplier = -1.0
    else:
        signal = _heuristic_signal_mask(out, hypothesis, thresholds)
        target = out["future_move_ticks"] >= min_success_ticks
        expected_move_values = None
        direction_multiplier = 1.0

    valid = out["future_move_ticks"].notna()
    event_mask = valid & signal
    event_count = int(event_mask.sum())
    success_count = int(target[event_mask].sum()) if event_count else 0
    accuracy = success_count / event_count if event_count else np.nan
    base_rate = float(target[valid].mean()) if valid.any() else np.nan
    avg_move = float(out.loc[event_mask, "future_move_ticks"].mean()) if event_count else np.nan
    ci_low, ci_high = _wilson_interval(success_count, event_count)

    return {
        "hypothesis": hypothesis,
        "horizon": int(horizon_ticks),
        "events": event_count,
        "successes": success_count,
        "accuracy": accuracy,
        "base_rate": base_rate,
        "lift": accuracy - base_rate if event_count else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "avg_move": avg_move,
        "expected_avg": (
            float(pd.Series(expected_move_values, index=out.index).loc[event_mask].mean())
            if event_count and _is_relative_velocity_hypothesis(hypothesis)
            else avg_move * direction_multiplier if event_count else np.nan
        ),
        "backend": "python_fallback" if cpp_fallback else "python",
    }


def _build_research_sweep(
    features: pd.DataFrame,
    min_success_ticks: float,
    t: dict,
    thresholds_by_hypothesis: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds_by_hypothesis = thresholds_by_hypothesis or {}
    raw_rows = [
        _evaluate_horizon_summary(
            features,
            horizon,
            hypothesis,
            min_success_ticks,
            thresholds_by_hypothesis.get(hypothesis),
        )
        for hypothesis in HYPOTHESIS_KEYS
        for horizon in RESEARCH_SWEEP_HORIZONS
    ]
    raw = pd.DataFrame(raw_rows)
    return raw, format_research_sweep_display(raw, t)


def format_research_sweep_display(raw: pd.DataFrame, t: dict) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=[
                t["sweep_hypothesis"],
                t["sweep_horizon"],
                t["sweep_events"],
                t["sweep_accuracy"],
                t["sweep_base_rate"],
                t["sweep_lift"],
                t["sweep_ci"],
                t["sweep_avg_move"],
                t["sweep_expected_avg"],
            ]
        )

    display = raw.copy()
    display[t["sweep_hypothesis"]] = display["hypothesis"].map(
        lambda key: t["hypothesis_labels"].get(key, str(key))
    )
    display[t["sweep_horizon"]] = display["horizon"].map(lambda value: f"{value} {t['ticks']}")
    display[t["sweep_events"]] = display["events"]
    display[t["sweep_accuracy"]] = display["accuracy"].map(_pct_text)
    display[t["sweep_base_rate"]] = display["base_rate"].map(_pct_text)
    display[t["sweep_lift"]] = display["lift"].map(_signed_pct_text)
    display[t["sweep_ci"]] = display.apply(
        lambda row: f"{_pct_text(row['ci_low'])} - {_pct_text(row['ci_high'])}",
        axis=1,
    )
    display[t["sweep_avg_move"]] = display["avg_move"].map(_tick_text)
    display[t["sweep_expected_avg"]] = display["expected_avg"].map(_tick_text)
    return display[
        [
            t["sweep_hypothesis"],
            t["sweep_horizon"],
            t["sweep_events"],
            t["sweep_accuracy"],
            t["sweep_base_rate"],
            t["sweep_lift"],
            t["sweep_ci"],
            t["sweep_avg_move"],
            t["sweep_expected_avg"],
        ]
    ]
