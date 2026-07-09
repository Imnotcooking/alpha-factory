from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(UI_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import BASE_DIR, DB_PATH, LOGS_DIR, get_plotly_template
from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar
from .text import PAGE_TEXT
from .constants import (
    BREAKDOWN_MAX_BOOK_IMBALANCE,
    BREAKDOWN_MAX_ROLLING_MID_MOVE,
    BREAKDOWN_MIN_PRICE_SHOCK,
    BREAKDOWN_MIN_VOLUME_INTENSITY,
    DEFAULT_HYPOTHESIS_KEY,
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
from .engine import (
    _build_research_sweep,
    _evaluate_horizon_summary,
    build_adaptive_thresholds,
    evaluate_microstructure_hypothesis,
    evaluate_seed_hypothesis,
    format_research_sweep_display,
)
from .hypothesis_seeds import format_seed_label, list_hypothesis_seeds
from .research_cache import (
    CACHE_SCHEMA_VERSION,
    get_or_compute_dataframe,
    load_cached_dataframe,
    make_cache_key,
)
from .views import (
    _render_calculation_audit,
    _render_event_ledger,
    _render_event_horizon_bars,
    _render_event_map,
    _render_outcome_overview,
    _select_visible_candidates,
)
from oqp.research.tick_pulse import (
    DEFAULT_TICK_FILE,
    contract_summary,
    load_tick_scope,
    load_ticks,
)
from oqp.data.runtime_paths import discover_futures_cn_tick_files
from oqp.research import make_evidence_ticket_id, record_evidence_ticket, stable_trial_hash
from oqp.research.tick_pulse import build_pulse_features_fast
from oqp.research.tick_pulse.sweeps import (
    compute_main_contract_file_sweep as run_main_contract_file_sweep,
)

@st.cache_data(show_spinner=False)
def load_tick_scope_data(path: str, mtime: float, product: str, symbol: str) -> pd.DataFrame:
    return load_tick_scope(path, product=product, symbol=symbol)


@st.cache_data(show_spinner=False)
def compute_contract_summary(path: str, mtime: float) -> pd.DataFrame:
    source_file = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "tick_contract_summary",
        "engine": "contract_summary_v1",
        "source_file": source_file,
        "source_mtime": float(mtime),
    }
    cache_key = make_cache_key(payload)

    def compute() -> pd.DataFrame:
        return contract_summary(load_ticks(path))

    return get_or_compute_dataframe(
        db_path=DB_PATH,
        logs_dir=LOGS_DIR,
        base_dir=BASE_DIR,
        cache_key=cache_key,
        metadata={
            "cache_type": "tick_contract_summary",
            "source_file": source_file,
            "source_mtime": float(mtime),
            "product": "ALL",
            "symbol": "ALL",
            "hypothesis": "",
            "threshold_mode": "",
            "window": 0,
            "min_success_ticks": 0.0,
            "horizon_set": "",
            "backend": "python",
        },
        compute_fn=compute,
    ).data


@st.cache_data(show_spinner=False)
def compute_feature_frame(path: str, mtime: float, product: str, symbol: str, window: int) -> pd.DataFrame:
    source_file = os.path.relpath(os.path.abspath(path), PROJECT_ROOT)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "tick_pulse_feature_frame",
        "engine": "pulse_feature_builder_v2",
        "source_file": source_file,
        "source_mtime": float(mtime),
        "product": product,
        "symbol": symbol,
        "window": int(window),
    }
    cache_key = make_cache_key(payload)

    def compute() -> pd.DataFrame:
        scoped = load_tick_scope(path, product=product, symbol=symbol)
        return build_pulse_features_fast(scoped, window=window)

    return get_or_compute_dataframe(
        db_path=DB_PATH,
        logs_dir=LOGS_DIR,
        base_dir=BASE_DIR,
        cache_key=cache_key,
        metadata={
            "cache_type": "tick_pulse_feature_frame",
            "source_file": source_file,
            "source_mtime": float(mtime),
            "product": product,
            "symbol": symbol,
            "hypothesis": "",
            "threshold_mode": "",
            "window": int(window),
            "min_success_ticks": 0.0,
            "horizon_set": "",
            "backend": "cpp_or_python",
        },
        compute_fn=compute,
    ).data


@st.cache_data(show_spinner=False)
def compute_features(path: str, mtime: float, product: str, symbol: str, window: int) -> pd.DataFrame:
    return compute_feature_frame(path, mtime, product, symbol, window)


@st.cache_data(show_spinner=False)
def compute_hypothesis_evaluation(
    path: str,
    mtime: float,
    product: str,
    symbol: str,
    window: int,
    horizon_ticks: int,
    hypothesis: str,
    min_success_ticks: float,
    threshold_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    features = compute_feature_frame(path, mtime, product, symbol, window)
    thresholds = build_adaptive_thresholds(features, hypothesis) if threshold_mode == "adaptive" else {}
    features, candidates = evaluate_microstructure_hypothesis(
        features,
        horizon_ticks,
        hypothesis,
        min_success_ticks,
        thresholds,
    )
    features.attrs["tick_pulse_thresholds"] = thresholds
    candidates.attrs["tick_pulse_thresholds"] = thresholds
    return features, candidates, thresholds


@st.cache_data(show_spinner=False)
def compute_seed_hypothesis_evaluation(
    path: str,
    mtime: float,
    product: str,
    symbol: str,
    window: int,
    horizon_ticks: int,
    seed: dict,
    min_success_ticks: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = compute_feature_frame(path, mtime, product, symbol, window)
    features, candidates = evaluate_seed_hypothesis(
        features,
        horizon_ticks,
        seed,
        min_success_ticks,
    )
    rule = seed.get("rule", {}) if isinstance(seed, dict) else {}
    features.attrs["tick_pulse_thresholds"] = rule
    candidates.attrs["tick_pulse_thresholds"] = rule
    return features, candidates, rule


def _effective_thresholds(thresholds: dict[str, float] | None) -> dict[str, float]:
    thresholds = thresholds or {}
    return {
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
        "rtv_min_fast_move_ticks": thresholds.get("rtv_min_fast_move_ticks", RTV_MIN_FAST_MOVE_TICKS),
    }



def _add_lift_stability(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    if out.empty or "lift" not in out.columns:
        out["lift_stability_code"] = "unstable"
        out["peak_lift_row"] = False
        return out

    codes = {}
    peak_indices = set()
    for key, group in out.groupby(["asset", "main_contract"], sort=False):
        lifts = pd.to_numeric(group["lift"], errors="coerce")
        if lifts.notna().all() and (lifts > 0).all():
            codes[key] = "stable_positive"
        elif lifts.notna().all() and (lifts < 0).all():
            codes[key] = "stable_negative"
        else:
            codes[key] = "unstable"
        abs_lifts = lifts.abs()
        if abs_lifts.notna().any():
            peak_indices.add(abs_lifts.idxmax())

    out["lift_stability_code"] = [
        codes.get((row.asset, row.main_contract), "unstable")
        for row in out[["asset", "main_contract"]].itertuples(index=False)
    ]
    out["peak_lift_row"] = out.index.isin(peak_indices)
    return out


def _style_cross_asset_stability(display: pd.DataFrame, stability_column: str, peak_column: str):
    def style_row(row: pd.Series) -> list[str]:
        value = str(row.get(stability_column, ""))
        is_peak = bool(str(row.get(peak_column, "")).strip())
        if value.startswith("Stable") or value.startswith("稳定"):
            color = "#A5D6A7" if is_peak else "#E8F5E9"
            return [f"background-color: {color}; color: #1B5E20;"] * len(row)
        color = "#FFCDD2" if is_peak else "#FFEBEE"
        return [f"background-color: {color}; color: #B71C1C;"] * len(row)

    return display.style.apply(style_row, axis=1)


def _direction_label_for_explainer(direction: str, t: dict) -> str:
    return t.get("direction_labels", {}).get(str(direction), str(direction))


def _target_success_mask(frame: pd.DataFrame, min_success_ticks: float) -> pd.Series:
    future = pd.to_numeric(frame["future_move_ticks"], errors="coerce")
    expected = frame.get("expected_direction", pd.Series("", index=frame.index)).astype(str)
    return (
        (expected.eq("Up") & future.ge(min_success_ticks))
        | (expected.eq("Down") & future.le(-min_success_ticks))
    )


def _pick_explainer_index(features: pd.DataFrame, candidates: pd.DataFrame) -> int | None:
    if not candidates.empty:
        return int(candidates.sort_values("datetime").index[0])
    valid = features[
        features["future_move_ticks"].notna()
        & features.get("expected_direction", pd.Series("", index=features.index)).isin(["Up", "Down"])
    ]
    if valid.empty:
        valid = features[features["future_move_ticks"].notna()]
    if valid.empty:
        return None
    return int(valid.index[len(valid) // 2])


def _relative_loc(index: pd.Index, value: int) -> int | None:
    matches = np.flatnonzero(index.to_numpy() == value)
    if len(matches) == 0:
        return None
    return int(matches[0])


def _format_criterion_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _criterion_bool(row: pd.Series, column: str, fallback: bool) -> bool:
    value = row.get(column, np.nan)
    if pd.isna(value):
        return bool(fallback)
    return bool(value)


def _rule_label(
    t: dict,
    feature_key: str,
    op: str,
    value: object,
    threshold: object,
    unit: str = "",
) -> str:
    name = t["criteria_feature_names"].get(feature_key, feature_key)
    return t["criteria_rule"].format(
        name=name,
        op=op,
        value=_format_criterion_value(value),
        threshold=_format_criterion_value(threshold),
        unit=unit,
    )


def _base_rate_criteria(
    row: pd.Series,
    selected_success: bool,
    horizon_ticks: int,
    min_success_ticks: float,
    t: dict,
) -> list[tuple[str, bool]]:
    future_move = row.get("future_move_ticks", np.nan)
    expected_direction = str(row.get("expected_direction", ""))
    expected_defined = expected_direction in {"Up", "Down"}
    direction_label = _direction_label_for_explainer(expected_direction, t)
    return [
        (
            t["criteria_valid_future"].format(horizon=int(horizon_ticks)),
            pd.notna(future_move),
        ),
        (
            t["criteria_expected_direction"].format(direction=direction_label),
            expected_defined,
        ),
        (
            t["criteria_future_success"].format(
                move=_format_criterion_value(future_move, 2),
                direction=direction_label,
                success=f"{min_success_ticks:g}",
            ),
            bool(selected_success),
        ),
    ]


def _event_criteria(
    row: pd.Series,
    hypothesis: str,
    thresholds: dict[str, float],
    t: dict,
) -> list[tuple[str, bool]]:
    signal = bool(row.get("hypothesis_signal", False))
    if hypothesis in {"relative_velocity", "relative_velocity_fade"}:
        direction = str(row.get("rtv_direction", ""))
        expected = str(row.get("expected_direction", ""))
        direction_defined = direction in {"Up", "Down"}
        min_fast = thresholds.get("rtv_min_fast_move_ticks", RTV_MIN_FAST_MOVE_TICKS)
        abs_move = row.get("rtv_abs_move_ticks", np.nan)
        rolling_threshold = row.get("rtv_threshold_ticks", np.nan)
        return [
            (
                t["criteria_rtv_direction"].format(
                    direction=_direction_label_for_explainer(direction, t),
                    expected=_direction_label_for_explainer(expected, t),
                ),
                direction_defined,
            ),
            (
                _rule_label(t, "rtv_abs_move_ticks", ">=", abs_move, min_fast, f" {t['price_ticks_unit']}"),
                _criterion_bool(row, "criterion_min_fast_move", pd.notna(abs_move) and abs_move >= min_fast),
            ),
            (
                t["criteria_rtv_percentile"].format(
                    value=_format_criterion_value(abs_move, 2),
                    threshold=_format_criterion_value(rolling_threshold, 2),
                    percentile=f"{RTV_PERCENTILE:.0%}",
                ),
                _criterion_bool(
                    row,
                    "criterion_velocity_percentile",
                    pd.notna(abs_move) and pd.notna(rolling_threshold) and abs_move >= rolling_threshold,
                ),
            ),
            (t["criteria_hypothesis_signal"], signal),
        ]

    if hypothesis == "bearish_breakdown":
        specs = [
            ("flow_imbalance", "criterion_flow", "<=", row.get("flow_imbalance", np.nan), thresholds["flow_sell_max"]),
            ("book_imbalance", "criterion_book", "<=", row.get("book_imbalance", np.nan), thresholds["breakdown_book_max"]),
            (
                "volume_intensity",
                "criterion_volume_burst",
                ">=",
                row.get("volume_intensity", np.nan),
                thresholds["breakdown_volume_burst_min"],
            ),
            (
                "rolling_mid_move_ticks",
                "criterion_price_resilience",
                "<=",
                row.get("rolling_mid_move_ticks", np.nan),
                thresholds["breakdown_rolling_mid_max"],
            ),
            ("price_shock", "criterion_price_shock", ">=", row.get("price_shock", np.nan), thresholds["price_shock_min"]),
        ]
    elif hypothesis == "bearish":
        specs = [
            ("flow_imbalance", "criterion_flow", "<=", row.get("flow_imbalance", np.nan), thresholds["flow_sell_max"]),
            ("book_imbalance", "criterion_book", "<=", row.get("book_imbalance", np.nan), thresholds["book_sell_max"]),
            (
                "volume_intensity",
                "criterion_volume_burst",
                ">=",
                row.get("volume_intensity", np.nan),
                thresholds["volume_burst_min"],
            ),
            (
                "rolling_mid_move_ticks",
                "criterion_price_resilience",
                "<=",
                row.get("rolling_mid_move_ticks", np.nan),
                thresholds["rolling_mid_down_max"],
            ),
        ]
    else:
        specs = [
            ("flow_imbalance", "criterion_flow", "<=", row.get("flow_imbalance", np.nan), thresholds["flow_sell_max"]),
            ("book_imbalance", "criterion_book", ">=", row.get("book_imbalance", np.nan), thresholds["book_buy_min"]),
            (
                "volume_intensity",
                "criterion_volume_burst",
                ">=",
                row.get("volume_intensity", np.nan),
                thresholds["volume_burst_min"],
            ),
            (
                "rolling_mid_move_ticks",
                "criterion_price_resilience",
                ">=",
                row.get("rolling_mid_move_ticks", np.nan),
                thresholds["rolling_mid_up_min"],
            ),
        ]

    criteria = []
    for feature_key, criterion_column, op, value, threshold in specs:
        fallback = pd.notna(value) and (value <= threshold if op == "<=" else value >= threshold)
        criteria.append(
            (
                _rule_label(t, feature_key, op, value, threshold),
                _criterion_bool(row, criterion_column, fallback),
            )
        )
    criteria.append((t["criteria_hypothesis_signal"], signal))
    return criteria


def _pick_event_example_indices(candidates: pd.DataFrame) -> dict[str, int | None]:
    examples: dict[str, int | None] = {"success": None, "failure": None}
    if candidates.empty or "is_correct" not in candidates.columns:
        return examples

    ordered = candidates.sort_values("datetime")
    outcome = ordered["is_correct"].fillna(False).astype(bool)
    if outcome.any():
        examples["success"] = int(ordered.loc[outcome].index[0])
    if (~outcome).any():
        examples["failure"] = int(ordered.loc[~outcome].index[0])
    return examples


def _row_success(row: pd.Series, min_success_ticks: float) -> bool:
    future_move = row.get("future_move_ticks", np.nan)
    expected_direction = str(row.get("expected_direction", ""))
    if pd.isna(future_move):
        return False
    if expected_direction == "Up":
        return float(future_move) >= min_success_ticks
    if expected_direction == "Down":
        return float(future_move) <= -min_success_ticks
    return False


def _snapshot_role(offset: int, horizon_ticks: int, t: dict) -> str:
    if offset == 0:
        return t["snapshot_role_event"]
    if offset == int(horizon_ticks):
        return t["snapshot_role_horizon"]
    if offset < 0:
        return t["snapshot_role_fast"]
    return t["snapshot_role_forward"]


def _event_example_context(
    features: pd.DataFrame,
    selected_idx: int,
    horizon_ticks: int,
) -> dict[str, object] | None:
    if selected_idx not in features.index:
        return None

    row_obj = features.loc[selected_idx]
    row = row_obj.iloc[0] if isinstance(row_obj, pd.DataFrame) else row_obj
    mask = features["symbol"].eq(row["symbol"])
    if "_session_id" in features.columns:
        mask &= features["_session_id"].eq(row["_session_id"])

    group = features.loc[mask].sort_values("datetime")
    row_loc = _relative_loc(group.index, selected_idx)
    if row_loc is None:
        return None

    start_loc = max(0, row_loc - RTV_FAST_WINDOW_TICKS)
    end_loc = min(len(group) - 1, row_loc + int(horizon_ticks))
    plot = group.iloc[start_loc : end_loc + 1].copy()
    plot["relative_snapshot"] = np.arange(start_loc - row_loc, end_loc - row_loc + 1)

    tick_size = float(row.get("tick_size_est", np.nan))
    current_price = float(row.get("mid_price", np.nan))
    future_price = row.get("future_mid_price", np.nan)
    future_datetime = row.get("future_datetime", pd.NaT)
    future_loc = row_loc + int(horizon_ticks)
    if future_loc < len(group):
        future_row = group.iloc[future_loc]
        if pd.isna(future_price):
            future_price = future_row.get("mid_price", np.nan)
        if pd.isna(future_datetime):
            future_datetime = future_row.get("datetime", pd.NaT)

    future_move = row.get("future_move_ticks", np.nan)
    if pd.isna(future_move) and pd.notna(future_price) and pd.notna(tick_size) and tick_size != 0:
        future_move = (float(future_price) - current_price) / tick_size

    fast_start_price = np.nan
    past_move = np.nan
    if row_loc >= RTV_FAST_WINDOW_TICKS:
        fast_start_price = float(group.iloc[row_loc - RTV_FAST_WINDOW_TICKS].get("mid_price", np.nan))
        if pd.notna(tick_size) and tick_size != 0:
            past_move = (current_price - fast_start_price) / tick_size

    return {
        "row": row,
        "group": group,
        "plot": plot,
        "row_loc": row_loc,
        "tick_size": tick_size,
        "current_price": current_price,
        "future_price": future_price,
        "future_datetime": future_datetime,
        "future_move": future_move,
        "past_move": past_move,
    }


def _render_event_example_card(
    features: pd.DataFrame,
    selected_idx: int | None,
    example_kind: str,
    hypothesis: str,
    horizon_ticks: int,
    min_success_ticks: float,
    thresholds: dict[str, float],
    t: dict,
    tpl: str,
) -> None:
    if selected_idx is None:
        st.info(t[f"missing_{example_kind}_example"])
        return

    context = _event_example_context(features, selected_idx, horizon_ticks)
    if context is None:
        st.info(t[f"missing_{example_kind}_example"])
        return

    row = context["row"]
    plot = context["plot"].copy()
    tick_size = float(context["tick_size"])
    current_price = float(context["current_price"])
    future_price = context["future_price"]
    future_move = context["future_move"]
    past_move = context["past_move"]
    expected_direction = str(row.get("expected_direction", ""))
    selected_success = _row_success(row, min_success_ticks)
    color = "#2E7D32" if selected_success else "#C62828"
    outcome_label = t["correct_label"] if selected_success else t["failed_label"]

    if pd.notna(tick_size) and tick_size != 0:
        plot["move_from_event_ticks"] = (plot["mid_price"] - current_price) / tick_size
    else:
        plot["move_from_event_ticks"] = np.nan
    plot["snapshot_role"] = [
        _snapshot_role(int(offset), horizon_ticks, t)
        for offset in plot["relative_snapshot"]
    ]
    for column in [
        "flow_imbalance",
        "book_imbalance",
        "volume_intensity",
        "rolling_mid_move_ticks",
        "price_shock",
        "rtv_abs_move_ticks",
        "rtv_threshold_ticks",
        "rtv_threshold_ratio",
    ]:
        if column not in plot.columns:
            plot[column] = np.nan

    customdata = np.column_stack(
        [
            plot["datetime"].astype(str),
            plot["move_from_event_ticks"],
            plot["snapshot_role"],
            plot["flow_imbalance"],
            plot["book_imbalance"],
            plot["volume_intensity"],
            plot["rolling_mid_move_ticks"],
            plot["price_shock"],
            plot["rtv_abs_move_ticks"],
            plot["rtv_threshold_ticks"],
            plot["rtv_threshold_ratio"],
        ]
    )

    st.markdown(f"##### {t[f'{example_kind}_example_title']}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot["relative_snapshot"],
            y=plot["mid_price"],
            mode="lines+markers",
            name=t["chart_mid_price"],
            line=dict(color="#607D8B", width=2),
            marker=dict(size=5),
            customdata=customdata,
            hovertemplate=t["event_chart_hover"],
        )
    )
    fig.add_vrect(
        x0=-RTV_FAST_WINDOW_TICKS,
        x1=0,
        fillcolor="#BBDEFB",
        opacity=0.28,
        line_width=0,
        annotation_text=t["fast_window_label"].format(fast_window=RTV_FAST_WINDOW_TICKS),
        annotation_position="top left",
    )
    fig.add_vline(x=0, line_color="#263238", line_width=2, annotation_text=t["event_point_label"])
    fig.add_vline(
        x=int(horizon_ticks),
        line_color="#7E57C2",
        line_dash="dash",
        annotation_text=t["horizon_label"].format(horizon=int(horizon_ticks)),
    )
    fig.add_hline(
        y=current_price,
        line_color="#90A4AE",
        line_dash="dot",
        annotation_text=t["entry_price_label"],
    )
    if pd.notna(tick_size) and tick_size != 0 and expected_direction in {"Up", "Down"}:
        success_price = current_price + (
            min_success_ticks * tick_size if expected_direction == "Up" else -min_success_ticks * tick_size
        )
        fig.add_hline(
            y=success_price,
            line_color="#00C853",
            line_dash="dot",
            annotation_text=t["success_price_label"].format(
                direction=_direction_label_for_explainer(expected_direction, t),
                success=f"{min_success_ticks:g}",
            ),
        )
    if pd.notna(future_price):
        fig.add_trace(
            go.Scatter(
                x=[int(horizon_ticks)],
                y=[float(future_price)],
                mode="markers",
                name=t["outcome_point_label"],
                marker=dict(color=color, size=13, symbol="diamond"),
                hovertemplate=t["outcome_hover"].format(
                    outcome=outcome_label,
                    move=_format_criterion_value(future_move, 2),
                    expected=_direction_label_for_explainer(expected_direction, t),
                    success=f"{min_success_ticks:g}",
                ),
            )
        )
    if hypothesis in {"relative_velocity", "relative_velocity_fade"}:
        threshold = row.get("rtv_threshold_ticks", np.nan)
        ratio = row.get("rtv_threshold_ratio", np.nan)
        fig.add_annotation(
            x=0,
            y=current_price,
            text=t["event_threshold_annotation"].format(
                past_move="N/A" if pd.isna(past_move) else f"{past_move:+.2f}",
                threshold="N/A" if pd.isna(threshold) else f"{threshold:.2f}",
                ratio="N/A" if pd.isna(ratio) else f"{ratio:.2f}",
            ),
            showarrow=True,
            arrowhead=2,
            ax=35,
            ay=-45,
            bgcolor="rgba(255,255,255,0.88)",
        )

    fig.update_layout(
        template=tpl,
        height=360,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title=t["snapshot_axis"],
        yaxis_title=t["price_axis"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    summary = pd.DataFrame(
        [
            {t["concept_col"]: t["event_datetime_label"], t["example_col"]: str(row.get("datetime", ""))},
            {t["concept_col"]: t["future_datetime_label"], t["example_col"]: str(context["future_datetime"])},
            {
                t["concept_col"]: t["expected_direction_label"],
                t["example_col"]: _direction_label_for_explainer(expected_direction, t),
            },
            {
                t["concept_col"]: t["future_move_formula_label"],
                t["example_col"]: t["future_move_formula"].format(
                    future=_format_criterion_value(future_price, 3),
                    current=_format_criterion_value(current_price, 3),
                    tick_size=_format_criterion_value(tick_size, 3),
                    move=_format_criterion_value(future_move, 2),
                ),
            },
            {t["concept_col"]: t["outcome_label"], t["example_col"]: outcome_label},
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown(f"**{t['base_checks_title']}**")
    for idx, (label, passed) in enumerate(
        _base_rate_criteria(row, selected_success, horizon_ticks, min_success_ticks, t)
    ):
        st.checkbox(label, value=bool(passed), disabled=True, key=f"base_rate_{example_kind}_{selected_idx}_{idx}")
    st.markdown(f"**{t['event_checks_title']}**")
    for idx, (label, passed) in enumerate(_event_criteria(row, hypothesis, thresholds, t)):
        st.checkbox(label, value=bool(passed), disabled=True, key=f"event_rule_{example_kind}_{selected_idx}_{idx}")


def _render_base_rate_visual_explainer(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    hypothesis: str,
    horizon_ticks: int,
    min_success_ticks: float,
    t: dict,
    lang: str,
    tpl: str,
) -> None:
    st.markdown(f"#### {t['base_rate_visual_title']}")
    st.caption(t["base_rate_visual_caption"])

    if features.empty or "future_move_ticks" not in features.columns:
        st.info(t["base_rate_visual_empty"])
        return

    target = _target_success_mask(features, min_success_ticks)
    valid = features["future_move_ticks"].notna()
    base_rate = float(target[valid].mean()) if valid.any() else np.nan
    event_accuracy = float(candidates["is_correct"].mean()) if not candidates.empty else np.nan
    event_count = int(len(candidates))
    thresholds = _effective_thresholds(features.attrs.get("tick_pulse_thresholds", {}))

    st.markdown(
        t["base_rate_mechanics"].format(
            fast_window=RTV_FAST_WINDOW_TICKS,
            horizon=int(horizon_ticks),
            success=f"{min_success_ticks:g}",
        )
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(t["base_rate_metric"], "N/A" if pd.isna(base_rate) else f"{base_rate:.1%}")
    metric_cols[1].metric(t["event_accuracy_metric"], "N/A" if pd.isna(event_accuracy) else f"{event_accuracy:.1%}")
    metric_cols[2].metric(t["base_rate_valid_rows"], f"{int(valid.sum()):,}")
    metric_cols[3].metric(t["base_rate_event_rows"], f"{event_count:,}")

    if candidates.empty:
        st.info(t["no_events"])
        return

    st.markdown(f"#### {t['event_examples_title']}")
    st.caption(t["event_examples_caption"])
    examples = _pick_event_example_indices(candidates)
    success_col, failure_col = st.columns(2)
    with success_col:
        _render_event_example_card(
            features,
            examples["success"],
            "success",
            hypothesis,
            horizon_ticks,
            min_success_ticks,
            thresholds,
            t,
            tpl,
        )
    with failure_col:
        _render_event_example_card(
            features,
            examples["failure"],
            "failure",
            hypothesis,
            horizon_ticks,
            min_success_ticks,
            thresholds,
            t,
            tpl,
        )


@st.cache_data(show_spinner=False)
def compute_main_contract_file_sweep(
    path: str,
    mtime: float,
    hypothesis: str,
    window: int,
    min_success_ticks: float,
    threshold_mode: str,
) -> pd.DataFrame:
    _ = mtime  # Kept in the signature so Streamlit cache invalidates on file refresh.
    return run_main_contract_file_sweep(
        path,
        hypothesis=hypothesis,
        window=window,
        min_success_ticks=min_success_ticks,
        threshold_mode=threshold_mode,
        default_thresholds=_effective_thresholds(None),
        source_base_dir=PROJECT_ROOT,
        horizons=RESEARCH_SWEEP_HORIZONS,
    )


def _discover_tick_files() -> list[dict]:
    files = []
    for path in discover_futures_cn_tick_files(patterns=("*tick_all_data*.parquet",)):
        filename = path.name
        rel_path = os.path.relpath(path, REPO_ROOT)
        stat = path.stat()
        metadata = _parse_tick_file_metadata(filename)
        label = _format_tick_file_label(filename, stat.st_size)
        files.append(
            {
                "path": rel_path,
                "label": label,
                "product_hint": metadata.get("product"),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    files.sort(key=lambda item: (item["size"], item["mtime"]), reverse=True)
    return files


def _parse_tick_file_metadata(filename: str) -> dict[str, str | None]:
    import re

    match = re.search(
        r"(?P<count>\d+)contract(?:_(?P<product>[A-Za-z]+))?_raw_(?P<start>\d{8})_(?P<end>\d{8})",
        filename,
    )
    if not match:
        return {"count": None, "product": None, "start": None, "end": None}
    return match.groupdict()


def _format_tick_file_label(filename: str, size: int) -> str:
    match = _parse_tick_file_metadata(filename)
    size_label = f"{size / 1024 / 1024:.1f} MB"
    if not match.get("count"):
        return f"{filename} ({size_label})"

    product = match.get("product") or "multi"
    return (
        f"{product} | {match.get('count')} contracts | "
        f"{match.get('start')}->{match.get('end')} | {size_label}"
    )


def _resolve_path(path_text: str) -> str:
    path_text = path_text.strip()
    if os.path.isabs(path_text):
        return path_text
    repo_path = os.path.join(REPO_ROOT, path_text)
    if os.path.exists(repo_path):
        return repo_path
    return os.path.join(PROJECT_ROOT, path_text)


def _ensure_option_state(key: str, options: list, default_value):
    if not options:
        return None
    if default_value not in options:
        default_value = options[0]
    if st.session_state.get(key) not in options:
        st.session_state[key] = default_value
    return st.session_state[key]


def _stateful_selectbox(label: str, options: list, key: str, default_value, **kwargs):
    if not options:
        return None
    if default_value not in options:
        default_value = options[0]
    if key in st.session_state:
        if st.session_state[key] in options:
            return st.selectbox(label, options, key=key, **kwargs)
        del st.session_state[key]
    return st.selectbox(label, options, index=options.index(default_value), key=key, **kwargs)


def _render_nav(options: dict[str, str]) -> str:
    option_keys = list(options.keys())
    _ensure_option_state("tick_pulse_nav", option_keys, option_keys[0])
    selected = st.pills(
        "Tick Event Study navigation",
        option_keys,
        selection_mode="single",
        format_func=lambda key: options[key],
        key="tick_pulse_nav",
        label_visibility="collapsed",
    )
    if selected is None:
        selected = option_keys[0]
    return selected


def _set_loading_progress(progress, percent: int, text: str) -> None:
    progress.progress(percent, text=f"{percent}% - {text}")


def _format_ci_cell(low: float, high: float) -> str:
    low_text = "N/A" if pd.isna(low) else f"{low:.1%}"
    high_text = "N/A" if pd.isna(high) else f"{high:.1%}"
    return f"{low_text} - {high_text}"


def _ticket_slug(value: object) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).lower())
    return cleaned.strip("_")[:48] or "tick_pulse"


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _base_rate_from_features(features: pd.DataFrame, min_success_ticks: float) -> float:
    if features.empty or "expected_direction" not in features or "future_move_ticks" not in features:
        return np.nan
    valid = features["future_move_ticks"].notna() & features["expected_direction"].isin(["Up", "Down"])
    if not bool(valid.any()):
        return np.nan
    success = (
        (features["expected_direction"].eq("Up") & (features["future_move_ticks"] >= float(min_success_ticks)))
        | (features["expected_direction"].eq("Down") & (features["future_move_ticks"] <= -float(min_success_ticks)))
    )
    return float(success.loc[valid].mean())


def _ticket_decision(events: int, accuracy: float, base_rate: float) -> str:
    if events <= 0 or not np.isfinite(accuracy):
        return "needs_more_data"
    if np.isfinite(base_rate) and accuracy > base_rate:
        return "promote_to_validation"
    if not np.isfinite(base_rate) and accuracy >= 0.5:
        return "promote_to_validation"
    return "needs_review"


def _ticket_confidence(events: int, accuracy: float, base_rate: float) -> float | None:
    if events <= 0 or not np.isfinite(accuracy):
        return None
    sample_component = min(np.log1p(events) / np.log1p(500), 1.0)
    if np.isfinite(base_rate):
        lift_component = max(min((accuracy - base_rate + 0.1) / 0.2, 1.0), 0.0)
    else:
        lift_component = max(min(accuracy, 1.0), 0.0)
    return float(0.35 * sample_component + 0.65 * lift_component)


def _repo_relative(path: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), REPO_ROOT)
    except Exception:
        return str(path)


def _build_tick_pulse_evidence_ticket_payload(
    *,
    selected_file: str,
    file_path: str,
    selected_product: str,
    selected_symbol: str,
    selected_hypothesis: str,
    hypothesis_label: str,
    window: int,
    horizon_ticks: int,
    min_success_ticks: float,
    threshold_mode: str,
    thresholds: dict,
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    backend: str,
    selected_seed: dict | None,
) -> dict[str, object]:
    events = int(len(candidates))
    rows = int(len(features))
    correct_moves = int(candidates["is_correct"].sum()) if "is_correct" in candidates and events else 0
    accuracy = correct_moves / events if events else np.nan
    event_rate = events / rows if rows else np.nan
    avg_move = candidates["future_move_ticks"].mean() if "future_move_ticks" in candidates and events else np.nan
    base_rate = _base_rate_from_features(features, min_success_ticks)
    lift = accuracy - base_rate if np.isfinite(accuracy) and np.isfinite(base_rate) else np.nan
    seed_id = str(selected_seed.get("seed_id", "")) if selected_seed else ""
    hypothesis_key = seed_id if seed_id else selected_hypothesis
    factor_id = f"tick_pulse_{_ticket_slug(hypothesis_key)}"
    signature_payload = {
        "source_page": "03_Tick_Event_Study",
        "source_file": selected_file,
        "symbol": selected_symbol,
        "hypothesis": selected_hypothesis,
        "seed_id": seed_id,
        "window": int(window),
        "horizon_ticks": int(horizon_ticks),
        "min_success_ticks": float(min_success_ticks),
        "threshold_mode": threshold_mode,
    }
    trial_signature = stable_trial_hash(signature_payload)
    ticket_id = make_evidence_ticket_id(signature_payload)
    decision = _ticket_decision(events, accuracy, base_rate)
    confidence = _ticket_confidence(events, accuracy, base_rate)

    return {
        "ticket_id": ticket_id,
        "title": f"{selected_symbol} {hypothesis_label} @ {int(horizon_ticks)} ticks",
        "source_page": "03_Tick_Event_Study",
        "evidence_type": "microstructure_hypothesis",
        "stage": "hypothesis_tested",
        "status": "ready_for_review" if events else "open",
        "decision": decision,
        "thesis": (
            f"{hypothesis_label} on {selected_symbol} produced {events:,} events "
            f"over {rows:,} cleaned tick rows."
        ),
        "factor_id": factor_id,
        "research_family": "tick_pulse_lab",
        "run_id": f"tick_ticket_{trial_signature[:16]}",
        "trial_signature": trial_signature,
        "metric_name": "event_accuracy",
        "metric_value": _finite_float(accuracy),
        "confidence_score": confidence,
        "priority": 1 if decision == "promote_to_validation" else 2,
        "metrics": {
            "rows": rows,
            "events": events,
            "event_rate": _finite_float(event_rate),
            "correct_moves": correct_moves,
            "accuracy": _finite_float(accuracy),
            "base_rate": _finite_float(base_rate),
            "lift": _finite_float(lift),
            "avg_future_move_ticks": _finite_float(avg_move),
            "window": int(window),
            "horizon_ticks": int(horizon_ticks),
            "min_success_ticks": float(min_success_ticks),
            "backend": backend,
        },
        "artifacts": [
            {"kind": "tick_data", "path": _repo_relative(file_path)},
            {"kind": "research_db", "path": _repo_relative(DB_PATH)},
        ],
        "context": {
            "product": selected_product,
            "symbol": selected_symbol,
            "source_file": selected_file,
            "hypothesis": selected_hypothesis,
            "hypothesis_label": hypothesis_label,
            "threshold_mode": threshold_mode,
            "thresholds": thresholds or {},
            "seed_id": seed_id,
            "seed_name": str(selected_seed.get("name", "")) if selected_seed else "",
        },
        "metadata": {
            "schema_version": "tick_pulse_evidence_ticket_v1",
            "next_page": "08_Factor_Review",
        },
    }


class TickPulseLabPage:
    def _render_evidence_ticket_action(self, payload: dict[str, object], events: int, t: dict) -> None:
        button_disabled = events <= 0
        if st.button(
            t["evidence_ticket_button"],
            width="stretch",
            key=f"tick_evidence_ticket_{payload['ticket_id']}",
            disabled=button_disabled,
        ):
            try:
                ticket = record_evidence_ticket(DB_PATH, **payload)
            except Exception as exc:
                st.error(t["evidence_ticket_error"].format(error=exc))
            else:
                st.success(t["evidence_ticket_saved"].format(ticket_id=ticket.ticket_id))
        if button_disabled:
            st.caption(t["evidence_ticket_no_events"])

    def _build_seed_math_sweep(
        self,
        *,
        features: pd.DataFrame,
        seed: dict,
        min_success_ticks: float,
    ) -> pd.DataFrame:
        rows = []
        seed_name = str(seed.get("name") or "Saved seed")
        for horizon in RESEARCH_SWEEP_HORIZONS:
            out, _ = evaluate_seed_hypothesis(features, int(horizon), seed, float(min_success_ticks))
            valid = out["future_move_ticks"].notna()
            signal = out["hypothesis_signal"].fillna(False).astype(bool)
            target = out["is_correct"].fillna(False).astype(bool)
            event_mask = valid & signal
            events = int(event_mask.sum())
            successes = int(target[event_mask].sum()) if events else 0
            accuracy = successes / events if events else np.nan
            base_rate = float(target[valid].mean()) if valid.any() else np.nan
            expected_move = np.where(
                out["expected_direction"].eq("Down"),
                -out["future_move_ticks"],
                out["future_move_ticks"],
            )
            ci_low, ci_high = self._wilson_interval(successes, events)
            rows.append(
                {
                    "hypothesis": seed_name,
                    "horizon": int(horizon),
                    "events": events,
                    "successes": successes,
                    "accuracy": accuracy,
                    "base_rate": base_rate,
                    "lift": accuracy - base_rate if events else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "avg_move": float(out.loc[event_mask, "future_move_ticks"].mean()) if events else np.nan,
                    "expected_avg": (
                        float(pd.Series(expected_move, index=out.index).loc[event_mask].mean())
                        if events
                        else np.nan
                    ),
                    "backend": out.attrs.get("tick_pulse_backend", "python_seed"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
        if total <= 0:
            return np.nan, np.nan
        p_hat = successes / total
        denom = 1.0 + z * z / total
        center = p_hat + z * z / (2.0 * total)
        margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
        return (center - margin) / denom, (center + margin) / denom

    def _load_or_compute_math_sweep(
        self,
        *,
        features: pd.DataFrame,
        file_path: str,
        mtime: float,
        product: str,
        symbol: str,
        window: int,
        min_success_ticks: float,
        threshold_mode: str,
        t: dict,
    ) -> pd.DataFrame:
        source_file = os.path.relpath(os.path.abspath(file_path), PROJECT_ROOT)
        horizon_set = ",".join(str(value) for value in RESEARCH_SWEEP_HORIZONS)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_type": "math_horizon_sweep",
            "source_file": source_file,
            "source_mtime": float(mtime),
            "product": product,
            "symbol": symbol,
            "window": int(window),
            "min_success_ticks": float(min_success_ticks),
            "threshold_mode": threshold_mode,
            "horizon_set": horizon_set,
            "hypotheses": HYPOTHESIS_KEYS,
            "engine": "tick_horizon_sweep_v5",
        }
        cache_key = make_cache_key(payload)
        cached = load_cached_dataframe(DB_PATH, cache_key, BASE_DIR)
        if cached is not None:
            raw, metadata = cached
            st.progress(100, text=f"100% - {t['math_cache_hit']}")
            st.success(
                f"{t['math_cache_hit']} | rows={len(raw):,} | "
                f"backend={metadata.get('backend', 'sqlite_cache')}"
            )
            return raw

        progress = st.progress(0, text=f"0% - {t['math_cache_miss']}")

        def compute() -> pd.DataFrame:
            progress.progress(25, text=f"25% - {t['loading_thresholds']}")
            if threshold_mode == "adaptive":
                thresholds_by_hypothesis = {
                    hypothesis: build_adaptive_thresholds(features, hypothesis)
                    for hypothesis in HYPOTHESIS_KEYS
                }
            else:
                thresholds_by_hypothesis = {}

            progress.progress(65, text=f"65% - {t['loading_events']}")
            raw, _ = _build_research_sweep(
                features,
                min_success_ticks,
                t,
                thresholds_by_hypothesis,
            )
            progress.progress(90, text=f"90% - {t['math_cache_store']}")
            return raw

        result = get_or_compute_dataframe(
            db_path=DB_PATH,
            logs_dir=LOGS_DIR,
            base_dir=BASE_DIR,
            cache_key=cache_key,
            metadata={
                "cache_type": "math_horizon_sweep",
                "source_file": source_file,
                "source_mtime": float(mtime),
                "product": product,
                "symbol": symbol,
                "hypothesis": "ALL",
                "threshold_mode": threshold_mode,
                "window": int(window),
                "min_success_ticks": float(min_success_ticks),
                "horizon_set": horizon_set,
                "backend": "python",
            },
            compute_fn=compute,
        )
        progress.progress(100, text=f"100% - {t['loading_done']}")
        st.success(
            f"{t['math_cache_backend']}: saved | rows={len(result.data):,} | "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )
        return result.data

    def _render_cross_asset_sweep(
        self,
        tick_files: list[dict],
        hypothesis: str,
        window: int,
        min_success_ticks: float,
        threshold_mode: str,
        t: dict,
    ) -> None:
        st.markdown(f"#### {t['cross_asset_title']}")
        st.caption(t["cross_asset_intro"])
        st.info(t["cross_asset_note"])

        product_files = []
        seen_products = set()
        for item in tick_files:
            product = item.get("product_hint")
            if not product:
                continue
            product_key = str(product).lower()
            if product_key in seen_products:
                continue
            seen_products.add(product_key)
            product_files.append(item)

        if not product_files:
            st.info(t["cross_asset_empty"])
            return

        source_manifest = "|".join(f"{item['path']}:{item['mtime']}" for item in product_files)
        cache_payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_type": "cross_asset_main_contract_sweep",
            "engine": "main_contract_horizon_sweep_v5",
            "hypothesis": hypothesis,
            "window": int(window),
            "min_success_ticks": float(min_success_ticks),
            "threshold_mode": threshold_mode,
            "horizon_set": ",".join(str(value) for value in RESEARCH_SWEEP_HORIZONS),
            "sources": source_manifest,
        }
        run_key = make_cache_key(cache_payload)

        cached = load_cached_dataframe(DB_PATH, run_key, BASE_DIR)
        if cached is not None and st.session_state.get("tick_cross_asset_sweep_key") != run_key:
            raw, metadata = cached
            st.session_state.tick_cross_asset_sweep_key = run_key
            st.session_state.tick_cross_asset_sweep = raw
            st.caption(
                f"{t['cross_asset_cache_hit']} | rows={len(raw):,} | "
                f"backend={metadata.get('backend', 'sqlite_cache')}"
            )

        if st.button(t["cross_asset_run"], width="stretch", key="tick_cross_asset_run"):
            progress = st.progress(0, text=f"0% - {t['cross_asset_loading']}")

            def compute_cross_asset() -> pd.DataFrame:
                frames = []
                total = len(product_files)
                for idx, item in enumerate(product_files, start=1):
                    percent = int((idx - 1) / total * 100)
                    progress.progress(
                        percent,
                        text=f"{percent}% - {t['cross_asset_loading']} {item['label']}",
                    )
                    path = _resolve_path(item["path"])
                    try:
                        frame = compute_main_contract_file_sweep(
                            path,
                            item["mtime"],
                            hypothesis,
                            int(window),
                            float(min_success_ticks),
                            threshold_mode,
                        )
                        if not frame.empty:
                            frames.append(frame)
                    except Exception as exc:
                        st.warning(f"{item['label']}: {exc}")
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            result = get_or_compute_dataframe(
                db_path=DB_PATH,
                logs_dir=LOGS_DIR,
                base_dir=BASE_DIR,
                cache_key=run_key,
                metadata={
                    "cache_type": "cross_asset_main_contract_sweep",
                    "source_file": source_manifest,
                    "source_mtime": max(float(item["mtime"]) for item in product_files),
                    "product": "ALL",
                    "symbol": "MAIN_CONTRACT",
                    "hypothesis": hypothesis,
                    "threshold_mode": threshold_mode,
                    "window": int(window),
                    "min_success_ticks": float(min_success_ticks),
                    "horizon_set": ",".join(str(value) for value in RESEARCH_SWEEP_HORIZONS),
                    "backend": "cpp_or_python",
                },
                compute_fn=compute_cross_asset,
            )
            progress.progress(100, text=f"100% - {t['loading_done']}")
            st.session_state.tick_cross_asset_sweep_key = run_key
            st.session_state.tick_cross_asset_sweep = result.data
            status_text = t["cross_asset_cache_hit"] if result.cache_hit else t["cross_asset_cache_saved"]
            st.success(
                f"{status_text} | rows={len(result.data):,} | "
                f"elapsed={result.elapsed_seconds:.2f}s"
            )

        raw = st.session_state.get("tick_cross_asset_sweep")
        if raw is None or st.session_state.get("tick_cross_asset_sweep_key") != run_key:
            st.caption(t["cross_asset_not_run"])
            return
        if raw.empty:
            st.info(t["cross_asset_empty"])
            return

        raw = _add_lift_stability(raw)
        display = raw.copy()
        display[t["cross_asset_asset"]] = display["asset"]
        display[t["cross_asset_contract"]] = display["main_contract"]
        display[t["cross_asset_lift_stability"]] = display["lift_stability_code"].map(
            t["cross_asset_stability_labels"]
        )
        display[t["cross_asset_peak_lift"]] = np.where(
            display["peak_lift_row"],
            t["cross_asset_peak_lift_label"],
            "",
        )
        display[t["sweep_horizon"]] = display["horizon"].map(lambda value: f"{value} {t['ticks']}")
        display[t["sweep_events"]] = display["events"]
        display[t["sweep_accuracy"]] = display["accuracy"]
        display[t["sweep_base_rate"]] = display["base_rate"]
        display[t["sweep_lift"]] = display["lift"]
        display[t["sweep_ci"]] = display.apply(lambda row: _format_ci_cell(row["ci_low"], row["ci_high"]), axis=1)
        display[t["sweep_avg_move"]] = display["avg_move"]
        display[t["sweep_expected_avg"]] = display["expected_avg"]

        columns = [
            t["cross_asset_asset"],
            t["cross_asset_contract"],
            t["cross_asset_lift_stability"],
            t["cross_asset_peak_lift"],
            t["sweep_horizon"],
            t["sweep_events"],
            t["sweep_accuracy"],
            t["sweep_base_rate"],
            t["sweep_lift"],
            t["sweep_ci"],
            t["sweep_avg_move"],
            t["sweep_expected_avg"],
        ]
        styled_display = _style_cross_asset_stability(
            display[columns],
            t["cross_asset_lift_stability"],
            t["cross_asset_peak_lift"],
        ).format(
            {
                t["sweep_accuracy"]: "{:.1%}",
                t["sweep_base_rate"]: "{:.1%}",
                t["sweep_lift"]: "{:+.1%}",
                t["sweep_avg_move"]: "{:.2f}",
                t["sweep_expected_avg"]: "{:.2f}",
            }
        )
        st.dataframe(
            styled_display,
            width="stretch",
            hide_index=True,
        )

    def render(self) -> None:
        init_global_ui_state()
        apply_global_style()
        render_global_controls_in_sidebar()

        lang = st.session_state.lang
        if lang == "CN":
            lang = "ZH"
        t = PAGE_TEXT.get(lang, PAGE_TEXT["EN"])
        tpl = get_plotly_template(st.session_state.theme_mode)

        st.title(t["title"])
        st.caption(t["subtitle"])

        section_labels = {
            "math": t["math_title"],
            "events": t["event_map"],
        }
        selected_section = _render_nav(section_labels)

        tick_files = _discover_tick_files()
        if tick_files:
            file_options = [item["path"] for item in tick_files]
            label_by_path = {item["path"]: item["label"] for item in tick_files}
            default_file_index = file_options.index(DEFAULT_TICK_FILE) if DEFAULT_TICK_FILE in file_options else 0
            selected_file = _stateful_selectbox(
                t["file"],
                file_options,
                key="tick_scope_file",
                default_value=file_options[default_file_index],
                format_func=lambda path: label_by_path.get(path, path),
            )
        else:
            selected_file = DEFAULT_TICK_FILE
            st.warning(t["no_tick_files"])

        file_path = _resolve_path(selected_file)

        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            st.stop()

        load_progress = st.progress(0, text=f"0% - {t['loading_file']}")
        mtime = os.path.getmtime(file_path)
        try:
            _set_loading_progress(load_progress, 5, t["loading_summary"])
            summary = compute_contract_summary(file_path, mtime)
        except Exception as exc:
            st.error(f"Could not load tick data: {exc}")
            st.stop()

        _set_loading_progress(load_progress, 35, t["loading_summary"])
        products = sorted(summary["product"].dropna().unique())
        if summary.empty or not products:
            st.error(t["cross_asset_empty"])
            st.stop()

        scope_cols = st.columns(2)
        contract_symbols = sorted(summary["symbol"].dropna().astype(str).unique())
        if not contract_symbols:
            st.error(t["cross_asset_empty"])
            st.stop()
        default_symbol = summary.sort_values("positive_volume_delta", ascending=False)["symbol"].astype(str).iloc[0]
        default_index = contract_symbols.index(default_symbol) if default_symbol in contract_symbols else 0
        with scope_cols[0]:
            selected_symbol = _stateful_selectbox(
                t["symbol"],
                contract_symbols,
                key="tick_scope_symbol",
                default_value=contract_symbols[default_index],
            )
        symbol_summary = summary.loc[summary["symbol"].astype(str).eq(selected_symbol)]
        selected_product = (
            str(symbol_summary["product"].iloc[0])
            if not symbol_summary.empty and "product" in symbol_summary
            else products[0]
        )
        seeds = list_hypothesis_seeds(DB_PATH)
        source_options = ["built_in", "saved_seed"] if seeds else ["built_in"]
        _ensure_option_state("tick_scope_hypothesis_source", source_options, "built_in")
        selected_seed = None
        with scope_cols[1]:
            if len(source_options) > 1:
                hypothesis_source = st.segmented_control(
                    t["hypothesis_source"],
                    source_options,
                    format_func=lambda key: t["hypothesis_source_options"].get(key, key),
                    key="tick_scope_hypothesis_source",
                )
                if hypothesis_source is None:
                    hypothesis_source = "built_in"
            else:
                hypothesis_source = "built_in"

            if hypothesis_source == "saved_seed":
                seed_ids = [str(seed["seed_id"]) for seed in seeds]
                selected_seed_id = _stateful_selectbox(
                    t["saved_seed_select"],
                    seed_ids,
                    key="tick_scope_seed_id",
                    default_value=seed_ids[0],
                    format_func=lambda seed_id: format_seed_label(
                        next(seed for seed in seeds if str(seed["seed_id"]) == str(seed_id))
                    ),
                )
                selected_seed = next(seed for seed in seeds if str(seed["seed_id"]) == str(selected_seed_id))
                selected_hypothesis = f"seed:{selected_seed_id}"
            else:
                selected_hypothesis = _stateful_selectbox(
                    t["hypothesis"],
                    HYPOTHESIS_KEYS,
                    key="tick_scope_hypothesis",
                    default_value=DEFAULT_HYPOTHESIS_KEY,
                    format_func=lambda key: t["hypothesis_labels"][key],
                )
        if not seeds:
            st.caption(t["saved_seed_empty"])

        test_cols = st.columns(4)
        with test_cols[0]:
            window = st.slider(t["window"], min_value=30, max_value=600, value=120, step=30, key="tick_scope_window")
        with test_cols[1]:
            horizon_ticks = st.slider(t["horizon"], min_value=30, max_value=600, value=120, step=30, key="tick_scope_horizon")
        with test_cols[2]:
            min_success_ticks = st.slider(
                t["success_move"],
                min_value=0.5,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key="tick_scope_success_move",
            )
        with test_cols[3]:
            top_n = st.slider(t["top"], min_value=25, max_value=1000, value=200, step=25, key="tick_scope_top_n")

        mode_options = ["adaptive", "fixed"]
        _ensure_option_state("tick_scope_threshold_mode", mode_options, "adaptive")
        threshold_mode = st.session_state.get("tick_scope_threshold_mode", "adaptive")

        raw_scope = load_tick_scope_data(file_path, mtime, selected_product, selected_symbol)
        _set_loading_progress(load_progress, 60, t["loading_features"])
        _set_loading_progress(load_progress, 75, t["loading_thresholds"])
        _set_loading_progress(load_progress, 88, t["loading_events"])
        if selected_seed is not None:
            seed_file = str(selected_seed.get("source_file") or "")
            seed_symbol = str(selected_seed.get("symbol") or "")
            if seed_file != selected_file or seed_symbol != selected_symbol:
                st.warning(
                    t["saved_seed_scope_warning"].format(
                        seed_file=seed_file,
                        seed_symbol=seed_symbol,
                        current_file=selected_file,
                        current_symbol=selected_symbol,
                    )
                )
            features, candidates, adaptive_thresholds = compute_seed_hypothesis_evaluation(
                file_path,
                mtime,
                selected_product,
                selected_symbol,
                int(window),
                int(horizon_ticks),
                selected_seed,
                float(min_success_ticks),
            )
        else:
            features, candidates, adaptive_thresholds = compute_hypothesis_evaluation(
                file_path,
                mtime,
                selected_product,
                selected_symbol,
                int(window),
                int(horizon_ticks),
                selected_hypothesis,
                float(min_success_ticks),
                threshold_mode,
            )
        _set_loading_progress(load_progress, 100, t["loading_done"])
        st.success(f"100% - {t['loading_complete']}")

        liquid = summary.sort_values("positive_volume_delta", ascending=False).iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["detected"], ", ".join(products))
        m2.metric(t["liquid"], str(liquid["symbol"]))
        m3.metric(t["rows"], f"{len(features):,}")
        m4.metric(t["events"], f"{len(candidates):,}")

        if len(features) > 0:
            event_rate = len(candidates) / len(features)
        else:
            event_rate = 0.0
        correct_moves = int(candidates["is_correct"].sum()) if not candidates.empty else 0
        accuracy = correct_moves / len(candidates) if len(candidates) else np.nan
        avg_move = candidates["future_move_ticks"].mean() if not candidates.empty else np.nan
        m5, m6, m7, m8 = st.columns(4)
        m5.metric(t["event_rate"], f"{event_rate:.2%}")
        m6.metric(t["correct_moves"], f"{correct_moves:,}")
        m7.metric(t["accuracy"], "N/A" if pd.isna(accuracy) else f"{accuracy:.1%}")
        m8.metric(t["avg_move"], "N/A" if pd.isna(avg_move) else f"{avg_move:.1f} {t['ticks']}")
        backend = features.attrs.get("tick_pulse_backend", "python")
        backend_labels = {
            "EN": {
                "cpp": "Backend: C++ quant_core",
                "python_fallback": "Backend: Python fallback",
                "python": "Backend: Python",
                "python_seed": "Backend: Python saved-seed evaluator",
            },
            "ZH": {
                "cpp": "计算后端：C++ quant_core",
                "python_fallback": "计算后端：Python 备用路径",
                "python": "计算后端：Python",
                "python_seed": "计算后端：Python 已保存种子评估器",
            },
        }
        st.caption(backend_labels.get(lang, backend_labels["EN"]).get(backend, backend))
        hypothesis_label = (
            str(selected_seed.get("name") or selected_seed.get("seed_id"))
            if selected_seed is not None
            else t["hypothesis_labels"].get(selected_hypothesis, selected_hypothesis)
        )
        ticket_payload = _build_tick_pulse_evidence_ticket_payload(
            selected_file=selected_file,
            file_path=file_path,
            selected_product=selected_product,
            selected_symbol=selected_symbol,
            selected_hypothesis=selected_hypothesis,
            hypothesis_label=hypothesis_label,
            window=int(window),
            horizon_ticks=int(horizon_ticks),
            min_success_ticks=float(min_success_ticks),
            threshold_mode=threshold_mode,
            thresholds=adaptive_thresholds,
            features=features,
            candidates=candidates,
            backend=backend,
            selected_seed=selected_seed,
        )
        self._render_evidence_ticket_action(ticket_payload, len(candidates), t)
        if selected_seed is not None:
            with st.expander(t["saved_seed_context"], expanded=False):
                st.write(
                    {
                        "seed_id": selected_seed.get("seed_id"),
                        "name": selected_seed.get("name"),
                        "source_file": selected_seed.get("source_file"),
                        "symbol": selected_seed.get("symbol"),
                        "event_time": selected_seed.get("event_time"),
                        "pulse_family": selected_seed.get("pulse_family"),
                        "quality_status": selected_seed.get("quality_status"),
                        "rule": selected_seed.get("rule", {}),
                    }
                )
        else:
            st.caption(t["threshold_mode_active"].format(mode=t["threshold_modes"].get(threshold_mode, threshold_mode)))
            with st.expander(t["threshold_mode_advanced_title"], expanded=False):
                st.markdown(t["threshold_mode_note"])
                selected_threshold_mode = st.segmented_control(
                    t["threshold_mode"],
                    mode_options,
                    format_func=lambda key: t["threshold_modes"].get(key, key),
                    key="tick_scope_threshold_mode",
                )
                if selected_threshold_mode is None:
                    selected_threshold_mode = threshold_mode
                st.caption(t["threshold_mode_debug_note"])

        if selected_section == "events":
            visible_candidates = _select_visible_candidates(candidates, top_n) if not candidates.empty else candidates
            _render_calculation_audit(raw_scope, features, candidates, visible_candidates, t)
            if candidates.empty:
                st.info(t["no_events"])
            else:
                _render_event_map(features, visible_candidates, selected_hypothesis, tpl, t)
                _render_event_horizon_bars(features, candidates, min_success_ticks, tpl, t)
                _render_outcome_overview(candidates, tpl, t)
                _render_event_ledger(candidates, t)
            st.divider()
            st.markdown(f"#### {t['sweep_title']}")
            if selected_seed is not None:
                sweep_raw = self._build_seed_math_sweep(
                    features=compute_feature_frame(file_path, mtime, selected_product, selected_symbol, int(window)),
                    seed=selected_seed,
                    min_success_ticks=min_success_ticks,
                )
            else:
                sweep_raw = self._load_or_compute_math_sweep(
                    features=features,
                    file_path=file_path,
                    mtime=mtime,
                    product=selected_product,
                    symbol=selected_symbol,
                    window=window,
                    min_success_ticks=min_success_ticks,
                    threshold_mode=threshold_mode,
                    t=t,
                )
            st.dataframe(
                format_research_sweep_display(sweep_raw, t),
                width="stretch",
                hide_index=True,
            )
            st.divider()
            if selected_seed is None:
                self._render_cross_asset_sweep(
                    tick_files,
                    selected_hypothesis,
                    window,
                    min_success_ticks,
                    threshold_mode,
                    t,
                )
            else:
                st.info(t["saved_seed_cross_asset_note"])

        elif selected_section == "math":
            if selected_seed is None:
                _render_base_rate_visual_explainer(
                    features,
                    candidates,
                    selected_hypothesis,
                    horizon_ticks,
                    min_success_ticks,
                    t,
                    lang,
                    tpl,
                )
            else:
                st.info(t["saved_seed_math_note"])
                st.json(selected_seed.get("rule", {}), expanded=True)
