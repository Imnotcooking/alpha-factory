from __future__ import annotations

import numpy as np
import pandas as pd

from oqp.research.tick_pulse.features import (
    add_session_ids,
    build_pulse_features,
    clean_tick_data,
    infer_tick_size,
    product_prefix,
)
from oqp.research.tick_pulse.native_bridge import load_tick_pulse_core


NUMERIC_FEATURE_COLUMNS = [
    "mid_price",
    "spread",
    "book_imbalance",
    "volume_delta_raw",
    "volume_delta",
    "oi_delta",
    "last_price_delta",
    "mid_price_delta",
    "mid_move_ticks",
    "trade_sign",
    "signed_volume",
    "rolling_signed_volume",
    "rolling_total_volume",
    "flow_imbalance",
    "rolling_pos_volume_median",
    "volume_intensity",
    "rolling_mid_move_ticks",
    "rolling_abs_tick_median",
    "price_shock",
    "pulse_score",
]

__all__ = [
    "NUMERIC_FEATURE_COLUMNS",
    "build_pulse_features_fast",
]


def _encode_int_codes(values: pd.Series) -> np.ndarray:
    return pd.Categorical(values).codes.astype(np.int32, copy=False)


def _prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_tick_data(df)
    out = add_session_ids(out)
    out["product"] = out["symbol"].map(product_prefix)

    tick_size_map = {
        symbol: infer_tick_size(group)
        for symbol, group in out.groupby("symbol", sort=False)
    }
    out["tick_size_est"] = out["symbol"].map(tick_size_map).replace(0, np.nan)
    return out


def build_pulse_features_fast(
    df: pd.DataFrame,
    window: int = 120,
    *,
    prefer_cpp: bool = True,
    raise_on_cpp_error: bool = False,
) -> pd.DataFrame:
    if not prefer_cpp:
        out = build_pulse_features(df, window=window)
        out.attrs["tick_pulse_feature_backend"] = "python"
        return out

    try:
        out = _build_pulse_features_cpp(df, window=window)
        out.attrs["tick_pulse_feature_backend"] = "cpp"
        return out
    except Exception as exc:
        if raise_on_cpp_error:
            raise
        out = build_pulse_features(df, window=window)
        out.attrs["tick_pulse_feature_backend"] = "python_fallback"
        out.attrs["tick_pulse_feature_backend_error"] = str(exc)
        return out


def _build_pulse_features_cpp(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    qc = load_tick_pulse_core(required_features=("compute_tick_pulse_features_core",))

    if not hasattr(qc, "compute_tick_pulse_features_core"):
        raise AttributeError("native quant core is missing compute_tick_pulse_features_core.")

    out = _prepare_feature_frame(df)
    min_periods = max(10, int(window) // 5)
    result = qc.compute_tick_pulse_features_core(
        np.ascontiguousarray(out["last_price"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["volume"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["bid_price_1"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["bid_volume_1"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["ask_price_1"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["ask_volume_1"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["oi"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(out["tick_size_est"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(_encode_int_codes(out["symbol"])),
        np.ascontiguousarray(_encode_int_codes(out["_session_id"])),
        int(window),
        int(min_periods),
    )

    for column in NUMERIC_FEATURE_COLUMNS:
        out[column] = np.asarray(result[column], dtype=float)

    direction_codes = np.asarray(result["pulse_direction_code"], dtype=np.int32)
    out["pulse_direction"] = np.select(
        [direction_codes > 0, direction_codes < 0],
        ["buy", "sell"],
        default="neutral",
    )
    out["flow_price_aligned"] = np.asarray(result["flow_price_aligned"], dtype=bool)
    out["book_flow_aligned"] = np.asarray(result["book_flow_aligned"], dtype=bool)
    pulse_type_codes = np.asarray(result["pulse_type_code"], dtype=np.int32)
    out["pulse_type"] = np.select(
        [
            pulse_type_codes == 0,
            pulse_type_codes == 1,
            pulse_type_codes == 2,
        ],
        [
            "momentum",
            "absorption",
            "queue_pressure",
        ],
        default="mixed",
    )
    return out
