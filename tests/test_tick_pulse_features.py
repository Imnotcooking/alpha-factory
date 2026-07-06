from pathlib import Path

import numpy as np
import pandas as pd

from oqp.research.tick_pulse import feature_bridge
from oqp.research.tick_pulse.features import build_pulse_features, load_tick_scope
from oqp.research.tick_pulse.feature_bridge import build_pulse_features_fast


def _raw_ticks(rows_per_session: int = 80) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-06-01 09:00:00")
    for symbol_idx, symbol in enumerate(["au2608", "ag2608"]):
        for session_id in [0, 1]:
            base = 900.0 + symbol_idx * 20.0 + session_id * 5.0
            cumulative_volume = 100
            for i in range(rows_per_session):
                cumulative_volume += 0 if i % 7 == 0 else (i % 5) + 1
                mid = base + 0.03 * i + 0.2 * np.sin(i / 5.0)
                bid = mid - 0.5
                ask = mid + 0.5
                rows.append(
                    {
                        "symbol": symbol,
                        "datetime": start + pd.Timedelta(days=session_id, minutes=symbol_idx, seconds=i),
                        "last_price": ask if i % 4 == 0 else bid if i % 6 == 0 else mid,
                        "volume": cumulative_volume,
                        "bid_price_1": bid,
                        "bid_volume_1": 10 + (i % 9),
                        "ask_price_1": ask,
                        "ask_volume_1": 12 + ((i * 2) % 11),
                        "oi": 1000.0 + i + session_id * 3,
                    }
                )

    rows.append(
        {
            "symbol": "au2608",
            "datetime": start + pd.Timedelta(seconds=999),
            "last_price": 0.0,
            "volume": 999,
            "bid_price_1": 0.0,
            "bid_volume_1": 1,
            "ask_price_1": 0.0,
            "ask_volume_1": 1,
            "oi": 1000.0,
        }
    )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_load_tick_scope_filters_symbol(tmp_path: Path):
    path = tmp_path / "ticks.parquet"
    _raw_ticks(rows_per_session=3).to_parquet(path, index=False)

    scoped = load_tick_scope(str(path), symbol="au2608")

    assert scoped["symbol"].unique().tolist() == ["au2608"]
    assert scoped["datetime"].is_monotonic_increasing


def test_build_pulse_features_fast_python_backend():
    features = build_pulse_features_fast(_raw_ticks(rows_per_session=20), window=20, prefer_cpp=False)

    assert features.attrs["tick_pulse_feature_backend"] == "python"
    assert {"pulse_score", "pulse_direction", "pulse_type", "_session_id"}.issubset(features.columns)
    assert features["last_price"].gt(0).all()


def test_feature_builder_falls_back_to_python(monkeypatch):
    raw = _raw_ticks(rows_per_session=20)

    def fail_cpp(*args, **kwargs):
        raise RuntimeError("simulated stale extension")

    monkeypatch.setattr(feature_bridge, "_build_pulse_features_cpp", fail_cpp)
    fallback = build_pulse_features_fast(raw, window=20)
    baseline = build_pulse_features(raw, window=20)

    assert fallback.attrs["tick_pulse_feature_backend"] == "python_fallback"
    assert "tick_pulse_feature_backend_error" in fallback.attrs
    np.testing.assert_allclose(
        fallback["pulse_score"].to_numpy(),
        baseline["pulse_score"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
