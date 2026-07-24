from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oqp.market import (
    forecast_volatility_models,
    load_latest_volatility_forecasts,
    select_forecast_vol,
    write_volatility_forecasts,
)
from oqp.market.vol_forecast import VolatilityForecast


class VolatilityForecastTests(unittest.TestCase):
    def sample_history(self, rows: int = 180) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-10-01", periods=rows)
        returns = rng.normal(loc=0.0004, scale=0.018, size=rows)
        close = 100 * np.exp(np.cumsum(returns))
        high = close * (1 + rng.uniform(0.002, 0.018, size=rows))
        low = close * (1 - rng.uniform(0.002, 0.018, size=rows))
        open_ = close * (1 + rng.normal(0, 0.002, size=rows))
        volume = rng.integers(1_000_000, 5_000_000, size=rows)
        return pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    def test_forecast_models_emit_expected_rows(self) -> None:
        forecasts = forecast_volatility_models(" aapl ", self.sample_history(), horizons=(1, 5, 21))

        self.assertEqual(set(forecasts["horizon_days"]), {1, 5, 21})
        self.assertEqual(set(forecasts["model"]), {"baseline_blend", "har_hv", "garch_1_1", "ensemble"})

        baseline = forecasts[forecasts["model"].eq("baseline_blend")]
        har = forecasts[forecasts["model"].eq("har_hv")]
        ensemble = forecasts[forecasts["model"].eq("ensemble")]
        self.assertTrue((baseline["forecast_vol"] > 0).all())
        self.assertTrue((har["forecast_vol"] > 0).all())
        self.assertTrue((ensemble["forecast_vol"] > 0).all())

    def test_select_forecast_vol_picks_nearest_ensemble_horizon(self) -> None:
        frame = pd.DataFrame(
            [
                {"horizon_days": 5, "model": "ensemble", "forecast_vol": 0.25},
                {"horizon_days": 21, "model": "ensemble", "forecast_vol": 0.33},
                {"horizon_days": 21, "model": "baseline_blend", "forecast_vol": 0.20},
            ]
        )

        self.assertEqual(select_forecast_vol(frame, horizon_days=30, fallback=0.10), 0.33)

    def test_persist_and_load_latest_forecasts(self) -> None:
        rows = [
            VolatilityForecast("AAPL", "2026-06-26", 21, "baseline_blend", 0.20),
            VolatilityForecast("AAPL", "2026-06-26", 21, "ensemble", 0.24),
            VolatilityForecast("AAPL", "2026-06-27", 21, "ensemble", 0.28),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "vol_forecasts.sqlite3"
            written = write_volatility_forecasts(rows, path=db_path)
            latest = load_latest_volatility_forecasts("aapl", path=db_path)

        self.assertEqual(written, 3)
        self.assertEqual(set(latest["as_of"]), {"2026-06-27"})
        self.assertEqual(float(latest.iloc[0]["forecast_vol"]), 0.28)


if __name__ == "__main__":
    unittest.main()
