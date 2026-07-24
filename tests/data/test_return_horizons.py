from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oqp.research.backtesting.return_horizons import (
    RETURN_CLOSE_TO_NEXT_CLOSE,
    RETURN_CLOSE_TO_NEXT_OPEN,
    RETURN_NEXT_OPEN_TO_NEXT_CLOSE,
    RETURN_NEXT_OPEN_TO_NEXT_OPEN,
    attach_return_horizon,
    normalize_return_horizon,
)
from oqp.research.contracts import resolve_factor_contract


class ReturnHorizonTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "ticker": ["A", "A", "A"],
                "open": [100.0, 110.0, 121.0],
                "close": [105.0, 120.0, 118.0],
            }
        )

    def test_next_open_to_next_close_horizon(self) -> None:
        out = attach_return_horizon(
            self._frame(),
            return_horizon="next_open_to_next_close",
            data_frequency="daily",
        )

        np.testing.assert_allclose(
            out["forward_return"].to_numpy()[:2],
            [120.0 / 110.0 - 1.0, 118.0 / 121.0 - 1.0],
        )
        np.testing.assert_allclose(out["execution_price"].to_numpy()[:2], [110.0, 121.0])
        self.assertEqual(out.attrs["return_horizon"], RETURN_NEXT_OPEN_TO_NEXT_CLOSE)
        self.assertEqual(out.attrs["benchmark_return_col"], "execution_period_return")

    def test_next_open_to_next_open_horizon(self) -> None:
        out = attach_return_horizon(
            self._frame(),
            return_horizon="next_open_to_next_open",
            data_frequency="daily",
        )

        self.assertAlmostEqual(float(out["forward_return"].iloc[0]), 121.0 / 110.0 - 1.0)
        self.assertTrue(pd.isna(out["forward_return"].iloc[1]))
        self.assertEqual(out.attrs["return_horizon"], RETURN_NEXT_OPEN_TO_NEXT_OPEN)

    def test_close_to_next_close_and_overnight_horizons(self) -> None:
        close_to_close = attach_return_horizon(
            self._frame(),
            return_horizon="close_to_next_close",
            data_frequency="daily",
        )
        overnight = attach_return_horizon(
            self._frame(),
            return_horizon="close_to_next_open",
            data_frequency="daily",
        )

        self.assertAlmostEqual(float(close_to_close["forward_return"].iloc[0]), 120.0 / 105.0 - 1.0)
        self.assertAlmostEqual(float(overnight["forward_return"].iloc[0]), 110.0 / 105.0 - 1.0)
        self.assertEqual(close_to_close.attrs["return_horizon"], RETURN_CLOSE_TO_NEXT_CLOSE)
        self.assertEqual(overnight.attrs["return_horizon"], RETURN_CLOSE_TO_NEXT_OPEN)

    def test_non_positive_open_prices_do_not_create_fake_returns(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
                ),
                "ticker": ["A", "A", "A", "A"],
                "open": [100.0, 110.0, 0.0, 130.0],
                "close": [105.0, 120.0, 125.0, 140.0],
            }
        )

        open_to_open = attach_return_horizon(
            frame,
            return_horizon="next_open_to_next_open",
            data_frequency="daily",
        )
        open_to_close = attach_return_horizon(
            frame,
            return_horizon="next_open_to_next_close",
            data_frequency="daily",
        )
        close_to_open = attach_return_horizon(
            frame,
            return_horizon="close_to_next_open",
            data_frequency="daily",
        )

        self.assertTrue(pd.isna(open_to_open["forward_return"].iloc[0]))
        self.assertTrue(pd.isna(open_to_open["forward_return"].iloc[1]))
        self.assertTrue(pd.isna(open_to_close["forward_return"].iloc[1]))
        self.assertTrue(pd.isna(close_to_open["forward_return"].iloc[1]))

    def test_contract_can_override_return_assumption_from_cli_horizon(self) -> None:
        class FactorModule:
            FACTOR_CONTRACT = {
                "evaluation_geometry": "time_series",
                "execution_mode": "direct",
                "alpha_signal_col": "factor_score",
                "execution_weight_col": "signal",
                "execution_lag": "next_open",
                "return_assumption": "close_signal_next_open_to_close",
                "supported_markets": ["EQUITY_CN"],
            }

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),
                "ticker": ["A"],
                "factor_score": [1.0],
                "signal": [1.0],
            }
        )

        contract = resolve_factor_contract(
            FactorModule,
            df,
            factor_id="test",
            requested_return_assumption="next_open_to_next_open",
            market_vertical="EQUITY_CN",
            strict=True,
        )

        self.assertEqual(contract.return_assumption, RETURN_NEXT_OPEN_TO_NEXT_OPEN)
        self.assertIn("return_assumption overridden", " ".join(contract.warnings))

    def test_normalize_return_horizon_rejects_unknown_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid return_horizon"):
            normalize_return_horizon("moon_to_mars")


if __name__ == "__main__":
    unittest.main()
