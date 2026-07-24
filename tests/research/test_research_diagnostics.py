from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oqp.research import compute_ic_decay as public_compute_ic_decay
from oqp.research.diagnostics import (
    compute_ic_decay,
    compute_shap_regime_dna,
    list_feature_columns,
)


class ResearchDiagnosticsTests(unittest.TestCase):
    def test_ic_decay_uses_next_open_to_future_close_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "matrix.parquet"
            self._matrix().to_parquet(matrix_path)

            features = list_feature_columns(matrix_path)
            decay = compute_ic_decay(
                feature="f_signal",
                matrix_path=matrix_path,
                horizons=(1, 2),
                min_assets_per_day=3,
            )

        self.assertIs(public_compute_ic_decay, compute_ic_decay)
        self.assertEqual(features, ["f_signal"])
        self.assertEqual(list(decay["horizon"]), [1, 2])
        self.assertGreater(float(decay.loc[decay["horizon"].eq(1), "ic"].iloc[0]), 0.9)
        self.assertGreater(int(decay["valid_days"].min()), 0)

    def test_shap_regime_helper_is_importable_without_running_optional_dependencies(self) -> None:
        self.assertEqual(compute_shap_regime_dna.__name__, "compute_shap_regime_dna")

    @staticmethod
    def _matrix() -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        rows = []
        for asset_idx in range(5):
            price = 100.0 + asset_idx
            for date in dates:
                daily_return = 0.001 * (asset_idx + 1)
                price *= 1.0 + daily_return
                rows.append(
                    {
                        "date": date,
                        "ticker": f"a{asset_idx}",
                        "open": price,
                        "close": price * (1.0 + daily_return),
                        "f_signal": float(asset_idx),
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
