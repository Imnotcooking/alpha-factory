from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_ROOT = REPO_ROOT / "alpha_research_lab"
if str(ALPHA_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHA_ROOT))

from public_examples.retired_factors.fac_retired_public_momentum_demo import (  # noqa: E402
    FACTOR_CONTRACT,
    FACTOR_ID,
    FACTOR_METADATA,
    compute_factor,
)


class AlphaPublicExampleTests(unittest.TestCase):
    def test_retired_public_momentum_demo_uses_synthetic_contract_shape(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=8).tolist() * 2,
                "ticker": ["AAA"] * 8 + ["BBB"] * 8,
                "close": [100, 101, 102, 103, 104, 105, 106, 107]
                + [50, 50, 49, 49, 48, 48, 47, 47],
            }
        )

        result = compute_factor(data, lookback=3)

        self.assertEqual(FACTOR_ID, "fac_retired_public_momentum_demo")
        self.assertEqual(FACTOR_METADATA["status"], "retired_public_example")
        self.assertEqual(FACTOR_CONTRACT["evaluation_geometry"], "cross_sectional")
        self.assertEqual(list(result.columns), ["date", "ticker", "factor_score"])
        self.assertEqual(len(result), 16)
        self.assertFalse(result["factor_score"].isna().any())
        self.assertEqual(result.attrs["factor_id"], FACTOR_ID)

    def test_retired_public_momentum_demo_rejects_missing_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            compute_factor(pd.DataFrame({"date": ["2026-01-01"], "close": [100]}))


if __name__ == "__main__":
    unittest.main()
