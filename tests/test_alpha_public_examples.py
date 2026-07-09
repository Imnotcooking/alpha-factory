from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EXAMPLE_ROOT = REPO_ROOT / "departments" / "research" / "retired_factors"
PRIVATE_TEMPLATE_PATH = (
    REPO_ROOT / "departments" / "research" / "factors" / "factor_template_private.py"
)
PUBLIC_TEMPLATE_PATH = PUBLIC_EXAMPLE_ROOT / "factor_template_retired_public.py"

_EXAMPLE_PATH = PUBLIC_EXAMPLE_ROOT / "fac_retired_public_momentum_demo.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_module(_EXAMPLE_PATH, "_public_momentum_demo")

FACTOR_CONTRACT = _MODULE.FACTOR_CONTRACT
FACTOR_ID = _MODULE.FACTOR_ID
FACTOR_METADATA = _MODULE.FACTOR_METADATA
compute_factor = _MODULE.compute_factor


def _synthetic_factor_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8).tolist() * 2,
            "ticker": ["AAA"] * 8 + ["BBB"] * 8,
            "close": [100, 101, 102, 103, 104, 105, 106, 107]
            + [50, 50, 49, 49, 48, 48, 47, 47],
        }
    )


class AlphaPublicExampleTests(unittest.TestCase):
    def test_retired_public_momentum_demo_uses_synthetic_contract_shape(self) -> None:
        data = _synthetic_factor_input()

        result = compute_factor(data, lookback=3)

        self.assertEqual(FACTOR_ID, "fac_retired_public_momentum_demo")
        self.assertEqual(FACTOR_METADATA["status"], "retired_public_example")
        self.assertEqual(FACTOR_METADATA["supported_markets"], ["FUTURES_CN"])
        self.assertEqual(FACTOR_CONTRACT["evaluation_geometry"], "cross_sectional")
        self.assertEqual(FACTOR_CONTRACT["supported_markets"], ["FUTURES_CN"])
        self.assertEqual(list(result.columns), ["date", "ticker", "factor_score"])
        self.assertEqual(len(result), 16)
        self.assertFalse(result["factor_score"].isna().any())
        self.assertEqual(result.attrs["factor_id"], FACTOR_ID)

    def test_retired_public_momentum_demo_rejects_missing_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            compute_factor(pd.DataFrame({"date": ["2026-01-01"], "close": [100]}))

    def test_factor_templates_are_importable_and_public_safe(self) -> None:
        for module_name, path, expected_status in [
            ("_private_factor_template", PRIVATE_TEMPLATE_PATH, "private_template"),
            (
                "_retired_public_factor_template",
                PUBLIC_TEMPLATE_PATH,
                "retired_public_template",
            ),
        ]:
            module = _load_module(path, module_name)
            result = module.compute_factor(_synthetic_factor_input(), lookback=3)

            self.assertEqual(module.FACTOR_METADATA["status"], expected_status)
            self.assertEqual(module.FACTOR_CONTRACT["alpha_signal_col"], "factor_score")
            if expected_status == "private_template":
                self.assertTrue(callable(module.compute))
            self.assertEqual(list(result.columns), ["date", "ticker", "factor_score"])
            self.assertFalse(result["factor_score"].isna().any())
            self.assertEqual(result.attrs["factor_id"], module.FACTOR_ID)


if __name__ == "__main__":
    unittest.main()
