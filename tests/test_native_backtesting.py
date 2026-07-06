from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from oqp.native import load_quant_core, quant_core_status
from oqp.research.backtesting import ExecutionBacktestRequest, PythonBacktestBackend


class NativeLoaderTests(unittest.TestCase):
    def test_quant_core_status_is_structured_without_raising(self) -> None:
        status = quant_core_status(("feature_that_should_not_exist",), allow_legacy=False)

        self.assertEqual(status.module_name, "oqp.native._quant_core")
        self.assertIsInstance(status.available, bool)

    def test_loader_can_use_explicit_legacy_path(self) -> None:
        old_module = sys.modules.pop("quant_core", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "quant_core.py"
                path.write_text("LEGACY_TEST_FEATURE = True\n", encoding="utf-8")

                module = load_quant_core(
                    ("LEGACY_TEST_FEATURE",),
                    allow_legacy=True,
                    legacy_paths=(tmp,),
                )

                self.assertTrue(module.LEGACY_TEST_FEATURE)
        finally:
            sys.modules.pop("quant_core", None)
            if old_module is not None:
                sys.modules["quant_core"] = old_module


class PythonBacktestBackendTests(unittest.TestCase):
    def test_python_backtest_backend_returns_equity_curve(self) -> None:
        request = ExecutionBacktestRequest(
            asset_ids=[0, 0, 1, 1],
            prices=[100.0, 101.0, 50.0, 49.5],
            target_weights=[0.2, 0.2, -0.1, -0.1],
            initial_capital=100_000.0,
            deadband=0.0,
        )

        result = PythonBacktestBackend().run(request)

        self.assertEqual(result.backend.backend_id, "python")
        self.assertEqual(len(result.equity_curve), request.n_rows)
        self.assertTrue(np.isfinite(result.final_equity))


if __name__ == "__main__":
    unittest.main()
