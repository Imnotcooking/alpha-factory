from __future__ import annotations

import unittest

import pandas as pd

from oqp.data.futures_cn import normalize_futures_cn_daily_frame


class FuturesCNDataTests(unittest.TestCase):
    def test_normalizes_vendor_daily_aliases(self) -> None:
        raw = pd.DataFrame(
            {
                "symbol": ["AP810", "AP811"],
                "datetime": ["2018-07-06", "2018-07-06"],
                "open": [9400.0, 9644.0],
                "high": [9490.0, 9695.0],
                "low": [9230.0, 9468.0],
                "close": [9373.0, 9607.0],
                "volume": [199637, 352],
                "open_interest": [84616.0, 3306.0],
            }
        )

        out = normalize_futures_cn_daily_frame(raw)

        self.assertEqual(out["ticker"].tolist(), ["AP810", "AP811"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["date"]))
        self.assertIn("oi", out.columns)
        self.assertEqual(out["oi"].tolist(), [84616.0, 3306.0])


if __name__ == "__main__":
    unittest.main()
