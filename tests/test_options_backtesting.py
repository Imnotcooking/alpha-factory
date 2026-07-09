from __future__ import annotations

import unittest

import pandas as pd

from oqp.options import (
    OptionBacktestEngine,
    OptionBacktestRequest,
    OptionLiquidityRule,
    intrinsic_value,
    normalize_option_chain_frame,
    option_result_to_execution_result,
)


def _chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "option_symbol": "AAPL260116C00100000",
                "underlying_symbol": "AAPL",
                "expiry": "2026-01-16",
                "right": "C",
                "strike": 100,
                "bid": 4.8,
                "ask": 5.2,
                "close": 5.0,
                "volume": 100,
                "open_interest": 500,
                "implied_volatility": 0.30,
                "delta": 0.55,
            },
            {
                "date": "2026-01-03",
                "option_symbol": "AAPL260116C00100000",
                "underlying_symbol": "AAPL",
                "expiry": "2026-01-16",
                "right": "call",
                "strike": 100,
                "bid": 5.8,
                "ask": 6.2,
                "close": 6.0,
                "volume": 80,
                "open_interest": 520,
                "implied_volatility": 0.31,
                "delta": 0.58,
            },
        ]
    )


class OptionsBacktestingTests(unittest.TestCase):
    def test_normalize_option_chain_frame_handles_massive_style_fields(self) -> None:
        raw = pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "contract_symbol": ["O:AAPL260116C00100000"],
                "underlying_ticker": ["aapl"],
                "expiration_date": ["2026-01-16"],
                "contract_type": ["call"],
                "strike_price": [100],
                "bid_price": [4.8],
                "ask_price": [5.2],
                "vol": [100],
                "oi": [500],
                "iv": [0.3],
            }
        )

        normalized = normalize_option_chain_frame(raw, market_vertical="OPTIONS_US", source="massive")

        self.assertEqual(normalized.loc[0, "underlying_symbol"], "AAPL")
        self.assertEqual(normalized.loc[0, "right"], "call")
        self.assertAlmostEqual(float(normalized.loc[0, "mid"]), 5.0)
        self.assertEqual(normalized.loc[0, "quote_source"], "massive")

    def test_normalize_option_chain_frame_handles_cn_static_parquet_shape(self) -> None:
        raw = pd.DataFrame(
            {
                "日期": ["2026-01-02"],
                "合约": ["au2604C600"],
                "标的代码": ["au2604"],
                "到期日": ["2026-04-25"],
                "类型": ["认购"],
                "行权价": [600],
                "结算价": [12.4],
                "成交量": [200],
                "持仓量": [1200],
            }
        )

        normalized = normalize_option_chain_frame(
            raw,
            market_vertical="OPTIONS_CN",
            source="cn_static_parquet",
            default_multiplier=1000,
        )

        self.assertEqual(normalized.loc[0, "market_vertical"], "OPTIONS_CN")
        self.assertEqual(normalized.loc[0, "underlying_symbol"], "AU2604")
        self.assertEqual(normalized.loc[0, "right"], "call")
        self.assertAlmostEqual(float(normalized.loc[0, "mark"]), 12.4)
        self.assertEqual(float(normalized.loc[0, "multiplier"]), 1000.0)

    def test_event_driven_engine_enters_and_exits_long_call(self) -> None:
        underlying = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-03"],
                "underlying_symbol": ["AAPL", "AAPL"],
                "close": [100.0, 103.0],
            }
        )
        signals = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-03"],
                "underlying_symbol": ["AAPL", "AAPL"],
                "direction": [1, 0],
            }
        )
        request = OptionBacktestRequest(
            chain=_chain(),
            underlying=underlying,
            signals=signals,
            initial_capital=10_000.0,
            liquidity=OptionLiquidityRule(max_spread_pct=0.2),
        )

        result = OptionBacktestEngine().run(request)

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades["reason"].tolist(), ["signal_entry", "signal_exit"])
        self.assertGreater(result.final_equity, 10_000.0)
        execution_result = option_result_to_execution_result(result)
        self.assertEqual(execution_result.backend.backend_id, "options_event_driven")

    def test_expiry_settlement_uses_intrinsic_value(self) -> None:
        chain = pd.DataFrame(
            [
                {
                    "date": "2026-01-15",
                    "option_symbol": "AAPL260116P00100000",
                    "underlying_symbol": "AAPL",
                    "expiry": "2026-01-16",
                    "right": "put",
                    "strike": 100,
                    "mark": 2.0,
                    "volume": 10,
                    "open_interest": 20,
                },
                {
                    "date": "2026-01-16",
                    "option_symbol": "AAPL260116P00100000",
                    "underlying_symbol": "AAPL",
                    "expiry": "2026-01-16",
                    "right": "put",
                    "strike": 100,
                    "mark": 7.0,
                    "volume": 10,
                    "open_interest": 20,
                },
            ]
        )
        underlying = pd.DataFrame(
            {
                "date": ["2026-01-15", "2026-01-16"],
                "underlying_symbol": ["AAPL", "AAPL"],
                "close": [98.0, 93.0],
            }
        )
        signals = pd.DataFrame(
            {
                "date": ["2026-01-15"],
                "underlying_symbol": ["AAPL"],
                "option_symbol": ["AAPL260116P00100000"],
                "direction": [-1],
            }
        )

        result = OptionBacktestEngine().run(
            OptionBacktestRequest(
                chain=chain,
                underlying=underlying,
                signals=signals,
                initial_capital=10_000.0,
            )
        )

        self.assertEqual(result.trades["reason"].tolist(), ["signal_entry", "expiry"])
        self.assertEqual(intrinsic_value(93.0, 100.0, "put"), 7.0)
        self.assertGreater(result.final_equity, 10_000.0)


if __name__ == "__main__":
    unittest.main()
