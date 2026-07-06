from __future__ import annotations

import unittest
from datetime import date

from oqp.data import OptionChainRequest, OptionContract, OptionQuote, OptionRight, Quote
from oqp.domain import AssetClass, Instrument
from oqp.options import fetch_option_mark, fetch_option_spread_mark


class FakeMassiveOptionsAdapter:
    def __init__(self, quotes: list[OptionQuote]) -> None:
        self.quotes = quotes

    def get_option_chain(self, request: OptionChainRequest) -> list[OptionQuote]:
        return [
            quote
            for quote in self.quotes
            if quote.contract.expiration == request.expiration
            and quote.contract.strike >= (request.min_strike or 0)
            and quote.contract.strike <= (request.max_strike or float("inf"))
        ]


class FakeYahoo:
    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def option_chain(self, expiry: str):  # pragma: no cover - fails if Massive primary is bypassed
            raise AssertionError("Yahoo should not be called when Massive has a usable mark.")


def quote(strike: float, mark: float, *, delta: float = 0.5) -> OptionQuote:
    underlying = Instrument(symbol="AVGO", asset_class=AssetClass.EQUITY)
    contract = OptionContract(
        underlying=underlying,
        expiration=date(2027, 1, 15),
        strike=strike,
        right=OptionRight.CALL,
        symbol=f"O:AVGO270115C{int(strike * 1000):08d}",
    )
    return OptionQuote(
        contract=contract,
        quote=Quote(instrument=underlying, mark=mark, source="massive_options"),
        implied_volatility=0.45,
        delta=delta,
        gamma=0.01,
        theta=-0.02,
        vega=0.12,
    )


class OptionPricingTests(unittest.TestCase):
    def test_fetch_option_mark_prefers_massive_and_stores_greeks(self) -> None:
        metadata: dict[str, object] = {}
        adapter = FakeMassiveOptionsAdapter([quote(410, 82.5, delta=0.72)])

        mark = fetch_option_mark(
            FakeYahoo,
            "AVGO",
            "2027-01-15",
            "call",
            410,
            options_adapter=adapter,  # type: ignore[arg-type]
            row_metadata=metadata,
        )

        self.assertEqual(mark, 82.5)
        self.assertEqual(metadata["pricing_method"], "massive")
        self.assertEqual(metadata["delta"], 0.72)
        self.assertEqual(metadata["implied_volatility"], 0.45)

    def test_fetch_option_spread_mark_nets_signed_legs(self) -> None:
        adapter = FakeMassiveOptionsAdapter([quote(410, 82.5), quote(430, 72.0, delta=0.40)])
        row = {
            "quote_symbol": "AVGO",
            "underlying": "AVGO",
            "expiry": "2027-01-15",
            "option_type": "call",
            "metadata": {
                "legs": [
                    {"side": "buy", "option_type": "call", "strike": 410, "quantity": 1, "average_cost": 78.37},
                    {"side": "sell", "option_type": "call", "strike": 430, "quantity": -1, "average_cost": 69.42},
                ]
            },
        }

        mark = fetch_option_spread_mark(FakeYahoo, row, adapter)  # type: ignore[arg-type]

        self.assertEqual(mark, 10.5)
        self.assertEqual(row["pricing_method"], "massive")
        self.assertEqual(row["metadata"]["legs"][0]["pricing_method"], "massive")
        self.assertEqual(row["metadata"]["legs"][1]["current_price"], 72.0)


if __name__ == "__main__":
    unittest.main()
