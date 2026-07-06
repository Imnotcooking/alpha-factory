from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oqp.accounts import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
    load_manual_external_positions_file,
    load_manual_external_positions,
    load_manual_external_positions_as_account_positions,
    load_latest_account_nav,
    load_latest_account_positions,
    materialize_unified_live_account_snapshot,
    sync_manual_external_positions_from_json,
    upsert_manual_external_positions,
    write_account_snapshot,
)
from oqp.domain.models import utc_now


class ManualExternalPositionsTests(unittest.TestCase):
    def test_upserts_and_loads_account_position_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "account_ledger.db"
            written = upsert_manual_external_positions(
                db_path,
                [
                    {
                        "position_id": "external-test-call",
                        "symbol": "TEST270115C00100000",
                        "asset_class": "option",
                        "quantity": 2,
                        "average_cost": 5.0,
                        "currency": "USD",
                        "multiplier": 100,
                        "underlying": "TEST",
                        "expiry": "2027-01-15",
                        "option_type": "call",
                        "strike": 100,
                    },
                    {
                        "position_id": "external-eur-equity",
                        "symbol": "EURTEST",
                        "asset_class": "equity",
                        "quantity": 10,
                        "average_cost": 20.0,
                        "current_price": 21.0,
                        "currency": "EUR",
                        "fx_rate_to_base": 1.2,
                    },
                ],
            )

            manual = load_manual_external_positions(db_path)
            account_positions = load_manual_external_positions_as_account_positions(db_path)

            self.assertEqual(written, 2)
            self.assertEqual(len(manual), 2)
            self.assertEqual(len(account_positions), 2)
            by_symbol = {row["symbol"]: row for row in account_positions.to_dict("records")}
            self.assertEqual(by_symbol["TEST270115C00100000"]["market_value"], 1000.0)
            self.assertEqual(by_symbol["TEST270115C00100000"]["currency"], "USD")
            self.assertEqual(by_symbol["EURTEST"]["market_value"], 252.0)
            self.assertEqual(by_symbol["EURTEST"]["currency"], "USD")

    def test_syncs_json_as_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "account_ledger.db"
            json_path = root / "manual_external_holdings.json"
            json_path.write_text(
                json.dumps(
                    {
                        "positions": [
                            {
                                "position_id": "keep-row",
                                "symbol": "KEEP",
                                "asset_class": "equity",
                                "quantity": 1,
                                "average_cost": 10,
                            },
                            {
                                "position_id": "remove-row",
                                "symbol": "DROP",
                                "asset_class": "equity",
                                "quantity": 1,
                                "average_cost": 20,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            first_sync = sync_manual_external_positions_from_json(db_path, json_path=json_path)
            first_rows = load_manual_external_positions(db_path)
            self.assertEqual(first_sync, 2)
            self.assertEqual(len(first_rows), 2)
            self.assertEqual(len(load_manual_external_positions_file(json_path)), 2)

            json_path.write_text(
                json.dumps(
                    {
                        "positions": [
                            {
                                "position_id": "keep-row",
                                "symbol": "KEEP",
                                "asset_class": "equity",
                                "quantity": 2,
                                "average_cost": 10,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            second_sync = sync_manual_external_positions_from_json(db_path, json_path=json_path)
            active_rows = load_manual_external_positions(db_path)
            all_rows = load_manual_external_positions(db_path, active_only=False)

            self.assertEqual(second_sync, 1)
            self.assertEqual(active_rows["position_id"].tolist(), ["keep-row"])
            self.assertEqual(int(all_rows.loc[all_rows["position_id"].eq("remove-row"), "active"].iloc[0]), 0)
            self.assertEqual(float(active_rows.iloc[0]["quantity"]), 2.0)

    def test_materializes_unified_live_snapshot_from_broker_and_manual_usd_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "account_ledger.db"
            now = utc_now()
            write_account_snapshot(
                db_path,
                AccountSnapshot(
                    snapshot_id="ibkr-live-test",
                    as_of=now,
                    account_id="U_TEST",
                    broker="ibkr",
                    profile="ibkr_live_readonly",
                    environment=AccountEnvironment.LIVE,
                    net_liquidation=1_000.0,
                    cash=100.0,
                    positions=(
                        PositionSnapshot(
                            symbol="AAPL",
                            asset_class="equity",
                            quantity=1,
                            market_price=900.0,
                            market_value=900.0,
                            currency="USD",
                        ),
                    ),
                    cash_balances=(CashSnapshot(currency="USD", cash=100.0),),
                ),
            )
            upsert_manual_external_positions(
                db_path,
                [
                    {
                        "position_id": "external-usd-cash",
                        "symbol": "USD Cash",
                        "asset_class": "cash",
                        "quantity": 50,
                        "average_cost": 1,
                        "currency": "USD",
                        "pricing_method": "manual_cash",
                    },
                    {
                        "position_id": "external-usd-option",
                        "symbol": "TEST270115C00100000",
                        "asset_class": "option",
                        "quantity": 1,
                        "average_cost": 2,
                        "current_price": 2,
                        "currency": "USD",
                        "multiplier": 100,
                    },
                    {
                        "position_id": "external-eur-no-fx",
                        "symbol": "EURTEST",
                        "asset_class": "equity",
                        "quantity": 1,
                        "average_cost": 10,
                        "currency": "EUR",
                    },
                ],
            )

            result = materialize_unified_live_account_snapshot(db_path)
            nav = load_latest_account_nav(db_path, environment="live", profile="unified_live")
            positions = load_latest_account_positions(db_path, environment="live", profile="unified_live")

            self.assertEqual(result.excluded_manual_rows, 1)
            self.assertEqual(result.manual_usd_value, 250.0)
            self.assertEqual(float(nav.iloc[0]["net_liquidation"]), 1_250.0)
            self.assertEqual(float(nav.iloc[0]["cash"]), 150.0)
            self.assertEqual(len(positions), 3)

    def test_materializes_unified_live_snapshot_in_broker_base_currency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "account_ledger.db"
            now = utc_now()
            write_account_snapshot(
                db_path,
                AccountSnapshot(
                    snapshot_id="ibkr-live-eur-test",
                    as_of=now,
                    account_id="U_EUR",
                    broker="ibkr",
                    profile="ibkr_live_readonly",
                    environment=AccountEnvironment.LIVE,
                    currency="EUR",
                    net_liquidation=1_000.0,
                    cash=100.0,
                    positions=(
                        PositionSnapshot(
                            symbol="AAPL",
                            asset_class="equity",
                            quantity=1,
                            market_price=900.0,
                            market_value=900.0,
                            currency="EUR",
                        ),
                    ),
                    cash_balances=(CashSnapshot(currency="EUR", cash=100.0),),
                ),
            )
            upsert_manual_external_positions(
                db_path,
                [
                    {
                        "position_id": "external-eur-equity",
                        "symbol": "EQAC",
                        "asset_class": "equity",
                        "quantity": 1,
                        "average_cost": 100,
                        "current_price": 120,
                        "currency": "EUR",
                        "base_currency": "USD",
                        "fx_rate_to_base": 1.2,
                    },
                    {
                        "position_id": "external-usd-cash",
                        "symbol": "USD Cash",
                        "asset_class": "cash",
                        "quantity": 120,
                        "average_cost": 1,
                        "currency": "USD",
                        "pricing_method": "manual_cash",
                    },
                ],
            )

            result = materialize_unified_live_account_snapshot(db_path)
            nav = load_latest_account_nav(db_path, environment="live", profile="unified_live")
            positions = load_latest_account_positions(db_path, environment="live", profile="unified_live")

            self.assertEqual(result.currency, "EUR")
            self.assertEqual(result.excluded_manual_rows, 0)
            self.assertAlmostEqual(result.manual_usd_value, 220.0, places=6)
            self.assertEqual(nav.iloc[0]["currency"], "EUR")
            self.assertAlmostEqual(float(nav.iloc[0]["net_liquidation"]), 1_220.0, places=6)
            self.assertAlmostEqual(float(nav.iloc[0]["cash"]), 200.0, places=6)
            self.assertEqual(set(positions["currency"]), {"EUR"})


if __name__ == "__main__":
    unittest.main()
