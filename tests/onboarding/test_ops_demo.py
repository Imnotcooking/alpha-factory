from pathlib import Path

from oqp.config import load_settings
from oqp.ops.status import collect_ops_status


def test_demo_status_does_not_contact_external_services(tmp_path: Path) -> None:
    snapshot = collect_ops_status(
        settings=load_settings(tmp_path / ".env"),
        account_ledger_path=tmp_path / "accounts.db",
        repo_root=tmp_path,
        demo_mode=True,
    )

    names = {item.name for item in snapshot.items}
    assert "Broker-free demo profile" in names
    assert "Live IBKR Gateway" not in names
    assert "IBKR Adapter Heartbeat" not in names
