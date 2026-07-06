"""Small account selection helpers shared by command entrypoints."""

from __future__ import annotations

from pathlib import Path

from oqp.accounts.ledger import load_latest_account_nav


def latest_account_id(
    account_ledger_path: str | Path,
    *,
    environment: str,
    profile: str,
) -> str | None:
    nav = load_latest_account_nav(
        account_ledger_path,
        environment=environment,
        profile=profile,
    )
    if nav.empty:
        return None
    value = nav.iloc[0].get("account_id")
    text = str(value).strip() if value is not None else ""
    return text or None
