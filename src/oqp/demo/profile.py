"""Runtime profile paths shared by onboarding commands and dashboards."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oqp.config.paths import resolve_repo_root


PROFILE_SCHEMA_VERSION = 1
DEMO_PROFILE = "demo"


@dataclass(frozen=True, slots=True)
class DemoPaths:
    repo_root: Path
    runtime_root: Path
    data_root: Path
    artifact_root: Path
    research_db: Path
    account_ledger: Path
    portfolio_ledger: Path
    paper_ledger: Path
    profile_marker: Path
    seed_manifest: Path


def profile_marker_path(repo_root: str | Path | None = None) -> Path:
    root = resolve_repo_root(configured_root=repo_root)
    return root / "runtime" / "state" / "platform" / "profile.json"


def demo_paths(repo_root: str | Path | None = None) -> DemoPaths:
    root = resolve_repo_root(configured_root=repo_root)
    runtime_root = root / "runtime" / "demo"
    return DemoPaths(
        repo_root=root,
        runtime_root=runtime_root,
        data_root=runtime_root / "data",
        artifact_root=runtime_root / "artifacts" / "research",
        research_db=runtime_root / "db" / "research" / "research_memory.db",
        account_ledger=runtime_root / "db" / "accounts" / "account_ledger.db",
        portfolio_ledger=runtime_root / "db" / "portfolio" / "portfolio_ledger.db",
        paper_ledger=runtime_root / "db" / "paper_trading" / "paper_trading.db",
        profile_marker=profile_marker_path(root),
        seed_manifest=runtime_root / "seed_manifest.json",
    )


def read_profile_marker(repo_root: str | Path | None = None) -> dict[str, Any]:
    path = profile_marker_path(repo_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def demo_environment(
    repo_root: str | Path | None = None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that isolates dashboards inside ``runtime/demo``."""

    paths = demo_paths(repo_root)
    env = dict(base or os.environ)
    src = paths.repo_root / "src"
    python_path = [str(src), str(paths.repo_root)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "OQP_PROFILE": DEMO_PROFILE,
            "OQP_REPO_ROOT": str(paths.repo_root),
            "OQP_RUNTIME_ROOT": str(paths.runtime_root),
            "DATA_ROOT": str(paths.data_root),
            "ALPHA_RUNTIME_DATA_ROOT": str(paths.data_root),
            "FUTURES_CN_DAILY_DATA_ROOT": str(paths.data_root / "futures_cn" / "daily"),
            "FUTURES_CN_INTRADAY_DATA_ROOT": str(paths.data_root / "futures_cn" / "intraday"),
            "FUTURES_CN_TICK_DATA_ROOT": str(paths.data_root / "futures_cn" / "tick"),
            "ALPHA_RESEARCH_DB_PATH": str(paths.research_db),
            "ALPHA_RUNTIME_ARTIFACT_ROOT": str(paths.artifact_root),
            "ALPHA_RESEARCH_ARTIFACT_ROOT": str(paths.artifact_root),
            "OQP_ACCOUNT_LEDGER_PATH": str(paths.account_ledger),
            "OQP_PORTFOLIO_LEDGER_PATH": str(paths.portfolio_ledger),
            "PAPER_TRADING_DB_PATH": str(paths.paper_ledger),
            "MPLCONFIGDIR": str(paths.runtime_root / "cache" / "matplotlib"),
            "PYTHONPATH": os.pathsep.join(python_path),
        }
    )
    return env
