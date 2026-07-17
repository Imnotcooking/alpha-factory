"""Environment diagnostics for the public onboarding path."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from oqp.config.paths import resolve_repo_root
from oqp.demo.profile import DEMO_PROFILE, demo_paths, read_profile_marker


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    required: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_doctor(repo_root: str | Path | None = None) -> tuple[DoctorCheck, ...]:
    root = resolve_repo_root(configured_root=repo_root)
    paths = demo_paths(root)
    marker = read_profile_marker(root)
    checks: list[DoctorCheck] = []

    checks.append(
        _check(
            "Python",
            sys.version_info >= (3, 11),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    checks.append(_check("Repository", (root / "pyproject.toml").is_file(), str(root)))
    checks.append(_module_check("Core package", "oqp"))
    checks.append(_module_check("Streamlit", "streamlit"))
    checks.append(_module_check("Plotly", "plotly"))
    checks.append(_module_check("Parquet engine", "pyarrow"))

    profile = str(marker.get("profile") or "uninitialized")
    checks.append(
        DoctorCheck(
            name="Runtime profile",
            status="pass" if profile == DEMO_PROFILE else "warn",
            detail=(
                "demo profile initialized"
                if profile == DEMO_PROFILE
                else "run `oqp init --profile demo` for the broker-free tour"
            ),
            required=False,
        )
    )
    if profile == DEMO_PROFILE:
        fixtures = (
            paths.research_db,
            paths.account_ledger,
            paths.data_root / "futures_cn" / "daily" / "demo_futures_cn_daily.parquet",
            paths.data_root / "options_us" / "api_cache" / "demo_options_us_chain.parquet",
            paths.seed_manifest,
        )
        missing = [str(path.relative_to(root)) for path in fixtures if not path.exists()]
        checks.append(
            _check(
                "Demo fixtures",
                not missing,
                "complete" if not missing else f"missing: {', '.join(missing)}",
            )
        )

    checks.append(
        DoctorCheck(
            name="Native C++ extension",
            status="pass" if importlib.util.find_spec("oqp.native._quant_core") else "warn",
            detail=(
                "available"
                if importlib.util.find_spec("oqp.native._quant_core")
                else "not built; Python fallback remains available"
            ),
            required=False,
        )
    )
    vendor_names = ("FMP_API_KEY", "MASSIVE_API_KEY", "POLYGON_API_KEY")
    configured_vendors = [name for name in vendor_names if os.getenv(name)]
    checks.append(
        DoctorCheck(
            name="Vendor APIs",
            status="pass" if configured_vendors else "warn",
            detail=(
                f"configured: {', '.join(configured_vendors)}"
                if configured_vendors
                else "none configured; demo mode does not require API keys"
            ),
            required=False,
        )
    )
    broker_configured = bool(os.getenv("IBKR_ACCOUNT") or os.getenv("QMT_ACCOUNT_ID"))
    checks.append(
        DoctorCheck(
            name="Broker adapters",
            status="pass" if broker_configured else "warn",
            detail=(
                "at least one broker profile is configured"
                if broker_configured
                else "none configured; demo mode uses a synthetic read-only ledger"
            ),
            required=False,
        )
    )
    live_enabled = str(os.getenv("ALLOW_LIVE_TRADING", "false")).strip().lower() == "true"
    checks.append(
        DoctorCheck(
            name="Live-trading gate",
            status="warn" if live_enabled else "pass",
            detail="enabled in environment" if live_enabled else "disabled",
            required=False,
        )
    )
    return tuple(checks)


def doctor_exit_code(checks: Iterable[DoctorCheck]) -> int:
    return 1 if any(check.required and check.status == "fail" for check in checks) else 0


def _module_check(name: str, module: str) -> DoctorCheck:
    available = importlib.util.find_spec(module) is not None
    return _check(name, available, "available" if available else f"missing module: {module}")


def _check(name: str, condition: bool, detail: str) -> DoctorCheck:
    return DoctorCheck(name=name, status="pass" if condition else "fail", detail=detail)
