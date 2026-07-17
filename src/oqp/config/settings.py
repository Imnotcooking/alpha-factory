"""Central runtime settings for Alpha Factory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from oqp.config.credentials import JsonCredentialSource, load_credential
from oqp.config.paths import REPO_ROOT


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _setting(name: str, env_values: dict[str, str], default: str | None = None) -> str | None:
    return os.getenv(name) or env_values.get(name) or default


def _credential(
    env_names: tuple[str, ...],
    env_values: dict[str, str],
    json_sources: tuple[JsonCredentialSource, ...] = (),
) -> str | None:
    return load_credential(env_names, env_values, json_sources).value


def _setting_int(name: str, env_values: dict[str, str], default: int) -> int:
    raw = _setting(name, env_values, str(default))
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _setting_float(name: str, env_values: dict[str, str]) -> float | None:
    raw = _setting(name, env_values)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _setting_float_default(
    name: str,
    env_values: dict[str, str],
    default: float | None,
) -> float | None:
    parsed = _setting_float(name, env_values)
    return default if parsed is None else parsed


def _setting_csv_tuple(
    name: str,
    env_values: dict[str, str],
    default: str | None = None,
) -> tuple[str, ...]:
    raw = _setting(name, env_values, default)
    if raw in (None, ""):
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _setting_bool(name: str, env_values: dict[str, str], default: bool) -> bool:
    raw = _setting(name, env_values)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _setting_path(name: str, env_values: dict[str, str], default: str) -> Path:
    raw = _setting(name, env_values, default) or default
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True, slots=True)
class OQPSettings:
    fmp_api_key: str | None
    polygon_api_key: str | None
    massive_api_key: str | None
    options_api_key: str | None
    rapid_api_key: str | None
    zai_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None
    anthropic_api_key: str | None
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_evidence_enabled: bool
    llm_timeout_seconds: int
    llm_cache_max_age_hours: float | None
    massive_flat_files_access_key_id: str | None
    massive_flat_files_secret_access_key: str | None
    massive_flat_files_endpoint: str
    massive_flat_files_bucket: str
    ibkr_host: str
    ibkr_paper_port: int
    ibkr_live_port: int
    ibkr_client_id: int
    ibkr_paper_client_id: int
    ibkr_paper_submit_client_id: int
    ibkr_live_client_id: int
    ibkr_live_monitor_enabled: bool
    qmt_connector_enabled: bool
    qmt_connector_url: str
    qmt_submit_connector_url: str
    qmt_api_token: str | None
    qmt_request_signing_secret: str | None
    qmt_audit_log_path: Path
    qmt_require_private_connector: bool
    qmt_account_id: str | None
    qmt_paper_account_id: str | None
    qmt_live_account_id: str | None
    qmt_account_type: str
    qmt_session_id: int
    qmt_timeout_seconds: float
    qmt_live_monitor_enabled: bool
    allow_qmt_paper_order_submit: bool
    allow_qmt_live_trading: bool
    ops_status_source: str
    allow_paper_trading: bool
    allow_paper_order_submit: bool
    paper_max_order_notional: float | None
    paper_max_daily_notional: float | None
    paper_allowed_symbols: tuple[str, ...]
    paper_allowed_asset_classes: tuple[str, ...]
    paper_options_enabled: bool
    paper_option_allowed_underlyings: tuple[str, ...]
    paper_option_allowed_strategies: tuple[str, ...]
    paper_option_max_contracts: float | None
    paper_option_max_premium: float | None
    paper_option_max_defined_risk: float | None
    paper_option_max_spread_width: float | None
    trading_mode: str
    allow_live_trading: bool
    max_daily_loss_pct: float | None
    max_gross_exposure: float | None
    database_url: str
    data_root: Path
    artifact_root: Path
    log_level: str

    @property
    def ibkr_port(self) -> int:
        if self.trading_mode.lower() == "live":
            return self.ibkr_live_port
        return self.ibkr_paper_port

    @property
    def has_massive_api_key(self) -> bool:
        return bool(self.massive_api_key or self.options_api_key)

    @property
    def llm_api_key(self) -> str | None:
        provider = self.llm_provider.strip().lower()
        if provider in {"zai", "z.ai", "glm", "glm-5.2"}:
            return self.zai_api_key
        if provider == "openai":
            return self.openai_api_key
        return self.zai_api_key or self.openai_api_key


def load_settings(env_file: Path | str | None = None) -> OQPSettings:
    env_path = Path(env_file) if env_file is not None else REPO_ROOT / ".env"
    env_values = _read_env_file(env_path)
    openai_api_key = _credential(("OPENAI_API_KEY",), env_values)
    zai_api_key = _credential(("ZAI_API_KEY", "GLM_API_KEY"), env_values)
    llm_provider = _setting(
        "LLM_PROVIDER",
        env_values,
        "zai" if zai_api_key else "openai",
    ) or ("zai" if zai_api_key else "openai")
    llm_provider_key = llm_provider.strip().lower()
    default_llm_base_url = (
        "https://api.openai.com/v1"
        if llm_provider_key == "openai"
        else "https://api.z.ai/api/paas/v4"
    )
    default_llm_model = "gpt-4.1-mini" if llm_provider_key == "openai" else "glm-5.2"

    return OQPSettings(
        fmp_api_key=_credential(("FMP_API_KEY", "FMP_KEY"), env_values),
        polygon_api_key=_credential(
            ("MASSIVE_API_KEY", "OPTIONS_API_KEY", "POLYGON_API_KEY"),
            env_values,
        ),
        massive_api_key=_credential(("MASSIVE_API_KEY",), env_values),
        options_api_key=_credential(("OPTIONS_API_KEY",), env_values),
        rapid_api_key=_credential(("RAPID_API_KEY",), env_values),
        zai_api_key=zai_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=_credential(("GEMINI_API_KEY", "GEMINI_KEY"), env_values),
        anthropic_api_key=_credential(("ANTHROPIC_API_KEY",), env_values),
        llm_provider=llm_provider,
        llm_base_url=_setting("LLM_BASE_URL", env_values, default_llm_base_url)
        or default_llm_base_url,
        llm_model=_setting("LLM_MODEL", env_values, default_llm_model)
        or default_llm_model,
        llm_evidence_enabled=_setting_bool(
            "LLM_EVIDENCE_ENABLED",
            env_values,
            bool(zai_api_key or openai_api_key),
        ),
        llm_timeout_seconds=_setting_int("LLM_TIMEOUT_SECONDS", env_values, 60),
        llm_cache_max_age_hours=_setting_float_default(
            "LLM_CACHE_MAX_AGE_HOURS",
            env_values,
            168.0,
        ),
        massive_flat_files_access_key_id=_setting(
            "MASSIVE_FLAT_FILES_ACCESS_KEY_ID", env_values
        ),
        massive_flat_files_secret_access_key=_setting(
            "MASSIVE_FLAT_FILES_SECRET_ACCESS_KEY", env_values
        ),
        massive_flat_files_endpoint=_setting(
            "MASSIVE_FLAT_FILES_ENDPOINT", env_values, "https://files.massive.com"
        )
        or "https://files.massive.com",
        massive_flat_files_bucket=_setting(
            "MASSIVE_FLAT_FILES_BUCKET", env_values, "flatfiles"
        )
        or "flatfiles",
        ibkr_host=_setting("IBKR_HOST", env_values, "127.0.0.1") or "127.0.0.1",
        ibkr_paper_port=_setting_int("IBKR_PAPER_PORT", env_values, 7497),
        ibkr_live_port=_setting_int("IBKR_LIVE_PORT", env_values, 7496),
        ibkr_client_id=_setting_int("IBKR_CLIENT_ID", env_values, 101),
        ibkr_paper_client_id=_setting_int(
            "IBKR_PAPER_CLIENT_ID",
            env_values,
            _setting_int("IBKR_CLIENT_ID", env_values, 101),
        ),
        ibkr_paper_submit_client_id=_setting_int(
            "IBKR_PAPER_SUBMIT_CLIENT_ID",
            env_values,
            121,
        ),
        ibkr_live_client_id=_setting_int(
            "IBKR_LIVE_CLIENT_ID",
            env_values,
            201,
        ),
        ibkr_live_monitor_enabled=_setting_bool(
            "IBKR_LIVE_MONITOR_ENABLED", env_values, False
        ),
        qmt_connector_enabled=_setting_bool("QMT_CONNECTOR_ENABLED", env_values, False),
        qmt_connector_url=_setting(
            "QMT_CONNECTOR_URL",
            env_values,
            "http://127.0.0.1:58668",
        )
        or "http://127.0.0.1:58668",
        qmt_submit_connector_url=_setting(
            "QMT_SUBMIT_CONNECTOR_URL",
            env_values,
            "http://127.0.0.1:58669",
        )
        or "http://127.0.0.1:58669",
        qmt_api_token=_credential(("QMT_API_TOKEN",), env_values),
        qmt_request_signing_secret=_credential(
            ("QMT_REQUEST_SIGNING_SECRET",),
            env_values,
        ),
        qmt_audit_log_path=_setting_path(
            "QMT_AUDIT_LOG_PATH",
            env_values,
            "runtime/logs/qmt_connector_audit.jsonl",
        ),
        qmt_require_private_connector=_setting_bool(
            "QMT_REQUIRE_PRIVATE_CONNECTOR",
            env_values,
            True,
        ),
        qmt_account_id=_setting("QMT_ACCOUNT_ID", env_values),
        qmt_paper_account_id=_setting("QMT_PAPER_ACCOUNT_ID", env_values),
        qmt_live_account_id=_setting("QMT_LIVE_ACCOUNT_ID", env_values),
        qmt_account_type=(
            _setting("QMT_ACCOUNT_TYPE", env_values, "STOCK") or "STOCK"
        ).strip().upper(),
        qmt_session_id=_setting_int("QMT_SESSION_ID", env_values, 880001),
        qmt_timeout_seconds=_setting_float_default(
            "QMT_TIMEOUT_SECONDS",
            env_values,
            5.0,
        )
        or 5.0,
        qmt_live_monitor_enabled=_setting_bool(
            "QMT_LIVE_MONITOR_ENABLED", env_values, False
        ),
        allow_qmt_paper_order_submit=_setting_bool(
            "ALLOW_QMT_PAPER_ORDER_SUBMIT", env_values, False
        ),
        allow_qmt_live_trading=_setting_bool(
            "ALLOW_QMT_LIVE_TRADING", env_values, False
        ),
        ops_status_source=(
            _setting("OQP_OPS_STATUS_SOURCE", env_values, "snapshot" if os.uname().sysname == "Darwin" else "direct")
            or "direct"
        ).strip().lower(),
        allow_paper_trading=_setting_bool("ALLOW_PAPER_TRADING", env_values, False),
        allow_paper_order_submit=_setting_bool(
            "ALLOW_PAPER_ORDER_SUBMIT", env_values, False
        ),
        paper_max_order_notional=_setting_float_default(
            "PAPER_MAX_ORDER_NOTIONAL",
            env_values,
            10_000.0,
        ),
        paper_max_daily_notional=_setting_float_default(
            "PAPER_MAX_DAILY_NOTIONAL",
            env_values,
            50_000.0,
        ),
        paper_allowed_symbols=_setting_csv_tuple(
            "PAPER_ALLOWED_SYMBOLS",
            env_values,
        ),
        paper_allowed_asset_classes=_setting_csv_tuple(
            "PAPER_ALLOWED_ASSET_CLASSES",
            env_values,
            "equity,etf",
        ),
        paper_options_enabled=_setting_bool("PAPER_OPTIONS_ENABLED", env_values, False),
        paper_option_allowed_underlyings=_setting_csv_tuple(
            "PAPER_OPTION_ALLOWED_UNDERLYINGS",
            env_values,
        ),
        paper_option_allowed_strategies=_setting_csv_tuple(
            "PAPER_OPTION_ALLOWED_STRATEGIES",
            env_values,
        ),
        paper_option_max_contracts=_setting_float_default(
            "PAPER_OPTION_MAX_CONTRACTS",
            env_values,
            1.0,
        ),
        paper_option_max_premium=_setting_float_default(
            "PAPER_OPTION_MAX_PREMIUM",
            env_values,
            500.0,
        ),
        paper_option_max_defined_risk=_setting_float_default(
            "PAPER_OPTION_MAX_DEFINED_RISK",
            env_values,
            1_000.0,
        ),
        paper_option_max_spread_width=_setting_float_default(
            "PAPER_OPTION_MAX_SPREAD_WIDTH",
            env_values,
            10.0,
        ),
        trading_mode=_setting("TRADING_MODE", env_values, "paper") or "paper",
        allow_live_trading=_setting_bool("ALLOW_LIVE_TRADING", env_values, False),
        max_daily_loss_pct=_setting_float("MAX_DAILY_LOSS_PCT", env_values),
        max_gross_exposure=_setting_float("MAX_GROSS_EXPOSURE", env_values),
        database_url=_setting(
            "DATABASE_URL", env_values, "sqlite:///runtime/db/oqp.sqlite3"
        )
        or "sqlite:///runtime/db/oqp.sqlite3",
        data_root=REPO_ROOT / (_setting("DATA_ROOT", env_values, "runtime/data") or "runtime/data"),
        artifact_root=REPO_ROOT
        / (_setting("ARTIFACT_ROOT", env_values, "runtime/artifacts") or "runtime/artifacts"),
        log_level=_setting("LOG_LEVEL", env_values, "INFO") or "INFO",
    )
