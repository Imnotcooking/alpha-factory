"""Small, read-only artifact loader for the quartile router lab."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "monthly": "paper_replication_monthly_returns.csv",
    "targets": "paper_replication_targets.parquet",
    "states": "paper_replication_volatility_states.csv",
    "costs": "paper_replication_cost_ledger.parquet",
    "summary": "paper_replication_summary.json",
    "anatomy": "market_anatomy/monthly_cross_sectional_anatomy.csv",
    "concentration": "market_anatomy/market_concentration_monthly.csv",
    "attribution": "market_anatomy/strategy_product_attribution.csv",
}

EXTENSION_REQUIRED_FILES = {
    "ema_targets": "ema_daily_targets.parquet",
    "ema_20_60_targets": "ema_20_60_daily_targets.parquet",
    "macd_targets": "macd_daily_targets.parquet",
    "product_states": "product_volatility_states.parquet",
    "alignment": "product_market_alignment.parquet",
    "breadth": "product_volatility_breadth.csv",
    "comparison": "sleeve_comparison_monthly.csv",
    "metrics": "sleeve_comparison_metrics.csv",
    "state_summary": "sleeve_state_summary.csv",
    "time_split": "sleeve_time_split.csv",
    "summary": "extension_summary.json",
}


@dataclass
class QuartileLabData:
    monthly: pd.DataFrame
    targets: pd.DataFrame
    states: pd.DataFrame
    costs: pd.DataFrame
    diagnostics: pd.DataFrame
    sectors: pd.DataFrame
    summary: dict
    source_id: str


@dataclass
class QuartileExtensionData:
    ema_targets: pd.DataFrame
    ema_20_60_targets: pd.DataFrame
    macd_targets: pd.DataFrame
    product_states: pd.DataFrame
    alignment: pd.DataFrame
    breadth: pd.DataFrame
    comparison: pd.DataFrame
    metrics: pd.DataFrame
    state_summary: pd.DataFrame
    time_split: pd.DataFrame
    summary: dict


def artifact_signature(artifact_root: str | Path) -> tuple[tuple[str, int, int], ...]:
    root = Path(artifact_root)
    signature = []
    for relative_path in REQUIRED_FILES.values():
        path = root / relative_path
        if path.exists():
            stat = path.stat()
            signature.append((relative_path, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            signature.append((relative_path, 0, 0))
    return tuple(signature)


def missing_artifacts(artifact_root: str | Path) -> list[Path]:
    root = Path(artifact_root)
    return [root / path for path in REQUIRED_FILES.values() if not (root / path).exists()]


def extension_artifact_signature(
    artifact_root: str | Path,
) -> tuple[tuple[str, int, int], ...]:
    root = Path(artifact_root)
    signature = []
    for relative_path in EXTENSION_REQUIRED_FILES.values():
        path = root / relative_path
        if path.exists():
            stat = path.stat()
            signature.append((relative_path, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            signature.append((relative_path, 0, 0))
    return tuple(signature)


def missing_extension_artifacts(artifact_root: str | Path) -> list[Path]:
    root = Path(artifact_root)
    return [
        root / path
        for path in EXTENSION_REQUIRED_FILES.values()
        if not (root / path).exists()
    ]


def load_quartile_lab_data(artifact_root: str | Path) -> QuartileLabData:
    root = Path(artifact_root)
    missing = missing_artifacts(root)
    if missing:
        raise FileNotFoundError(
            "quartile router artifacts are missing: "
            + ", ".join(str(path) for path in missing)
        )

    monthly = pd.read_csv(root / REQUIRED_FILES["monthly"])
    targets = pd.read_parquet(root / REQUIRED_FILES["targets"])
    states = pd.read_csv(root / REQUIRED_FILES["states"])
    costs = pd.read_parquet(root / REQUIRED_FILES["costs"])
    anatomy = pd.read_csv(root / REQUIRED_FILES["anatomy"])
    concentration = pd.read_csv(root / REQUIRED_FILES["concentration"])
    diagnostics = anatomy.merge(
        concentration, on="month", how="outer", validate="one_to_one"
    )
    attribution = pd.read_csv(root / REQUIRED_FILES["attribution"])
    sectors = attribution[["root", "sector"]].drop_duplicates().reset_index(drop=True)
    summary = json.loads((root / REQUIRED_FILES["summary"]).read_text(encoding="utf-8"))
    source_id = ":".join(
        [
            str(summary.get("version", "unknown")),
            str(summary.get("config_sha256", "unknown")),
            str(summary.get("daily_cache_sha256", "unknown")),
        ]
    )
    return QuartileLabData(
        monthly=monthly,
        targets=targets,
        states=states,
        costs=costs,
        diagnostics=diagnostics,
        sectors=sectors,
        summary=summary,
        source_id=source_id,
    )


def load_quartile_extension_data(
    artifact_root: str | Path,
) -> QuartileExtensionData:
    root = Path(artifact_root)
    missing = missing_extension_artifacts(root)
    if missing:
        raise FileNotFoundError(
            "quartile extension artifacts are missing: "
            + ", ".join(str(path) for path in missing)
        )
    return QuartileExtensionData(
        ema_targets=pd.read_parquet(root / EXTENSION_REQUIRED_FILES["ema_targets"]),
        ema_20_60_targets=pd.read_parquet(
            root / EXTENSION_REQUIRED_FILES["ema_20_60_targets"]
        ),
        macd_targets=pd.read_parquet(root / EXTENSION_REQUIRED_FILES["macd_targets"]),
        product_states=pd.read_parquet(
            root / EXTENSION_REQUIRED_FILES["product_states"]
        ),
        alignment=pd.read_parquet(root / EXTENSION_REQUIRED_FILES["alignment"]),
        breadth=pd.read_csv(root / EXTENSION_REQUIRED_FILES["breadth"]),
        comparison=pd.read_csv(root / EXTENSION_REQUIRED_FILES["comparison"]),
        metrics=pd.read_csv(root / EXTENSION_REQUIRED_FILES["metrics"]),
        state_summary=pd.read_csv(
            root / EXTENSION_REQUIRED_FILES["state_summary"]
        ),
        time_split=pd.read_csv(root / EXTENSION_REQUIRED_FILES["time_split"]),
        summary=json.loads(
            (root / EXTENSION_REQUIRED_FILES["summary"]).read_text(encoding="utf-8")
        ),
    )
