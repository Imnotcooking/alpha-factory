"""Export alpha-research DB rows as StrategyCandidate artifacts."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oqp.config import load_settings
from oqp.contracts.artifact_io import (
    strategy_candidate_directory,
    write_strategy_candidate_artifact,
)
from oqp.contracts.market_vertical import normalize_market_vertical
from oqp.contracts.strategy_candidate import (
    CandidateMetrics,
    CandidateSafetyLimits,
    CandidateStatus,
    StrategyCandidate,
)


class AlphaCandidateExportError(ValueError):
    """Raised when a research DB row cannot be exported as a candidate."""


def load_latest_candidate_from_research_db(
    db_path: Path,
    *,
    run_id: str | None = None,
    factor_id: str | None = None,
    status: CandidateStatus | str = CandidateStatus.RESEARCH_ONLY,
    target_market_vertical: str | None = None,
) -> StrategyCandidate:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = _select_backtest_row(conn, run_id=run_id, factor_id=factor_id)
        diagnostics = _diagnostics_for_run(conn, row.get("run_id"))

    return candidate_from_backtest_row(
        row,
        diagnostics=diagnostics,
        status=status,
        target_market_vertical=target_market_vertical,
    )


def candidate_from_backtest_row(
    row: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any] | None = None,
    status: CandidateStatus | str = CandidateStatus.RESEARCH_ONLY,
    target_market_vertical: str | None = None,
) -> StrategyCandidate:
    run_id = _text(row.get("run_id"))
    factor_id = _text(row.get("factor_id"))
    if not run_id:
        raise AlphaCandidateExportError("backtest row is missing run_id")
    if not factor_id:
        raise AlphaCandidateExportError("backtest row is missing factor_id")

    tested_market = normalize_market_vertical(
        row.get("market_vertical") or row.get("asset_class")
    )
    native_market = tested_market
    target_market = normalize_market_vertical(target_market_vertical or tested_market)
    diagnostic_payload = dict(diagnostics or {})
    diagnostic_code = _text(diagnostic_payload.get("failure_code"))
    diagnostic_action = _text(diagnostic_payload.get("suggested_action"))

    notes = None
    if diagnostic_code and diagnostic_code.upper() != "NONE":
        notes = f"{diagnostic_code}: {diagnostic_action or 'review required'}"

    return StrategyCandidate(
        candidate_id=f"candidate-{run_id}",
        strategy_id=factor_id,
        source="alpha_lab",
        promotion_status=status,
        native_market_vertical=native_market,
        tested_market_vertical=tested_market,
        target_market_vertical=target_market,
        intended_market_verticals=(native_market,),
        research_run_id=run_id,
        dataset_id=_text(row.get("dataset_id")),
        universe_id=_text(row.get("universe_id")),
        data_frequency=_text(row.get("data_frequency")),
        data_vendor=_text(row.get("data_vendor")),
        execution_assumption=_text(row.get("execution_assumption")),
        evaluation_geometry=_text(row.get("evaluation_geometry")),
        ic_metric=_text(row.get("ic_metric")),
        metrics=CandidateMetrics(
            validation_ic=_float(row.get("validation_ic")),
            holdout_ic=_float(row.get("holdout_ic")),
            crisis_ic=_float(row.get("crisis_ic")),
            validation_hit_rate=_float(row.get("validation_hit_rate")),
            holdout_hit_rate=_float(row.get("holdout_hit_rate")),
            sharpe_ratio=_float(row.get("sharpe_ratio")),
            annualized_return=_float(row.get("annualized_return")),
            max_drawdown=_float(row.get("max_drawdown")),
            turnover_rate=_float(row.get("turnover_rate")),
            avg_daily_cost_bps=_float(row.get("avg_daily_cost_bps")),
            metric_p_value=_float(row.get("stat_metric_p_value")),
            sharpe_p_value=_float(row.get("stat_sharpe_p_value")),
            significance=_text(row.get("stat_significance")),
        ),
        safety_limits=CandidateSafetyLimits(paper_only=True, allow_live_trading=False),
        instrument_mapping_required=target_market != tested_market,
        notes=notes,
        tags=("alpha_lab", native_market.lower()),
        metadata={
            "asset_class": _text(row.get("asset_class")),
            "split_mode": _text(row.get("split_mode")),
            "split_boundary": _text(row.get("split_boundary")),
            "validation_rows": _int(row.get("validation_rows")),
            "holdout_rows": _int(row.get("holdout_rows")),
            "crisis_rows": _int(row.get("crisis_rows")),
            "research_family": _text(row.get("stat_research_family")),
            "trial_signature": _text(row.get("stat_trial_signature")),
            "trial_count": _int(row.get("stat_trial_count")),
            "timestamp": _text(row.get("timestamp")),
            "diagnostics": diagnostic_payload,
        },
    )


def write_candidate_from_research_db(
    db_path: Path,
    *,
    output_dir: Path | None = None,
    run_id: str | None = None,
    factor_id: str | None = None,
    status: CandidateStatus | str = CandidateStatus.RESEARCH_ONLY,
    target_market_vertical: str | None = None,
    overwrite: bool = False,
) -> tuple[StrategyCandidate, Path]:
    candidate = load_latest_candidate_from_research_db(
        db_path,
        run_id=run_id,
        factor_id=factor_id,
        status=status,
        target_market_vertical=target_market_vertical,
    )
    directory = output_dir or strategy_candidate_directory(load_settings())
    path = write_strategy_candidate_artifact(
        candidate,
        directory,
        overwrite=overwrite,
    )
    return candidate, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export an alpha-lab backtest row as a strategy candidate."
    )
    parser.add_argument("--db", type=Path, default=Path("runtime/db/research/alpha_lab/research_memory.db"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--factor-id", type=str, default=None)
    parser.add_argument(
        "--status",
        type=str,
        default=CandidateStatus.RESEARCH_ONLY.value,
        choices=[status.value for status in CandidateStatus],
    )
    parser.add_argument("--target-market-vertical", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    candidate, path = write_candidate_from_research_db(
        args.db,
        output_dir=args.output_dir,
        run_id=args.run_id,
        factor_id=args.factor_id,
        status=args.status,
        target_market_vertical=args.target_market_vertical,
        overwrite=args.overwrite,
    )
    print(f"wrote {candidate.candidate_id} -> {path}")
    return 0


def _select_backtest_row(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    factor_id: str | None,
) -> dict[str, Any]:
    if not _table_exists(conn, "backtest_runs"):
        raise AlphaCandidateExportError("research DB has no backtest_runs table")

    clauses: list[str] = []
    params: list[str] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if factor_id:
        clauses.append("factor_id = ?")
        params.append(factor_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    row = conn.execute(
        f"""
        SELECT *
        FROM backtest_runs
        {where_sql}
        ORDER BY timestamp DESC, rowid DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        raise AlphaCandidateExportError("no matching backtest run found")
    return dict(row)


def _diagnostics_for_run(conn: sqlite3.Connection, run_id: str | None) -> dict[str, Any]:
    if not run_id or not _table_exists(conn, "diagnostics"):
        return {}
    row = conn.execute(
        "SELECT * FROM diagnostics WHERE run_id = ? LIMIT 1",
        (run_id,),
    ).fetchone()
    return dict(row) if row is not None else {}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
