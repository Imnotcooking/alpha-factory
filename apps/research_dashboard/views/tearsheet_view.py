import os
import sys
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import LOGS_DIR, TEXT


@dataclass(frozen=True)
class MetricQuality:
    label_key: str
    color: str
    help_key: str


class TearSheetView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(
        self,
        run_id: str,
        run_metadata: pd.Series,
        returns_path: str | None = None,
        lang: str = "EN",
        show_test_scope: bool = True,
    ):
        if isinstance(returns_path, str) and returns_path in TEXT and lang == "EN":
            lang = returns_path
            returns_path = None

        if returns_path is None:
            returns_path = run_metadata.get('returns_file_path', None)

        t = TEXT[lang]
        
        # 1. Fetch Cleaned Data
        df = self.dm.get_run_returns(run_id, returns_path=returns_path)
        if df.empty:
            st.warning(t["ts_no_returns"])
            return
        manifest = self._load_manifest(run_id)

        days = pd.to_datetime(df['date'], errors="coerce")
        daily_returns = pd.to_numeric(df['net_return'], errors="coerce").fillna(0.0).values
        daily_returns_bench = pd.to_numeric(df['benchmark_return'], errors="coerce").fillna(0.0).values
        daily_leverage = pd.to_numeric(df['portfolio_leverage'], errors="coerce").fillna(0.0).values
        avg_turnover = pd.to_numeric(df['daily_turnover'], errors="coerce").fillna(0.0).mean() * 100

        # 2. Risk Math (Strategy)
        cum_returns = (1 + daily_returns).cumprod()
        ann_return = (cum_returns[-1]) ** (252 / len(days)) - 1 if len(days) > 0 else 0
        ann_vol = np.std(daily_returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # 3. Risk Math (Benchmark)
        cum_returns_bench = (1 + daily_returns_bench).cumprod()
        extra_benchmark_curves = {}
        for column in df.columns:
            if not str(column).startswith("benchmark_return_"):
                continue
            returns = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy()
            extra_benchmark_curves[self._benchmark_column_label(column)] = (1 + returns).cumprod()
        ann_ret_bench = (cum_returns_bench[-1]) ** (252 / len(days)) - 1 if len(days) > 0 else 0
        ann_vol_bench = np.std(daily_returns_bench) * np.sqrt(252)
        sharpe_bench = ann_ret_bench / ann_vol_bench if ann_vol_bench != 0 else 0
        
        rolling_max_bench = np.maximum.accumulate(cum_returns_bench)
        drawdown_bench = (cum_returns_bench - rolling_max_bench) / rolling_max_bench
        max_dd_bench = np.min(drawdown_bench) if len(drawdown_bench) > 0 else 0
        calmar_bench = ann_ret_bench / abs(max_dd_bench) if max_dd_bench != 0 else 0

        # --- EXTRACT NEW METRICS ---
        holdout_ic = run_metadata.get('holdout_ic', 0.0)
        total_trades = run_metadata.get('total_trades', 'N/A')
        execution_diagnostics = self._execution_diagnostics(run_metadata, manifest)

        # --- RENDER TOP DASHBOARD ---
        if show_test_scope:
            self._render_test_scope(df, run_metadata, manifest, t)
        st.markdown(t["strategy_net"])
        r1, r2, r3, r4 = st.columns(4)
        self._render_quality_metric(
            r1,
            t["risk_metrics"][0],
            f"{ann_return * 100:.2f}%",
            self._classify_annual_return(ann_return, ann_ret_bench),
            t,
        )
        self._render_quality_metric(
            r2,
            t["risk_metrics"][1],
            f"{ann_vol * 100:.2f}%",
            self._classify_volatility(ann_vol, ann_vol_bench),
            t,
        )
        self._render_quality_metric(
            r3,
            t["risk_metrics"][2],
            f"{sharpe:.2f}",
            self._classify_sharpe(sharpe),
            t,
        )
        self._render_quality_metric(
            r4,
            t["risk_metrics"][3],
            f"{max_dd * 100:.2f}%",
            self._classify_drawdown(max_dd),
            t,
        )

        r5, r6, r7, r8 = st.columns(4)
        self._render_quality_metric(
            r5,
            t["risk_metrics"][4],
            f"{calmar:.2f}",
            self._classify_calmar(calmar),
            t,
        )
        self._render_quality_metric(
            r6,
            t["avg_turnover"],
            f"{avg_turnover:.1f}%",
            self._classify_turnover(avg_turnover / 100.0),
            t,
        )
        self._render_quality_metric(
            r7,
            t["holdout_ic"],
            f"{holdout_ic:.4f}",
            self._classify_holdout_ic(holdout_ic),
            t,
        )
        self._render_quality_metric(
            r8,
            t["total_trades"],
            self._format_count(total_trades),
            self._classify_trade_count(total_trades),
            t,
        )

        r9, r10, r11, r12 = st.columns(4)
        raw_entries = execution_diagnostics.get("raw_entry_count")
        raw_exits = execution_diagnostics.get("raw_exit_count")
        target_changes = execution_diagnostics.get("target_weight_change_count")
        signal_rows = execution_diagnostics.get("active_signal_row_count")
        self._render_quality_metric(
            r9,
            t.get("raw_entries", "Raw Entries"),
            self._format_count(raw_entries),
            self._classify_record_count(raw_entries),
            t,
        )
        self._render_quality_metric(
            r10,
            t.get("raw_exits", "Raw Exits"),
            self._format_count(raw_exits),
            self._classify_record_count(raw_exits),
            t,
        )
        self._render_quality_metric(
            r11,
            t.get("target_weight_changes", "Target-Weight Changes"),
            self._format_count(target_changes),
            self._classify_record_count(target_changes),
            t,
        )
        self._render_quality_metric(
            r12,
            t.get("active_signal_rows", "Active Signal Rows"),
            self._format_count(signal_rows),
            self._classify_signal_rows(signal_rows),
            t,
        )

        st.markdown(self._benchmark_title(run_id, t, df=df))
        b1, b2, b3, b4 = st.columns(4)
        self._render_quality_metric(
            b1,
            t["risk_metrics"][0],
            f"{ann_ret_bench * 100:.2f}%",
            self._classify_annual_return(ann_ret_bench),
            t,
        )
        self._render_quality_metric(
            b2,
            t["risk_metrics"][1],
            f"{ann_vol_bench * 100:.2f}%",
            self._classify_volatility(ann_vol_bench),
            t,
        )
        self._render_quality_metric(
            b3,
            t["risk_metrics"][2],
            f"{sharpe_bench:.2f}",
            self._classify_sharpe(sharpe_bench),
            t,
        )
        self._render_quality_metric(
            b4,
            t["risk_metrics"][3],
            f"{max_dd_bench * 100:.2f}%",
            self._classify_drawdown(max_dd_bench),
            t,
        )

        b5, b6, b7, b8 = st.columns(4)
        self._render_quality_metric(
            b5,
            t["risk_metrics"][4],
            f"{calmar_bench:.2f}",
            self._classify_calmar(calmar_bench),
            t,
        )
        reference = MetricQuality("reference", "#3b82f6", "reference_control")
        not_applicable = MetricQuality("not_applicable", "#64748b", "not_applicable")
        self._render_quality_metric(b6, t["avg_turnover"], "Passive", reference, t)
        self._render_quality_metric(b7, t["holdout_ic"], "N/A", not_applicable, t)
        self._render_quality_metric(b8, t["total_trades"], "Passive", reference, t)

        self._render_benchmark_guide(manifest, df, t)

        st.markdown("---")

        # --- RENDER MAIN PLOTLY CHARTS ---
        self._render_charts(
            days,
            cum_returns,
            cum_returns_bench,
            drawdown,
            daily_leverage,
            daily_returns,
            ann_vol,
            t,
            extra_benchmark_curves=extra_benchmark_curves,
        )

    def _load_manifest(self, run_id: str) -> dict:
        manifest_path = Path(LOGS_DIR) / "assumptions" / f"assumptions_{run_id}.json"
        if not manifest_path.exists():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _benchmark_title(self, run_id: str, copy: dict, df: pd.DataFrame | None = None) -> str:
        return copy.get("benchmark_eq", "#### 🟡 Benchmark")

    def _render_benchmark_guide(self, manifest: dict, returns: pd.DataFrame, copy: dict) -> None:
        entries = self._benchmark_guide_entries(manifest, returns)
        if not entries:
            return

        with st.expander(copy.get("tearsheet_benchmark_guide", "What the benchmarks mean")):
            intro = copy.get(
                "tearsheet_benchmark_guide_intro",
                "The benchmark card uses the primary benchmark_return series. Extra benchmark columns appear as dotted chart lines.",
            )
            headers = copy.get(
                "tearsheet_benchmark_guide_headers",
                ["Slot", "Benchmark", "Return column", "Meaning"],
            )
            if len(headers) != 4:
                headers = ["Slot", "Benchmark", "Return column", "Meaning"]

            lines = [
                str(intro).strip(),
                "",
                "| " + " | ".join(self._markdown_cell(header) for header in headers) + " |",
                "|---|---|---|---|",
            ]
            for entry in entries:
                mode = entry.get("return_mode", "unknown")
                mode_label = copy.get(
                    f"benchmark_mode_label_{mode}",
                    self._humanize_label(mode),
                )
                slot_label = copy.get(f"benchmark_slot_{entry['slot_key']}", entry["slot_label"])
                description = self._benchmark_description(entry, copy)
                meaning = f"{mode_label}. {description}" if mode_label else description
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            self._markdown_cell(slot_label),
                            self._markdown_cell(entry["label"]),
                            f"`{self._markdown_cell(entry['column'])}`",
                            self._markdown_cell(meaning),
                        ]
                    )
                    + " |"
                )
            st.markdown("\n".join(lines))

    def _benchmark_guide_entries(self, manifest: dict, returns: pd.DataFrame) -> list[dict[str, str]]:
        benchmark = manifest.get("benchmark", {}) if isinstance(manifest, dict) else {}
        if not isinstance(benchmark, dict):
            benchmark = {}

        entries: list[dict[str, str]] = []
        seen: set[str] = set()
        available = set(map(str, returns.columns)) if isinstance(returns, pd.DataFrame) else set()

        def add(policy_item: dict, slot_key: str, default_column: str | None = None) -> None:
            if not isinstance(policy_item, dict):
                return
            column = str(policy_item.get("benchmark_column") or default_column or "").strip()
            if not column or column in seen:
                return
            if available and column not in available:
                return

            role = str(policy_item.get("benchmark_role") or "").strip()
            return_mode = str(policy_item.get("return_mode") or "").strip()
            if not return_mode:
                return_mode = self._infer_benchmark_return_mode(column, role)
            label = str(policy_item.get("benchmark_label") or "").strip()
            if not label:
                label = self._benchmark_column_label(column)

            seen.add(column)
            entries.append(
                {
                    "slot_key": slot_key,
                    "slot_label": self._benchmark_slot_label(slot_key),
                    "label": label,
                    "column": column,
                    "benchmark_role": role,
                    "return_mode": return_mode,
                }
            )

        primary = benchmark or {"benchmark_column": "benchmark_return", "benchmark_label": "Benchmark"}
        add(primary, "primary", "benchmark_return")

        for secondary in benchmark.get("secondary_benchmarks", []) or []:
            add(secondary, "secondary")
        for control in benchmark.get("same_horizon_controls", []) or []:
            add(control, "control")

        if isinstance(returns, pd.DataFrame):
            if "benchmark_return" not in seen and "benchmark_return" in returns.columns:
                add({"benchmark_column": "benchmark_return", "benchmark_label": "Benchmark"}, "primary")
            for column in returns.columns:
                column = str(column)
                if column.startswith("benchmark_return_") and column not in seen:
                    add({"benchmark_column": column}, "additional")
        return entries

    @staticmethod
    def _benchmark_slot_label(slot_key: str) -> str:
        return {
            "primary": "Primary",
            "secondary": "Secondary",
            "control": "Control",
            "additional": "Additional",
        }.get(slot_key, "Benchmark")

    @staticmethod
    def _infer_benchmark_return_mode(column: str, role: str = "") -> str:
        role_text = str(role).lower()
        column_text = str(column).lower()
        if "same_horizon" in role_text or "same_horizon" in column_text:
            return "same_horizon"
        if "passive" in column_text or "close" in column_text:
            return "passive_close_to_close"
        return "passive_close_to_close" if role_text else "unknown"

    @staticmethod
    def _benchmark_description(entry: dict[str, str], copy: dict) -> str:
        role = str(entry.get("benchmark_role") or "").strip()
        mode = str(entry.get("return_mode") or "").strip()
        for key in (f"benchmark_role_{role}", f"benchmark_mode_{mode}"):
            if key in copy and copy[key]:
                return str(copy[key])
        return str(
            copy.get(
                "benchmark_mode_unknown",
                "Return construction was not recorded in the manifest; use this as context and audit the source column.",
            )
        )

    @staticmethod
    def _humanize_label(value: str) -> str:
        return str(value).replace("_", " ").replace("-", " ").strip().title()

    @staticmethod
    def _markdown_cell(value) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|").strip()

    def _render_test_scope(self, returns: pd.DataFrame, run_metadata: pd.Series, manifest: dict, copy: dict) -> None:
        summary = self.test_scope_summary(returns, run_metadata, manifest)
        if not summary:
            return
        st.info(self.format_test_scope(summary, copy))

    def test_scope_summary(self, returns: pd.DataFrame, run_metadata: pd.Series, manifest: dict) -> dict[str, object] | None:
        if returns is None or returns.empty:
            return None
        date_column = next(
            (
                column
                for column in (
                    "date",
                    "datetime",
                    "timestamp",
                    "trading_day",
                    "trade_date",
                    "session_date",
                )
                if column in returns.columns
            ),
            None,
        )
        if date_column is None:
            return None
        dates = pd.to_datetime(returns[date_column], errors="coerce").dropna()
        if dates.empty:
            return None
        data_section = manifest.get("data", {}) if isinstance(manifest.get("data"), dict) else {}
        signal_section = (
            manifest.get("signal_and_execution_mode", {})
            if isinstance(manifest.get("signal_and_execution_mode"), dict)
            else {}
        )
        start = dates.min().strftime("%Y-%m-%d")
        end = dates.max().strftime("%Y-%m-%d")
        observations = len(returns)
        years = observations / 252.0 if observations else 0.0
        prepared_start = self._short_date(data_section.get("prepared_data_start"))
        prepared_end = self._short_date(data_section.get("prepared_data_end"))
        prepared_rows = data_section.get("prepared_data_rows")
        prepared_window = (
            f"{prepared_start} to {prepared_end}"
            if prepared_start and prepared_end
            else "unknown"
        )
        prepared_rows_value = pd.to_numeric(pd.Series([prepared_rows]), errors="coerce").iloc[0]
        if pd.notna(prepared_rows_value):
            prepared_window = f"{prepared_window} ({int(prepared_rows_value):,} rows)"
        requested_start = data_section.get("requested_start_date")
        requested_end = data_section.get("requested_end_date")
        requested_window = (
            f"{requested_start or 'start'} to {requested_end or 'end'}"
            if requested_start or requested_end
            else "full prepared window"
        )
        frequency = (
            data_section.get("frequency")
            or run_metadata.get("data_frequency")
            or "unknown"
        )
        dataset_role = (
            data_section.get("dataset_role")
            or run_metadata.get("dataset_role")
            or "unknown"
        )
        tradability = (
            data_section.get("tradability")
            or run_metadata.get("data_tradability")
            or ""
        )
        return_clock = (
            data_section.get("return_horizon")
            or signal_section.get("return_assumption")
            or run_metadata.get("return_assumption")
            or run_metadata.get("execution_assumption")
            or "unknown"
        )
        source_path = str(data_section.get("source_path") or "").strip()
        source_label = Path(source_path).name if source_path else "unknown"
        role_text = f"{dataset_role}/{tradability}" if tradability else str(dataset_role)
        return {
            "run_id": str(run_metadata.get("run_id") or ""),
            "start": start,
            "end": end,
            "rows": observations,
            "years": years,
            "prepared_window": prepared_window,
            "requested_window": requested_window,
            "frequency": frequency,
            "role": role_text,
            "return_clock": return_clock,
            "source": source_label,
        }

    @staticmethod
    def format_test_scope(summary: dict[str, object], copy: dict) -> str:
        return copy.get(
            "test_scope",
            "Run: {run_id} | Backtest result window used by widgets/charts: {start} to {end} | Return rows: {rows:,} (~{years:.2f}y annualization) | Prepared data: {prepared_window} | Requested filter: {requested_window} | Frequency: {frequency} | Data: {role} | Return clock: {return_clock} | Source: {source}",
        ).format(
            run_id=summary.get("run_id", ""),
            start=summary.get("start", ""),
            end=summary.get("end", ""),
            rows=int(summary.get("rows", 0) or 0),
            years=float(summary.get("years", 0.0) or 0.0),
            prepared_window=summary.get("prepared_window", ""),
            requested_window=summary.get("requested_window", ""),
            frequency=summary.get("frequency", ""),
            role=summary.get("role", ""),
            return_clock=summary.get("return_clock", ""),
            source=summary.get("source", ""),
        )

    @staticmethod
    def _execution_diagnostics(run_metadata: pd.Series, manifest: dict) -> dict:
        signal_diag = manifest.get("signal_diagnostics", {}) if isinstance(manifest, dict) else {}
        if not isinstance(signal_diag, dict):
            signal_diag = {}

        def first_count(*values):
            for value in values:
                if value is None or value is pd.NA:
                    continue
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric) and float(numeric) > 0:
                    return int(numeric)
            return None

        return {
            "raw_entry_count": first_count(signal_diag.get("raw_entry_count")),
            "raw_exit_count": first_count(signal_diag.get("raw_exit_count")),
            "target_weight_change_count": first_count(
                signal_diag.get("target_weight_change_count"),
                signal_diag.get("executed_weight_change_count"),
            ),
            "active_signal_row_count": first_count(
                signal_diag.get("active_signal_row_count"),
                signal_diag.get("active_state_row_count"),
                run_metadata.get("active_tick_count"),
            ),
        }

    @staticmethod
    def _format_count(value) -> str:
        if value is None or value is pd.NA:
            return "N/A"
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "N/A"
        return f"{int(numeric):,}"

    @staticmethod
    def _numeric(value) -> float | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric) or not np.isfinite(numeric):
            return None
        return float(numeric)

    @classmethod
    def _classify_annual_return(cls, value, benchmark=None) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "return_unavailable")
        if numeric <= 0:
            return MetricQuality("negative", "#ef4444", "return_negative")

        reference = cls._numeric(benchmark)
        if reference is not None:
            spread = numeric - reference
            if spread < -0.002:
                return MetricQuality("below_benchmark", "#f59e0b", "return_below_benchmark")
            if spread > 0.002:
                return MetricQuality("ahead", "#22c55e", "return_ahead")
            return MetricQuality("in_line", "#3b82f6", "return_in_line")
        if numeric < 0.02:
            return MetricQuality("low", "#f59e0b", "return_low")
        if numeric < 0.30:
            return MetricQuality("positive", "#22c55e", "return_positive")
        return MetricQuality("high_audit", "#8b5cf6", "return_high_audit")

    @classmethod
    def _classify_volatility(cls, value, benchmark=None) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "volatility_unavailable")
        if numeric <= 0.001:
            return MetricQuality("audit", "#8b5cf6", "volatility_flat_audit")

        reference = cls._numeric(benchmark)
        if reference is not None and reference > 0.001:
            ratio = numeric / reference
            if ratio < 0.75:
                return MetricQuality("lower_risk", "#22c55e", "volatility_lower")
            if ratio <= 1.25:
                return MetricQuality("in_line", "#3b82f6", "volatility_in_line")
            if ratio <= 1.75:
                return MetricQuality("elevated", "#f59e0b", "volatility_elevated")
            return MetricQuality("high", "#ef4444", "volatility_high")
        if numeric < 0.02:
            return MetricQuality("very_low", "#64748b", "volatility_very_low")
        if numeric < 0.10:
            return MetricQuality("controlled", "#22c55e", "volatility_controlled")
        if numeric < 0.20:
            return MetricQuality("moderate", "#3b82f6", "volatility_moderate")
        if numeric < 0.35:
            return MetricQuality("elevated", "#f59e0b", "volatility_elevated")
        return MetricQuality("high", "#ef4444", "volatility_high")

    @classmethod
    def _classify_sharpe(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "sharpe_unavailable")
        if numeric <= 0:
            return MetricQuality("negative", "#ef4444", "sharpe_negative")
        if numeric < 0.5:
            return MetricQuality("weak", "#ef4444", "sharpe_weak")
        if numeric < 1.0:
            return MetricQuality("marginal", "#f59e0b", "sharpe_marginal")
        if numeric < 2.0:
            return MetricQuality("good", "#22c55e", "sharpe_good")
        if numeric < 3.0:
            return MetricQuality("strong", "#14b8a6", "sharpe_strong")
        if numeric < 5.0:
            return MetricQuality("high_audit", "#8b5cf6", "sharpe_high_audit")
        return MetricQuality("extreme_audit", "#ef4444", "sharpe_extreme_audit")

    @classmethod
    def _classify_drawdown(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "drawdown_unavailable")
        magnitude = abs(min(numeric, 0.0))
        if magnitude <= 0.001:
            return MetricQuality("audit", "#8b5cf6", "drawdown_none_audit")
        if magnitude <= 0.05:
            return MetricQuality("controlled", "#22c55e", "drawdown_controlled")
        if magnitude <= 0.10:
            return MetricQuality("moderate", "#3b82f6", "drawdown_moderate")
        if magnitude <= 0.20:
            return MetricQuality("elevated", "#f59e0b", "drawdown_elevated")
        return MetricQuality("severe", "#ef4444", "drawdown_severe")

    @classmethod
    def _classify_calmar(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "calmar_unavailable")
        if numeric <= 0:
            return MetricQuality("negative", "#ef4444", "calmar_negative")
        if numeric < 0.5:
            return MetricQuality("weak", "#ef4444", "calmar_weak")
        if numeric < 1.0:
            return MetricQuality("modest", "#f59e0b", "calmar_modest")
        if numeric < 2.0:
            return MetricQuality("good", "#22c55e", "calmar_good")
        if numeric < 5.0:
            return MetricQuality("strong", "#14b8a6", "calmar_strong")
        return MetricQuality("high_audit", "#8b5cf6", "calmar_high_audit")

    @classmethod
    def _classify_turnover(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "turnover_unavailable")
        if numeric < 0.05:
            return MetricQuality("low", "#22c55e", "turnover_low")
        if numeric < 0.20:
            return MetricQuality("moderate", "#3b82f6", "turnover_moderate")
        if numeric < 0.50:
            return MetricQuality("high", "#f59e0b", "turnover_high")
        return MetricQuality("very_high", "#ef4444", "turnover_very_high")

    @classmethod
    def _classify_holdout_ic(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "ic_unavailable")
        if numeric <= 0:
            return MetricQuality("negative", "#ef4444", "ic_negative")
        if numeric < 0.01:
            return MetricQuality("weak", "#ef4444", "ic_weak")
        if numeric < 0.03:
            return MetricQuality("modest", "#f59e0b", "ic_modest")
        if numeric < 0.05:
            return MetricQuality("good", "#22c55e", "ic_good")
        if numeric < 0.10:
            return MetricQuality("strong", "#14b8a6", "ic_strong")
        return MetricQuality("extreme_audit", "#ef4444", "ic_extreme_audit")

    @classmethod
    def _classify_trade_count(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("not_available", "#64748b", "trades_unavailable")
        if numeric <= 0:
            return MetricQuality("none", "#ef4444", "trades_none")
        if numeric < 30:
            return MetricQuality("sparse", "#ef4444", "trades_sparse")
        if numeric < 100:
            return MetricQuality("limited", "#f59e0b", "trades_limited")
        if numeric < 300:
            return MetricQuality("adequate", "#22c55e", "trades_adequate")
        return MetricQuality("broad", "#14b8a6", "trades_broad")

    @classmethod
    def _classify_record_count(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("missing", "#64748b", "diagnostic_missing")
        if numeric <= 0:
            return MetricQuality("none", "#f59e0b", "diagnostic_none")
        return MetricQuality("recorded", "#3b82f6", "diagnostic_recorded")

    @classmethod
    def _classify_signal_rows(cls, value) -> MetricQuality:
        numeric = cls._numeric(value)
        if numeric is None:
            return MetricQuality("missing", "#64748b", "signal_rows_missing")
        if numeric < 100:
            return MetricQuality("limited", "#ef4444", "signal_rows_limited")
        if numeric < 500:
            return MetricQuality("moderate", "#f59e0b", "signal_rows_moderate")
        return MetricQuality("broad", "#14b8a6", "signal_rows_broad")

    @staticmethod
    def _render_quality_metric(column, metric_label: str, metric_value: str, quality: MetricQuality, copy: dict) -> None:
        labels = copy.get("tearsheet_quality_labels", {})
        help_texts = copy.get("tearsheet_quality_help", {})
        label = labels.get(quality.label_key, quality.label_key.replace("_", " ").title())
        help_text = help_texts.get(quality.help_key, "")
        column.html(
            f"""
            <div style="min-height: 5.2rem; padding-top: 0.1rem;">
                <div style="font-size: 0.875rem; line-height: 1.2; margin-bottom: 0.55rem;">
                    {escape(str(metric_label))}
                </div>
                <div style="white-space: nowrap; overflow: visible; line-height: 1.1;">
                    <span style="display: inline-block; font-size: 1.75rem; line-height: 1.1; font-weight: 400; vertical-align: middle;">
                        {escape(str(metric_value))}
                    </span>
                    <span title="{escape(help_text)}" style="
                        display: inline-flex;
                        margin-left: 0.35rem;
                        padding: 0.18rem 0.3rem;
                        border: 1px solid {quality.color}66;
                        border-radius: 4px;
                        background: {quality.color}14;
                        font-size: 0.62rem;
                        font-weight: 700;
                        line-height: 1.15;
                        white-space: nowrap;
                        vertical-align: middle;
                    ">
                        {escape(label)}
                    </span>
                </div>
            </div>
            """
        )

    @staticmethod
    def _short_date(value) -> str:
        if value is None or value is pd.NA or not str(value).strip():
            return ""
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return str(value)
        return parsed.strftime("%Y-%m-%d")

    @staticmethod
    def _benchmark_column_label(column: str) -> str:
        suffix = str(column).removeprefix("benchmark_return_").replace("_", " ").strip()
        if not suffix:
            return "Benchmark"
        return suffix.upper() if len(suffix) <= 4 else suffix.title()

    def _render_charts(
        self,
        days,
        cum_returns,
        cum_returns_bench,
        drawdown,
        daily_leverage,
        daily_returns,
        ann_vol,
        t,
        extra_benchmark_curves=None,
    ):
        # 1. Main Tear Sheet Grid
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(
            go.Scatter(
                x=days,
                y=(cum_returns - 1) * 100,
                name="Net Strategy",
                line=dict(color='#00E676', width=2),
                hovertemplate="<b>Net Strategy</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=(cum_returns_bench - 1) * 100,
                name="Benchmark",
                line=dict(color='#FFCA28', width=2, dash='dash'),
                hovertemplate="<b>Benchmark</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
        palette = ["#42A5F5", "#AB47BC", "#26A69A", "#EF5350"]
        for idx, (label, curve) in enumerate((extra_benchmark_curves or {}).items()):
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=(curve - 1) * 100,
                    name=f"Benchmark {label}",
                    line=dict(color=palette[idx % len(palette)], width=1.6, dash='dot'),
                    hovertemplate=(
                        f"<b>Benchmark {label}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=drawdown * 100,
                name="Drawdown",
                fill='tozeroy',
                line=dict(color='#D32F2F', width=1),
                fillcolor='rgba(211, 47, 47, 0.3)',
                hovertemplate="<b>Drawdown</b><br>Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=daily_leverage,
                name="Gross Leverage",
                fill='tozeroy',
                line=dict(color='#2196F3', width=1),
                fillcolor='rgba(33, 150, 243, 0.2)',
                hovertemplate="<b>Gross Leverage</b><br>Date: %{x|%Y-%m-%d}<br>Leverage: %{y:.2f}x<extra></extra>",
            ),
            row=3,
            col=1,
        )

        fig.update_yaxes(title_text=t.get("axis_cum_return", "Cumulative return (%)"), ticksuffix="%", row=1, col=1)
        fig.update_yaxes(title_text=t.get("axis_drawdown", "Drawdown (%)"), ticksuffix="%", row=2, col=1)
        fig.update_yaxes(title_text=t.get("axis_gross_leverage", "Gross exposure (x)"), tickformat=".2f", row=3, col=1)
        fig.update_xaxes(title_text=t.get("axis_date", "Date"), row=3, col=1)
        fig.update_layout(
            template="plotly_dark",
            height=700,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(hoverformat="%Y-%m-%d")
        st.plotly_chart(fig, width="stretch")
        st.markdown("---")

        # 2. Monthly Heatmap & Volatility Matrix
        df_monthly = pd.DataFrame({'date': days, 'return': daily_returns})
        df_monthly['year'] = df_monthly['date'].dt.year
        df_monthly['month'] = df_monthly['date'].dt.month
        monthly_pivot = df_monthly.groupby(['year', 'month'])['return'].apply(lambda x: (1 + x).prod() - 1).reset_index().pivot(index='year', columns='month', values='return').fillna(0)
        for m in range(1, 13):
            if m not in monthly_pivot.columns: monthly_pivot[m] = 0.0
        monthly_pivot = monthly_pivot[range(1, 13)]

        df_vol = pd.DataFrame({'date': days, 'rolling_vol': df_monthly['return'].rolling(30).std() * np.sqrt(252)})

        c_heat, c_vol = st.columns([1.5, 1])
        with c_heat:
            st.markdown(f"#### {t.get('heatmap_title', '🗓️ Monthly Return Matrix')}")
            fig_hm = go.Figure(data=go.Heatmap(z=monthly_pivot.values * 100, x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], y=monthly_pivot.index, colorscale='RdYlGn', zmid=0, text=np.round(monthly_pivot.values * 100, 1), texttemplate="%{text}%", hoverinfo="z"))
            fig_hm.update_layout(template="plotly_dark", height=350, yaxis=dict(autorange="reversed", type='category'), margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hm, width="stretch")
            with st.expander(t.get('heatmap_help_title', '💡 Insight')): st.markdown(t.get('heatmap_help', ''))
            
        with c_vol:
            st.markdown(f"#### {t.get('vol_title', '📉 Rolling 30-Day Volatility')}")
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=df_vol['date'], y=df_vol['rolling_vol'] * 100, mode='lines', line=dict(color='#FF9800', width=2), fill='tozeroy', fillcolor='rgba(255, 152, 0, 0.2)'))
            fig_v.add_hline(y=ann_vol*100, line_dash="dash", line_color="gray")
            fig_v.update_layout(template="plotly_dark", height=350, yaxis_title="Ann. Volatility (%)", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_v, width="stretch")
            with st.expander(t.get('vol_help_title', '💡 Insight')): st.markdown(t.get('vol_help', ''))
