import os
import sys
import json
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

class TearSheetView:
    def __init__(self, data_manager):
        self.dm = data_manager

    def render(
        self,
        run_id: str,
        run_metadata: pd.Series,
        returns_path: str | None = None,
        lang: str = "EN",
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
        self._render_test_scope(df, run_metadata, manifest, t)
        st.markdown(t["strategy_net"])
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(t["risk_metrics"][0], f"{ann_return * 100:.2f}%")
        r2.metric(t["risk_metrics"][1], f"{ann_vol * 100:.2f}%")
        r3.metric(t["risk_metrics"][2], f"{sharpe:.2f}")
        r4.metric(t["risk_metrics"][3], f"{max_dd * 100:.2f}%")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric(t["risk_metrics"][4], f"{calmar:.2f}")
        r6.metric(t["avg_turnover"], f"{avg_turnover:.1f}%")
        r7.metric(t["holdout_ic"], f"{holdout_ic:.4f}")
        r8.metric(t["total_trades"], self._format_count(total_trades))

        r9, r10, r11, r12 = st.columns(4)
        r9.metric(t.get("raw_entries", "Raw Entries"), self._format_count(execution_diagnostics.get("raw_entry_count")))
        r10.metric(t.get("raw_exits", "Raw Exits"), self._format_count(execution_diagnostics.get("raw_exit_count")))
        r11.metric(
            t.get("target_weight_changes", "Target-Weight Changes"),
            self._format_count(execution_diagnostics.get("target_weight_change_count")),
        )
        r12.metric(
            t.get("active_signal_rows", "Active Signal Rows"),
            self._format_count(execution_diagnostics.get("active_signal_row_count")),
        )

        st.markdown(self._benchmark_title(run_id, t, df=df))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(t["risk_metrics"][0], f"{ann_ret_bench * 100:.2f}%")
        b2.metric(t["risk_metrics"][1], f"{ann_vol_bench * 100:.2f}%")
        b3.metric(t["risk_metrics"][2], f"{sharpe_bench:.2f}")
        b4.metric(t["risk_metrics"][3], f"{max_dd_bench * 100:.2f}%")

        b5, b6, b7, b8 = st.columns(4)
        b5.metric(t["risk_metrics"][4], f"{calmar_bench:.2f}")
        b6.metric(t["avg_turnover"], "Passive")
        b7.metric(t["holdout_ic"], "N/A")
        b8.metric(t["total_trades"], "Passive")

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
        payload = self._load_manifest(run_id)
        if not payload:
            return copy["benchmark_eq"]
        benchmark = payload.get("benchmark")
        if not isinstance(benchmark, dict):
            return copy["benchmark_eq"]
        label = str(benchmark.get("benchmark_label") or "").strip()
        if not label:
            return copy["benchmark_eq"]
        column = str(benchmark.get("benchmark_column") or "benchmark_return").strip()
        return_mode = str(benchmark.get("return_mode") or "").replace("_", " ").strip()
        suffix_parts = []
        if return_mode:
            suffix_parts.append(return_mode)
        if column and column != "benchmark_return":
            suffix_parts.append(column)
        if (
            isinstance(df, pd.DataFrame)
            and str(column) == "benchmark_return"
            and "benchmark_return_same_horizon" in df.columns
            and str(benchmark.get("return_mode") or "") != "same_horizon"
        ):
            suffix_parts.append("same-horizon control also available")
        suffix = f" | {'; '.join(suffix_parts)}" if suffix_parts else ""
        return f"### 🟡 Benchmark ({label}{suffix})"

    def _render_test_scope(self, returns: pd.DataFrame, run_metadata: pd.Series, manifest: dict, copy: dict) -> None:
        dates = pd.to_datetime(returns.get("date"), errors="coerce").dropna()
        if dates.empty:
            return
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
        st.info(
            copy.get(
                "test_scope",
                "Run: {run_id} | Backtest result window used by widgets/charts: {start} to {end} | Return rows: {rows:,} (~{years:.2f}y annualization) | Prepared data: {prepared_window} | Requested filter: {requested_window} | Frequency: {frequency} | Data: {role} | Return clock: {return_clock} | Source: {source}",
            ).format(
                run_id=str(run_metadata.get("run_id") or ""),
                start=start,
                end=end,
                rows=observations,
                years=years,
                prepared_window=prepared_window,
                requested_window=requested_window,
                frequency=frequency,
                role=role_text,
                return_clock=return_clock,
                source=source_label,
            )
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
