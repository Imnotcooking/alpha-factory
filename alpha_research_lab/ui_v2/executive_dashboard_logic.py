import pandas as pd
import plotly.express as px
import streamlit as st

from config import TEXT, get_plotly_template


class ExecutiveDashboardView:
    def __init__(self, data_manager):
        self.dm = data_manager

    @staticmethod
    def _first_match(df: pd.DataFrame, mask: pd.Series):
        c = df[mask].copy()
        if c.empty:
            return None
        c["timestamp"] = pd.to_datetime(c["timestamp"], errors="coerce")
        return c.sort_values("timestamp", ascending=False).iloc[0]

    def _max_dd_from_run(self, row: pd.Series) -> float:
        if row is None:
            return float("nan")
        if "max_drawdown" in row.index and pd.notna(row.get("max_drawdown")):
            try:
                return float(row["max_drawdown"])
            except Exception:
                pass
        run_id = row.get("run_id")
        ret_df = self.dm.get_run_returns(str(run_id)) if pd.notna(run_id) else pd.DataFrame()
        if ret_df.empty:
            return float("nan")
        r = pd.to_numeric(ret_df.get("net_return", ret_df.get("gross_return", 0.0)), errors="coerce").fillna(0.0)
        eq = (1.0 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        return float(dd.min())

    def _pick_drawdown_df(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        name = runs_df["name"].astype(str).str.lower()
        fid = runs_df.get("factor_id", pd.Series([""] * len(runs_df), index=runs_df.index)).astype(str).str.lower()

        sma = self._first_match(runs_df, ((name.str.contains("sma") & ~name.str.contains("binary|continuous|discretized|router|layer 4")) | fid.eq("fac_039")))
        boll = self._first_match(runs_df, ((name.str.contains("bollinger") & ~name.str.contains("router|layer 4")) | fid.eq("fac_040")))
        router = self._first_match(runs_df, (name.str.contains("layer 4|router") | fid.eq("fac_038")))

        rows = []
        if sma is not None:
            rows.append(("Baseline SMA", self._max_dd_from_run(sma), sma.get("name", "SMA")))
        if boll is not None:
            rows.append(("Baseline Bollinger", self._max_dd_from_run(boll), boll.get("name", "Bollinger")))
        if router is not None:
            rows.append(("Layer 4 Router", self._max_dd_from_run(router), router.get("name", "Router")))
        out = pd.DataFrame(rows, columns=["Strategy", "Max Drawdown", "Source"])
        return out.dropna(subset=["Max Drawdown"]) if not out.empty else out

    def _pick_turnover_df(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        name = runs_df["name"].astype(str).str.lower()
        fid = runs_df.get("factor_id", pd.Series([""] * len(runs_df), index=runs_df.index)).astype(str).str.lower()
        rows = []
        for label, f_id, pattern in [
            ("Binary", "fac_050", "sma binary"),
            ("Continuous", "fac_051", "sma continuous"),
            ("Discretized", "fac_052", "sma discretized"),
        ]:
            row = self._first_match(runs_df, (fid.eq(f_id) | name.str.contains(pattern)))
            if row is not None and pd.notna(row.get("turnover_rate")):
                rows.append((label, float(row["turnover_rate"]), row.get("name", label)))
        out = pd.DataFrame(rows, columns=["Evolution", "Turnover Rate", "Source"])
        return out.dropna(subset=["Turnover Rate"]) if not out.empty else out

    def render(self, lang: str = "EN", theme_mode: str = "LIGHT"):
        t = TEXT[lang]
        tpl = get_plotly_template(theme_mode)
        runs_df = self.dm.get_all_runs()

        st.markdown(t["exec_dd_title"])
        st.caption(t["exec_dd_desc"])

        if runs_df.empty:
            st.warning(t["exec_missing"])
            return

        dd_df = self._pick_drawdown_df(runs_df)
        to_df = self._pick_turnover_df(runs_df)

        if not dd_df.empty and (dd_df["Strategy"] == "Layer 4 Router").any():
            router_dd = dd_df.loc[dd_df["Strategy"] == "Layer 4 Router", "Max Drawdown"].iloc[0]
            dd_df["Drawdown Savings vs Router"] = dd_df["Max Drawdown"] - router_dd

            fig_dd = px.bar(
                dd_df, x="Strategy", y="Max Drawdown", color="Strategy",
                text=dd_df["Max Drawdown"].map(lambda x: f"{x:.2%}"),
                hover_data={"Source": True, "Drawdown Savings vs Router": ":.2%"},
                template=tpl
            )
            fig_dd.update_layout(showlegend=False, yaxis_title="Max Drawdown", xaxis_title="")
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.warning(t["exec_missing"])

        st.markdown(t["exec_to_title"])
        st.caption(t["exec_to_desc"])
        if not to_df.empty:
            order = ["Binary", "Continuous", "Discretized"]
            to_df["Evolution"] = pd.Categorical(to_df["Evolution"], categories=order, ordered=True)
            to_df = to_df.sort_values("Evolution")
            fig_to = px.bar(
                to_df, x="Evolution", y="Turnover Rate", color="Evolution",
                text=to_df["Turnover Rate"].map(lambda x: f"{x:.1f}%"),
                hover_data={"Source": True}, template=tpl
            )
            fig_to.update_layout(showlegend=False, yaxis_title="Turnover Rate (%)", xaxis_title="")
            st.plotly_chart(fig_to, use_container_width=True)
        else:
            st.warning(t["exec_missing"])

        with st.expander("Debug: Ledger Preview", expanded=False):
            st.dataframe(runs_df.head(30), use_container_width=True, hide_index=True)
