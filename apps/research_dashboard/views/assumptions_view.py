from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import BASE_DIR, DB_PATH, LOGS_DIR, TEXT  # noqa: E402

try:
    from oqp.contracts.market_vertical import ASSET_TAXONOMY
except Exception:  # pragma: no cover - keeps standalone dashboard launches alive.
    ASSET_TAXONOMY = {}

try:
    from oqp.execution.transaction_costs import TransactionCostRegistry
except Exception:  # pragma: no cover - dashboard can still render legacy assumptions.
    TransactionCostRegistry = None

try:
    from views.system_health_view import SystemHealthView
except Exception:  # pragma: no cover - assumptions view can run without health module.
    SystemHealthView = None


class AssumptionsView:
    """Render the recorded assumption manifest for a research run."""

    def __init__(self, data_manager, assumptions_dir: str | os.PathLike[str] | None = None):
        self.dm = data_manager
        self.assumptions_dir = Path(assumptions_dir) if assumptions_dir else Path(LOGS_DIR) / "assumptions"

    def render(self, run_id: str, run_metadata: pd.Series, lang: str = "EN") -> None:
        copy = TEXT.get(lang, TEXT["EN"])
        st.markdown(f"### {copy.get('assumptions_title', copy.get('tab_assumptions', 'Assumptions'))}")

        index = self.manifest_index()
        selected_manifest = self.load_manifest(run_id)
        selected_run_id = run_id

        if not index.empty:
            options = index["run_id"].astype(str).tolist()
            if run_id not in options:
                options = [run_id, *options]
            default_index = options.index(run_id) if run_id in options else 0
            selected_run_id = st.selectbox(
                copy.get("assumptions_select_run", "Inspect run"),
                options=options,
                index=default_index,
                format_func=lambda value: self._format_run_option(value, index, run_metadata),
                key=f"assumptions_manifest_select_{run_id}",
            )
            selected_manifest = self.load_manifest(selected_run_id)

        if selected_manifest is None:
            path_hint = self.candidate_paths(selected_run_id)[0]
            reconstructed = self.reconstruct_legacy_manifest(selected_run_id, run_metadata)
            if reconstructed is not None:
                st.warning(
                    copy.get(
                        "assumptions_reconstructed",
                        "No saved JSON manifest exists for this legacy run, so this view is reconstructed from the research ledger.",
                    )
                )
                st.caption(f"{copy.get('assumptions_expected_path', 'Expected JSON path')}: `{self._display_path(path_hint)}`")
                selected_manifest = reconstructed, path_hint
            else:
                st.warning(copy.get("assumptions_missing", "No assumption manifest found for this run."))
                st.code(str(path_hint))
                if not index.empty:
                    st.markdown(f"#### {copy.get('assumptions_available', 'Available manifests')}")
                    st.dataframe(
                        index.loc[:, ["run_id", "factor_id", "asset_class", "execution_mode", "modified_at"]],
                        use_container_width=True,
                        hide_index=True,
                        height=260,
                    )
                return

        if selected_manifest is None:
            return

        manifest, manifest_path = selected_manifest
        if not manifest_path.exists() and manifest.get("manifest_source") == "db_reconstructed_legacy":
            st.caption(f"{copy.get('assumptions_manifest_path', 'Manifest')}: `{copy.get('assumptions_not_saved', 'not saved; reconstructed in memory')}`")
        else:
            st.caption(f"{copy.get('assumptions_manifest_path', 'Manifest')}: `{self._display_path(manifest_path)}`")

        self._render_topline(manifest, copy)
        self._render_success_criterion(manifest, copy)
        self._render_transaction_cost_readiness(manifest, copy)
        if manifest.get("manifest_source") == "db_reconstructed_legacy":
            self._render_section(
                copy.get("assumptions_reconstruction", "Reconstruction Note"),
                manifest.get("reconstruction_note", {}),
                copy,
            )
        self._render_section(copy.get("assumptions_data", "Data"), manifest.get("data", {}), copy)
        self._render_data_health_assumptions(manifest, copy)
        self._render_section(copy.get("assumptions_signal", "Signal & execution"), manifest.get("signal_and_execution_mode", {}), copy)
        self._render_section(copy.get("assumptions_engine", "Execution engine"), manifest.get("execution_engine", {}), copy)
        if manifest.get("risk_and_allocation"):
            self._render_section(
                copy.get("assumptions_risk", "Risk & allocation"),
                manifest.get("risk_and_allocation", {}),
                copy,
            )
        if manifest.get("costs_and_slippage"):
            self._render_costs_and_slippage(
                copy.get("assumptions_costs", "Costs & slippage"),
                manifest.get("costs_and_slippage", {}),
                copy,
            )
        self._render_section(copy.get("assumptions_liquidity", "Liquidity policy"), manifest.get("liquidity_policy", {}), copy)

        if manifest.get("option_contract_selection"):
            self._render_section(
                copy.get("assumptions_option_selection", "Option contract selection"),
                manifest.get("option_contract_selection", {}),
                copy,
            )

        self._render_section(copy.get("assumptions_realized", "Realized summary"), manifest.get("realized_summary", {}), copy)

        raw_text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
        with st.expander(copy.get("assumptions_raw_json", "Raw JSON")):
            st.download_button(
                copy.get("assumptions_download", "Download manifest"),
                data=raw_text,
                file_name=manifest_path.name,
                mime="application/json",
                key=f"download_assumptions_{selected_run_id}",
            )
            st.json(manifest)

    def manifest_index(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for directory in self.assumption_directories():
            if not directory.exists():
                continue
            for path in sorted(directory.glob("assumptions_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                payload = self._read_json(path)
                if not isinstance(payload, dict):
                    continue
                rows.append(
                    {
                        "run_id": str(payload.get("run_id") or path.stem.removeprefix("assumptions_")),
                        "factor_id": str(payload.get("factor_id") or ""),
                        "asset_class": str(payload.get("asset_class") or ""),
                        "execution_mode": str(
                            (payload.get("signal_and_execution_mode") or {}).get("execution_mode")
                            or ""
                        ),
                        "modified_at": pd.to_datetime(path.stat().st_mtime, unit="s"),
                        "path": str(path),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["run_id", "factor_id", "asset_class", "execution_mode", "modified_at", "path"])
        return pd.DataFrame(rows).sort_values("modified_at", ascending=False).reset_index(drop=True)

    def load_manifest(self, run_id: str) -> tuple[dict[str, Any], Path] | None:
        for path in self.candidate_paths(run_id):
            payload = self._read_json(path)
            if isinstance(payload, dict):
                return payload, path
        return None

    def candidate_paths(self, run_id: str) -> list[Path]:
        filename = f"assumptions_{run_id}.json"
        return [directory / filename for directory in self.assumption_directories()]

    def reconstruct_legacy_manifest(self, run_id: str, run_metadata: pd.Series) -> dict[str, Any] | None:
        record = self._run_record(run_id, run_metadata)
        if not record:
            return None

        factor_params = self._parse_json_dict(record.get("factor_params"))
        asset_class = str(record.get("asset_class") or record.get("market_vertical") or "")
        market_vertical = str(record.get("market_vertical") or asset_class)
        signal = {
            "alpha_signal_col": record.get("alpha_signal_col"),
            "execution_weight_col": record.get("execution_weight_col"),
            "execution_signal_col_used": record.get("execution_weight_col") or record.get("alpha_signal_col"),
            "execution_mode": record.get("execution_mode"),
            "execution_assumption": record.get("execution_assumption"),
            "return_assumption": record.get("return_assumption"),
            "execution_lag": record.get("execution_lag"),
        }
        traded_tickers = str(record.get("traded_tickers") or "")
        traded_ticker_count = len([item for item in traded_tickers.split(",") if item.strip()])

        return {
            "manifest_source": "db_reconstructed_legacy",
            "run_id": run_id,
            "factor_id": record.get("factor_id"),
            "asset_class": asset_class,
            "factor_contract": {
                "contract_source": record.get("factor_contract_source"),
                "evaluation_geometry": record.get("evaluation_geometry"),
                "alpha_signal_col": record.get("alpha_signal_col"),
                "execution_weight_col": record.get("execution_weight_col"),
                "execution_mode": record.get("execution_mode"),
                "execution_lag": record.get("execution_lag"),
                "return_assumption": record.get("return_assumption"),
            },
            "factor_params": factor_params,
            "market_taxonomy": ASSET_TAXONOMY.get(market_vertical, ASSET_TAXONOMY.get(asset_class, {})),
            "reconstruction_note": {
                "source": "backtest_runs database row",
                "reason": "This run was created before assumption JSON artifacts were enabled.",
                "confidence": "High for fields stored in the run ledger; unknown engine internals are left blank.",
            },
            "data": {
                "source_path": record.get("source_path"),
                "frequency": record.get("data_frequency"),
                "dataset_id": record.get("dataset_id"),
                "universe_id": record.get("universe_id"),
                "dataset_role": record.get("dataset_role"),
                "tradability": record.get("data_tradability"),
                "price_source": record.get("data_price_source"),
                "roll_model": record.get("data_roll_model"),
                "liquidity_model": record.get("data_liquidity_model"),
                "execution_reality": record.get("data_execution_reality"),
                "vendor": record.get("data_vendor"),
            },
            "signal_and_execution_mode": signal,
            "execution_engine": {
                "engine": "legacy_db_reconstruction",
                "tca_model": "see realized cost fields",
                "margin_model": "",
                "price_limit_enabled": "",
                "price_limit_model": "",
                "t1_enabled": "",
                "hurst_input_col": "",
                "hurst_default": "",
                "lot_mode": record.get("execution_lot_mode"),
                "initial_capital": record.get("initial_capital"),
                "capital_currency": record.get("capital_currency"),
                "capital_profile": record.get("capital_profile"),
                "capital_source": record.get("capital_source"),
                "min_trade_weight_delta": record.get("min_trade_weight_delta"),
            },
            "liquidity_policy": {
                "min_daily_traded_value": factor_params.get("min_avg_traded_value"),
                "direct_weight_row_grid_preserved": record.get("execution_mode") == "direct",
                "avg_daily_cost_bps": record.get("avg_daily_cost_bps"),
                "total_exchange_fees": record.get("total_exchange_fees"),
                "total_slippage_cost": record.get("total_slippage_cost"),
                "total_execution_cost": record.get("total_execution_cost"),
            },
            "costs_and_slippage": {
                "summary": (
                    "Legacy reconstruction: cost assumptions are inferred from ledger "
                    "fields. For current futures runs, the engine records the explicit "
                    "0.5 tick per-side slippage overlay and instrument fee source."
                ),
                "tca_model": "see legacy engine row",
                "fixed_slippage_ticks_per_side": "",
                "round_trip_slippage_ticks_assumed": "",
                "exchange_fee_source": "legacy DB realized cost fields",
                "avg_daily_cost_bps": record.get("avg_daily_cost_bps"),
                "total_exchange_fees": record.get("total_exchange_fees"),
                "total_slippage_cost": record.get("total_slippage_cost"),
                "total_execution_cost": record.get("total_execution_cost"),
                "avg_round_trip_fee_bps": record.get("avg_round_trip_fee_bps"),
                "avg_round_trip_fee_ticks": record.get("avg_round_trip_fee_ticks"),
                "fee_constrained_rate": record.get("fee_constrained_rate"),
                "lot_constrained_rate": record.get("lot_constrained_rate"),
            },
            "realized_summary": {
                "returns_file_path": record.get("returns_file_path"),
                "trading_days": "",
                "annualized_return": record.get("annualized_return"),
                "sharpe_ratio": record.get("sharpe_ratio"),
                "max_drawdown": record.get("max_drawdown"),
                "avg_daily_turnover": record.get("turnover_rate"),
                "avg_daily_cost_bps": record.get("avg_daily_cost_bps"),
                "total_trades": record.get("total_trades"),
                "universe_size": record.get("universe_size"),
                "traded_ticker_count": traded_ticker_count,
                "traded_tickers_preview": self._preview_csv(traded_tickers),
            },
        }

    def assumption_directories(self) -> list[Path]:
        roots = [
            self.assumptions_dir,
            Path(LOGS_DIR) / "assumptions",
            Path(LOGS_DIR) / "alpha_lab" / "assumptions",
        ]
        if Path(LOGS_DIR).name == "alpha_lab":
            roots.append(Path(LOGS_DIR).parent / "assumptions")
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.expanduser().resolve()
            if resolved not in seen:
                unique.append(resolved)
                seen.add(resolved)
        return unique

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _run_record(self, run_id: str, run_metadata: pd.Series) -> dict[str, Any]:
        record = pd.Series(dtype=object)
        if self.dm is not None and hasattr(self.dm, "get_run_record"):
            try:
                record = self.dm.get_run_record(run_id)
            except Exception:
                record = pd.Series(dtype=object)
        if record.empty:
            record = run_metadata
        return {
            str(key): self._clean_value(value)
            for key, value in record.to_dict().items()
        }

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _parse_json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else {"value": payload}
        except Exception:
            return {"value": value}

    @staticmethod
    def _preview_csv(value: str, limit: int = 18) -> str:
        items = [item.strip() for item in str(value or "").split(",") if item.strip()]
        if not items:
            return ""
        preview = ", ".join(items[:limit])
        if len(items) > limit:
            preview = f"{preview}, ... (+{len(items) - limit} more)"
        return preview

    def _render_topline(self, manifest: dict[str, Any], copy: dict[str, Any]) -> None:
        signal = manifest.get("signal_and_execution_mode") or {}
        realized = manifest.get("realized_summary") or {}
        cols = st.columns(4)
        cols[0].metric(copy.get("assumptions_asset", "Asset"), str(manifest.get("asset_class") or ""))
        cols[1].metric(copy.get("assumptions_mode", "Mode"), str(signal.get("execution_mode") or ""))
        cols[2].metric(copy.get("assumptions_alpha_col", "Alpha"), str(signal.get("alpha_signal_col") or ""))
        cols[3].metric(copy.get("assumptions_trades", "Trades"), self._format_value(realized.get("total_trades", "")))

    def _render_transaction_cost_readiness(
        self,
        manifest: dict[str, Any],
        copy: dict[str, Any],
    ) -> None:
        status = self._transaction_cost_status(manifest)
        if status is None:
            return

        st.markdown(
            f"#### {copy.get('assumptions_cost_readiness', 'Transaction Cost Readiness')}"
        )
        rows = [
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_market", "Market"
                ),
                copy.get("assumptions_cost_value", "Value"): status["market_vertical"],
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_profile", "Profile"
                ),
                copy.get("assumptions_cost_value", "Value"): status["profile_id"],
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_source", "Source"
                ),
                copy.get("assumptions_cost_value", "Value"): (
                    copy.get("assumptions_cost_source_frozen", "Frozen with run")
                    if status["frozen_with_run"]
                    else copy.get(
                        "assumptions_cost_source_current",
                        "Current default; not frozen with run",
                    )
                ),
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_use_case", "Claim level"
                ),
                copy.get("assumptions_cost_value", "Value"): status["use_case"],
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_schedule", "Schedule status"
                ),
                copy.get("assumptions_cost_value", "Value"): status["profile_status"],
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_completeness", "Completeness"
                ),
                copy.get("assumptions_cost_value", "Value"): status["completeness"],
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_engine_support", "Engine support"
                ),
                copy.get("assumptions_cost_value", "Value"): status["engine_support"],
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_net_ready", "Research-net ready"
                ),
                copy.get("assumptions_cost_value", "Value"): self._yes_no(
                    status["research_net_ready"], copy
                ),
            },
            {
                copy.get("assumptions_cost_field", "Field"): copy.get(
                    "assumptions_cost_production_ready", "Production ready"
                ),
                copy.get("assumptions_cost_value", "Value"): self._yes_no(
                    status["production_ready"], copy
                ),
            },
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if not status["frozen_with_run"]:
            st.warning(
                copy.get(
                    "assumptions_cost_historical_unfrozen",
                    "This historical run did not freeze a transaction-cost profile. "
                    "The panel shows the current default for this market and does not validate the old net result.",
                )
            )

        if status["frozen_with_run"] and status["production_ready"]:
            st.success(
                copy.get(
                    "assumptions_cost_ready_production",
                    "This run used a profile approved for research-net and production claims.",
                )
            )
        elif status["frozen_with_run"] and status["research_net_ready"]:
            st.info(
                copy.get(
                    "assumptions_cost_ready_research",
                    "This profile supports research-net results but is not approved for production claims.",
                )
            )
        elif status["frozen_with_run"] and status["gross_only"]:
            st.warning(
                copy.get(
                    "assumptions_cost_gross_only",
                    "This is a gross-only run. It must not be reported as net or production-ready.",
                )
            )
        elif status["frozen_with_run"]:
            st.error(
                copy.get(
                    "assumptions_cost_not_net_ready",
                    "The fee schedule is not fully wired into the engine, so net performance is not validated.",
                )
            )

        limitations = status["limitations"]
        if limitations and not status["production_ready"]:
            st.markdown(
                f"**{copy.get('assumptions_cost_required_work', 'Required before promotion')}**"
            )
            for item in limitations:
                st.markdown(f"- {item}")

        fingerprint = status.get("fingerprint")
        if fingerprint:
            fingerprint_label = (
                copy.get("assumptions_cost_fingerprint", "Frozen profile fingerprint")
                if status["frozen_with_run"]
                else copy.get(
                    "assumptions_cost_current_fingerprint",
                    "Current default profile fingerprint",
                )
            )
            st.caption(
                f"{fingerprint_label}: `{fingerprint}`"
            )

    def _render_success_criterion(
        self,
        manifest: dict[str, Any],
        copy: dict[str, Any],
    ) -> None:
        status = self._success_criterion_status(manifest)
        if status is None:
            return
        st.markdown(
            f"#### {copy.get('assumptions_success_criterion', 'Primary Success Criterion')}"
        )
        rows = [
            {
                "Field": "Status",
                "Value": status["status"],
            },
            {
                "Field": "Profile",
                "Value": status["profile_id"],
            },
            {
                "Field": "Research object",
                "Value": status["research_object"],
            },
            {
                "Field": "Decision sample",
                "Value": status["decision_sample"],
            },
            {
                "Field": "Primary metric",
                "Value": status["primary_metric"],
            },
            {
                "Field": "Frozen comparator",
                "Value": status["comparator_metric"],
            },
            {
                "Field": "Minimum improvement",
                "Value": status["minimum_improvement"],
            },
            {
                "Field": "Absolute floor",
                "Value": status["absolute_floor"],
            },
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if status["economic_question"]:
            st.caption(status["economic_question"])

        decision = status["status"]
        if decision == "pass":
            st.success("The recorded metrics satisfy the frozen criterion.")
        elif decision == "fail":
            st.error("The recorded metrics do not satisfy the frozen criterion.")
        elif decision == "incomplete":
            missing = ", ".join(status["missing_metrics"])
            st.warning(
                "The criterion cannot be decided because required evidence is "
                f"missing or non-finite: {missing or 'see evaluation record'}."
            )
        elif decision == "declared_not_evaluated":
            st.info(
                "The objective was frozen before the run, but its complete "
                "decision metrics have not yet been attached."
            )
        else:
            st.warning(
                "No primary success criterion was frozen for this run. Its "
                "metrics are exploratory and cannot support a promotion claim."
            )

        if status["gates"]:
            st.markdown("**Hard gates**")
            st.dataframe(
                pd.DataFrame(status["gates"]),
                use_container_width=True,
                hide_index=True,
            )
        if status["profile_fingerprint"]:
            st.caption(
                "Frozen criterion fingerprint: "
                f"`{status['profile_fingerprint']}`"
            )

    @staticmethod
    def _success_criterion_status(
        manifest: dict[str, Any],
    ) -> dict[str, Any] | None:
        payload = manifest.get("success_criterion")
        if not isinstance(payload, dict):
            return None
        definition = payload.get("definition") or {}
        evaluation = payload.get("evaluation") or {}
        if not isinstance(definition, dict):
            definition = {}
        if not isinstance(evaluation, dict):
            evaluation = {}
        primary = definition.get("primary") or {}
        if not isinstance(primary, dict):
            primary = {}
        gates = evaluation.get("gates") or definition.get("gates") or []
        if not isinstance(gates, list):
            gates = []
        missing_metrics = evaluation.get("missing_metrics") or []
        if isinstance(missing_metrics, str):
            missing_metrics = [missing_metrics]
        return {
            "status": str(
                evaluation.get("decision")
                or payload.get("status")
                or "not_declared"
            ),
            "profile_id": str(payload.get("profile_id") or ""),
            "profile_fingerprint": str(
                payload.get("profile_fingerprint") or ""
            ),
            "research_object": str(definition.get("research_object") or ""),
            "decision_sample": str(definition.get("decision_sample") or ""),
            "economic_question": str(
                definition.get("economic_question") or ""
            ),
            "primary_metric": str(primary.get("metric") or ""),
            "comparator_metric": str(primary.get("comparator_metric") or ""),
            "minimum_improvement": primary.get("minimum_improvement"),
            "absolute_floor": primary.get("absolute_floor"),
            "gates": gates,
            "missing_metrics": list(missing_metrics),
        }

    @staticmethod
    def _transaction_cost_status(manifest: dict[str, Any]) -> dict[str, Any] | None:
        costs = manifest.get("costs_and_slippage") or {}
        if not isinstance(costs, dict):
            return None
        assumptions = costs.get("profile_assumptions") or {}
        if not isinstance(assumptions, dict):
            assumptions = {}
        profile_id = str(costs.get("profile_id") or assumptions.get("profile_id") or "").strip()
        if not profile_id:
            market_vertical = str(
                manifest.get("asset_class")
                or (manifest.get("data") or {}).get("market_vertical")
                or ""
            ).strip()
            if not market_vertical or TransactionCostRegistry is None:
                return None
            try:
                profile = TransactionCostRegistry.load().resolve(market_vertical)
            except Exception:
                return None
            return {
                "market_vertical": profile.market_vertical,
                "profile_id": profile.profile_id,
                "use_case": "not_recorded",
                "profile_status": profile.status,
                "completeness": profile.completeness,
                "engine_support": profile.engine_support,
                "research_net_ready": profile.research_net_ready,
                "production_ready": profile.production_ready,
                "gross_only": False,
                "limitations": list(profile.readiness_actions()),
                "fingerprint": profile.fingerprint,
                "frozen_with_run": False,
            }
        limitations = assumptions.get("limitations") or []
        if isinstance(limitations, str):
            limitations = [limitations]
        return {
            "market_vertical": str(
                assumptions.get("market_vertical")
                or manifest.get("asset_class")
                or ""
            ),
            "profile_id": profile_id,
            "use_case": str(costs.get("use_case") or "unknown"),
            "profile_status": str(
                costs.get("profile_status") or assumptions.get("status") or "unknown"
            ),
            "completeness": str(
                costs.get("profile_completeness")
                or assumptions.get("completeness")
                or "unknown"
            ),
            "engine_support": str(
                costs.get("engine_support")
                or assumptions.get("engine_support")
                or "unknown"
            ),
            "research_net_ready": bool(
                costs.get("research_net_ready", assumptions.get("research_net_ready", False))
            ),
            "production_ready": bool(
                costs.get("production_ready", assumptions.get("production_ready", False))
            ),
            "gross_only": bool(costs.get("gross_only", False)),
            "limitations": [str(item) for item in limitations if str(item).strip()],
            "fingerprint": str(costs.get("profile_fingerprint") or ""),
            "frozen_with_run": True,
        }

    @staticmethod
    def _yes_no(value: bool, copy: dict[str, Any]) -> str:
        return (
            copy.get("assumptions_cost_yes", "Yes")
            if value
            else copy.get("assumptions_cost_no", "No")
        )

    def _render_section(self, title: str, payload: Any, copy: dict[str, Any]) -> None:
        st.markdown(f"#### {title}")
        frame = self._key_value_frame(payload)
        if frame.empty:
            st.info(copy.get("assumptions_no_values", "No recorded values."))
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)

    def _render_costs_and_slippage(self, title: str, payload: Any, copy: dict[str, Any]) -> None:
        st.markdown(f"#### {title}")
        if not isinstance(payload, dict) or not payload:
            st.info(copy.get("assumptions_no_values", "No recorded values."))
            return
        summary = payload.get("summary") or payload.get("paragraph") or payload.get("costs_paragraph")
        if summary:
            st.info(str(summary))
        table_payload = {key: value for key, value in payload.items() if key not in {"summary", "paragraph", "costs_paragraph"}}
        frame = self._key_value_frame(table_payload)
        if frame.empty:
            st.info(copy.get("assumptions_no_values", "No recorded values."))
        else:
            st.dataframe(frame, use_container_width=True, hide_index=True)

    def _render_data_health_assumptions(self, manifest: dict[str, Any], copy: dict[str, Any]) -> None:
        st.markdown(f"#### {copy.get('assumptions_data_health', 'Data Health')}")
        st.caption(
            copy.get(
                "assumptions_data_health_note",
                "Accounting views may forward-fill stale marks, while alpha views must block synthetic inputs.",
            )
        )
        if SystemHealthView is None:
            st.info(copy.get("assumptions_data_health_missing", "Data health snapshot is unavailable."))
            return

        try:
            snapshot = SystemHealthView._load_snapshot(str(BASE_DIR), str(DB_PATH))
        except Exception as exc:
            st.info(f"{copy.get('assumptions_data_health_missing', 'Data health snapshot is unavailable.')}: {exc}")
            return

        folders = snapshot.get("data_folders", pd.DataFrame())
        if folders.empty or "fresh_pct" not in folders.columns:
            st.info(copy.get("assumptions_data_health_missing", "Data health snapshot is unavailable."))
            return

        data_section = manifest.get("data") or {}
        source_path = str(data_section.get("source_path") or "")
        asset_class = str(manifest.get("asset_class") or data_section.get("asset_class") or "")
        scoped = folders.dropna(subset=["fresh_pct"]).copy()
        if "asset_class" in scoped.columns:
            scoped = scoped[scoped["asset_class"].astype(str).ne("CORE")]
        if asset_class and "asset_class" in scoped.columns:
            matched = scoped[scoped["asset_class"].astype(str).eq(asset_class)]
            if not matched.empty:
                scoped = matched
        if source_path and "sample_file" in scoped.columns:
            file_matched = scoped[scoped["sample_file"].astype(str).map(lambda value: bool(value and value in source_path))]
            if not file_matched.empty:
                scoped = file_matched

        if scoped.empty:
            st.info(copy.get("assumptions_data_health_missing", "Data health snapshot is unavailable."))
            return

        display_cols = [
            "freshness_status",
            "asset_class",
            "timeframe",
            "sample_file",
            "fresh_pct",
            "synthetic_pct",
            "expired_rows",
            "fill_policy",
        ]
        display = scoped[[col for col in display_cols if col in scoped.columns]].rename(
            columns={
                "freshness_status": copy.get("assumptions_health_status", "Status"),
                "asset_class": copy.get("assumptions_health_asset", "Asset"),
                "timeframe": copy.get("assumptions_health_timeframe", "Timeframe"),
                "sample_file": copy.get("assumptions_health_file", "Sample file"),
                "fresh_pct": copy.get("assumptions_health_fresh", "Fresh %"),
                "synthetic_pct": copy.get("assumptions_health_synthetic", "Synthetic %"),
                "expired_rows": copy.get("assumptions_health_expired", "Expired rows"),
                "fill_policy": copy.get("assumptions_health_policy", "Fill policy"),
            }
        )
        for col in [
            copy.get("assumptions_health_fresh", "Fresh %"),
            copy.get("assumptions_health_synthetic", "Synthetic %"),
        ]:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce").map(
                    lambda value: "" if pd.isna(value) else f"{value:.1%}"
                )
        expired_col = copy.get("assumptions_health_expired", "Expired rows")
        if expired_col in display.columns:
            display[expired_col] = pd.to_numeric(display[expired_col], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{int(value):,}"
            )
        st.dataframe(display, use_container_width=True, hide_index=True, height=180)

    def _key_value_frame(self, payload: Any) -> pd.DataFrame:
        if not isinstance(payload, dict) or not payload:
            return pd.DataFrame(columns=["Assumption", "Value"])
        rows = [
            {"Assumption": str(key), "Value": self._format_value(value)}
            for key, value in payload.items()
        ]
        return pd.DataFrame(rows)

    def _format_run_option(self, run_id: str, index: pd.DataFrame, run_metadata: pd.Series) -> str:
        rows = index[index["run_id"].astype(str).eq(str(run_id))]
        if rows.empty:
            factor = str(run_metadata.get("factor_id") or "")
            name = str(run_metadata.get("name") or "")
            return f"{run_id} | {factor or name or 'selected run'}"
        row = rows.iloc[0]
        return f"{row['run_id']} | {row['factor_id']} | {row['asset_class']} | {row['execution_mode']}"

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        text = str(value)
        if len(text) > 700:
            return f"{text[:700]}..."
        return text

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path(BASE_DIR).resolve()))
        except Exception:
            return str(path)
