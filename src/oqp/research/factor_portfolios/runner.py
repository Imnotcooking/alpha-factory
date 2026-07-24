"""Reusable orchestration for building and evaluating factor portfolios."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import ModuleType

import numpy as np
import pandas as pd

from oqp.optimization.contracts import stable_optimization_hash
from oqp.optimization.parameter_spaces import (
    resolve_component_parameter_schema,
    resolve_component_parameter_values,
)
from oqp.research.backtesting import (
    ExecutionModeConfig,
    ExecutionModeFactory,
    RETURN_HORIZON_AUTO,
    attach_capital_attrs,
    normalize_return_horizon,
    resolve_execution_capital,
)
from oqp.research.backtesting.margin_policy import apply_margin_utilization_cap
from oqp.research.backtesting import AlphaEvaluator
from oqp.research.contracts import (
    FactorContract,
    attach_factor_contract_attrs,
    resolve_factor_contract,
    validate_factor_market_compatibility,
)
from oqp.research.factor_portfolios.composer import (
    CompositionResult,
    FactorPortfolioComposer,
)
from oqp.research.factor_portfolios.contracts import FactorPortfolioConfig
from oqp.research.factors import load_factor_module
from oqp.research.liquidity_eligibility import (
    apply_liquidity_gate,
    ensure_liquidity_eligibility,
)
from oqp.research.parameter_schema import (
    attach_factor_parameter_attrs,
    fingerprint_parameter_payload,
    resolve_factor_parameter_schema,
    resolve_parameter_values,
)
from oqp.research.temporal_policy import (
    ensure_signal_holding_policy,
    synchronize_temporal_targets,
)
from oqp.research.strategy_routing import (
    RoutedSleeveResult,
    load_router_module,
    resolve_router_contract,
    route_sleeve_targets,
)
from oqp.research.strategy_risk_overlays import (
    StrategyRiskOverlayContract,
    apply_strategy_risk_overlay,
    load_strategy_risk_overlay_module,
)
from oqp.research.sleeves import (
    ExtractedSleeveConfig,
    PersistentSleeveConfig,
    SleeveConstructionConfig,
    build_extracted_sleeve_targets,
    build_persistent_sleeve_targets,
    build_sleeve_targets,
    load_sleeve_module,
    supports_extracted_sleeve_execution,
)
from oqp.research.success_criteria import (
    SuccessCriterionRegistry,
    attach_success_criterion_attrs,
)


@dataclass(frozen=True)
class FactorPortfolioBuildResult:
    config: FactorPortfolioConfig
    composition: CompositionResult | None
    frame: pd.DataFrame
    factor_contracts: dict[str, FactorContract]
    execution_detail: str
    sleeve_compositions: dict[str, CompositionResult]
    router_result: RoutedSleeveResult | None
    risk_overlay_contracts: dict[str, StrategyRiskOverlayContract]


class FactorPortfolioRunner:
    """Compute factor recipes, blend their scores, then reuse shared execution."""

    def __init__(self, config: FactorPortfolioConfig):
        self.config = config

    def build(
        self,
        base_frame: pd.DataFrame,
        *,
        strict_factor_contracts: bool = True,
        router_states: pd.DataFrame | None = None,
    ) -> FactorPortfolioBuildResult:
        prepared = self._prepare_base_frame(base_frame)
        prepared = self._prepare_liquidity_eligibility(prepared)
        if self.config.router is not None:
            return self._build_routed(
                prepared,
                router_states=router_states,
                strict_factor_contracts=strict_factor_contracts,
            )
        factor_frames: dict[str, pd.DataFrame] = {}
        signal_columns: dict[str, str] = {}
        factor_contracts: dict[str, FactorContract] = {}

        for spec in self.config.factors:
            module = load_factor_module(spec.factor_id)
            factor_frame, contract = self._compute_factor(
                module,
                prepared,
                factor_id=spec.factor_id,
                parameters=spec.parameters,
                strict=strict_factor_contracts,
            )
            factor_frames[spec.factor_id] = factor_frame
            signal_columns[spec.factor_id] = spec.signal_col or contract.alpha_signal_col
            factor_contracts[spec.factor_id] = contract

        self._validate_return_horizon_contracts(factor_contracts)

        composition = FactorPortfolioComposer(self.config).compose(
            prepared,
            factor_frames,
            signal_columns=signal_columns,
        )
        execution_config = ExecutionModeConfig(
            max_gross_leverage=self.config.max_gross_leverage,
            max_weight_per_asset=self.config.max_weight_per_asset,
            source_col="composite_score",
            neutralize=self.config.neutralize,
            sizing_modules=self.config.sizing_modules,
            kelly_fraction=self.config.kelly_fraction,
        )
        execution = ExecutionModeFactory.create(
            self.config.execution_mode,
            execution_config,
        ).apply(composition.frame)
        frame = execution.df
        frame, risk_overlay_contracts = self._apply_risk_overlays(frame)
        frame = self._apply_temporal_and_liquidity_policies(frame)

        reference_contract = next(iter(factor_contracts.values()))
        portfolio_contract = FactorContract(
            factor_id=self.config.strategy_id,
            evaluation_geometry=reference_contract.evaluation_geometry,
            execution_mode=self.config.execution_mode,
            alpha_signal_col="composite_score",
            execution_weight_col="final_target_weight",
            execution_lag=reference_contract.execution_lag,
            return_assumption=reference_contract.return_assumption,
            supported_markets=(self.config.market_vertical,),
            contract_source="factor_portfolio",
        )
        attach_factor_contract_attrs(frame, portfolio_contract)
        frame.attrs["strategy_id"] = self.config.strategy_id
        frame.attrs["strategy_name"] = self.config.name
        frame.attrs["market_vertical"] = self.config.market_vertical
        frame.attrs["factor_portfolio"] = self.config.to_dict()
        frame.attrs["factor_params"] = {
            "component_type": "factor_portfolio",
            "factor_portfolio": self.config.to_dict(),
        }
        self._attach_component_parameter_provenance(frame, factor_frames)
        frame.attrs["factor_metadata"] = {
            "name": self.config.name,
            "category": "Factor Portfolio",
            "economic_rationale": self.config.description,
            "component_type": "strategy",
        }
        frame.attrs["strategy_risk_overlay_contracts"] = {
            overlay_id: contract.to_dict()
            for overlay_id, contract in risk_overlay_contracts.items()
        }
        self._attach_success_criterion(frame, routed=False)
        return FactorPortfolioBuildResult(
            config=self.config,
            composition=composition,
            frame=frame,
            factor_contracts=factor_contracts,
            execution_detail=self._execution_detail_with_overlays(execution.detail),
            sleeve_compositions={},
            router_result=None,
            risk_overlay_contracts=risk_overlay_contracts,
        )

    def build_with_sleeve(
        self,
        base_frame: pd.DataFrame,
        *,
        factor_id: str,
        sleeve_id: str,
        parameters: dict | None = None,
        strict_factor_contracts: bool = True,
    ) -> FactorPortfolioBuildResult:
        """Translate one governed factor score through one frozen sleeve recipe."""

        prepared = self._prepare_base_frame(base_frame)
        prepared = self._prepare_liquidity_eligibility(prepared)
        factor_module = load_factor_module(factor_id)
        factor_frame, factor_contract = self._compute_factor(
            factor_module,
            prepared,
            factor_id=factor_id,
            parameters=parameters,
            strict=strict_factor_contracts,
        )
        factor_frame = self._attach_reusable_sleeve_panel(
            factor_frame,
            prepared=prepared,
        )
        self._validate_return_horizon_contracts(
            {factor_id: factor_contract}
        )

        factor_metadata = getattr(factor_module, "FACTOR_METADATA", {}) or {}
        orientation = str(
            getattr(factor_module, "SIGNAL_ORIENTATION", None)
            or factor_metadata.get("signal_orientation")
            or ""
        ).strip().lower()
        if orientation not in {"higher_is_bullish", "higher_is_bearish"}:
            raise ValueError(
                f"{factor_id} must declare a valid SIGNAL_ORIENTATION before "
                "entering a reusable sleeve"
            )

        sleeve_module = load_sleeve_module(sleeve_id)
        sleeve_config = sleeve_module.build_config(
            factor_id,
            market_vertical=self.config.market_vertical,
            signal_orientation=orientation,
        )
        if isinstance(sleeve_config, ExtractedSleeveConfig):
            if not supports_extracted_sleeve_execution(sleeve_config):
                support = (
                    "declares execution_supported=False"
                    if not sleeve_config.execution_supported
                    else "does not have a registered execution adapter"
                )
                raise ValueError(
                    f"{sleeve_config.sleeve_id} uses ExtractedSleeveConfig and "
                    f"{support}; extracted sleeve contracts cannot be executed "
                    "by the reusable factor-sleeve runner"
                )
        if not isinstance(
            sleeve_config,
            (
                SleeveConstructionConfig,
                PersistentSleeveConfig,
                ExtractedSleeveConfig,
            ),
        ):
            raise TypeError(
                f"{sleeve_id} build_config() returned unsupported config type "
                f"{type(sleeve_config).__name__}"
            )
        if isinstance(sleeve_config, SleeveConstructionConfig):
            if (
                sleeve_config.construction_geometry
                != factor_contract.evaluation_geometry
            ):
                raise ValueError(
                    "factor and sleeve evaluation geometries differ: "
                    f"{factor_contract.evaluation_geometry!r} != "
                    f"{sleeve_config.construction_geometry!r}"
                )
            if (
                sleeve_config.return_assumption
                != factor_contract.return_assumption
            ):
                raise ValueError(
                    "factor and sleeve return assumptions differ: "
                    f"{factor_contract.return_assumption!r} != "
                    f"{sleeve_config.return_assumption!r}"
                )
        elif isinstance(sleeve_config, PersistentSleeveConfig):
            if factor_contract.evaluation_geometry == "cross_sectional":
                pass
            else:
                raise ValueError(
                    f"{sleeve_id} persistent construction requires a "
                    f"cross-sectional factor, but {factor_id} declares "
                    f"{factor_contract.evaluation_geometry!r}"
                )
        else:
            construction_geometry = str(
                (sleeve_config.parameters or {}).get(
                    "construction_geometry"
                )
                or ""
            ).strip().lower()
            required_geometry = {
                "time_series_stateful": "time_series",
                "cross_sectional": "cross_sectional",
                "cross_sectional_stateful": "cross_sectional",
            }.get(construction_geometry)
            if required_geometry is None:
                raise ValueError(
                    f"{sleeve_id} extracted construction declares unsupported "
                    f"construction_geometry={construction_geometry!r}"
                )
            if factor_contract.evaluation_geometry != required_geometry:
                raise ValueError(
                    "factor and extracted sleeve evaluation geometries differ: "
                    f"{factor_contract.evaluation_geometry!r} != "
                    f"{construction_geometry!r}"
                )
        if not isinstance(sleeve_config, ExtractedSleeveConfig):
            sleeve_config = replace(
                sleeve_config,
                signal_col=factor_contract.alpha_signal_col,
            )
        sleeve_input = self._attest_reusable_sleeve_alignment(
            factor_frame,
            prepared=prepared,
            factor_contract=factor_contract,
            sleeve_config=sleeve_config,
        )
        if isinstance(sleeve_config, PersistentSleeveConfig):
            sleeve_input = self._prepare_persistent_sleeve_input(
                sleeve_input,
                prepared=prepared,
                config=sleeve_config,
            )
            construction = build_persistent_sleeve_targets(
                sleeve_input,
                sleeve_config,
            )
        elif isinstance(sleeve_config, ExtractedSleeveConfig):
            construction = build_extracted_sleeve_targets(
                sleeve_input,
                sleeve_config,
            )
        else:
            construction = build_sleeve_targets(sleeve_input, sleeve_config)
        alignment_attestation = dict(
            construction.positions.attrs["sleeve_alignment_attestation"]
        )

        execution_config = ExecutionModeConfig(
            max_gross_leverage=self.config.max_gross_leverage,
            max_weight_per_asset=self.config.max_weight_per_asset,
            source_col=(
                sleeve_config.output_col
                if isinstance(sleeve_config, ExtractedSleeveConfig)
                else "target_weight"
            ),
            neutralize=False,
        )
        execution = ExecutionModeFactory.create(
            "direct", execution_config
        ).apply(construction.positions)
        frame = execution.df
        frame["composite_score"] = pd.to_numeric(
            frame[factor_contract.alpha_signal_col], errors="coerce"
        )
        frame, risk_overlay_contracts = self._apply_risk_overlays(frame)
        frame = self._apply_temporal_and_liquidity_policies(frame)

        strategy_contract = FactorContract(
            factor_id=self.config.strategy_id,
            evaluation_geometry=factor_contract.evaluation_geometry,
            execution_mode="direct",
            alpha_signal_col="composite_score",
            execution_weight_col="final_target_weight",
            execution_lag=factor_contract.execution_lag,
            return_assumption=factor_contract.return_assumption,
            supported_markets=(self.config.market_vertical,),
            contract_source="factor_sleeve_strategy",
        )
        attach_factor_contract_attrs(frame, strategy_contract)
        frame.attrs["strategy_id"] = self.config.strategy_id
        frame.attrs["strategy_name"] = self.config.name
        frame.attrs["market_vertical"] = self.config.market_vertical
        frame.attrs["factor_portfolio"] = self.config.to_dict()
        frame.attrs["sleeve_id"] = sleeve_config.sleeve_id
        frame.attrs["sleeve_config"] = sleeve_config.to_dict()
        frame.attrs["sleeve_config_fingerprint"] = sleeve_config.fingerprint
        frame.attrs["sleeve_alignment_attestation"] = alignment_attestation
        frame.attrs["causal_signal_alignment_verified"] = True
        frame.attrs["causal_return_alignment_verified"] = True
        frame.attrs["factor_params"] = {
            "component_type": "factor_sleeve_strategy",
            "factor_portfolio": self.config.to_dict(),
            "sleeve_config": sleeve_config.to_dict(),
        }
        self._attach_component_parameter_provenance(
            frame, {factor_id: factor_frame}
        )
        frame.attrs["factor_metadata"] = {
            "name": self.config.name,
            "category": "Factor Sleeve Strategy",
            "economic_rationale": self.config.description,
            "component_type": "strategy",
        }
        frame.attrs["strategy_risk_overlay_contracts"] = {
            overlay_id: contract.to_dict()
            for overlay_id, contract in risk_overlay_contracts.items()
        }
        self._attach_success_criterion(frame, routed=False)
        return FactorPortfolioBuildResult(
            config=self.config,
            composition=None,
            frame=frame,
            factor_contracts={factor_id: factor_contract},
            execution_detail=self._execution_detail_with_overlays(
                f"{sleeve_config.sleeve_id}; {execution.detail}"
            ),
            sleeve_compositions={},
            router_result=None,
            risk_overlay_contracts=risk_overlay_contracts,
        )

    def _build_routed(
        self,
        prepared: pd.DataFrame,
        *,
        router_states: pd.DataFrame | None,
        strict_factor_contracts: bool,
    ) -> FactorPortfolioBuildResult:
        if self.config.router is None:
            raise ValueError("routed strategy configuration is missing a router")
        if router_states is None:
            raise ValueError("routed strategy requires a causal router state frame")

        unique_specs = {}
        for sleeve in self.config.sleeves:
            for spec in sleeve.factors:
                existing = unique_specs.get(spec.factor_id)
                if existing is not None and existing != spec:
                    raise ValueError(
                        f"routed sleeves configure {spec.factor_id} inconsistently"
                    )
                unique_specs.setdefault(spec.factor_id, spec)

        factor_frames: dict[str, pd.DataFrame] = {}
        factor_contracts: dict[str, FactorContract] = {}
        signal_columns: dict[str, str] = {}
        for factor_id, spec in unique_specs.items():
            module = load_factor_module(factor_id)
            factor_frame, contract = self._compute_factor(
                module,
                prepared,
                factor_id=factor_id,
                parameters=spec.parameters,
                strict=strict_factor_contracts,
            )
            factor_frames[factor_id] = factor_frame
            factor_contracts[factor_id] = contract
            signal_columns[factor_id] = spec.signal_col or contract.alpha_signal_col

        self._validate_routed_factor_contracts(factor_contracts)
        self._validate_return_horizon_contracts(factor_contracts)
        sleeve_compositions: dict[str, CompositionResult] = {}
        sleeve_targets: dict[str, pd.DataFrame] = {}
        execution_details: list[str] = []
        for sleeve in self.config.sleeves:
            sleeve_config = FactorPortfolioConfig(
                strategy_id=f"{self.config.strategy_id}__{sleeve.sleeve_id}",
                name=f"{self.config.name}: {sleeve.sleeve_id}",
                market_vertical=self.config.market_vertical,
                factors=sleeve.factors,
                weighting_method=sleeve.weighting_method,
                normalization=sleeve.normalization,
                missing_policy=sleeve.missing_policy,
                min_available_factors=sleeve.min_available_factors,
                winsor_limit=sleeve.winsor_limit,
                execution_mode=self.config.execution_mode,
                max_gross_leverage=self.config.max_gross_leverage,
                max_weight_per_asset=self.config.max_weight_per_asset,
                neutralize=self.config.neutralize,
                sizing_modules=self.config.sizing_modules,
                kelly_fraction=self.config.kelly_fraction,
                liquidity_policy=self.config.liquidity_policy,
                temporal_policy=self.config.temporal_policy,
                return_horizon=self.config.return_horizon,
                description=sleeve.description,
            )
            sleeve_factor_frames = {
                spec.factor_id: factor_frames[spec.factor_id]
                for spec in sleeve.factors
            }
            sleeve_signal_columns = {
                spec.factor_id: signal_columns[spec.factor_id]
                for spec in sleeve.factors
            }
            composition = FactorPortfolioComposer(sleeve_config).compose(
                prepared,
                sleeve_factor_frames,
                signal_columns=sleeve_signal_columns,
            )
            execution_config = ExecutionModeConfig(
                max_gross_leverage=self.config.max_gross_leverage,
                max_weight_per_asset=self.config.max_weight_per_asset,
                source_col="composite_score",
                neutralize=self.config.neutralize,
                sizing_modules=self.config.sizing_modules,
                kelly_fraction=self.config.kelly_fraction,
            )
            execution = ExecutionModeFactory.create(
                self.config.execution_mode,
                execution_config,
            ).apply(composition.frame)
            sleeve_compositions[sleeve.sleeve_id] = composition
            sleeve_targets[sleeve.sleeve_id] = execution.df[
                ["date", "ticker", "final_target_weight"]
            ].rename(columns={"final_target_weight": "target_weight"})
            execution_details.append(f"{sleeve.sleeve_id}: {execution.detail}")

        router_module = load_router_module(self.config.router.router_id)
        router_contract = resolve_router_contract(
            router_module,
            router_id=self.config.router.router_id,
        )
        supported = set(router_contract.supported_markets)
        if "*" not in supported and self.config.market_vertical not in supported:
            raise ValueError(
                f"{router_contract.router_id} does not support "
                f"{self.config.market_vertical}"
            )
        route_function = getattr(router_module, "route", None)
        if not callable(route_function):
            raise ValueError(f"{router_contract.router_id} must expose route()")
        router_parameters = dict(self.config.router.parameters)
        router_parameter_schema = None
        if getattr(router_module, "ROUTER_PARAMETERS", None) is not None:
            router_parameter_schema = resolve_component_parameter_schema(
                router_module,
                component_type="router",
            )
            router_parameters = resolve_component_parameter_values(
                router_parameter_schema,
                router_parameters,
            )
        allocations = route_function(
            router_states.copy(),
            sleeve_ids=tuple(sleeve_targets),
            parameters=router_parameters,
        )
        if not isinstance(allocations, pd.DataFrame):
            raise TypeError(f"{router_contract.router_id} route() must return a DataFrame")
        routed = route_sleeve_targets(
            prepared,
            sleeve_targets,
            allocations,
            router_contract,
        )
        frame = routed.frame
        frame["composite_score"] = frame["routed_target_weight"]
        frame["final_target_weight"] = frame["routed_target_weight"]
        frame["target_weight"] = frame["routed_target_weight"]
        frame["signal"] = frame["routed_target_weight"]
        frame["execution_mode"] = "direct"
        frame.attrs["execution_mode"] = "direct"
        frame, risk_overlay_contracts = self._apply_risk_overlays(frame)
        frame = self._apply_temporal_and_liquidity_policies(frame)

        reference_contract = next(iter(factor_contracts.values()))
        strategy_contract = FactorContract(
            factor_id=self.config.strategy_id,
            evaluation_geometry=reference_contract.evaluation_geometry,
            execution_mode="direct",
            alpha_signal_col="composite_score",
            execution_weight_col="final_target_weight",
            execution_lag=reference_contract.execution_lag,
            return_assumption=reference_contract.return_assumption,
            supported_markets=(self.config.market_vertical,),
            contract_source="routed_factor_portfolio",
        )
        attach_factor_contract_attrs(frame, strategy_contract)
        frame.attrs["strategy_id"] = self.config.strategy_id
        frame.attrs["strategy_name"] = self.config.name
        frame.attrs["market_vertical"] = self.config.market_vertical
        frame.attrs["factor_portfolio"] = self.config.to_dict()
        frame.attrs["router_id"] = router_contract.router_id
        frame.attrs["router_contract"] = router_contract.to_dict()
        if router_parameter_schema is not None:
            frame.attrs["router_parameter_schema"] = (
                router_parameter_schema.to_dict()
            )
            frame.attrs["router_parameter_schema_fingerprint"] = (
                router_parameter_schema.fingerprint
            )
            frame.attrs["router_parameter_values"] = router_parameters
            frame.attrs["router_parameter_values_fingerprint"] = (
                stable_optimization_hash(router_parameters)
            )
        frame.attrs["factor_params"] = {
            "component_type": "routed_strategy",
            "factor_portfolio": self.config.to_dict(),
            "router_contract": router_contract.to_dict(),
            "router_parameters": router_parameters,
        }
        self._attach_component_parameter_provenance(frame, factor_frames)
        frame.attrs["factor_metadata"] = {
            "name": self.config.name,
            "category": "Routed Strategy",
            "economic_rationale": self.config.description,
            "component_type": "strategy",
        }
        frame.attrs["strategy_risk_overlay_contracts"] = {
            overlay_id: contract.to_dict()
            for overlay_id, contract in risk_overlay_contracts.items()
        }
        self._attach_success_criterion(frame, routed=True)
        return FactorPortfolioBuildResult(
            config=self.config,
            composition=None,
            frame=frame,
            factor_contracts=factor_contracts,
            execution_detail=self._execution_detail_with_overlays(
                "; ".join(execution_details)
            ),
            sleeve_compositions=sleeve_compositions,
            router_result=routed,
            risk_overlay_contracts=risk_overlay_contracts,
        )

    def _attach_success_criterion(
        self,
        frame: pd.DataFrame,
        *,
        routed: bool,
    ) -> None:
        profile_id = self.config.success_criterion_profile
        if not profile_id:
            frame.attrs["success_criterion_status"] = "not_declared"
            return
        criterion = SuccessCriterionRegistry.load().resolve(profile_id)
        allowed_objects = {"strategy", "router"} if routed else {"strategy"}
        if criterion.research_object not in allowed_objects:
            raise ValueError(
                f"Success criterion {profile_id!r} targets "
                f"{criterion.research_object!r}, but this configuration is "
                f"{'routed' if routed else 'an unrouted strategy'}"
            )
        attach_success_criterion_attrs(frame, criterion)

    @staticmethod
    def _validate_routed_factor_contracts(
        factor_contracts: dict[str, FactorContract],
    ) -> None:
        if not factor_contracts:
            raise ValueError("routed strategy has no factor contracts")
        fields = ("evaluation_geometry", "execution_lag", "return_assumption")
        for field in fields:
            values = {getattr(contract, field) for contract in factor_contracts.values()}
            if len(values) > 1:
                raise ValueError(
                    f"routed sleeves have incompatible {field} values: {sorted(values)}"
                )

    def _validate_return_horizon_contracts(
        self,
        factor_contracts: dict[str, FactorContract],
    ) -> None:
        configured = normalize_return_horizon(self.config.return_horizon)
        if configured == RETURN_HORIZON_AUTO:
            return
        declared = {
            normalize_return_horizon(contract.return_assumption)
            for contract in factor_contracts.values()
        }
        if declared != {configured}:
            raise ValueError(
                "strategy return_horizon does not match factor contracts: "
                f"strategy={configured}, factors={sorted(declared)}"
            )

    def evaluate(
        self,
        result: FactorPortfolioBuildResult,
        *,
        db_path,
        logs_dir,
        crisis_period=None,
        split_date: str | None = "2023-01-01",
        split_mode: str = "auto",
        validation_fraction: float = 0.70,
        purge_periods: int = 0,
        embargo_periods: int = 0,
        purge_unit: str = "auto",
    ) -> str:
        evaluator = AlphaEvaluator(
            db_path=db_path,
            logs_dir=logs_dir,
            asset_class=self.config.market_vertical,
        )
        reference_contract = next(iter(result.factor_contracts.values()))
        return evaluator.run_evaluation(
            self.config.strategy_id,
            result.frame,
            crisis_period=crisis_period,
            split_date=split_date or "2023-01-01",
            split_mode=split_mode,
            validation_fraction=validation_fraction,
            purge_periods=purge_periods,
            embargo_periods=embargo_periods,
            purge_unit=purge_unit,
            factor_category="Factor Portfolio",
            strategy_geometry=reference_contract.evaluation_geometry,
            alpha_signal_col="composite_score",
        )

    def _compute_factor(
        self,
        module: ModuleType,
        base_frame: pd.DataFrame,
        *,
        factor_id: str,
        parameters: dict | None,
        strict: bool,
    ) -> tuple[pd.DataFrame, FactorContract]:
        validate_factor_market_compatibility(
            module,
            self.config.market_vertical,
            factor_id=factor_id,
        )
        factor_input = base_frame.copy()
        factor_input.attrs.update(base_frame.attrs)
        if hasattr(module, "prepare_data"):
            factor_input = module.prepare_data(factor_input)
        supplied_parameters = dict(parameters or {})
        if getattr(module, "FACTOR_PARAMETERS", None) is not None:
            parameter_schema = resolve_factor_parameter_schema(module)
            execution_parameters = resolve_parameter_values(
                parameter_schema,
                supplied_parameters,
            )
        else:
            execution_parameters = supplied_parameters
        factor_frame = module.compute(factor_input, **execution_parameters)
        if not isinstance(factor_frame, pd.DataFrame):
            raise TypeError(f"{factor_id} compute() must return a pandas DataFrame")
        factor_frame.attrs.update(base_frame.attrs)
        factor_frame.attrs["market_vertical"] = self.config.market_vertical
        attach_factor_parameter_attrs(
            factor_frame,
            module,
            supplied_parameters=execution_parameters,
        )
        factor_metadata = getattr(module, "FACTOR_METADATA", None)
        if factor_metadata is not None:
            if not isinstance(factor_metadata, dict):
                raise ValueError(f"{factor_id} FACTOR_METADATA must be a dict")
            factor_frame.attrs["factor_metadata"] = dict(factor_metadata)
        temporal_policy = getattr(module, "TEMPORAL_POLICY", None)
        if temporal_policy is not None:
            if not isinstance(temporal_policy, dict):
                raise ValueError(f"{factor_id} TEMPORAL_POLICY must be a dict")
            factor_frame.attrs["temporal_policy_overrides"] = dict(
                temporal_policy
            )
        contract = resolve_factor_contract(
            module,
            factor_frame,
            factor_id=factor_id,
            requested_execution_mode="auto",
            default_return_assumption=str(
                base_frame.attrs.get("execution_assumption")
                or base_frame.attrs.get("return_assumption")
                or "custom_forward_return"
            ),
            market_vertical=self.config.market_vertical,
            strict=strict,
        )
        attach_factor_contract_attrs(factor_frame, contract)
        factor_frame.attrs["market_vertical"] = self.config.market_vertical
        for key in ("data_frequency", "return_horizon"):
            if base_frame.attrs.get(key) not in (None, ""):
                factor_frame.attrs[key] = base_frame.attrs[key]
        return factor_frame, contract

    @staticmethod
    def _attach_component_parameter_provenance(
        frame: pd.DataFrame,
        factor_frames: dict[str, pd.DataFrame],
    ) -> None:
        components: dict[str, dict] = {}
        for factor_id, factor_frame in factor_frames.items():
            schema_fingerprint = factor_frame.attrs.get(
                "factor_parameter_schema_fingerprint"
            )
            if not schema_fingerprint:
                continue
            components[factor_id] = {
                "schema_version": factor_frame.attrs.get(
                    "factor_parameter_schema_version"
                ),
                "schema_fingerprint": schema_fingerprint,
                "values": factor_frame.attrs.get("factor_parameter_values", {}),
                "values_fingerprint": factor_frame.attrs.get(
                    "factor_parameter_values_fingerprint"
                ),
                "overrides": factor_frame.attrs.get(
                    "factor_parameter_overrides", {}
                ),
            }
        if not components:
            return
        frame.attrs["factor_parameter_components"] = components
        frame.attrs["factor_parameter_components_fingerprint"] = (
            fingerprint_parameter_payload(components)
        )

    @staticmethod
    def _attest_reusable_sleeve_alignment(
        factor_frame: pd.DataFrame,
        *,
        prepared: pd.DataFrame,
        factor_contract: FactorContract,
        sleeve_config: (
            SleeveConstructionConfig
            | PersistentSleeveConfig
            | ExtractedSleeveConfig
        ),
    ) -> pd.DataFrame:
        """Prove that a decision-row target is paired with its future return.

        Daily factor data stores the close-time signal on row ``t`` and attaches
        the return earned from the next executable open to that same row.  A
        reusable sleeve must therefore construct the target on row ``t`` and
        must not shift it again merely because execution occurs at the next
        open.
        """

        if "forward_return" not in factor_frame.columns:
            raise ValueError(
                "reusable sleeve execution requires the factor panel to carry "
                "its causally attached forward_return"
            )

        lag = factor_contract.execution_lag
        try:
            factor_return = normalize_return_horizon(
                factor_contract.return_assumption
            )
        except ValueError:
            factor_return = factor_contract.return_assumption
        source_horizon_raw = (
            prepared.attrs.get("execution_assumption")
            or prepared.attrs.get("return_horizon")
        )
        source_horizon: str | None = None
        if source_horizon_raw not in (None, "", RETURN_HORIZON_AUTO):
            try:
                source_horizon = normalize_return_horizon(
                    str(source_horizon_raw)
                )
            except ValueError:
                source_horizon = None

        if lag == "next_open":
            frequency = (
                str(prepared.attrs.get("data_frequency") or "")
                .strip()
                .lower()
                .replace("-", "_")
            )
            if frequency in {"1d", "day"}:
                frequency = "daily"
            if frequency != "daily":
                raise ValueError(
                    "next-open reusable sleeve alignment requires daily data; "
                    f"received data_frequency={frequency or 'missing'!r}"
                )
            if factor_return not in {
                "close_signal_next_open_to_close",
                "close_signal_next_open_to_next_open",
            }:
                raise ValueError(
                    "next-open reusable sleeve alignment requires a named "
                    "close-signal/next-open return horizon; received "
                    f"{factor_contract.return_assumption!r}"
                )
            if source_horizon != factor_return:
                raise ValueError(
                    "factor and dataset return horizons are not causally "
                    "aligned: "
                    f"factor={factor_return!r}, "
                    f"dataset={source_horizon_raw!r}"
                )
            if (
                isinstance(sleeve_config, SleeveConstructionConfig)
                and sleeve_config.execution_delay_periods != 0
            ):
                raise ValueError(
                    f"{sleeve_config.sleeve_id} would add "
                    f"{sleeve_config.execution_delay_periods} extra row shift(s) "
                    "to a next-open return already attached to the decision row"
                )
            signal_row_semantics = "decision_close"
        elif lag == "already_lagged":
            signal_row_semantics = "pre_aligned_execution_decision_row"
        else:
            raise ValueError(
                f"{sleeve_config.sleeve_id} supports explicit already-lagged "
                "scores or causally aligned daily next-open factors; "
                f"{factor_contract.factor_id} declares execution_lag={lag!r}"
            )

        out = factor_frame.copy()
        out.attrs.update(factor_frame.attrs)
        out.attrs["causal_signal_alignment_verified"] = True
        out.attrs["causal_return_alignment_verified"] = True
        out.attrs["sleeve_alignment_attestation"] = {
            "schema_version": 1,
            "verified": True,
            "factor_id": factor_contract.factor_id,
            "sleeve_id": sleeve_config.sleeve_id,
            "data_frequency": str(
                prepared.attrs.get("data_frequency") or "unknown"
            ),
            "factor_execution_lag": lag,
            "factor_return_assumption": factor_return,
            "dataset_return_horizon": source_horizon,
            "signal_row_semantics": signal_row_semantics,
            "target_row_semantics": (
                "same_decision_row_paired_with_attached_forward_return"
            ),
            "additional_row_shift_periods": 0,
            "forward_return_col": "forward_return",
            "future_return_used_for_selection": False,
        }
        return out

    @staticmethod
    def _attach_reusable_sleeve_panel(
        factor_frame: pd.DataFrame,
        *,
        prepared: pd.DataFrame,
    ) -> pd.DataFrame:
        """Restore market-panel columns that a pure factor may omit.

        Pure factor implementations are allowed to return only keys and their
        score.  Sleeve construction and evaluation still need the independently
        prepared market data, so missing columns are joined only after the
        factor has finished computing its signal.
        """

        keys = ["date", "ticker"]
        missing_keys = sorted(set(keys).difference(factor_frame.columns))
        if missing_keys:
            raise ValueError(
                "reusable sleeve factor output is missing key columns: "
                f"{missing_keys}"
            )
        if factor_frame.duplicated(keys).any():
            raise ValueError(
                "reusable sleeve factor output requires unique date/ticker rows"
            )
        attrs = dict(factor_frame.attrs)
        missing_columns = [
            column
            for column in prepared.columns
            if column not in factor_frame.columns and column not in keys
        ]
        if missing_columns:
            out = factor_frame.merge(
                prepared[[*keys, *missing_columns]],
                on=keys,
                how="left",
                sort=False,
                validate="one_to_one",
            )
        else:
            out = factor_frame.copy()
        out.attrs.update(attrs)
        out.attrs["sleeve_market_panel_columns_restored"] = missing_columns
        return out

    @staticmethod
    def _prepare_persistent_sleeve_input(
        factor_frame: pd.DataFrame,
        *,
        prepared: pd.DataFrame,
        config: PersistentSleeveConfig,
    ) -> pd.DataFrame:
        """Derive the frozen, causal inputs required by persistent sleeves."""

        required_keys = {config.date_col, config.product_col, config.signal_col}
        missing_keys = sorted(required_keys.difference(factor_frame.columns))
        if missing_keys:
            raise ValueError(
                "persistent sleeve factor output is missing columns: "
                f"{missing_keys}"
            )

        attrs = dict(factor_frame.attrs)
        out = factor_frame.copy()
        reference = prepared.rename(
            columns={
                "date": config.date_col,
                "ticker": config.product_col,
            }
        )
        support_columns = ("close", "sector", "liquidity_eligible")
        for column in support_columns:
            if column in out.columns or column not in reference.columns:
                continue
            out = out.merge(
                reference[
                    [config.date_col, config.product_col, column]
                ],
                on=[config.date_col, config.product_col],
                how="left",
                sort=False,
                validate="one_to_one",
            )

        if "close" not in out.columns:
            raise ValueError(
                "persistent sleeve execution requires close prices to derive "
                "causal trailing volatility"
            )

        out[config.date_col] = pd.to_datetime(
            out[config.date_col],
            errors="coerce",
        ).dt.normalize()
        out[config.product_col] = (
            out[config.product_col].astype("string").str.strip()
        )
        if (
            out[config.date_col].isna().any()
            or out[config.product_col].isna().any()
            or out[config.product_col].eq("").any()
        ):
            raise ValueError(
                "persistent sleeve input contains invalid date/product keys"
            )
        if out.duplicated([config.date_col, config.product_col]).any():
            raise ValueError(
                "persistent sleeve input requires unique date/product rows"
            )
        out = out.sort_values(
            [config.product_col, config.date_col],
            kind="mergesort",
        ).reset_index(drop=True)

        if "liquidity_eligible" in out.columns:
            liquidity = out["liquidity_eligible"]
            if pd.api.types.is_bool_dtype(liquidity.dtype):
                liquidity = liquidity.fillna(False).astype(bool)
            elif pd.api.types.is_numeric_dtype(liquidity.dtype):
                liquidity = (
                    pd.to_numeric(liquidity, errors="coerce")
                    .fillna(0.0)
                    .ne(0.0)
                )
            else:
                liquidity = (
                    liquidity.astype("string")
                    .str.strip()
                    .str.lower()
                    .isin({"1", "true", "t", "yes", "y"})
                )
            liquidity_source = "liquidity_eligible"
        else:
            liquidity = pd.Series(True, index=out.index, dtype=bool)
            liquidity_source = "default_true_missing_liquidity_eligible"

        signal = pd.to_numeric(
            out[config.signal_col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        out[config.signal_col] = signal
        out[config.rank_eligible_col] = liquidity & signal.notna()
        out[config.tradable_col] = liquidity

        close = pd.to_numeric(out["close"], errors="coerce").where(
            lambda values: values.gt(0.0)
        )
        products = out[config.product_col]
        close_return = close.groupby(products, sort=False).pct_change(
            fill_method=None
        )
        lagged_close_return = close_return.groupby(
            products,
            sort=False,
        ).shift(1)
        trailing_volatility = lagged_close_return.groupby(
            products,
            sort=False,
        ).transform(
            lambda values: values.rolling(
                window=20,
                min_periods=20,
            ).std(ddof=0)
        )
        out[config.volatility_col] = trailing_volatility.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if "sector" in out.columns:
            sector = out["sector"]
        else:
            sector = pd.Series("Unknown", index=out.index, dtype="string")
        out[config.sector_col] = (
            sector.astype("string").str.strip().fillna("Unknown")
        )
        out.loc[out[config.sector_col].eq(""), config.sector_col] = "Unknown"

        out.attrs.update(attrs)
        out.attrs["persistent_sleeve_input_contract"] = {
            "schema_version": 1,
            "rank_eligible_rule": (
                "liquidity_eligible AND finite_non_null_factor_signal"
            ),
            "tradable_rule": "liquidity_eligible",
            "liquidity_source": liquidity_source,
            "trailing_volatility_price": "close",
            "trailing_volatility_return": "close_to_close",
            "trailing_volatility_window_sessions": 20,
            "trailing_volatility_min_observations": 20,
            "trailing_volatility_lag_sessions": 1,
            "execution_sector_source": (
                "sector" if "sector" in out.columns else "Unknown"
            ),
            "future_return_used": False,
        }
        return out

    def _apply_risk_overlays(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, StrategyRiskOverlayContract]]:
        out = frame
        contracts: dict[str, StrategyRiskOverlayContract] = {}
        for spec in self.config.risk_overlays:
            module = load_strategy_risk_overlay_module(spec.overlay_id)
            result = apply_strategy_risk_overlay(
                out,
                module,
                overlay_id=spec.overlay_id,
                parameters=spec.parameters,
                market_vertical=self.config.market_vertical,
            )
            out = result.frame
            contracts[spec.overlay_id] = result.contract
        return out, contracts

    def _execution_detail_with_overlays(self, detail: str) -> str:
        if not self.config.risk_overlays:
            return detail
        overlay_ids = ", ".join(
            spec.overlay_id for spec in self.config.risk_overlays
        )
        return f"{detail}; strategy risk overlays: {overlay_ids}"

    def _prepare_liquidity_eligibility(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not ({"volume", "vol"} & set(frame.columns)):
            frame.attrs["liquidity_policy_status"] = "deferred_missing_volume"
            return frame
        if "initial_capital" not in frame.attrs:
            attach_capital_attrs(
                frame,
                resolve_execution_capital(asset_class=self.config.market_vertical),
            )
        frame.attrs["max_weight_per_asset"] = self.config.max_weight_per_asset or 0.05
        return ensure_liquidity_eligibility(
            frame,
            market_vertical=self.config.market_vertical,
            initial_capital=float(frame.attrs["initial_capital"]),
            capital_currency=str(frame.attrs["capital_currency"]),
            max_position_weight=self.config.max_weight_per_asset or 0.05,
            overrides=self.config.liquidity_policy or None,
        )

    def _apply_temporal_and_liquidity_policies(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        candidate_col = next(
            (
                column
                for column in (
                    "final_target_weight",
                    "routed_target_weight",
                    "target_weight",
                    "signal",
                )
                if column in frame.columns
            ),
            None,
        )
        if candidate_col is None:
            return frame
        out = ensure_signal_holding_policy(
            frame,
            candidate_col=candidate_col,
            overrides=self.config.temporal_policy or None,
        )
        out = synchronize_temporal_targets(out, effective_col=candidate_col)
        if "liquidity_eligible" in out.columns:
            out = apply_liquidity_gate(out)
        return apply_margin_utilization_cap(
            out,
            market_vertical=self.config.market_vertical,
            source_weight_col=candidate_col,
            max_margin_utilization=self.config.max_margin_utilization,
        )

    @staticmethod
    def _prepare_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "ticker" not in out.columns and "symbol" in out.columns:
            out["ticker"] = out["symbol"].astype(str)
        if "date" not in out.columns and "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
        if "close" not in out.columns and "last_price" in out.columns:
            out["close"] = pd.to_numeric(out["last_price"], errors="coerce")
        required = {"date", "ticker", "close"}
        missing = sorted(required - set(out.columns))
        if missing:
            raise ValueError(
                f"factor portfolio data is missing required columns: {', '.join(missing)}"
            )
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if out["date"].isna().any():
            raise ValueError("factor portfolio data contains invalid dates")
        out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
        if out.duplicated(["date", "ticker"]).any():
            raise ValueError("factor portfolio data must have one row per date and ticker")
        out.attrs.update(frame.attrs)
        return out


__all__ = ["FactorPortfolioBuildResult", "FactorPortfolioRunner"]
