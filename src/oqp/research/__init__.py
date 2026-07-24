"""Research-layer contracts and engines shared by alpha-lab surfaces.

The package-level compatibility API is dependency-lazy.  Importing a focused
module such as :mod:`oqp.research.ml.regimes.filtering` must not initialize every
backtester, dataframe engine, ML backend, and dashboard helper in research.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "oqp.research.contracts": frozenset(
        {
            "FactorContract",
            "attach_factor_contract_attrs",
            "resolve_factor_contract",
            "resolve_factor_supported_markets",
            "validate_factor_market_compatibility",
        }
    ),
    "oqp.research.backtesting": frozenset(
        {
            "AbsoluteReturnBenchmark",
            "BaseBenchmark",
            "BenchmarkFactory",
            "BuyAndHoldBenchmark",
            "CSI300Benchmark",
            "DEFAULT_MIN_TRADE_WEIGHT_DELTA",
            "DirectExecutionMode",
            "ExecutionCapitalProfile",
            "ExecutionModeConfig",
            "ExecutionModeFactory",
            "ExecutionModeResult",
            "ExecutionTradePolicy",
            "DXYBenchmark",
            "HSIBenchmark",
            "NanhuaBenchmark",
            "QQQBenchmark",
            "RiskFreeRateBenchmark",
            "RiskDeskExecutionMode",
            "SPYBenchmark",
            "SectorNeutralBenchmark",
            "StatArbExecutionMode",
            "TickerBenchmark",
            "attach_capital_attrs",
            "attach_trade_policy_attrs",
            "dynamic_equal_weight_benchmark",
            "resolve_default_benchmark_policy",
            "resolve_execution_capital",
            "resolve_execution_trade_policy",
        }
    ),
    "oqp.research.artifacts": frozenset(
        {
            "FileFingerprint",
            "ModelArtifactStore",
            "StoredArtifact",
            "fingerprint_file",
            "normalize_workspace_path",
            "sha256_file",
            "slugify",
        }
    ),
    "oqp.research.datasets": frozenset(
        {
            "DatasetTradabilityProfile",
            "attach_dataset_tradability_attrs",
            "base_symbol_from_ticker",
            "infer_dataset_tradability",
        }
    ),
    "oqp.research.dataset_fingerprints": frozenset(
        {
            "DATASET_MANIFEST_SCHEMA_VERSION",
            "DEFAULT_DATASET_MANIFEST_ROOT",
            "DatasetFingerprintError",
            "DatasetFrameProfile",
            "DatasetManifest",
            "DatasetSourceFingerprint",
            "DatasetVerificationResult",
            "attach_dataset_manifest_attrs",
            "discover_dataset_files",
            "ensure_dataset_manifest_attrs",
            "load_dataset_manifest",
            "register_dataset_manifest",
            "verify_dataset_manifest",
        }
    ),
    "oqp.research.liquidity_eligibility": frozenset(
        {
            "LIQUIDITY_POLICY_VERSION",
            "LiquidityEligibilityPolicy",
            "LiquidityEligibilitySummary",
            "apply_liquidity_gate",
            "assess_liquidity_eligibility",
            "ensure_liquidity_eligibility",
            "liquidity_eligible_rows",
            "resolve_liquidity_policy",
        }
    ),
    "oqp.research.temporal_policy": frozenset(
        {
            "HOLD_FACTOR_MANAGED",
            "HOLD_FIXED_PERIOD",
            "HOLD_SESSION_FLAT",
            "HOLD_UNTIL_NEXT_DECISION",
            "SIGNAL_EVENT_DRIVEN",
            "SIGNAL_EVERY_BAR",
            "SIGNAL_FIXED_INTERVAL",
            "SIGNAL_SESSION_CLOSE",
            "TEMPORAL_POLICY_VERSION",
            "SignalHoldingPolicy",
            "TemporalPolicySummary",
            "apply_signal_holding_policy",
            "ensure_signal_holding_policy",
            "resolve_signal_holding_policy",
            "synchronize_temporal_targets",
            "temporal_metric_rows",
        }
    ),
    "oqp.research.success_criteria": frozenset(
        {
            "DEFAULT_SUCCESS_CRITERIA_PATH",
            "SUCCESS_CRITERIA_SCHEMA_VERSION",
            "CriterionDecision",
            "GateEvaluation",
            "MetricGate",
            "SuccessCriterionRegistry",
            "SuccessCriterionResult",
            "SuccessCriterionSpec",
            "attach_success_criterion_attrs",
            "attach_success_criterion_result_attrs",
            "evaluate_success_criterion",
            "success_criterion_manifest_payload",
        }
    ),
    "oqp.research.factor_definitions": frozenset(
        {
            "ExpectedHoldingHorizon",
            "FACTOR_DEFINITION_SCHEMA_VERSION",
            "FactorDefinition",
            "FactorDefinitionInspection",
            "inspect_factor_definition",
            "resolve_factor_definition",
        }
    ),
    "oqp.research.factor_deduplication": frozenset(
        {
            "DuplicateThresholds",
            "FactorQualityProfile",
            "QualityTolerances",
            "RepresentativeSelection",
            "is_near_duplicate_edge",
            "select_cluster_representative",
        }
    ),
    "oqp.research.predictive_evidence": frozenset(
        {
            "CausalAlignmentError",
            "PREDICTIVE_EVIDENCE_SCHEMA_VERSION",
            "PredictiveEvidenceBundle",
            "PredictiveEvidenceConfig",
            "build_predictive_evidence",
            "load_predictive_evidence_bundle",
            "write_predictive_evidence_bundle",
        }
    ),
    "oqp.research.sleeves": frozenset(
        {
            "ANNUALIZATION_DAYS",
            "CONDITIONAL_BEHAVIOUR_SCHEMA_VERSION",
            "CONDITION_DEFINITIONS",
            "SLEEVE_CONSTRUCTION_SCHEMA_VERSION",
            "SLEEVE_EVIDENCE_SCHEMA_VERSION",
            "STANDALONE_SLEEVE_TEST_SCHEMA_VERSION",
            "ConditionalBehaviourBundle",
            "ConditionalBehaviourConfig",
            "ObservableConditionsBundle",
            "SleeveAlignmentError",
            "SleeveConstructionConfig",
            "SleeveConstructionResult",
            "SleeveEvidenceBundle",
            "StandaloneSleeveTestBundle",
            "StandaloneSleeveTestConfig",
            "build_sleeve_evidence",
            "build_conditional_behaviour",
            "build_observable_conditions",
            "build_sleeve_targets",
            "build_standalone_sleeve_test",
            "execute_intraday_session_targets",
            "load_sleeve_evidence_bundle",
            "load_conditional_behaviour_bundle",
            "load_observable_conditions_bundle",
            "load_standalone_sleeve_test_bundle",
            "summarize_executed_positions",
            "write_sleeve_evidence_bundle",
            "write_conditional_behaviour_bundle",
            "write_observable_conditions_bundle",
            "write_standalone_sleeve_test_bundle",
        }
    ),
    "oqp.research.strategy_routing": frozenset(
        {
            "ATTAINABLE_STRATEGIES",
            "ORACLE_STRATEGY",
            "ROUTER_HYPOTHESIS_SCHEMA_VERSION",
            "RouterHypothesisConfig",
            "RouterHypothesisEvidenceBundle",
            "audit_router_readiness",
            "build_router_hypothesis_evidence",
            "load_router_hypothesis_evidence_bundle",
            "write_router_hypothesis_evidence_bundle",
            "write_router_readiness_snapshot",
        }
    ),
    "oqp.research.diagnostics": frozenset(
        {"compute_ic_decay", "compute_shap_regime_dna", "list_feature_columns"}
    ),
    "oqp.research.evaluation": frozenset(
        {
            "AlphaMetricEvaluator",
            "EvaluationGeometry",
            "EvaluationMetricResult",
            "StrategyGeometryClassifier",
        }
    ),
    "oqp.research.model_registry": frozenset(
        {
            "ModelArtifactRecord",
            "build_data_fingerprint",
            "ensure_model_registry_tables",
            "latest_model_artifact",
            "list_model_artifacts",
            "record_from_artifact",
            "register_model_artifact",
        }
    ),
    "oqp.research.ml": frozenset(
        {
            "BaseMLModel",
            "FeatureGovernanceConfig",
            "GMMHMM",
            "GaussianHMM",
            "LGBMModel",
            "LGBMModelConfig",
            "LightGBMRegressorTrainer",
            "MODEL_CATALOG_VERSION",
            "MLExperimentResult",
            "MLModelFactory",
            "ModelRuntimeStatus",
            "PurgedMDAConfig",
            "SupervisedModelBase",
            "StudentTHMM",
            "ValidationConfig",
            "WalkForwardConfig",
            "XGBoostModelConfig",
            "XGBoostRegressorTrainer",
            "XGBoostTrainingEngine",
            "build_purged_time_folds",
            "compute_feature_governance",
            "compute_oos_mda",
            "detect_feature_columns",
            "ensure_ml_experiment_table",
            "latest_ml_experiment",
            "list_ml_experiments",
            "list_matrix_files",
            "load_matrix",
            "rank_ic_score",
            "register_failed_ml_experiment",
            "register_ml_experiment",
            "research_experiment_catalog",
            "research_model_catalog",
            "probe_model_runtime",
            "resolve_model_artifact_path",
            "require_model_runtime",
            "tag_feature_family",
        }
    ),
    "oqp.research.splits": frozenset(
        {
            "EvaluationSplitPolicy",
            "EvaluationSplitResult",
            "build_chronological_split",
        }
    ),
    "oqp.research.ml.state_space": frozenset(
        {
            "DualKalmanRegression",
            "DualKalmanRegressionConfig",
            "StateSpaceArtifact",
            "StateSpaceFilter",
            "StateSpaceSchema",
            "coefficient_columns",
            "summarize_dual_kalman_output",
        }
    ),
    "oqp.research.statistics": frozenset(
        {
            "AlphaStatisticalTester",
            "MultipleTestingAdjustment",
            "StatisticalEvidence",
            "benjamini_hochberg_q_values",
            "bonferroni_p_value",
            "holm_bonferroni_adjust",
            "sharpe_p_value_from_returns",
            "significance_label",
            "stable_trial_hash",
        }
    ),
    "oqp.research.trials": frozenset(
        {
            "EvidenceTicketRecord",
            "ResearchTrialRecord",
            "ensure_research_evidence_ticket_ledger",
            "ensure_research_trial_ledger",
            "get_evidence_ticket",
            "list_evidence_tickets",
            "make_evidence_ticket_id",
            "record_evidence_ticket",
            "record_research_trial",
            "refresh_multiple_testing_adjustments",
            "update_evidence_ticket_status",
        }
    ),
    "oqp.research_runtime": frozenset(
        {"AlphaResearchRuntimePaths", "alpha_research_runtime_paths"}
    ),
    "oqp.research.tick_pulse": frozenset(
        {
            "PulseThresholds",
            "TICK_GATEKEEPER_FEATURES",
            "TICK_ML_FEATURES",
            "TICK_ML_MODEL_VERSION",
            "TickGatekeeperConfig",
            "TickGatekeeperResearchEngine",
            "TickPulseCalibrationConfig",
            "TickPulseHeuristicOptimizer",
            "TickXGBoostBayesianOptimizer",
            "TickXGBoostConfig",
            "TickXGBoostResearchEngine",
            "build_pulse_features",
            "build_pulse_features_fast",
            "build_tick_ml_study_params",
            "compute_rtv_frame",
            "contract_summary",
            "discover_daily_universe_files",
            "ensure_tick_ml_tables",
            "filter_ranked_assets",
            "flag_pulses",
            "load_daily_universe",
            "load_tick_ml_study",
            "load_tick_scope",
            "load_ticks",
            "make_tick_ml_study_key",
            "rank_daily_asset_volatility",
            "save_tick_ml_study",
            "score_probability_threshold",
        }
    ),
}

_EXPORT_MODULE_BY_NAME: dict[str, str] = {}
for _module_name, _export_names in _MODULE_EXPORTS.items():
    for _export_name in _export_names:
        if _export_name in _EXPORT_MODULE_BY_NAME:
            raise RuntimeError(f"duplicate oqp.research export: {_export_name}")
        _EXPORT_MODULE_BY_NAME[_export_name] = _module_name

__all__ = sorted(_EXPORT_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULE_BY_NAME[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
