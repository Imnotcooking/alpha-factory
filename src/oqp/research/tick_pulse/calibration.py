from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oqp.optimization import (
    ConstraintSpec,
    FrozenResearchInputs,
    ObjectiveSpec,
    OptimizationComparisonResult,
    OptimizationStudyResult,
    OptimizationStudyRunner,
    OptimizationStudySpec,
    OptimizationStudyStore,
    SearchBudget,
    TrialEvaluation,
    build_component_parameter_schema,
    require_dataset_fingerprint,
    stable_optimization_hash,
)
from oqp.research.tick_pulse.cpp_bridge import compute_rtv_frame


@dataclass(frozen=True)
class TickPulseCalibrationConfig:
    hypothesis: str = "relative_velocity_fade"
    n_trials: int = 50
    min_events: int = 200
    min_fold_events: int = 30
    n_folds: int = 4
    holdout_fraction: float = 0.30
    random_state: int = 42
    fast_min: int = 3
    fast_max: int = 12
    slow_min: int = 1_000
    slow_max: int = 10_000
    slow_step: int = 500
    percentile_min: float = 0.95
    percentile_max: float = 0.995
    min_fast_move_min: float = 2.0
    min_fast_move_max: float = 40.0
    min_fast_move_step: float = 0.5
    horizon_choices: tuple[int, ...] = (30, 60, 90, 120, 180, 240)
    min_success_min: float = 0.5
    min_success_max: float = 5.0
    min_success_step: float = 0.5
    sampler_id: str = "tpe"
    compare_random_baseline: bool = True


class TickPulseHeuristicOptimizer:
    """
    Optuna optimizer for the pure relative-tick-velocity heuristic.

    This deliberately does not train a predictive model. Each trial proposes a
    small set of C++-portable math parameters, evaluates the selected hypothesis
    across chronological day folds, and scores only the pre-holdout folds. The
    last holdout block of dates is reported after optimization and is not used
    by Optuna.
    """

    def __init__(
        self,
        config: TickPulseCalibrationConfig,
        *,
        store: OptimizationStudyStore | None = None,
        study_id: str | None = None,
    ):
        if config.hypothesis not in {"relative_velocity", "relative_velocity_fade"}:
            raise ValueError("Tick pulse calibration currently supports relative_velocity hypotheses only.")
        self.config = config
        self.store = store or OptimizationStudyStore()
        self.study_id = study_id
        self.study: OptimizationStudyResult | None = None
        self.optimization_comparison: OptimizationComparisonResult | None = None
        self.best_events: pd.DataFrame | None = None
        self.feature_frame: pd.DataFrame | None = None
        self._development_feature_frame: pd.DataFrame | None = None
        self._calibration_dates: np.ndarray = np.array([])
        self._holdout_dates: np.ndarray = np.array([])
        self._frozen_split_info: dict = {}
        self.backend_counts: dict[str, int] = {}

    def run(self, features: pd.DataFrame) -> dict:
        self.backend_counts = {}
        self.feature_frame = self._prepare_features(features)
        dataset_fingerprint = require_dataset_fingerprint(features)
        self._freeze_temporal_split(self.feature_frame)
        event_dates = pd.to_datetime(self.feature_frame["datetime"]).dt.normalize()
        self._development_feature_frame = self.feature_frame.loc[
            event_dates.isin(self._calibration_dates)
        ].copy()

        schema = self._parameter_schema()
        identity_payload = {
            "component": "tick_pulse_heuristic",
            "dataset_fingerprint": dataset_fingerprint,
            "config": asdict(self.config),
            "split": self._frozen_split_info,
            "schema": schema.to_dict(),
        }
        resolved_study_id = self.study_id or (
            "tick_pulse_heuristic_"
            + stable_optimization_hash(identity_payload)[:16]
        )
        spec = OptimizationStudySpec(
            study_id=resolved_study_id,
            purpose="factor_parameter",
            component_id=schema.component_id,
            sampler_id=self.config.sampler_id,
            objectives=(ObjectiveSpec("walk_forward_score", "score", "maximize"),),
            constraints=(
                ConstraintSpec("minimum_events", "events", ">=", self.config.min_events),
                ConstraintSpec("minimum_folds", "fold_count", ">=", 2),
                ConstraintSpec(
                    "minimum_events_per_fold",
                    "min_fold_events",
                    ">=",
                    self.config.min_fold_events,
                ),
            ),
            frozen_inputs=FrozenResearchInputs(
                dataset_fingerprint=dataset_fingerprint,
                holdout_fingerprint=stable_optimization_hash(
                    self._frozen_split_info
                ),
            ),
            budget=SearchBudget(max_trials=self.config.n_trials, n_jobs=1),
            seed=self.config.random_state,
            metadata={"hypothesis": self.config.hypothesis},
        )
        runner = OptimizationStudyRunner(self.store)
        if self.config.compare_random_baseline and self.config.sampler_id != "random":
            comparison = runner.run_with_random_baseline(
                spec, schema, self._evaluate_candidate
            )
        else:
            result = runner.run(spec, schema, self._evaluate_candidate)
            comparison = OptimizationComparisonResult(result, result)
        self.optimization_comparison = comparison
        self.study = comparison.challenger
        if not self.study.candidates:
            raise ValueError(
                "No tick-pulse parameter set satisfied the frozen event and fold constraints"
            )

        best_candidate = self.study.candidates[0]
        best_params = dict(best_candidate.parameters)
        best_events = self._build_event_dataset(self.feature_frame, best_params)
        calibration_events, holdout_events, fold_metrics, split_info = self._split_events(best_events)
        self.best_events = best_events

        trials = self._trial_dataframe(self.study)
        return {
            "optimizer": f"shared_{self.config.sampler_id}_walk_forward_heuristic",
            "hypothesis": self.config.hypothesis,
            "best_value": float(best_candidate.objective_values[0]),
            "best_params": best_params,
            "split": split_info,
            "fold_metrics": fold_metrics,
            "calibration_metrics": self._score_events(calibration_events),
            "holdout_metrics": self._score_events(holdout_events),
            "events": best_events,
            "trials": trials,
            "backend": best_events.attrs.get("tick_pulse_backend", "unknown"),
            "backend_counts": self.backend_counts.copy(),
            "optimization": self.study.to_dict(),
            "random_baseline": comparison.baseline.to_dict(),
            "objective_note": (
                "Objective scores chronological day folds before the final holdout. It rewards "
                "Wilson lower-bound accuracy and expected move, then penalizes fold instability. "
                "The final holdout block of dates is untouched during Optuna search."
            ),
        }

    def _evaluate_candidate(self, params: dict, context) -> TrialEvaluation:
        context.require_development_only()
        if self._development_feature_frame is None:
            raise RuntimeError("Tick-pulse development frame has not been frozen")
        events = self._build_event_dataset(self._development_feature_frame, params)
        calibration_events, _, fold_metrics, _ = self._split_events(events)
        metrics = self._score_events(calibration_events)
        fold_score = self._score_walk_forward_folds(fold_metrics)
        combined = {
            **metrics,
            **fold_score,
        }
        combined = {
            name: float(value) if np.isfinite(value) else 0.0
            for name, value in combined.items()
        }
        return TrialEvaluation(
            metrics=combined,
            fold_metrics=tuple(fold_metrics.to_dict(orient="records")),
            metadata={
                "backend": events.attrs.get("tick_pulse_backend", "unknown")
            },
        )

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "datetime", "mid_price", "tick_size_est"}
        missing = required.difference(features.columns)
        if missing:
            raise ValueError(f"Tick pulse calibration missing required columns: {sorted(missing)}")

        out = features.sort_values(["symbol", "datetime"]).copy()
        group_keys = _group_keys(out)
        out["_event_row_pos"] = out.groupby(group_keys, sort=False).cumcount()
        return out

    def _parameter_schema(self):
        midpoint = lambda low, high: (float(low) + float(high)) / 2.0
        return build_component_parameter_schema(
            "tick_pulse_relative_velocity_heuristic",
            "factor",
            {
                "fast_window_ticks": {
                    "default": int(self.config.fast_min),
                    "type": "int",
                    "low": self.config.fast_min,
                    "high": self.config.fast_max,
                    "tunable": True,
                },
                "slow_window_ticks": {
                    "default": int(self.config.slow_min),
                    "type": "int",
                    "low": self.config.slow_min,
                    "high": self.config.slow_max,
                    "step": self.config.slow_step,
                    "tunable": True,
                },
                "percentile": {
                    "default": midpoint(
                        self.config.percentile_min,
                        self.config.percentile_max,
                    ),
                    "type": "float",
                    "low": self.config.percentile_min,
                    "high": self.config.percentile_max,
                    "tunable": True,
                },
                "min_fast_move_ticks": {
                    "default": float(self.config.min_fast_move_min),
                    "type": "float",
                    "low": self.config.min_fast_move_min,
                    "high": self.config.min_fast_move_max,
                    "step": self.config.min_fast_move_step,
                    "tunable": True,
                },
                "horizon_ticks": {
                    "default": int(self.config.horizon_choices[0]),
                    "type": "categorical",
                    "choices": list(self.config.horizon_choices),
                    "tunable": True,
                },
                "min_success_ticks": {
                    "default": float(self.config.min_success_min),
                    "type": "float",
                    "low": self.config.min_success_min,
                    "high": self.config.min_success_max,
                    "step": self.config.min_success_step,
                    "tunable": True,
                },
            },
        )

    def _freeze_temporal_split(self, features: pd.DataFrame) -> None:
        feature_dates = pd.to_datetime(features["datetime"]).dt.normalize()
        unique_dates = np.array(sorted(feature_dates.dropna().unique()))
        if len(unique_dates) < 4:
            raise ValueError(
                "Tick-pulse optimization requires at least four trading days "
                "so one fixed chronological holdout can be reserved"
            )
        holdout_days = int(np.ceil(len(unique_dates) * self.config.holdout_fraction))
        holdout_days = max(1, min(holdout_days, len(unique_dates) - 2))
        self._calibration_dates = unique_dates[:-holdout_days]
        self._holdout_dates = unique_dates[-holdout_days:]
        self._frozen_split_info = {
            "calibration_days": int(len(self._calibration_dates)),
            "holdout_days": int(len(self._holdout_dates)),
            "calibration_start": str(
                pd.Timestamp(self._calibration_dates[0]).date()
            ),
            "calibration_end": str(
                pd.Timestamp(self._calibration_dates[-1]).date()
            ),
            "holdout_start": str(pd.Timestamp(self._holdout_dates[0]).date()),
            "holdout_end": str(pd.Timestamp(self._holdout_dates[-1]).date()),
            "split_type": "frozen_walk_forward_by_day",
        }

    def _build_event_dataset(self, features: pd.DataFrame, params: dict) -> pd.DataFrame:
        try:
            events = self._build_event_dataset_cpp(features, params)
            backend = "cpp"
        except Exception as exc:
            events = self._build_event_dataset_python(features, params)
            backend = "python_fallback"
            events.attrs["tick_pulse_backend_error"] = str(exc)

        events.attrs["tick_pulse_backend"] = backend
        self.backend_counts[backend] = self.backend_counts.get(backend, 0) + 1
        return events

    def _build_event_dataset_cpp(self, features: pd.DataFrame, params: dict) -> pd.DataFrame:
        fast_window = int(params["fast_window_ticks"])
        slow_window = int(params["slow_window_ticks"])
        percentile = float(params["percentile"])
        min_fast_move = float(params["min_fast_move_ticks"])
        horizon = int(params["horizon_ticks"])
        min_success = float(params["min_success_ticks"])
        min_periods = min(500, max(50, slow_window // 10))

        _, events = compute_rtv_frame(
            features,
            horizon_ticks=horizon,
            fast_window=fast_window,
            slow_window=slow_window,
            percentile=percentile,
            min_periods=min_periods,
            min_fast_move_ticks=min_fast_move,
            min_success_ticks=min_success,
            fade=self.config.hypothesis == "relative_velocity_fade",
            gap_ticks=1,
        )
        events = events.copy()
        events["pulse_fast_move_ticks"] = events["rtv_fast_move_ticks"]
        events["pulse_abs_move_ticks"] = events["rtv_abs_move_ticks"]
        events["pulse_threshold_ticks"] = events["rtv_threshold_ticks"]
        events["pulse_threshold_ratio"] = events["rtv_threshold_ratio"]
        events["pulse_direction"] = events["rtv_direction"]
        events["target"] = events["is_correct"].astype(int)
        events["expected_move_ticks"] = np.where(
            events["expected_direction"] == "Down",
            -events["future_move_ticks"],
            events["future_move_ticks"],
        )
        events = events.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        events.attrs["tick_pulse_backend"] = "cpp"
        return events

    def _build_event_dataset_python(self, features: pd.DataFrame, params: dict) -> pd.DataFrame:
        out = features.copy()
        group_keys = _group_keys(out)
        grouped = out.groupby(group_keys, sort=False)
        tick_size = out["tick_size_est"].replace(0, np.nan)

        fast_window = int(params["fast_window_ticks"])
        slow_window = int(params["slow_window_ticks"])
        percentile = float(params["percentile"])
        min_fast_move = float(params["min_fast_move_ticks"])
        horizon = int(params["horizon_ticks"])
        min_success = float(params["min_success_ticks"])

        out["pulse_fast_move_ticks"] = (out["mid_price"] - grouped["mid_price"].shift(fast_window)) / tick_size
        out["pulse_abs_move_ticks"] = out["pulse_fast_move_ticks"].abs()
        min_periods = min(500, max(50, slow_window // 10))
        out["pulse_threshold_ticks"] = grouped["pulse_abs_move_ticks"].transform(
            lambda s: s.shift(1).rolling(slow_window, min_periods=min_periods).quantile(percentile)
        )
        out["pulse_threshold_ratio"] = (
            out["pulse_abs_move_ticks"] / out["pulse_threshold_ticks"].replace(0, np.nan)
        )
        out["pulse_direction"] = np.select(
            [out["pulse_fast_move_ticks"] > 0, out["pulse_fast_move_ticks"] < 0],
            ["Up", "Down"],
            default="Flat",
        )
        out["future_datetime"] = grouped["datetime"].shift(-horizon)
        out["future_mid_price"] = grouped["mid_price"].shift(-horizon)
        out["future_move_ticks"] = (out["future_mid_price"] - out["mid_price"]) / tick_size

        if self.config.hypothesis == "relative_velocity_fade":
            out["expected_direction"] = np.select(
                [out["pulse_direction"] == "Up", out["pulse_direction"] == "Down"],
                ["Down", "Up"],
                default="Flat",
            )
        else:
            out["expected_direction"] = out["pulse_direction"]

        signal = (
            out["pulse_threshold_ticks"].notna()
            & out["pulse_direction"].isin(["Up", "Down"])
            & (out["pulse_abs_move_ticks"] >= min_fast_move)
            & (out["pulse_abs_move_ticks"] >= out["pulse_threshold_ticks"])
            & out["future_move_ticks"].notna()
        )
        correct = (
            ((out["expected_direction"] == "Up") & (out["future_move_ticks"] >= min_success))
            | ((out["expected_direction"] == "Down") & (out["future_move_ticks"] <= -min_success))
        )
        expected_move = np.where(
            out["expected_direction"] == "Down",
            -out["future_move_ticks"],
            out["future_move_ticks"],
        )

        events = out.loc[signal].copy()
        events["is_correct"] = correct.loc[events.index].astype(bool)
        events["target"] = events["is_correct"].astype(int)
        events["outcome"] = np.where(events["is_correct"], "Correct", "Failed")
        events["expected_move_ticks"] = pd.Series(expected_move, index=out.index).loc[events.index]
        events = _collapse_event_episodes(events)
        events = events.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        events.attrs["tick_pulse_backend"] = "python"
        return events

    def _split_events(self, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        events = events.sort_values(["datetime", "symbol"]).copy()
        events["_event_date"] = pd.to_datetime(events["datetime"]).dt.normalize()
        calibration_events = events.loc[
            events["_event_date"].isin(self._calibration_dates)
        ].copy()
        holdout_events = events.loc[
            events["_event_date"].isin(self._holdout_dates)
        ].copy()
        fold_metrics = self._build_day_fold_metrics(
            calibration_events, self._calibration_dates
        )
        split_info = {
            **self._frozen_split_info,
            "calibration_rows": int(len(calibration_events)),
            "holdout_rows": int(len(holdout_events)),
            "fold_count": int(len(fold_metrics)),
            "embargo_events": 0,
        }
        return (
            calibration_events.drop(columns=["_event_date"], errors="ignore"),
            holdout_events.drop(columns=["_event_date"], errors="ignore"),
            fold_metrics,
            split_info,
        )

    def _build_day_fold_metrics(self, calibration_events: pd.DataFrame, calibration_dates: np.ndarray) -> pd.DataFrame:
        if calibration_events.empty or len(calibration_dates) == 0:
            return pd.DataFrame()

        fold_count = max(1, min(self.config.n_folds, len(calibration_dates)))
        date_chunks = [chunk for chunk in np.array_split(calibration_dates, fold_count) if len(chunk)]
        rows = []
        for fold_idx, fold_dates in enumerate(date_chunks, start=1):
            fold_events = calibration_events.loc[calibration_events["_event_date"].isin(fold_dates)]
            metrics = self._score_events(fold_events)
            rows.append(
                {
                    "fold": fold_idx,
                    "start_date": str(pd.Timestamp(fold_dates[0]).date()),
                    "end_date": str(pd.Timestamp(fold_dates[-1]).date()),
                    "days": int(len(fold_dates)),
                    **metrics,
                }
            )
        return pd.DataFrame(rows)

    def _build_event_fallback_fold_metrics(self, calibration_events: pd.DataFrame) -> pd.DataFrame:
        if calibration_events.empty:
            return pd.DataFrame()

        fold_count = max(1, min(self.config.n_folds, len(calibration_events)))
        chunks = np.array_split(np.arange(len(calibration_events)), fold_count)
        rows = []
        for fold_idx, idx in enumerate(chunks, start=1):
            if len(idx) == 0:
                continue
            fold_events = calibration_events.iloc[idx]
            metrics = self._score_events(fold_events)
            rows.append(
                {
                    "fold": fold_idx,
                    "start_date": str(pd.to_datetime(fold_events["datetime"]).min().date()),
                    "end_date": str(pd.to_datetime(fold_events["datetime"]).max().date()),
                    "days": int(pd.to_datetime(fold_events["datetime"]).dt.normalize().nunique()),
                    **metrics,
                }
            )
        return pd.DataFrame(rows)

    def _score_walk_forward_folds(self, fold_metrics: pd.DataFrame) -> dict:
        if fold_metrics.empty:
            return {
                "score": -1.0,
                "fold_count": 0,
                "min_fold_events": 0,
                "min_fold_accuracy": np.nan,
                "mean_fold_accuracy": np.nan,
                "fold_accuracy_std": np.nan,
            }

        valid = fold_metrics.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["events", "accuracy", "ci_low", "avg_expected_move_ticks"]
        )
        if valid.empty:
            return {
                "score": -1.0,
                "fold_count": 0,
                "min_fold_events": 0,
                "min_fold_accuracy": np.nan,
                "mean_fold_accuracy": np.nan,
                "fold_accuracy_std": np.nan,
            }

        fold_events = valid["events"].astype(float)
        fold_accuracy = valid["accuracy"].astype(float)
        fold_ci_low = valid["ci_low"].astype(float)
        fold_move = valid["avg_expected_move_ticks"].astype(float)

        mean_ci_low = float(fold_ci_low.mean())
        accuracy_std = float(fold_accuracy.std(ddof=0))
        mean_move_bonus = 0.02 * float(np.tanh(fold_move.mean() / 5.0))
        negative_move_penalty = 0.015 * float((fold_move < 0).mean())
        weak_fold_penalty = 0.02 * float((fold_events < self.config.min_fold_events).mean())
        sample_bonus = min(0.015, 0.003 * np.log10(max(fold_events.sum(), 1.0)))
        score = mean_ci_low + mean_move_bonus + sample_bonus - (0.50 * accuracy_std) - negative_move_penalty - weak_fold_penalty

        return {
            "score": float(score),
            "fold_count": int(len(valid)),
            "min_fold_events": int(fold_events.min()),
            "min_fold_accuracy": float(fold_accuracy.min()),
            "mean_fold_accuracy": float(fold_accuracy.mean()),
            "fold_accuracy_std": accuracy_std,
        }

    @staticmethod
    def _score_events(events: pd.DataFrame) -> dict:
        total = int(len(events))
        successes = int(events["is_correct"].sum()) if total else 0
        accuracy = successes / total if total else np.nan
        ci_low, ci_high = _wilson_interval(successes, total)
        return {
            "events": total,
            "correct": successes,
            "accuracy": float(accuracy) if total else np.nan,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "avg_future_move_ticks": float(events["future_move_ticks"].mean()) if total else np.nan,
            "avg_expected_move_ticks": float(events["expected_move_ticks"].mean()) if total else np.nan,
        }

    def _trial_dataframe(
        self, result: OptimizationStudyResult | None
    ) -> pd.DataFrame:
        if result is None or not result.artifact_path:
            return pd.DataFrame()
        trials_path = Path(result.artifact_path).with_name("trials.json")
        if not trials_path.exists():
            return pd.DataFrame()
        payload = json.loads(trials_path.read_text(encoding="utf-8"))
        rows = []
        for trial in payload:
            user_attrs = dict(trial.get("user_attrs") or {})
            metrics = dict(user_attrs.get("metrics") or {})
            row = {
                "trial": trial.get("number"),
                "score": (trial.get("values") or [np.nan])[0],
                **dict(trial.get("resolved_parameters") or trial.get("params") or {}),
                **metrics,
            }
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )


def _group_keys(df: pd.DataFrame) -> list[str]:
    return ["symbol", "_session_id"] if "_session_id" in df.columns else ["symbol"]


def _collapse_event_episodes(candidates: pd.DataFrame, gap_ticks: int = 1) -> pd.DataFrame:
    row_pos_column = "_event_row_pos"
    if candidates.empty or row_pos_column not in candidates.columns:
        return candidates

    keys = _group_keys(candidates)
    pieces = []
    cluster_offset = 0
    for _, group in candidates.sort_values([*keys, row_pos_column]).groupby(keys, sort=False):
        group = group.copy()
        new_episode = group[row_pos_column].diff().gt(gap_ticks).fillna(True)
        local_cluster = new_episode.cumsum().astype(int) - 1
        group["_event_cluster_id"] = local_cluster + cluster_offset
        group["_event_cluster_size"] = group.groupby("_event_cluster_id")["_event_cluster_id"].transform("size")
        pieces.append(group[group["_event_cluster_id"].ne(group["_event_cluster_id"].shift(1))])
        cluster_offset += int(local_cluster.max()) + 1

    if not pieces:
        return candidates.iloc[0:0].copy()
    return pd.concat(pieces).sort_values("datetime").copy()


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan

    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = p_hat + z * z / (2.0 * total)
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
    return (center - margin) / denom, (center + margin) / denom
