from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oqp.research.state_space.base_filter import StateSpaceFilter, StateSpaceSchema


__all__ = ["DualKalmanRegression", "DualKalmanRegressionConfig"]


@dataclass(frozen=True)
class DualKalmanRegressionConfig:
    """
    Dynamic linear regression with random-walk coefficients.

    Observation equation:
        y_t = x_t beta_t + epsilon_t

    State equation:
        beta_t = beta_{t-1} + eta_t

    The "dual" use here is practical rather than mystical: the filter updates
    both the latent regression coefficients and their uncertainty online.
    """

    schema: StateSpaceSchema
    include_intercept: bool = True
    process_noise: float = 1e-5
    observation_noise: float = 1e-3
    initial_state_mean: tuple[float, ...] | None = None
    initial_state_covariance: float = 1.0
    min_abs_observation_variance: float = 1e-12
    prefix: str = "dkf"

    def __post_init__(self) -> None:
        self.schema.validate()
        if self.process_noise <= 0:
            raise ValueError("process_noise must be positive.")
        if self.observation_noise <= 0:
            raise ValueError("observation_noise must be positive.")
        if self.initial_state_covariance <= 0:
            raise ValueError("initial_state_covariance must be positive.")


class DualKalmanRegression(StateSpaceFilter):
    """Panel-safe Kalman regression for adaptive hedge ratios/exposures."""

    def __init__(self, config: DualKalmanRegressionConfig):
        self.config = config
        self.schema = config.schema

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [self.schema.date_col, self.schema.y_col, *self.schema.x_cols, *self.schema.group_cols]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        work = df.copy()
        work[self.schema.date_col] = pd.to_datetime(work[self.schema.date_col])
        work["_dkf_input_order"] = np.arange(len(work))
        work = work.sort_values([*self.schema.group_cols, self.schema.date_col, "_dkf_input_order"])

        if self.schema.group_cols:
            frames = [
                self._filter_group(group)
                for _, group in work.groupby(list(self.schema.group_cols), sort=False, dropna=False)
            ]
            out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            out = self._filter_group(work)

        if out.empty:
            return out
        out = out.sort_values("_dkf_input_order").drop(columns=["_dkf_input_order"]).reset_index(drop=True)
        return out

    def _filter_group(self, group: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        schema = self.schema
        coef_names = self._coefficient_names()
        state_dim = len(coef_names)

        beta = self._initial_state(state_dim)
        covariance = np.eye(state_dim, dtype=float) * float(cfg.initial_state_covariance)
        process_cov = np.eye(state_dim, dtype=float) * float(cfg.process_noise)

        rows = []
        for _, row in group.iterrows():
            x_vec = self._design_vector(row)
            y_value = row[schema.y_col]
            valid = np.isfinite(y_value) and np.isfinite(x_vec).all()

            pred_beta = beta.copy()
            pred_cov = covariance + process_cov
            predicted_y = float(x_vec @ pred_beta) if valid else np.nan
            innovation = float(y_value - predicted_y) if valid else np.nan
            innovation_var = float(x_vec @ pred_cov @ x_vec + cfg.observation_noise) if valid else np.nan
            innovation_var = max(innovation_var, cfg.min_abs_observation_variance) if valid else np.nan

            if valid:
                kalman_gain = (pred_cov @ x_vec) / innovation_var
                beta = pred_beta + kalman_gain * innovation
                covariance = pred_cov - np.outer(kalman_gain, x_vec) @ pred_cov
                covariance = 0.5 * (covariance + covariance.T)
            else:
                beta = pred_beta
                covariance = pred_cov

            posterior_y = float(x_vec @ beta) if np.isfinite(x_vec).all() else np.nan
            residual = float(y_value - posterior_y) if np.isfinite(y_value) and np.isfinite(posterior_y) else np.nan
            residual_std = float(np.sqrt(max(innovation_var, cfg.min_abs_observation_variance))) if valid else np.nan

            out_row = {
                "_dkf_input_order": int(row["_dkf_input_order"]),
                schema.date_col: row[schema.date_col],
                schema.y_col: y_value,
                f"{cfg.prefix}_predicted_y": posterior_y,
                f"{cfg.prefix}_residual": residual,
                f"{cfg.prefix}_innovation": innovation,
                f"{cfg.prefix}_innovation_std": residual_std,
                f"{cfg.prefix}_innovation_z": innovation / residual_std if residual_std and np.isfinite(residual_std) else np.nan,
                f"{cfg.prefix}_state_uncertainty": float(np.trace(covariance)),
                f"{cfg.prefix}_observation_loglik": self._gaussian_loglik(innovation, innovation_var) if valid else np.nan,
            }

            for col in schema.group_cols:
                out_row[col] = row[col]
            for feature, value in zip(coef_names, beta):
                out_row[f"{cfg.prefix}_beta_{feature}"] = float(value)
            for idx, feature in enumerate(coef_names):
                out_row[f"{cfg.prefix}_beta_{feature}_var"] = float(covariance[idx, idx])
            rows.append(out_row)

        result = pd.DataFrame(rows)
        result = self._add_parameter_drift(result, coef_names)
        return result

    def _coefficient_names(self) -> list[str]:
        names = list(self.schema.x_cols)
        if self.config.include_intercept:
            names = ["intercept", *names]
        return names

    def _design_vector(self, row: pd.Series) -> np.ndarray:
        values = [float(row[col]) for col in self.schema.x_cols]
        if self.config.include_intercept:
            values = [1.0, *values]
        return np.asarray(values, dtype=float)

    def _initial_state(self, state_dim: int) -> np.ndarray:
        if self.config.initial_state_mean is None:
            return np.zeros(state_dim, dtype=float)
        initial = np.asarray(self.config.initial_state_mean, dtype=float)
        if len(initial) != state_dim:
            raise ValueError(f"initial_state_mean length {len(initial)} != state dimension {state_dim}")
        return initial.copy()

    def _add_parameter_drift(self, result: pd.DataFrame, coef_names: list[str]) -> pd.DataFrame:
        if result.empty:
            return result
        beta_cols = [f"{self.config.prefix}_beta_{name}" for name in coef_names]
        result = result.copy()
        result[f"{self.config.prefix}_beta_l1_change"] = result[beta_cols].diff().abs().sum(axis=1).fillna(0.0)
        return result

    @staticmethod
    def _gaussian_loglik(innovation: float, innovation_var: float) -> float:
        return float(-0.5 * (np.log(2.0 * np.pi * innovation_var) + (innovation * innovation) / innovation_var))
