"""Constrained convex allocation using explicit alpha, risk, and trading costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ConvexAllocationConfig:
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.0
    gross_limit: float = 1.0
    max_weight_per_asset: float | None = 0.10
    long_only: bool = False
    net_target: float | None = 0.0
    min_net_exposure: float | None = None
    max_net_exposure: float | None = None
    covariance_eigen_floor: float = 1e-8
    solver_order: tuple[str, ...] = ("OSQP", "CLARABEL", "SCS")

    def __post_init__(self) -> None:
        if float(self.risk_aversion) < 0:
            raise ValueError("risk_aversion cannot be negative")
        if float(self.turnover_penalty) < 0:
            raise ValueError("turnover_penalty cannot be negative")
        if float(self.gross_limit) <= 0:
            raise ValueError("gross_limit must be positive")
        if self.max_weight_per_asset is not None and float(
            self.max_weight_per_asset
        ) <= 0:
            raise ValueError("max_weight_per_asset must be positive or null")
        if float(self.covariance_eigen_floor) < 0:
            raise ValueError("covariance_eigen_floor cannot be negative")
        if (
            self.min_net_exposure is not None
            and self.max_net_exposure is not None
            and float(self.min_net_exposure) > float(self.max_net_exposure)
        ):
            raise ValueError("min_net_exposure cannot exceed max_net_exposure")
        if not self.solver_order:
            raise ValueError("solver_order cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class AllocationResult:
    weights: pd.Series
    status: str
    solver: str
    objective_value: float
    estimated_variance: float
    estimated_alpha: float
    turnover: float
    config_fingerprint: str
    diagnostics: dict[str, Any]


class ConvexPortfolioAllocator:
    """Solve one rebalance allocation without estimating alpha or covariance."""

    def __init__(self, config: ConvexAllocationConfig | None = None) -> None:
        self.config = config or ConvexAllocationConfig()

    def allocate(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        *,
        previous_weights: pd.Series | None = None,
        linear_trading_costs: pd.Series | None = None,
        eligible: pd.Series | None = None,
    ) -> AllocationResult:
        assets = pd.Index(expected_returns.index.astype(str), dtype=str)
        if assets.empty or assets.has_duplicates:
            raise ValueError("expected_returns requires unique asset labels")
        alpha = pd.to_numeric(expected_returns, errors="coerce").reindex(assets)
        if alpha.isna().any() or not np.isfinite(alpha.to_numpy(dtype=float)).all():
            raise ValueError("expected_returns must be finite for every asset")
        covariance = covariance.copy()
        covariance.index = covariance.index.astype(str)
        covariance.columns = covariance.columns.astype(str)
        covariance = covariance.reindex(index=assets, columns=assets)
        if covariance.isna().any().any() or not np.isfinite(
            covariance.to_numpy(dtype=float)
        ).all():
            raise ValueError("covariance must cover every expected-return asset")
        covariance_values = _nearest_psd(
            covariance.to_numpy(dtype=float),
            floor=float(self.config.covariance_eigen_floor),
        )
        previous = _aligned_series(previous_weights, assets, default=0.0)
        costs = _aligned_series(linear_trading_costs, assets, default=0.0)
        if not np.isfinite(previous.to_numpy(dtype=float)).all():
            raise ValueError("previous_weights must be finite")
        if not np.isfinite(costs.to_numpy(dtype=float)).all():
            raise ValueError("linear_trading_costs must be finite")
        if (costs < 0).any():
            raise ValueError("linear_trading_costs cannot be negative")
        eligibility = (
            pd.Series(True, index=assets)
            if eligible is None
            else eligible.reindex(assets).fillna(False).astype(bool)
        )
        if not eligibility.any():
            raise ValueError("at least one asset must be eligible")

        weights = cp.Variable(len(assets))
        delta = weights - previous.to_numpy(dtype=float)
        objective = cp.Maximize(
            alpha.to_numpy(dtype=float) @ weights
            - float(self.config.risk_aversion)
            * cp.quad_form(weights, cp.psd_wrap(covariance_values))
            - float(self.config.turnover_penalty) * cp.norm1(delta)
            - cp.sum(cp.multiply(costs.to_numpy(dtype=float), cp.abs(delta)))
        )
        constraints = [cp.norm1(weights) <= float(self.config.gross_limit)]
        if self.config.max_weight_per_asset is not None:
            cap = float(self.config.max_weight_per_asset)
            constraints.extend([weights <= cap, weights >= -cap])
        if self.config.long_only:
            constraints.append(weights >= 0.0)
        if self.config.net_target is not None:
            constraints.append(cp.sum(weights) == float(self.config.net_target))
        else:
            if self.config.min_net_exposure is not None:
                constraints.append(
                    cp.sum(weights) >= float(self.config.min_net_exposure)
                )
            if self.config.max_net_exposure is not None:
                constraints.append(
                    cp.sum(weights) <= float(self.config.max_net_exposure)
                )
        ineligible = np.flatnonzero(~eligibility.to_numpy())
        if len(ineligible):
            constraints.append(weights[ineligible] == 0.0)

        problem = cp.Problem(objective, constraints)
        if not problem.is_dcp():
            raise ValueError("allocation problem does not satisfy CVXPY DCP rules")
        solver_used = ""
        errors: dict[str, str] = {}
        installed = set(cp.installed_solvers())
        for solver in self.config.solver_order:
            solver = str(solver).upper()
            if solver not in installed:
                errors[solver] = "not installed"
                continue
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as exc:  # pragma: no cover - solver-specific
                errors[solver] = str(exc)
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                solver_used = solver
                break
            errors[solver] = str(problem.status)
        if not solver_used or weights.value is None:
            raise RuntimeError(
                "Convex allocation failed; solver outcomes: "
                + json.dumps(errors, sort_keys=True)
            )

        solved = pd.Series(np.asarray(weights.value).reshape(-1), index=assets)
        solved = solved.where(solved.abs() > 1e-10, 0.0)
        delta_values = solved - previous
        variance = float(solved.to_numpy() @ covariance_values @ solved.to_numpy())
        estimated_alpha = float(alpha.to_numpy() @ solved.to_numpy())
        turnover = float(delta_values.abs().sum())
        return AllocationResult(
            weights=solved,
            status=str(problem.status),
            solver=solver_used,
            objective_value=float(problem.value),
            estimated_variance=variance,
            estimated_alpha=estimated_alpha,
            turnover=turnover,
            config_fingerprint=self.config.fingerprint,
            diagnostics={
                "gross_exposure": float(solved.abs().sum()),
                "net_exposure": float(solved.sum()),
                "eligible_assets": int(eligibility.sum()),
                "solver_errors": errors,
            },
        )


def _aligned_series(
    values: pd.Series | None,
    assets: pd.Index,
    *,
    default: float,
) -> pd.Series:
    if values is None:
        return pd.Series(float(default), index=assets, dtype=float)
    out = pd.to_numeric(values, errors="coerce")
    out.index = out.index.astype(str)
    out = out.reindex(assets).fillna(float(default)).astype(float)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("allocation inputs must be finite")
    return out


def _nearest_psd(covariance: np.ndarray, *, floor: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, float(floor), None)
    return (eigenvectors * clipped) @ eigenvectors.T


__all__ = [
    "AllocationResult",
    "ConvexAllocationConfig",
    "ConvexPortfolioAllocator",
]
