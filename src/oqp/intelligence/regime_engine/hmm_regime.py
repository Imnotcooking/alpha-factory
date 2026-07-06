"""HMM-style market regime models.

This module mirrors the public API shape used by the alpha research lab:
``MarketHMM`` and ``MarketGMMHMM`` expose ``fit``, ``predict``,
``predict_proba``, ``save``, and ``load`` while aligning output states by
volatility so state 0 is quiet and the highest state is panic.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

try:  # model persistence is optional until HMM artifacts are promoted
    import joblib
except Exception:  # pragma: no cover - depends on deployment environment
    joblib = None

try:  # keep Ops import-safe on environments without hmmlearn installed
    from hmmlearn.hmm import GMMHMM, GaussianHMM
except Exception:  # pragma: no cover - depends on deployment environment
    GMMHMM = None
    GaussianHMM = None


warnings.filterwarnings("ignore")


class MarketHMM:
    """Unsupervised Hidden Markov Model for macro regime detection."""

    def __init__(self, n_components=3, covariance_type="full", random_state=42):
        if GaussianHMM is None:
            raise ImportError("hmmlearn is required to use MarketHMM.")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=1000,
            random_state=self.random_state,
        )
        self.is_fitted = False
        self.state_map = {}

    def _prepare_emissions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract emission sequences.

        Expects a DataFrame with ``returns`` and ``volatility`` columns.
        """

        _require_columns(df, ("returns", "volatility"))
        emissions = np.column_stack([df["returns"].values, df["volatility"].values])
        return np.nan_to_num(emissions)

    def fit(self, df: pd.DataFrame):
        """Fit the HMM to historical macro data."""

        emissions = self._prepare_emissions(df)
        self.model.fit(emissions)
        self.is_fitted = True
        self._align_states(emissions)

    def _align_states(self, emissions: np.ndarray):
        """Ensure State 0 = quiet, State 1 = chop, State 2 = panic."""

        hidden_states = self.model.predict(emissions)
        state_vols = []

        for i in range(self.n_components):
            state_data = emissions[hidden_states == i]
            if len(state_data) > 0:
                state_vols.append((i, np.mean(state_data[:, 1])))
            else:
                state_vols.append((i, 0))

        sorted_states = sorted(state_vols, key=lambda item: item[1])
        self.state_map = {
            old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_states)
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the aligned discrete regime for each row."""

        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")

        emissions = self._prepare_emissions(df)
        raw_states = self.model.predict(emissions)
        return np.array([self.state_map[state] for state in raw_states])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return aligned probability matrix [P(State0), P(State1), P(State2)]."""

        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")

        emissions = self._prepare_emissions(df)
        raw_proba = self.model.predict_proba(emissions)
        aligned_proba = np.zeros_like(raw_proba)
        for old_idx, new_idx in self.state_map.items():
            aligned_proba[:, new_idx] = raw_proba[:, old_idx]
        return aligned_proba

    def save(self, filepath: str):
        """Save the fitted model to disk."""

        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        if joblib is None:
            raise ImportError("joblib is required to save MarketHMM artifacts.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "state_map": self.state_map}, filepath)

    def load(self, filepath: str):
        """Load a fitted model from disk."""

        if joblib is None:
            raise ImportError("joblib is required to load MarketHMM artifacts.")
        data = joblib.load(filepath)
        self.model = data["model"]
        self.state_map = data["state_map"]
        self.is_fitted = True


class MarketGMMHMM:
    """Gaussian Mixture Model HMM for fatter-tailed regime emissions."""

    def __init__(self, n_components=3, n_mix=2, covariance_type="diag", random_state=42):
        if GMMHMM is None:
            raise ImportError("hmmlearn is required to use MarketGMMHMM.")
        self.n_components = n_components
        self.n_mix = n_mix
        self.model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=100,
        )
        self.is_fitted = False
        self.state_map = {}

    def fit(self, emissions: pd.DataFrame):
        X = emissions.values
        self.model.fit(X)
        self._align_states(X)
        self.is_fitted = True

    def _align_states(self, X):
        """
        Align states by volatility.

        Expected emission order for this institutional shape is:
        ``amihud_z=0``, ``gk_vol=1``, ``ker_20d=2``.
        """

        hidden_states = self.model.predict(X)
        gk_col = 1

        state_vol = {}
        for i in range(self.n_components):
            mask = hidden_states == i
            if np.sum(mask) > 0:
                state_vol[i] = np.mean(X[mask, gk_col])
            else:
                state_vol[i] = np.inf

        sorted_states = sorted(state_vol.items(), key=lambda item: item[1])
        self.state_map = {
            old_state: new_state
            for new_state, (old_state, _) in enumerate(sorted_states)
        }

    def predict(self, emissions: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("GMM-HMM is not fitted yet.")

        raw_states = self.model.predict(emissions.values)
        return np.array([self.state_map[state] for state in raw_states])

    def predict_proba(self, emissions: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("GMM-HMM is not fitted yet.")

        raw_proba = self.model.predict_proba(emissions.values)
        aligned_proba = np.zeros_like(raw_proba)
        for old_idx, new_idx in self.state_map.items():
            aligned_proba[:, new_idx] = raw_proba[:, old_idx]
        return aligned_proba


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Regime emissions missing columns: {', '.join(missing)}")
