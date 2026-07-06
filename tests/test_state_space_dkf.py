import json
from pathlib import Path

import numpy as np
import pandas as pd

from oqp.research.state_space import (
    DualKalmanRegression,
    DualKalmanRegressionConfig,
    RelationshipLabConfig,
    StateSpaceSchema,
    coefficient_columns,
    run_relationship_dkf,
    save_dual_kalman_feature_artifact,
    summarize_dual_kalman_output,
)


def _drifting_beta_panel(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    x = rng.normal(size=rows)
    beta = np.linspace(0.4, 1.6, rows)
    y = 0.25 + beta * x + rng.normal(scale=0.03, size=rows)
    return pd.DataFrame({"date": dates, "ticker": "au", "y": y, "x": x, "true_beta": beta})


def test_dual_kalman_regression_tracks_drifting_beta():
    df = _drifting_beta_panel()
    config = DualKalmanRegressionConfig(
        schema=StateSpaceSchema(date_col="date", y_col="y", x_cols=("x",), group_cols=("ticker",)),
        process_noise=2e-3,
        observation_noise=2e-3,
        initial_state_covariance=10.0,
    )

    out = DualKalmanRegression(config).fit_transform(df)

    assert len(out) == len(df)
    assert "dkf_beta_x" in out.columns
    assert "dkf_beta_intercept" in out.columns
    assert out["dkf_beta_x"].iloc[-40:].mean() > out["dkf_beta_x"].iloc[:40].mean()
    assert abs(out["dkf_beta_x"].iloc[-1] - df["true_beta"].iloc[-1]) < 0.35
    assert out["dkf_state_uncertainty"].iloc[-1] < out["dkf_state_uncertainty"].iloc[0]


def test_dual_kalman_regression_filters_groups_independently():
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    x = np.linspace(0.5, 2.0, len(dates))
    rows = []
    for ticker, beta in [("a", 1.2), ("b", -0.8)]:
        for date, x_value in zip(dates, x):
            rows.append({"date": date, "ticker": ticker, "y": beta * x_value, "x": x_value})
    df = pd.DataFrame(rows)
    config = DualKalmanRegressionConfig(
        schema=StateSpaceSchema(date_col="date", y_col="y", x_cols=("x",), group_cols=("ticker",)),
        include_intercept=False,
        process_noise=1e-4,
        observation_noise=1e-4,
        initial_state_covariance=20.0,
    )

    out = DualKalmanRegression(config).fit_transform(df)
    last = out.groupby("ticker")["dkf_beta_x"].last()

    assert last["a"] > 1.0
    assert last["b"] < -0.6
    assert out[out["ticker"] == "b"]["dkf_beta_x"].iloc[0] < 0.0


def test_dual_kalman_artifact_is_reproducible_and_diagnostic(tmp_path: Path):
    df = _drifting_beta_panel(rows=60)
    source_path = tmp_path / "source.csv"
    df.to_csv(source_path, index=False)
    config = DualKalmanRegressionConfig(
        schema=StateSpaceSchema(date_col="date", y_col="y", x_cols=("x",), group_cols=("ticker",)),
        process_noise=1e-3,
        observation_noise=1e-3,
    )

    artifact = save_dual_kalman_feature_artifact(
        df,
        config,
        artifact_name="unit_test_dkf",
        source_path=source_path,
        output_root=tmp_path / "state_space",
    )

    assert Path(artifact.output_path).exists()
    assert Path(artifact.metadata_path).exists()
    assert artifact.row_count == len(df)
    assert "dkf_beta_x" in artifact.feature_columns

    metadata = json.loads(Path(artifact.metadata_path).read_text(encoding="utf-8"))
    assert metadata["row_count"] == len(df)
    assert metadata["source_fingerprint"]["exists"] is True
    assert metadata["source_fingerprint"]["sha256"]
    saved = pd.read_parquet(artifact.output_path)
    summary = summarize_dual_kalman_output(saved)
    assert not summary.empty
    assert "dkf_beta_x" in coefficient_columns(saved)


def test_relationship_lab_builds_pair_diagnostics_from_prices():
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    x_ret = np.full(len(dates), 0.001)
    y_ret = 0.0002 + 1.5 * x_ret
    x_close = 100 * np.exp(np.cumsum(x_ret))
    y_close = 80 * np.exp(np.cumsum(y_ret))
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["x"] * len(dates) + ["y"] * len(dates),
            "close": list(x_close) + list(y_close),
        }
    )
    result = run_relationship_dkf(
        df,
        RelationshipLabConfig(
            y_ticker="y",
            x_ticker="x",
            process_noise=1e-4,
            observation_noise=1e-5,
        ),
    )

    pair = result["pair"]
    assert not pair.empty
    assert {"dynamic_beta", "residual_z", "state_uncertainty"}.issubset(pair.columns)
    assert result["summary"]["rows"] > 50
