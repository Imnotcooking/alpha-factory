import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Oracle ground-truth thresholds (20-day forward horizon)
ORACLE_TREND_RETURN_MIN = 0.02
ORACLE_TREND_KER_MIN = 0.25
ORACLE_PANIC_MDD_MAX = -0.08
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALPHA_RUNTIME_DATA_ROOT = _REPO_ROOT / "runtime" / "data"
DEFAULT_MATRIX_PATH = os.path.join(_ALPHA_RUNTIME_DATA_ROOT, "feature_store", "ML_Feature_Matrix.parquet")
DEFAULT_PROBS_PATH = os.path.join(_ALPHA_RUNTIME_DATA_ROOT, "regime", "GMM_Rolling_Probabilities.parquet")


@dataclass
class OracleEvaluationResult:
    eval_df: pd.DataFrame
    confusion: pd.DataFrame
    report: pd.DataFrame
    accuracy: float
    panic_auc: float | None
    state_map: dict[int, str]


def build_oracle_labels(df, window=20):
    """
    Calculates the 'Perfect Human Hindsight' labels looking forward N days.
    """
    df = df.sort_values("date").copy()
    prices = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    n = len(prices)

    fwd_return = np.full(n, np.nan)
    fwd_mdd = np.full(n, np.nan)
    fwd_ker = np.full(n, np.nan)

    for i in range(0, max(0, n - window)):
        start_price = prices[i]
        end_price = prices[i + window]
        path = prices[i : i + window + 1]

        if not np.isfinite(start_price) or abs(start_price) <= 1e-12:
            continue
        if not np.all(np.isfinite(path)):
            continue

        fwd_return[i] = end_price / start_price - 1
        running_peak = np.maximum.accumulate(path)
        fwd_mdd[i] = np.min(path / running_peak - 1)

        total_path = np.abs(np.diff(path)).sum()
        fwd_ker[i] = abs(end_price - start_price) / total_path if total_path > 1e-12 else 0.0

    df["fwd_return_20d"] = fwd_return
    df["fwd_mdd_20d"] = fwd_mdd
    df["fwd_ker_20d"] = fwd_ker

    # 4. Apply The Oracle Rules (Ground Truth)
    df["Oracle_State"] = 1

    df.loc[df["fwd_mdd_20d"] < ORACLE_PANIC_MDD_MAX, "Oracle_State"] = 2
    df.loc[
        (df["fwd_return_20d"] > ORACLE_TREND_RETURN_MIN)
        & (df["fwd_ker_20d"] > ORACLE_TREND_KER_MIN)
        & (df["Oracle_State"] != 2),
        "Oracle_State",
    ] = 0

    return df.dropna(subset=["fwd_return_20d"])


def align_ai_states(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    Map raw GMM state indices to semantic Quiet / Chop / Panic using microstructure stress.
    """
    out = df.copy()
    prob_cols = ["p_state_0", "p_state_1", "p_state_2"]
    out["raw_dom"] = out[prob_cols].values.argmax(axis=1)

    stress_cols = [
        col
        for col in ["amihud_z", "gk_vol_z", "f_macro_amihud_z", "f_macro_gk_vol"]
        if col in out.columns
    ]
    if stress_cols:
        out["stress_metric"] = out[stress_cols].sum(axis=1)
    else:
        out["stress_metric"] = 0.0

    state_scores = {}
    fallback_score = out["stress_metric"].max() + 1.0
    for state in range(3):
        state_slice = out.loc[out["raw_dom"] == state, "stress_metric"]
        state_scores[state] = state_slice.mean() if not state_slice.empty else fallback_score + state

    quiet_idx, chop_idx, panic_idx = sorted(state_scores, key=state_scores.get)

    out["prob_Quiet"] = out[f"p_state_{quiet_idx}"]
    out["prob_Chop"] = out[f"p_state_{chop_idx}"]
    out["prob_Panic"] = out[f"p_state_{panic_idx}"]
    out["AI_Predicted_State"] = out[["prob_Quiet", "prob_Chop", "prob_Panic"]].values.argmax(axis=1)
    state_map = {quiet_idx: "Quiet", chop_idx: "Chop", panic_idx: "Panic"}
    return out, state_map


class OracleEvaluator:
    """
    Evaluates unsupervised GMM/HMM regime probabilities against hindsight labels.
    This is a regime-model diagnostic, not a normal factor PnL evaluator.
    """
    def __init__(
        self,
        matrix_path: str = DEFAULT_MATRIX_PATH,
        probs_path: str = DEFAULT_PROBS_PATH,
        window: int = 20,
    ):
        self.matrix_path = matrix_path
        self.probs_path = probs_path
        self.window = window

    def evaluate(self, tickers: Iterable[str] | None = None) -> OracleEvaluationResult:
        prices = pd.read_parquet(self.matrix_path)
        probs = pd.read_parquet(self.probs_path)

        prices["date"] = pd.to_datetime(prices["date"])
        probs["date"] = pd.to_datetime(probs["date"])

        price_cols = ["date", "ticker", "close"]
        feature_cols = [
            col
            for col in [
                "amihud_z",
                "gk_vol_z",
                "ker_20d",
                "f_macro_amihud_z",
                "f_macro_gk_vol",
                "f_macro_ker_20d",
            ]
            if col in prices.columns
        ]

        merge_keys = ["date", "ticker"] if "ticker" in probs.columns else ["date"]
        df = pd.merge(
            prices[price_cols + feature_cols],
            probs,
            on=merge_keys,
            how="inner",
        )

        if tickers:
            ticker_set = set(tickers)
            df = df[df["ticker"].isin(ticker_set)].copy()

        required_cols = {"p_state_0", "p_state_1", "p_state_2", "close", "ticker", "date"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Oracle evaluation missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Oracle evaluation has no overlapping price/probability rows.")

        df, state_map = align_ai_states(df)

        results = []
        for _, group in df.groupby("ticker"):
            oracle_df = build_oracle_labels(group, window=self.window)
            if not oracle_df.empty:
                results.append(oracle_df)
        if not results:
            raise ValueError("Oracle evaluation produced no labeled rows.")

        eval_df = pd.concat(results, ignore_index=True)
        y_true = eval_df["Oracle_State"]
        y_pred = eval_df["AI_Predicted_State"]
        y_panic = (y_true == 2).astype(int)
        panic_scores = eval_df["prob_Panic"]

        labels = [0, 1, 2]
        label_names = ["Trend", "Chop", "Panic"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion = pd.DataFrame(
            cm,
            index=[f"True {name}" for name in label_names],
            columns=[f"Pred {name}" for name in label_names],
        )
        report = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=label_names,
                zero_division=0,
                output_dict=True,
            )
        ).transpose()

        panic_auc = None
        if y_panic.nunique() >= 2:
            panic_auc = float(roc_auc_score(y_panic, panic_scores))

        return OracleEvaluationResult(
            eval_df=eval_df,
            confusion=confusion,
            report=report,
            accuracy=float((y_true == y_pred).mean()),
            panic_auc=panic_auc,
            state_map=state_map,
        )


def evaluate_model():
    print("🚁 Launching Oracle Evaluator...")
    print(
        f"   Oracle rules: Trend if fwd_return>{ORACLE_TREND_RETURN_MIN:.0%} & "
        f"fwd_ker>{ORACLE_TREND_KER_MIN}; Panic if fwd_mdd<{ORACLE_PANIC_MDD_MAX:.0%}"
    )

    matrix_path = DEFAULT_MATRIX_PATH
    probs_path = DEFAULT_PROBS_PATH

    try:
        result = OracleEvaluator(matrix_path, probs_path).evaluate()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: Could not find oracle parquet inputs. ({exc})")
        return

    print("\n==================================================")
    print("📊 ORACLE EVALUATION RESULTS (All Assets)")
    print("==================================================\n")

    print("1. CONFUSION MATRIX (Rows: True Oracle, Columns: AI Predicted)")
    print("State 0: Trend | State 1: Chop | State 2: Panic")
    print(result.confusion)

    print("\n2. CLASSIFICATION REPORT")
    print(result.report)

    print(f"\n   Overall accuracy: {result.accuracy:.1%}")
    print(f"   Raw state map: {result.state_map}")

    print("\n3. ROC-AUC (Binary Panic Detection: Oracle Crisis vs AI Panic Probability)")
    if result.panic_auc is None:
        print("   Skipped: only one class present in Oracle panic labels.")
    else:
        print(f"   Panic ROC-AUC: {result.panic_auc:.4f}")
        print("   (1.0 = perfect separation, 0.5 = random guessing)")

    print("\n==================================================")
    print("Note: If 'True 2' (Panic) aligns well with 'Pred 2', the execution shields are mathematically valid.")
    print("==================================================")


if __name__ == "__main__":
    evaluate_model()
