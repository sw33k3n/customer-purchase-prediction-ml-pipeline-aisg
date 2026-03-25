from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))

    return metrics


def test_best_model(best_model_path, X_test, y_test):
    best_model_path = Path(best_model_path)
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path.resolve()}")

    model = joblib.load(best_model_path)
    return evaluate_model(model, X_test, y_test)


def get_permutation_importance(
    model,
    X_test,
    y_test,
    *,
    scoring: str = "f1",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")

    pi = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    alpha = 1.0 - ci_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    importances = pi.importances
    means = importances.mean(axis=1)
    stds = importances.std(axis=1, ddof=1 if n_repeats > 1 else 0)
    sem = stds / np.sqrt(n_repeats)

    ci_lower_normal = means - z * sem
    ci_upper_normal = means + z * sem
    ci_lower_percentile = np.quantile(importances, alpha / 2.0, axis=1)
    ci_upper_percentile = np.quantile(importances, 1.0 - alpha / 2.0, axis=1)

    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": means,
                "importance_std": stds,
                "ci_lower_normal": ci_lower_normal,
                "ci_upper_normal": ci_upper_normal,
                "ci_lower_percentile": ci_lower_percentile,
                "ci_upper_percentile": ci_upper_percentile,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
