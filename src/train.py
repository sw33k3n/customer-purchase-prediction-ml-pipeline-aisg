from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluate import evaluate_model

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover
    SMOTE = None
    ImbPipeline = None


def _apply_log_columns(X: pd.DataFrame, log_cols: list[str] | None) -> pd.DataFrame:
    if log_cols is None or len(log_cols) == 0:
        return X

    X = X.copy()
    for col in log_cols:
        if col not in X.columns:
            raise ValueError(f"log_cols contains unknown column: {col}")
        values = pd.to_numeric(X[col], errors="coerce")
        if (values < 0).any():
            raise ValueError(f"Negative values found in '{col}', cannot apply log1p.")
        X[col] = np.log1p(values)
    return X


class ColumnLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_cols: list[str] | None):
        self.log_cols = log_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _apply_log_columns(X, self.log_cols)


def build_full_pipeline(
    *,
    num_cols: list[str],
    cat_cols: list[str],
    log_cols: list[str] | None,
    imputation_strategy: str,
    rebalance: bool,
    model,
):
    if imputation_strategy not in {"simple", "rf"}:
        raise ValueError("imputation_strategy must be either 'simple' or 'rf'.")

    if imputation_strategy == "simple":
        num_imputer = SimpleImputer(strategy="median")
    else:
        num_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            random_state=42,
            max_iter=10,
        )

    num_pipe = Pipeline(
        steps=[
            ("imputer", num_imputer),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    steps = [
        ("log_transform", ColumnLogTransformer(log_cols)),
        ("preprocessor", preprocessor),
    ]

    if rebalance:
        if ImbPipeline is None or SMOTE is None:
            raise ImportError(
                "rebalance=True requires imbalanced-learn. Add 'imbalanced-learn' to dependencies."
            )
        steps.extend([("smote", SMOTE(random_state=42)), ("model", model)])
        return ImbPipeline(steps=steps)

    steps.append(("model", model))
    return Pipeline(steps=steps)


@dataclass
class ExperimentResult:
    model_name: str
    imputation_strategy: str
    best_estimator: Any
    best_params: Dict[str, Any]
    best_cv_score: float
    cv_scoring: str
    cv_folds: int
    test_metrics: Dict[str, float]
    fit_seconds: float


def train_experiment(
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    num_cols,
    cat_cols,
    log_cols,
    model_name: str,
    model_obj,
    imputation_strategy: str,
    param_grid: Dict[str, Any],
    cv: int = 5,
    scoring: str = "f1",
    rebalance: bool = True,
    n_jobs: int = -1,
) -> ExperimentResult:
    """
    Train one experiment: preprocess + (optional) SMOTE + model, with GridSearchCV.

    Returns a consistent ExperimentResult object for any model.
    """
    if not isinstance(param_grid, dict) or len(param_grid) == 0:
        raise ValueError(f"param_grid must be a non-empty dict for {model_name}.")

    pipeline = build_full_pipeline(
        num_cols=num_cols,
        cat_cols=cat_cols,
        log_cols=log_cols,
        imputation_strategy=imputation_strategy,  # "simple" or "rf"
        rebalance=rebalance,  # SMOTE inside pipeline
        model=model_obj
    )

    start = time.time()

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    test_metrics = evaluate_model(best_pipeline, X_test, y_test)

    end = time.time()

    return ExperimentResult(
        model_name=model_name,
        imputation_strategy=imputation_strategy,
        best_estimator=best_pipeline,
        best_params=dict(grid.best_params_),
        best_cv_score=float(grid.best_score_),
        cv_scoring=scoring,
        cv_folds=cv,
        test_metrics=test_metrics,
        fit_seconds=float(end - start),
    )


def summarize_and_select_best(
    experiments: Dict[str, ExperimentResult],
    selection_metric: str = "best_cv_score",
) -> Tuple[str, ExperimentResult]:
    """
    Summarize all experiments and select best based on CV score (default).

    selection_metric:
      - "best_cv_score" (recommended)
      - You can extend to other metrics if needed.
    """
    if not experiments:
        raise ValueError("No experiments to summarize.")

    print("\n==============================")
    print("        EXPERIMENT SUMMARY     ")
    print("==============================")

    # Print each experiment in a consistent format
    for exp_id, res in experiments.items():
        print(f"\n[{exp_id}]")
        print(f"Model: {res.model_name}")
        print(f"Imputation: {res.imputation_strategy}")
        print(f"CV: {res.cv_folds}-fold | Scoring: {res.cv_scoring}")
        print(f"Best CV Score: {res.best_cv_score:.4f}")
        print(f"Fit Time (s): {res.fit_seconds:.1f}")
        print(f"Best Params: {res.best_params}")
        print(f"Test Metrics: {res.test_metrics}")

    # Select best model by CV score (no test leakage)
    if selection_metric == "best_cv_score":
        best_exp_id = max(experiments, key=lambda k: experiments[k].best_cv_score)
    else:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    best_res = experiments[best_exp_id]

    print("\n==============================")
    print("        SELECTED BEST MODEL    ")
    print("==============================")
    print(f"Experiment ID: {best_exp_id}")
    print(f"Model: {best_res.model_name}")
    print(f"Imputation: {best_res.imputation_strategy}")
    print(f"Best CV Score: {best_res.best_cv_score:.4f}")
    print(f"Best Params: {best_res.best_params}")
    print(f"Final Test Metrics: {best_res.test_metrics}")

    return best_exp_id, best_res
