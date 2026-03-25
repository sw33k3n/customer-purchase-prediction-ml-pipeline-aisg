from __future__ import annotations

import argparse
import importlib.util
import math
import os
import warnings
from pathlib import Path

from data_loader import load_data
from evaluate import get_permutation_importance, test_best_model
from experiments import get_model_specs
from preprocess import clean_data
from train import summarize_and_select_best, train_experiment
from train_test_split import split_data
from utils import save_results


def _derive_columns(df, target_col="PurchaseCompleted"):
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target_col]
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
    default_log_candidates = ["SpecialDayProximity", "ExitRate", "PageValue", "BounceRate", "ProductPageTime"]
    log_cols = [c for c in default_log_candidates if c in num_cols]
    return num_cols, cat_cols, log_cols


def _parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate shopping intent models.")
    parser.add_argument(
        "--db-path",
        default=os.getenv("ONLINE_SHOPPING_DB_PATH", "data/online_shopping.db"),
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--table-name",
        default=os.getenv("ONLINE_SHOPPING_TABLE_NAME", "online_shopping"),
        help="Source table name inside the SQLite database.",
    )
    return parser.parse_args()


def _sitrep(message: str):
    print(f"[SITREP] {message}")


def _grid_size(param_grid):
    if not param_grid:
        return 0
    return math.prod(len(values) for values in param_grid.values())


def _configure_warning_filters():
    # Suppress noisy third-party warning floods (including joblib workers).
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")


def main():
    _configure_warning_filters()
    _sitrep("Pipeline started.")
    args = _parse_args()
    _sitrep(f"Loading data from db='{args.db_path}', table='{args.table_name}'.")
    df = load_data(db_path=args.db_path, table_name=args.table_name)
    _sitrep(f"Loaded raw dataset with {len(df)} rows and {len(df.columns)} columns.")

    df = clean_data(df)
    _sitrep("Data cleaning completed.")

    num_cols, cat_cols, log_cols = _derive_columns(df)
    _sitrep(
        f"Derived columns: {len(num_cols)} numeric, {len(cat_cols)} categorical, {len(log_cols)} log-transform."
    )
    X_train, X_test, y_train, y_test = split_data(df)
    _sitrep(
        f"Train/test split completed: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows."
    )

    model_specs = get_model_specs()
    experiments = {}
    rebalance_enabled = importlib.util.find_spec("imblearn") is not None
    if not rebalance_enabled:
        _sitrep("imbalanced-learn not installed; continuing without SMOTE.")
    else:
        _sitrep("SMOTE is enabled for training.")

    total_experiments = len(model_specs) * 2
    completed = 0

    for model_name, spec in model_specs.items():
        for imputation_strategy in ["simple", "rf"]:
            grid_key = (
                "param_grid_simple" if imputation_strategy == "simple" else "param_grid_rf"
            )
            exp_id = f"{model_name}__{imputation_strategy}"
            completed += 1
            _sitrep(
                f"[{completed}/{total_experiments}] Training {exp_id} "
                f"(grid={_grid_size(spec[grid_key])} configs)."
            )

            experiments[exp_id] = train_experiment(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                num_cols=num_cols,
                cat_cols=cat_cols,
                log_cols=log_cols,
                model_name=model_name,
                model_obj=spec["model"],
                imputation_strategy=imputation_strategy,
                param_grid=spec[grid_key],
                cv=5,
                scoring="f1",
                rebalance=rebalance_enabled,
            )
            _sitrep(
                f"Finished {exp_id}: cv_f1={experiments[exp_id].best_cv_score:.4f}, "
                f"test_f1={experiments[exp_id].test_metrics['f1']:.4f}, "
                f"fit={experiments[exp_id].fit_seconds:.1f}s."
            )

    _sitrep("Selecting best model based on CV score.")
    best_id, best_res = summarize_and_select_best(experiments)
    _sitrep(f"Selected best experiment: {best_id} (cv_f1={best_res.best_cv_score:.4f}).")
    models_to_save = {best_id: best_res.best_estimator}

    results_to_save = {
        exp_id: {
            "model": res.model_name,
            "imputation": res.imputation_strategy,
            "best_params": res.best_params,
            "best_cv_score": res.best_cv_score,
            "cv_scoring": res.cv_scoring,
            "cv_folds": res.cv_folds,
            "test_metrics": res.test_metrics,
            "fit_seconds": res.fit_seconds,
        }
        for exp_id, res in experiments.items()
    }
    _sitrep("Saving model and metrics artifacts.")
    save_results(models=models_to_save, metrics=results_to_save)
    _sitrep("Artifacts saved to outputs/models and outputs/metrics.")

    best_model_path = f"outputs/models/{best_id}.pkl"
    _sitrep("Running final evaluation from saved best model.")
    final_test_metrics = test_best_model(best_model_path, X_test, y_test)
    _sitrep("Computing permutation importance with confidence intervals on test set.")
    ci_level = 0.95
    pi_df = get_permutation_importance(
        best_res.best_estimator,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=30,
        random_state=42,
        n_jobs=1,
        ci_level=ci_level,
    )
    pi_path = Path(f"outputs/metrics/permutation_importance_ci_{best_id}.csv")
    pi_path.parent.mkdir(parents=True, exist_ok=True)
    pi_df.to_csv(pi_path, index=False)

    print("\n==============================")
    print("   FINAL TEST EVALUATION")
    print("==============================")
    print(f"Best Model ID: {best_id}")
    print(f"Saved Model Path: {best_model_path}")
    print(f"Final Test Metrics: {final_test_metrics}")
    print(f"Permutation Importance Path: {pi_path}")
    print(f"Permutation Importance CI Level: {ci_level:.0%}")
    print("Top 10 Permutation Importances with CI:")
    print(
        pi_df[
            [
                "feature",
                "importance_mean",
                "importance_std",
                "ci_lower_normal",
                "ci_upper_normal",
                "ci_lower_percentile",
                "ci_upper_percentile",
            ]
        ].head(10).to_string(index=False)
    )
    _sitrep("Pipeline completed.")


if __name__ == "__main__":
    main()
