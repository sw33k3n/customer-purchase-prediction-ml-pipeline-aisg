# AIAP23 Machine Learning Pipeline (Online Shopping Intent)

## 1) Problem Statement
This project predicts whether a web session ends in purchase (`PurchaseCompleted`) using behavioral and session features from a SQLite dataset.

The solution is implemented fully in Python scripts (`.py`) and executed via `run.sh`.

## 2) Repository Structure
```
.
├── run.sh
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_test_split.py
│   ├── experiments.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── pipeline.py
└── outputs/
    ├── models/
    └── metrics/
```

## 3) Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) How To Run
Make script executable once:
```bash
chmod +x run.sh
```

Default run:
```bash
./run.sh
```

Notes:
- Default `run.sh` behavior downloads `online_shopping.db` from the assessment URL to a temporary file in `/tmp`.
- If your environment has no network access, pass a local DB path instead:

```bash
./run.sh /absolute/path/to/online_shopping.db online_shopping
```

Or via env vars:
```bash
ONLINE_SHOPPING_DB_PATH=/absolute/path/to/online_shopping.db \
ONLINE_SHOPPING_TABLE_NAME=online_shopping \
./run.sh
```

## 5) Pipeline Components (Step-by-Step)
`run.sh` launches `src/pipeline.py`.

1. Data loading (`src/data_loader.py`)
- Loads from SQLite DB and validates table existence.

2. Data preprocessing (`src/preprocess.py`)
- Validates required columns.
- Standardizes/cleans categorical values. `CustomerType` is reclassified into three categories: `Returning_Visitor`, `New_Visitor`, and `Other`.
- Converts target to boolean.
- Coerces numeric columns and handles invalid negative values for key rate/time/value fields.

3. Train/test split (`src/train_test_split.py`)
- Stratified split (`test_size=0.2`, `random_state=42`) to preserve class distribution.

4. Feature processing + model pipeline (`src/train.py`)
- Log transform selected numeric features to mitigate extreme right skewness.
- Numerical pipeline: imputation + standardization.
- Categorical pipeline: most-frequent imputation + one-hot encoding.
- Optional SMOTE in training pipeline (if `imbalanced-learn` available).

5. Model search (`src/experiments.py` + `src/train.py`)
- Models: Random Forest, SVM, CART.
- Two imputation strategies evaluated per model:
  - `simple`: median/most-frequent
  - `rf`: iterative imputer with random forest regressor
- Hyperparameter optimization via `GridSearchCV` (`cv=5`, `scoring="f1"`).

6. Evaluation (`src/evaluate.py`)
- Reports `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (when probability exists).

7. Model selection and persistence (`src/train.py` + `src/utils.py`)
- Selects best experiment by cross-validation F1 (not test score).
- Saves best model and full metrics summary.

8. Final evaluation + permutation importance (`src/pipeline.py` + `src/evaluate.py`)
- Re-evaluates the saved best model on test set.
- Computes permutation importance on test set.

## 6) Why These Preprocessing / Feature Engineering Choices
- Stratified split: prevents distorted class representation in train/test.
- Log transform: reduces skew impact for positive, heavy-tailed numeric variables.
- Numeric scaling: important for SVM margin optimization.
- One-hot encoding: converts categorical variables to model-usable numeric form.
- Dual imputation strategy (`simple` vs `rf`): compares robustness of lightweight vs model-based imputation.
- SMOTE inside pipeline: addresses class imbalance while preventing leakage outside training flow.

## 7) Why These Models
- Random Forest: robust nonlinear baseline, handles interactions well.
- SVM (`class_weight="balanced"`): strong classifier for potentially high-dimensional transformed feature space.
- CART: interpretable tree baseline for comparison.

Using three algorithm families improves robustness of model selection instead of relying on one model type.

## 8) Why These Evaluation Metrics
- Primary metric: **F1-score** (class imbalance and balanced precision/recall tradeoff).
- Precision and recall: explicitly track false positives vs false negatives.
- ROC-AUC: threshold-independent separability measure.
- Accuracy: included for completeness and broad comparability.

## 9) Outputs
After run completion:
- Best model: `outputs/models/<best_experiment_id>.pkl`
- Full experiment results: `outputs/metrics/results.json`
- Permutation importance: `outputs/metrics/permutation_importance_<best_experiment_id>.csv`

## 10) Code Quality / Reusability Notes
- Pipeline is modularized by responsibility (load, preprocess, split, train, evaluate, save).
- Reusable functions and classes are used across scripts.
- `pipeline.py` provides SITREP-style runtime updates for traceability.
- All MLP logic is implemented in `.py` scripts (not notebook-executed pipeline logic).

## 11) Actionable Insights Provision
- Assessed on: SVM with RBF kernel
- Permutation Importances:
            feature  importance_mean  importance_std
          PageValue         0.445739        0.015028
           ExitRate         0.015802        0.005046
         BounceRate         0.014938        0.005132
    ProductPageTime         0.000999        0.001169
SpecialDayProximity         0.000268        0.000893
   GeographicRegion         0.000154        0.000462
       CustomerType         0.000000        0.000000
      TrafficSource        -0.000903        0.000670

- Insights:
  - PageValue is a dominant driver in sales. It is reccomended that the web pages are linked to pages with high value. This works hand in hand with deep exploratory behaviour of consumers, thereby helping with conversion
  - ExitRate and BounceRate are secondary signals that suggest accurate targetting for awareness and engagement of customers are important. To do so, identify interests of visitors and promote the right products to them through Meta Ads or other platforms that give specific targetting ability.
  - Looking at which features are stastistically significant, it suggests that the biggest problem faced in the purchase funnel (awareness, consideration, purchase) is in fact the consideration stage. This should be where actionable steps should be taken to boost sales.