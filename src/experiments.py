from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model_specs():
    """
    Returns a dict of model specs:
      - constructor (model object)
      - param grids (different grids per imputation strategy if you want)
    """
    specs = {
        "random_forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid_simple": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.5],
            },
            "param_grid_rf": {
                # RF-imputation is slower → keep grid smaller
                "model__n_estimators": [300, 500],
                "model__max_depth": [None, 15],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
        },

        "svm": {
            # IMPORTANT: for imbalanced data, class_weight helps a lot
            "model": SVC(probability=True, random_state=42, class_weight="balanced"),
            "param_grid_simple": {
                "model__C": [0.5, 1, 2],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
            "param_grid_rf": {
                # keep smaller
                "model__C": [1, 2],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale"],
            },
        },

        "cart": {
            "model": DecisionTreeClassifier(random_state=42),
            "param_grid_simple": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
                "model__criterion": ["gini", "entropy"],
            },
            "param_grid_rf": {
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__criterion": ["gini", "entropy"],
            },
        },
    }

    return specs