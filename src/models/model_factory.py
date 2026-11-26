"""
Model Factory Module for BEANWISE.

Constructs standardized Scikit-learn Pipeline objects for machine learning
classification algorithms, ensuring uniform interfaces and leakage-free scaling.
"""

from typing import Any, Dict, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_model_pipeline(model_key: str, random_state: int = 42) -> Pipeline:
    """
    Constructs a Scikit-learn Pipeline for the requested algorithm.

    Parameters:
        model_key: Identifier string for the algorithm.
        random_state: Seed for reproducibility.

    Returns:
        Pipeline instance configured with appropriate scaling and classifier.
    """
    key = model_key.lower().strip()

    if key in ["logistic_regression", "lr"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1500, random_state=random_state))
        ])

    elif key in ["knn", "k_nearest_neighbors"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ])

    elif key in ["decision_tree", "dt"]:
        return Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=random_state))
        ])

    elif key in ["random_forest", "rf"]:
        return Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=4,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

    elif key in ["extra_trees", "et"]:
        return Pipeline([
            ("classifier", ExtraTreesClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=4,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

    elif key in ["svm_rbf", "svm", "svc"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=2.0,
                gamma="scale",
                random_state=random_state
            ))
        ])

    elif key in ["gradient_boosting", "gb"]:
        return Pipeline([
            ("classifier", GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.1,
                max_depth=4,
                random_state=random_state
            ))
        ])

    elif key in ["hist_gradient_boosting", "hgb"]:
        return Pipeline([
            ("classifier", HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=random_state
            ))
        ])

    elif key in ["xgboost", "xgb"]:
        return Pipeline([
            ("classifier", XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1
            ))
        ])

    else:
        raise ValueError(
            f"Unsupported model key: '{model_key}'. Supported keys: "
            "['logistic_regression', 'knn', 'decision_tree', 'random_forest', "
            "'extra_trees', 'svm_rbf', 'gradient_boosting', 'hist_gradient_boosting', 'xgboost']"
        )


def get_all_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Returns a dictionary of all 9 baseline candidate model pipelines.
    """
    model_keys = [
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "svm_rbf",
        "gradient_boosting",
        "hist_gradient_boosting",
        "xgboost",
    ]
    return {k: get_model_pipeline(k, random_state=random_state) for k in model_keys}


def get_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """
    Returns hyperparameter search spaces for candidate models.
    """
    return {
        "random_forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [15, 25, None],
            "classifier__min_samples_split": [2, 5],
        },
        "svm_rbf": {
            "classifier__C": [0.5, 2.0, 10.0],
            "classifier__gamma": ["scale", "auto", 0.05],
        },
        "xgboost": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 6],
            "classifier__learning_rate": [0.05, 0.1],
        },
    }
