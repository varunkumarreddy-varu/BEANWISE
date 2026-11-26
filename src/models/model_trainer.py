"""
Model Trainer Module for BEANWISE.

Executes 5-fold Stratified Cross-Validation on the training partition,
hyperparameter tuning for top candidate models, and comprehensive evaluation
on the held-out validation partition (strictly without touching test data).
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline


def cross_validate_pipelines(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Performs Stratified K-Fold cross validation on multiple pipelines.

    Returns:
        DataFrame summarizing mean and std metrics for all models,
        and raw dictionary of fold metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
    }

    summary_rows = []
    raw_results = {}

    for model_name, pipeline in pipelines.items():
        start_t = time.perf_counter()
        cv_res = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        total_wall_time = time.perf_counter() - start_t

        mean_acc = float(np.mean(cv_res["test_accuracy"]))
        std_acc = float(np.std(cv_res["test_accuracy"]))
        mean_macro_f1 = float(np.mean(cv_res["test_macro_f1"]))
        std_macro_f1 = float(np.std(cv_res["test_macro_f1"]))
        mean_weighted_f1 = float(np.mean(cv_res["test_weighted_f1"]))
        mean_train_acc = float(np.mean(cv_res["train_accuracy"]))
        mean_fit_time = float(np.mean(cv_res["fit_time"]))
        mean_score_time = float(np.mean(cv_res["score_time"]))

        summary_rows.append({
            "Model": model_name,
            "CV Mean Accuracy": round(mean_acc, 4),
            "CV Std Accuracy": round(std_acc, 4),
            "CV Mean Macro F1": round(mean_macro_f1, 4),
            "CV Std Macro F1": round(std_macro_f1, 4),
            "CV Mean Weighted F1": round(mean_weighted_f1, 4),
            "Train Accuracy": round(mean_train_acc, 4),
            "Overfitting Gap": round(mean_train_acc - mean_acc, 4),
            "Mean Fit Time (s)": round(mean_fit_time, 4),
            "Mean Score Time (s)": round(mean_score_time, 4),
            "Total CV Time (s)": round(total_wall_time, 2),
        })

        raw_results[model_name] = {
            "test_accuracy": cv_res["test_accuracy"].tolist(),
            "test_macro_f1": cv_res["test_macro_f1"].tolist(),
            "test_weighted_f1": cv_res["test_weighted_f1"].tolist(),
            "train_accuracy": cv_res["train_accuracy"].tolist(),
            "fit_time": cv_res["fit_time"].tolist(),
            "score_time": cv_res["score_time"].tolist(),
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("CV Mean Macro F1", ascending=False).reset_index(drop=True)
    return summary_df, raw_results


def tune_pipelines(
    candidate_pipelines: Dict[str, Pipeline],
    param_grids: Dict[str, Dict[str, List[Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Performs GridSearchCV with StratifiedKFold on candidate models.

    Returns:
        Dictionary of best estimator pipelines and DataFrame comparison.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    tuned_models = {}
    tuning_records = []

    for model_name, pipeline in candidate_pipelines.items():
        if model_name not in param_grids:
            continue

        grid = param_grids[model_name]
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )

        start_t = time.perf_counter()
        search.fit(X_train, y_train)
        duration = time.perf_counter() - start_t

        best_pipeline = search.best_estimator_
        tuned_models[f"{model_name}_tuned"] = best_pipeline

        tuning_records.append({
            "Base Model": model_name,
            "Tuned Model": f"{model_name}_tuned",
            "Best CV Macro F1": round(float(search.best_score_), 4),
            "Best Parameters": str(search.best_params_),
            "Tuning Search Time (s)": round(duration, 2),
            "Total Candidates Evaluated": len(search.cv_results_["params"]),
        })

    tuning_df = pd.DataFrame(tuning_records).sort_values("Best CV Macro F1", ascending=False).reset_index(drop=True)
    return tuned_models, tuning_df


def evaluate_on_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Trains a pipeline on full training data and evaluates on validation set.
    Strictly touches only Train and Validation sets.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    # Fit on full training set
    start_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit

    # Predict on validation set
    start_pred = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    pred_time = time.perf_counter() - start_pred

    # Predict probabilities if supported
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_val)
        except Exception:
            pass

    acc = float(accuracy_score(y_val, y_pred))
    macro_prec = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    labels = sorted(list(np.unique(np.concatenate([y_val, y_pred]))))
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    # Classification report dict
    report_dict = classification_report(
        y_val,
        y_pred,
        labels=labels,
        target_names=[str(c) for c in class_names] if class_names else None,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "fit_time": round(fit_time, 4),
        "prediction_time": round(pred_time, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": report_dict,
        "y_pred": y_pred.tolist(),
    }
