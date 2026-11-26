"""
Prediction Analysis and Error Investigation Module for BEANWISE.

Analyzes individual misclassified samples, confidence distributions,
and error concentrations across dry bean varieties.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def extract_misclassified_samples(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    y_proba: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Extracts all misclassified observations with true and predicted labels,
    predicted probability/confidence, and input feature values.
    """
    mis_mask = (y_true != y_pred)
    mis_indices = np.where(mis_mask)[0]

    rows = []
    for idx in mis_indices:
        true_cls = class_names[y_true[idx]]
        pred_cls = class_names[y_pred[idx]]
        conf = float(y_proba[idx, y_pred[idx]]) if y_proba is not None else None

        row_dict = {
            "Sample_Index": int(X_test.index[idx]) if hasattr(X_test, "index") else int(idx),
            "True_Class": true_cls,
            "Predicted_Class": pred_cls,
            "Prediction_Confidence": round(conf, 4) if conf is not None else None,
        }
        # Add key input features for review
        for col in ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "roundness", "Compactness"]:
            if col in X_test.columns:
                row_dict[col] = round(float(X_test.iloc[idx][col]), 4)

        rows.append(row_dict)

    df_mis = pd.DataFrame(rows)
    if not df_mis.empty and "Prediction_Confidence" in df_mis.columns and df_mis["Prediction_Confidence"].notnull().any():
        df_mis = df_mis.sort_values("Prediction_Confidence", ascending=False).reset_index(drop=True)
    return df_mis


def analyze_confidence_by_correctness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compares the mean and median confidence for correct vs incorrect predictions.
    """
    correct_mask = (y_true == y_pred)
    correct_conf = np.max(y_proba[correct_mask], axis=1) if np.any(correct_mask) else np.array([0.0])
    error_conf = np.max(y_proba[~correct_mask], axis=1) if np.any(~correct_mask) else np.array([0.0])

    return {
        "mean_confidence_correct": round(float(np.mean(correct_conf)), 4),
        "median_confidence_correct": round(float(np.median(correct_conf)), 4),
        "mean_confidence_error": round(float(np.mean(error_conf)), 4),
        "median_confidence_error": round(float(np.median(error_conf)), 4),
        "confidence_separation_gap": round(float(np.mean(correct_conf) - np.mean(error_conf)), 4),
    }
