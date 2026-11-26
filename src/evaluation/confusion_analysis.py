"""
Confusion Analysis Module for BEANWISE.

Computes raw and row-normalized confusion matrices, isolates top confused
cultivar pairs, and quantifies inter-variety misclassification patterns.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Computes raw and normalized confusion matrix DataFrames.
    """
    labels = list(range(len(class_names)))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize by true row sums
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_raw.astype(float),
        row_sums,
        out=np.zeros_like(cm_raw, dtype=float),
        where=row_sums != 0,
    )

    df_raw = pd.DataFrame(cm_raw, index=class_names, columns=class_names)
    df_norm = pd.DataFrame(np.round(cm_norm, 4), index=class_names, columns=class_names)

    return {
        "raw_confusion_matrix": df_raw,
        "normalized_confusion_matrix": df_norm,
    }


def find_top_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identifies the most frequent misclassification pairs (True != Predicted).
    """
    labels = list(range(len(class_names)))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)

    confused_records = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_raw[i, j] > 0:
                confused_records.append({
                    "True Cultivar": class_names[i],
                    "Predicted Cultivar": class_names[j],
                    "Misclassified Count": int(cm_raw[i, j]),
                    "Misclassified % of Class": round((cm_raw[i, j] / cm_raw[i, :].sum()) * 100, 2),
                })

    confused_df = pd.DataFrame(confused_records)
    if not confused_df.empty:
        confused_df = confused_df.sort_values("Misclassified Count", ascending=False).head(top_n).reset_index(drop=True)
    return confused_df
