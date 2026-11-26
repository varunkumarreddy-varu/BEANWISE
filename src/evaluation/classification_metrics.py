"""
Classification Metrics Module for BEANWISE.

Calculates comprehensive classification metrics for multiclass dry bean evaluation,
including Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall,
Matthews Correlation Coefficient (MCC), Cohen's Kappa, and per-class breakdowns.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    log_loss,
    roc_auc_score,
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes a complete suite of multiclass evaluation metrics.
    """
    labels = sorted(list(np.unique(np.concatenate([y_true, y_pred]))))
    target_names = [class_names[i] if class_names and i < len(class_names) else str(i) for i in labels]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # Optional Log Loss and Multiclass ROC-AUC (One-vs-Rest)
    loss = None
    roc_auc = None
    if y_proba is not None and y_proba.shape[1] == len(labels):
        try:
            loss = float(log_loss(y_true, y_proba))
        except Exception:
            pass
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            pass

    # Per-Class Breakdown
    rep_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_list = []
    for cls in target_names:
        if cls in rep_dict:
            per_class_list.append({
                "Cultivar": cls,
                "Precision": round(rep_dict[cls]["precision"], 4),
                "Recall": round(rep_dict[cls]["recall"], 4),
                "F1-Score": round(rep_dict[cls]["f1-score"], 4),
                "Support": int(rep_dict[cls]["support"]),
            })

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
        "matthews_corrcoef": round(mcc, 4),
        "cohens_kappa": round(kappa, 4),
        "log_loss": round(loss, 4) if loss is not None else None,
        "roc_auc_ovr": round(roc_auc, 4) if roc_auc is not None else None,
        "per_class_metrics": per_class_list,
        "classification_report_dict": rep_dict,
    }
