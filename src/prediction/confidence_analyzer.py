"""
Confidence and Probability Analyzer Module for BEANWISE.

Analyzes model-estimated class probabilities, identifies margin gaps,
and structures probability distributions across the seven bean cultivars.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.special import softmax


def analyze_probabilities(
    model: Any,
    X_processed: pd.DataFrame,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Computes class probability distributions and confidence indicators.

    Returns:
        List of result dictionaries (one per input sample).
    """
    # 1. Compute probabilities or softmax over decision function
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_processed)
        except Exception:
            probas = None
    else:
        probas = None

    if probas is None and hasattr(model, "decision_function"):
        dec = model.decision_function(X_processed)
        if dec.ndim == 1:
            dec = np.vstack([-dec, dec]).T
        probas = softmax(dec, axis=1)

    results = []
    n_samples = len(X_processed)

    for i in range(n_samples):
        if probas is not None:
            p_row = probas[i]
            sorted_indices = np.argsort(p_row)[::-1]
            top_idx = sorted_indices[0]
            top_prob = float(p_row[top_idx])
            second_prob = float(p_row[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = top_prob - second_prob

            # Formulate structured ranking
            class_prob_pairs = [
                {
                    "class": class_names[idx],
                    "probability": round(float(p_row[idx]), 4),
                    "percentage": round(float(p_row[idx]) * 100, 2),
                }
                for idx in sorted_indices
            ]

            # Categorize confidence level
            if top_prob >= 0.75 and margin >= 0.20:
                conf_level = "High Confidence"
                conf_color = "#2ca02c"
            elif top_prob >= 0.50:
                conf_level = "Moderate Confidence"
                conf_color = "#3182ce"
            else:
                conf_level = "Borderline / Low Confidence"
                conf_color = "#d95f02"

            results.append({
                "predicted_index": int(top_idx),
                "predicted_class": class_names[top_idx],
                "top_probability": round(top_prob, 4),
                "margin_to_second": round(margin, 4),
                "confidence_level": conf_level,
                "confidence_color": conf_color,
                "class_probabilities": class_prob_pairs,
                "probabilities_array": p_row.tolist(),
            })
        else:
            # Fallback for models without probability or decision function
            pred = model.predict(X_processed)[i]
            results.append({
                "predicted_index": int(pred),
                "predicted_class": class_names[int(pred)],
                "top_probability": None,
                "margin_to_second": None,
                "confidence_level": "Standard Prediction",
                "confidence_color": "#3182ce",
                "class_probabilities": [],
                "probabilities_array": [],
            })

    return results
