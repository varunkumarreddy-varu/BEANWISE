"""
Batch Prediction Module for BEANWISE.

Processes bulk CSV uploads containing 16 morphological measurements,
validates schema integrity, executes pipeline batch inference, and produces
structured tabular exports with per-cultivar probabilities.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.features.feature_engineering import add_engineered_features
from src.prediction.input_validator import validate_batch_dataframe, EXPECTED_16_FEATURES
from src.prediction.confidence_analyzer import analyze_probabilities
from src.prediction.prediction_engine import get_prediction_engine


def predict_batch_csv(df_upload: pd.DataFrame, model_name: str = "best_model") -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any], List[str]]:
    """
    Executes end-to-end batch validation and prediction.

    Returns:
        (success, result_df, summary_metrics, error_list)
    """
    t0 = time.perf_counter()

    # 1. Validation
    is_valid, clean_df, errors, warnings = validate_batch_dataframe(df_upload)
    if not is_valid or clean_df is None:
        return False, None, {}, errors

    # 2. Add domain features
    engine = get_prediction_engine(model_name)
    df_augmented = add_engineered_features(clean_df)

    # 3. Model Inference and Probabilities
    analysis_results = analyze_probabilities(engine.model, df_augmented, engine.class_names)

    # 4. Construct Output DataFrame
    output_df = clean_df.copy()
    output_df["Predicted_Cultivar"] = [r["predicted_class"] for r in analysis_results]
    output_df["Prediction_Confidence"] = [r["top_probability"] for r in analysis_results]
    output_df["Confidence_Level"] = [r["confidence_level"] for r in analysis_results]

    # Add per-class probability columns
    for cls in engine.class_names:
        output_df[f"Prob_{cls}"] = [
            round(next((p["probability"] for p in r["class_probabilities"] if p["class"] == cls), 0.0), 4)
            for r in analysis_results
        ]

    duration_sec = time.perf_counter() - t0
    class_distribution = output_df["Predicted_Cultivar"].value_counts().to_dict()

    summary = {
        "total_records": len(output_df),
        "inference_duration_seconds": round(duration_sec, 3),
        "per_sample_latency_ms": round((duration_sec / len(output_df)) * 1000, 3),
        "class_distribution": class_distribution,
        "warnings": warnings,
    }

    return True, output_df, summary, []
