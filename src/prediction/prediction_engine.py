"""
Real-Time Prediction Engine for BEANWISE.

Loads the saved machine learning pipeline and metadata, validates user inputs,
computes domain engineered features, and executes low-latency inference.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.features.feature_engineering import add_engineered_features
from src.models.model_persistence import load_model_pipeline, load_label_encoder, load_model_metadata
from src.prediction.input_validator import validate_single_input, EXPECTED_16_FEATURES
from src.prediction.confidence_analyzer import analyze_probabilities
from src.prediction.variety_info import get_variety_info


class PredictionEngine:
    """
    Stateful and reusable prediction engine that holds the loaded pipeline and encoder.
    """

    def __init__(self, model_name: str = "best_model"):
        self.model_name = model_name
        self.model = None
        self.encoder = None
        self.metadata = None
        self.class_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model pipeline, label encoder, and metadata from disk."""
        self.model = load_model_pipeline(self.model_name)
        self.encoder = load_label_encoder("target_label_encoder")
        self.class_names = list(self.encoder.classes_)
        try:
            self.metadata = load_model_metadata(self.model_name)
        except Exception:
            self.metadata = {}

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end validation, feature augmentation, and prediction for a single observation.
        """
        t0 = time.perf_counter()

        # 1. Validation
        is_valid, errors, warnings, sanitized = validate_single_input(input_dict)
        if not is_valid:
            return {
                "success": False,
                "errors": errors,
                "warnings": warnings,
                "predicted_class": None,
                "variety_info": None,
                "top_probability": None,
                "class_probabilities": [],
                "confidence_level": None,
                "inference_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # 2. DataFrame formulation and domain feature engineering
        df_raw = pd.DataFrame([sanitized])[EXPECTED_16_FEATURES]
        df_augmented = add_engineered_features(df_raw)

        # 3. Model Prediction and Confidence Analysis
        analysis_list = analyze_probabilities(self.model, df_augmented, self.class_names)
        res = analysis_list[0]
        predicted_class = res["predicted_class"]

        # 4. Variety Metadata Profile
        variety_info = get_variety_info(predicted_class)

        t_elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "success": True,
            "errors": [],
            "warnings": warnings,
            "predicted_class": predicted_class,
            "variety_info": variety_info,
            "top_probability": res["top_probability"],
            "margin_to_second": res["margin_to_second"],
            "confidence_level": res["confidence_level"],
            "confidence_color": res["confidence_color"],
            "class_probabilities": res["class_probabilities"],
            "sanitized_inputs": sanitized,
            "model_identifier": self.model_name,
            "inference_time_ms": round(t_elapsed_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Global singleton instance for streamlined reuse
_ENGINE_INSTANCE: Optional[PredictionEngine] = None


def get_prediction_engine(model_name: str = "best_model") -> PredictionEngine:
    """Returns singleton PredictionEngine instance."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None or _ENGINE_INSTANCE.model_name != model_name:
        _ENGINE_INSTANCE = PredictionEngine(model_name)
    return _ENGINE_INSTANCE
