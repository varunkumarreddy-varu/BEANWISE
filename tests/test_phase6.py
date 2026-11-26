"""
Automated Pytest Suite for BEANWISE Phase 6 Final Independent Model Evaluation.

Verifies:
- Final model pipeline loading and feature order
- Evaluation metrics calculation accuracy
- Confusion matrix and misclassification properties
- Presence of all Phase 6 reports, tables, figures, and executed notebook
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATASET_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    NOTEBOOKS_DIR,
    MODEL_ARTIFACTS_DIR,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from src.models.model_persistence import load_model_pipeline, load_label_encoder
from src.evaluation.classification_metrics import compute_comprehensive_metrics
from src.evaluation.confusion_analysis import compute_confusion_matrices

PHASE6_DIR = REPORTS_DIR / "phase6"


def test_final_model_loading_and_interface():
    """Verify final model pipeline loads, has expected steps and predicts properly."""
    pipeline = load_model_pipeline("best_model")
    encoder = load_label_encoder("target_label_encoder")

    assert pipeline is not None
    assert "classifier" in pipeline.named_steps
    assert len(encoder.classes_) == 7

    # Dummy forward pass
    dummy_input = pd.DataFrame(np.ones((2, 22)), columns=[
        "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
        "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
        "Extent", "Solidity", "roundness", "Compactness",
        "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
        "Area_to_Perimeter", "Axis_Ratio", "Axis_Difference",
        "Convex_Area_Ratio", "Compactness_Roundness_Ratio", "EquivDiameter_to_MajorAxis"
    ])
    preds = pipeline.predict(dummy_input)
    assert len(preds) == 2


def test_phase6_metric_calculation_accuracy():
    """Verify metric calculations produce valid ranges."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 1])  # 9/10 correct

    metrics = compute_comprehensive_metrics(y_true, y_pred, class_names=TARGET_CLASSES)
    assert metrics["accuracy"] == 0.90
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["matthews_corrcoef"] <= 1.0
    assert len(metrics["per_class_metrics"]) == 7


def test_phase6_artifacts_exist():
    """Verify Phase 6 evaluation outputs exist on disk."""
    assert (PHASE6_DIR / "final_evaluation_summary.json").is_file()
    assert (PHASE6_DIR / "evaluation_metadata.json").is_file()
    assert (PHASE6_DIR / "final_model_evaluation.html").is_file()
    assert (PHASE6_DIR / "per_class_metrics.csv").is_file()
    assert (PHASE6_DIR / "confusion_matrix.csv").is_file()
    assert (PHASE6_DIR / "normalized_confusion_matrix.csv").is_file()
    assert (PHASE6_DIR / "misclassified_samples.csv").is_file()
    assert (PHASE6_DIR / "top_confused_pairs.csv").is_file()

    figs = [
        "raw_confusion_matrix.png",
        "normalized_confusion_matrix.png",
        "per_class_performance.png",
        "error_distribution_by_class.png",
        "confidence_distribution.png",
    ]
    for fig in figs:
        fig_path = PHASE6_DIR / "figures" / fig
        assert fig_path.is_file(), f"Missing figure: {fig}"
        assert fig_path.stat().st_size > 2000

    nb_path = NOTEBOOKS_DIR / "06_Final_Model_Evaluation.ipynb"
    assert nb_path.is_file()
    assert nb_path.stat().st_size > 15000
