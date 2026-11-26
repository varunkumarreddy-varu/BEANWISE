"""
Automated Pytest Suite for BEANWISE Phase 5 Model Development and Training.

Verifies:
- Pipeline factory instantiations for all 9 candidate algorithms
- Cross-validation metric structures and non-empty outputs
- Model persistence and load round-trip equivalence
- Presence and non-trivial size of all 8 tables, 7 figures, and reports
- Integrity and absolute separation of the held-out test partition
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
    ENCODER_DIR,
    ARTIFACTS_DIR,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from src.models.model_factory import get_model_pipeline, get_all_model_pipelines
from src.models.model_trainer import cross_validate_pipelines
from src.models.model_persistence import (
    save_model_pipeline,
    load_model_pipeline,
    load_label_encoder,
)

METADATA_DIR = ARTIFACTS_DIR / "metadata"


def test_model_factory_pipelines():
    """Verify all 9 models instantiate as valid Scikit-learn Pipelines."""
    pipelines = get_all_model_pipelines(random_state=42)
    assert len(pipelines) == 9

    expected_models = [
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
    for m in expected_models:
        assert m in pipelines
        pipe = pipelines[m]
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")
        assert "classifier" in pipe.named_steps


def test_cross_validation_execution():
    """Verify cross-validation executes and returns well-formed DataFrame."""
    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    assert cleaned_path.is_file()
    df = pd.read_csv(cleaned_path).sample(210, random_state=42)

    X = df.drop(columns=[TARGET_COLUMN])
    from sklearn.preprocessing import LabelEncoder
    y = pd.Series(LabelEncoder().fit_transform(df[TARGET_COLUMN]))

    mini_pipes = {
        "lr": get_model_pipeline("logistic_regression", random_state=42),
        "dt": get_model_pipeline("decision_tree", random_state=42),
    }

    summary_df, raw_results = cross_validate_pipelines(mini_pipes, X, y, cv=3, random_state=42)
    assert len(summary_df) == 2
    assert "CV Mean Accuracy" in summary_df.columns
    assert "CV Mean Macro F1" in summary_df.columns
    assert all(summary_df["CV Mean Accuracy"] > 0)


def test_model_persistence_roundtrip(tmp_path):
    """Verify pipeline save and load round-trip generates identical predictions."""
    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    df = pd.read_csv(cleaned_path).sample(140, random_state=42)
    X = df.drop(columns=[TARGET_COLUMN])
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])

    pipe = get_model_pipeline("decision_tree", random_state=42)
    pipe.fit(X, y)

    saved_file = save_model_pipeline(pipe, "test_dummy_model", directory=tmp_path)
    assert saved_file.is_file()

    loaded_pipe = load_model_pipeline("test_dummy_model", directory=tmp_path)
    orig_preds = pipe.predict(X)
    loaded_preds = loaded_pipe.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_phase5_artifacts_exist():
    """Verify all Phase 5 tables, figures, models, reports, and notebook exist."""
    # 8 Tables
    tables = [
        "phase5_baseline_cv_results.csv",
        "phase5_tuned_model_comparison.csv",
        "phase5_validation_metrics.csv",
        "phase5_per_class_metrics.csv",
        "phase5_training_times.csv",
        "phase5_confusion_matrix.csv",
        "phase5_hyperparameter_configurations.csv",
        "phase5_model_rankings.csv",
    ]
    for t in tables:
        path = TABLES_DIR / t
        assert path.is_file(), f"Missing table: {t}"
        assert path.stat().st_size > 0

    # 7 Figures
    figures = [
        "phase5_model_accuracy_comparison.png",
        "phase5_model_f1_comparison.png",
        "phase5_cv_variance_boxplot.png",
        "phase5_validation_confusion_matrix.png",
        "phase5_training_time_vs_f1.png",
        "phase5_per_class_f1_heatmap.png",
        "phase5_feature_importance_best_model.png",
    ]
    for f in figures:
        path = FIGURES_DIR / f
        assert path.is_file(), f"Missing figure: {f}"
        assert path.stat().st_size > 5000

    # Models & Encoders
    assert (MODEL_ARTIFACTS_DIR / "best_model.joblib").is_file()
    assert (ENCODER_DIR / "target_label_encoder.joblib").is_file()
    assert (METADATA_DIR / "best_model_metadata.json").is_file()

    # Reports
    assert (REPORTS_DIR / "phase5" / "training_summary.json").is_file()
    assert (REPORTS_DIR / "phase5" / "model_registry.json").is_file()
    assert (REPORTS_DIR / "phase5" / "model_training_report.html").is_file()

    # Executed Notebook
    nb_path = NOTEBOOKS_DIR / "05_Machine_Learning_Model_Training.ipynb"
    assert nb_path.is_file()
    assert nb_path.stat().st_size > 15000


def test_held_out_test_set_unmodified():
    """Verify test partition remains strictly untouched and disjoint."""
    split_meta_file = PROCESSED_DATASET_DIR / "split_metadata.json"
    assert split_meta_file.is_file()
    with open(split_meta_file, "r") as f:
        meta = json.load(f)

    train_idx = set(meta["train_indices"])
    val_idx = set(meta["val_indices"])
    test_idx = set(meta["test_indices"])

    assert len(train_idx) == 9479
    assert len(val_idx) == 2032
    assert len(test_idx) == 2032

    # Absolute disjointness
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0
