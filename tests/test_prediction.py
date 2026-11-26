"""
Automated Pytest Suite for BEANWISE Prediction Engine, Validation, and Variety Assets.

Verifies:
- PredictionEngine instantiation and inference consistency
- Field validation and physical constraint enforcement
- Batch CSV prediction functionality
- All 7 dry bean photographs presence and integrity
- Botanical variety metadata lookups
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TARGET_CLASSES, PROCESSED_DATASET_DIR
from src.prediction import (
    get_prediction_engine,
    predict_batch_csv,
    validate_single_input,
    validate_batch_dataframe,
    get_variety_info,
    get_all_varieties,
    EXPECTED_16_FEATURES,
)


@pytest.fixture(scope="module")
def valid_sample():
    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    assert cleaned_path.is_file()
    df = pd.read_csv(cleaned_path)
    return df.iloc[0].drop("Class").to_dict()


def test_prediction_engine_single(valid_sample):
    """Verify single prediction runs and returns structured dictionary."""
    engine = get_prediction_engine("best_model")
    res = engine.predict_single(valid_sample)

    assert res["success"] is True
    assert res["predicted_class"] in TARGET_CLASSES
    assert res["top_probability"] is not None
    assert 0.0 <= res["top_probability"] <= 1.0
    assert len(res["class_probabilities"]) == 7
    assert res["variety_info"]["name"] == res["predicted_class"]
    assert res["inference_time_ms"] > 0


def test_input_validator_rejects_invalid_values(valid_sample):
    """Verify input validator flags non-numeric, negative, and inconsistent values."""
    # Negative value
    bad_sample_1 = valid_sample.copy()
    bad_sample_1["Area"] = -100.0
    is_valid, errors, _, _ = validate_single_input(bad_sample_1)
    assert not is_valid
    assert any("strictly positive" in e for e in errors)

    # Inconsistent major < minor
    bad_sample_2 = valid_sample.copy()
    bad_sample_2["MajorAxisLength"] = 150.0
    bad_sample_2["MinorAxisLength"] = 250.0
    is_valid, errors, _, _ = validate_single_input(bad_sample_2)
    assert not is_valid
    assert any("Major Axis Length" in e for e in errors)

    # Inconsistent ConvexArea < Area
    bad_sample_3 = valid_sample.copy()
    bad_sample_3["Area"] = 50000.0
    bad_sample_3["ConvexArea"] = 40000.0
    is_valid, errors, _, _ = validate_single_input(bad_sample_3)
    assert not is_valid
    assert any("Convex Area" in e for e in errors)


def test_batch_prediction_csv():
    """Verify batch prediction processes DataFrame and appends predictions."""
    template_path = PROJECT_ROOT / "assets" / "templates" / "batch_prediction_template.csv"
    assert template_path.is_file()
    df_template = pd.read_csv(template_path)

    success, out_df, summary, errors = predict_batch_csv(df_template)
    assert success is True
    assert len(out_df) == len(df_template)
    assert "Predicted_Cultivar" in out_df.columns
    assert "Prediction_Confidence" in out_df.columns
    assert all(c in TARGET_CLASSES for c in out_df["Predicted_Cultivar"])
    assert summary["total_records"] == len(df_template)


def test_bean_variety_images_exist():
    """Verify all 7 bean photographs exist and are non-empty."""
    img_dir = PROJECT_ROOT / "assets" / "bean_images"
    assert img_dir.is_dir()

    expected_files = [
        "barbunya.jpg",
        "bombay.jpg",
        "cali.jpg",
        "dermason.jpg",
        "horoz.jpg",
        "seker.jpg",
        "sira.jpg",
    ]
    for fn in expected_files:
        p = img_dir / fn
        assert p.is_file(), f"Missing image asset: {fn}"
        assert p.stat().st_size > 10000, f"Image {fn} appears too small or empty"

    assert (img_dir / "IMAGE_SOURCES.md").is_file()


def test_variety_info_metadata():
    """Verify variety metadata profiles exist for all 7 classes."""
    varieties = get_all_varieties()
    assert len(varieties) == 7

    for cls in TARGET_CLASSES:
        info = get_variety_info(cls)
        assert info["name"] == cls
        assert "scientific_name" in info
        assert "description" in info
        assert info["image_path"].is_file()
