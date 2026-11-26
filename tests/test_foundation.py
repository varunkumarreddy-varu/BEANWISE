"""
Automated Pytest Suite for BEANWISE Phase 1 Foundation.

Verifies:
- Directory existence and integrity
- Configuration constants and feature definitions
- Dependency imports
- Logging utilities
- Dataset inspection and non-destructive SHA-256 integrity
- App compilation
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from config.settings import (
    PROJECT_NAME,
    TARGET_COLUMN,
    TARGET_CLASSES,
    FEATURE_NAMES,
    ALL_DIRECTORIES,
    verify_required_directories,
    get_raw_dataset_path,
)
from src.utils.logger import get_logger, setup_logging
from src.data.dataset_inspector import inspect_dataset, compute_file_sha256

EXPECTED_DATASET_SHA256 = "a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714"


def test_project_root_exists():
    """Verify that the project root directory is accessible."""
    assert PROJECT_ROOT.is_dir(), f"Project root does not exist: {PROJECT_ROOT}"


def test_required_directories_exist():
    """Verify that all required project directories are created."""
    all_ok, missing = verify_required_directories(PROJECT_ROOT)
    assert all_ok, f"Missing directories: {missing}"


def test_configuration_constants():
    """Verify core constants and feature schemas in config.settings."""
    assert PROJECT_NAME == "BEANWISE"
    assert TARGET_COLUMN == "Class"
    assert len(TARGET_CLASSES) == 7
    assert set(TARGET_CLASSES) == {
        "SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"
    }
    assert len(FEATURE_NAMES) == 16
    assert "Area" in FEATURE_NAMES
    assert "AspectRation" in FEATURE_NAMES
    assert "roundness" in FEATURE_NAMES


def test_dependency_imports():
    """Verify that all critical packages import without errors."""
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import xgboost
    import streamlit
    import streamlit_option_menu
    import plotly
    import seaborn
    import matplotlib
    import joblib
    import PIL
    import dotenv
    import missingno

    assert np.__version__ is not None
    assert pd.__version__ is not None


def test_logging_utility(tmp_path):
    """Verify that structured logging initializes safely and writes logs."""
    test_log_path = setup_logging(
        log_file_name="test_audit.log",
        logs_dir=tmp_path,
        force_new_file=True,
    )
    assert test_log_path.parent.is_dir()

    logger = get_logger("test_module")
    logger.info("Verification log message - Test event")
    logger.warning("Verification warning message")

    for handler in logging.getLogger(PROJECT_NAME).handlers:
        handler.flush()

    assert test_log_path.exists()
    content = test_log_path.read_text(encoding="utf-8")
    assert "Verification log message" in content
    assert "Verification warning message" in content


def test_dataset_location_and_inspection():
    """Verify non-destructive dataset inspection and schema compliance."""
    raw_path = get_raw_dataset_path()
    assert raw_path is not None, "Raw Dry Bean dataset could not be located."
    assert raw_path.is_file(), f"Dataset path is not a file: {raw_path}"

    inspection = inspect_dataset(raw_path)
    assert inspection.exists, "Dataset reported as not existing."
    assert inspection.is_readable, f"Failed to read dataset: {inspection.error_message}"
    assert inspection.row_count == 13611, f"Unexpected row count: {inspection.row_count}"
    assert inspection.column_count == 17, f"Unexpected column count: {inspection.column_count}"
    assert inspection.has_target_column, "Missing target column 'Class'"
    assert inspection.all_expected_classes_present, f"Missing classes: {inspection.missing_classes}"
    assert len(inspection.missing_expected_features) == 0, f"Missing features: {inspection.missing_expected_features}"


def test_dataset_sha256_integrity():
    """Verify that the raw dataset has not been modified using SHA-256 comparison."""
    raw_path = get_raw_dataset_path()
    assert raw_path is not None
    current_sha = compute_file_sha256(raw_path)
    assert current_sha == EXPECTED_DATASET_SHA256, (
        f"Dataset checksum mismatch! Expected {EXPECTED_DATASET_SHA256}, got {current_sha}"
    )


def test_app_compilation():
    """Verify that app.py has valid syntax and compiles without error."""
    app_path = PROJECT_ROOT / "app.py"
    assert app_path.is_file(), "app.py does not exist"
    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()
    compile(source, str(app_path), "exec")
