"""
Automated Pytest Suite for BEANWISE Phase 3 Data Cleaning and Preprocessing.

Verifies:
- Cleaned dataset existence, shape, and integrity
- Absence of duplicate records in cleaned dataset
- Absence of null, infinite, or negative values
- Preservation of all 7 target classes
- Presence of all required Phase 3 reports, tables, and figures
- Raw dataset cryptographic checksum unchanged
"""

import sys
from pathlib import Path
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
    FEATURE_NAMES,
    TARGET_COLUMN,
    TARGET_CLASSES,
    get_raw_dataset_path,
)
from src.data.dataset_inspector import compute_file_sha256
from src.preprocessing.data_validator import validate_dataset

EXPECTED_RAW_SHA256 = "a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714"


@pytest.fixture(scope="module")
def cleaned_df():
    path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    assert path.is_file(), f"Cleaned dataset does not exist at {path}"
    return pd.read_csv(path)


def test_cleaned_dataset_dimensions(cleaned_df):
    """Verify cleaned dataset has exactly 13,543 rows and 17 columns."""
    assert len(cleaned_df) == 13543
    assert len(cleaned_df.columns) == 17
    assert TARGET_COLUMN in cleaned_df.columns


def test_cleaned_dataset_duplicates(cleaned_df):
    """Verify zero duplicate rows exist in cleaned dataset."""
    assert cleaned_df.duplicated().sum() == 0


def test_cleaned_dataset_integrity(cleaned_df):
    """Verify programmatic validation passes without errors."""
    res = validate_dataset(cleaned_df)
    assert res["is_valid"] is True
    assert set(res["target_classes_present"]) == set(TARGET_CLASSES)


def test_phase3_artifacts_exist():
    """Verify Phase 3 reports, tables, figures, and executed notebook exist."""
    assert (REPORTS_DIR / "phase3" / "data_cleaning_report.html").is_file()
    assert (REPORTS_DIR / "phase3" / "cleaning_summary.json").is_file()

    tables = [
        "phase3_quality_summary.csv",
        "phase3_missing_values.csv",
        "phase3_duplicate_summary.csv",
        "phase3_invalid_values.csv",
        "phase3_outlier_decisions.csv",
        "phase3_class_distribution.csv",
    ]
    for t in tables:
        assert (TABLES_DIR / t).is_file(), f"Missing table: {t}"

    figures = [
        "phase3_data_quality.png",
        "phase3_class_comparison.png",
        "phase3_cleaning_comparison.png",
    ]
    for f in figures:
        assert (FIGURES_DIR / f).is_file(), f"Missing figure: {f}"

    nb = NOTEBOOKS_DIR / "03_Data_Cleaning_Preprocessing.ipynb"
    assert nb.is_file(), "Missing executed Phase 3 notebook"
    assert nb.stat().st_size > 5000


def test_raw_dataset_remains_unmodified():
    """Verify original raw dataset checksum remains completely unchanged."""
    raw_path = get_raw_dataset_path()
    assert raw_path is not None
    current_sha = compute_file_sha256(raw_path)
    assert current_sha == EXPECTED_RAW_SHA256
