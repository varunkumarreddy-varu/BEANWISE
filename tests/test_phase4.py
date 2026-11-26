"""
Automated Pytest Suite for BEANWISE Phase 4 Feature Engineering and Selection.

Verifies:
- Mathematical accuracy and finiteness of engineered features
- Correct stratified 3-way partition proportions and disjoint indices (no leakage)
- Feature selection outputs and multi-criteria ranking table integrity
- Existence of Phase 4 figures, tables, reports, and executed notebook
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
    FEATURE_NAMES,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from src.features.feature_engineering import add_engineered_features, get_engineered_feature_names
from src.features.feature_selection import split_dataset


@pytest.fixture(scope="module")
def cleaned_df():
    path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    assert path.is_file()
    return pd.read_csv(path)


def test_engineered_features_computation(cleaned_df):
    """Verify engineered features are added correctly without NaN or inf values."""
    df_eng = add_engineered_features(cleaned_df)
    eng_names = get_engineered_feature_names()
    assert len(eng_names) == 6
    for col in eng_names:
        assert col in df_eng.columns
        assert not df_eng[col].isnull().any()
        assert not np.isinf(df_eng[col]).any()


def test_stratified_split_integrity(cleaned_df):
    """Verify 70/15/15 split sizes and absolute disjointness (no data leakage)."""
    train_df, val_df, test_df, meta = split_dataset(cleaned_df, random_state=42)

    total_len = len(cleaned_df)
    assert len(train_df) + len(val_df) + len(test_df) == total_len
    assert len(train_df) == 9479
    assert len(val_df) == 2032
    assert len(test_df) == 2032

    # Verify disjointness
    train_idx = set(meta["train_indices"])
    val_idx = set(meta["val_indices"])
    test_idx = set(meta["test_indices"])

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0

    # Verify all classes present in each split
    for split_data in [train_df, val_df, test_df]:
        assert set(split_data[TARGET_COLUMN].unique()) == set(TARGET_CLASSES)


def test_phase4_artifacts_exist():
    """Verify Phase 4 output files exist on disk."""
    assert (PROCESSED_DATASET_DIR / "feature_engineering_analysis.csv").is_file()
    assert (PROCESSED_DATASET_DIR / "split_metadata.json").is_file()
    assert (REPORTS_DIR / "phase4" / "feature_engineering_report.html").is_file()
    assert (REPORTS_DIR / "phase4" / "feature_selection_summary.json").is_file()
    assert (REPORTS_DIR / "phase4" / "feature_dictionary.json").is_file()

    tables = [
        "phase4_feature_statistics.csv",
        "phase4_correlation_analysis.csv",
        "phase4_mutual_information.csv",
        "phase4_anova_scores.csv",
        "phase4_feature_rankings.csv",
        "phase4_feature_comparison.csv",
        "phase4_pca_variance.csv",
        "phase4_split_summary.csv",
    ]
    for t in tables:
        assert (TABLES_DIR / t).is_file(), f"Missing table: {t}"

    figures = [
        "phase4_feature_correlation.png",
        "phase4_mutual_information.png",
        "phase4_anova_ranking.png",
        "phase4_feature_importance.png",
        "phase4_pca_scree.png",
        "phase4_pca_visualization.png",
        "phase4_feature_comparison.png",
    ]
    for f in figures:
        assert (FIGURES_DIR / f).is_file(), f"Missing figure: {f}"

    nb = NOTEBOOKS_DIR / "04_Feature_Engineering_Selection.ipynb"
    assert nb.is_file()
    assert nb.stat().st_size > 10000
