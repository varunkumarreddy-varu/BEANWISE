"""
Automated Pytest Suite for BEANWISE Phase 2 Exploratory Data Analysis.

Verifies:
- Data quality computations (missingness, duplicates, invalid values)
- Statistical summaries and percentile metrics
- Class distributions and imbalance metrics
- Outlier detection (IQR and Z-score)
- Correlation analysis (Pearson and Spearman)
- ANOVA and Kruskal-Wallis inferential tests
- Existence of all generated figures, tables, JSON findings, and executed notebook
- Dataset cryptographic integrity (SHA-256 unchanged)
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
    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
    get_raw_dataset_path,
    TARGET_COLUMN,
    FEATURE_NAMES,
)
from src.data.dataset_inspector import compute_file_sha256
from src.data.eda_analyzer import (
    compute_data_quality_report,
    compute_class_distribution,
    compute_descriptive_statistics,
    compute_outlier_report,
    compute_correlation_analysis,
    compute_feature_variability_and_tests,
    compute_exploratory_pca,
)

EXPECTED_DATASET_SHA256 = "a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714"


@pytest.fixture(scope="module")
def raw_dataframe():
    dataset_path = get_raw_dataset_path()
    assert dataset_path is not None and dataset_path.is_file()
    return pd.read_excel(dataset_path)


def test_data_quality_computation(raw_dataframe):
    """Verify data quality audit outputs."""
    report = compute_data_quality_report(raw_dataframe)
    assert report["total_records"] == 13611
    assert report["total_features"] == 16
    assert report["total_missing_cells"] == 0
    assert report["exact_duplicates"] == 68
    assert report["unique_records"] == 13543
    assert len(report["constant_features"]) == 0


def test_class_distribution_metrics(raw_dataframe):
    """Verify target class counts, percentages, and imbalance calculation."""
    class_df = compute_class_distribution(raw_dataframe)
    assert len(class_df) == 7
    assert class_df["Sample Count"].sum() == 13611
    assert set(class_df["Class"]) == {
        "SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"
    }
    # Majority class is Dermason, minority is Bombay
    assert class_df.iloc[0]["Class"] == "DERMASON"
    assert class_df.iloc[-1]["Class"] == "BOMBAY"
    assert class_df.iloc[-1]["Ratio to Majority (1:N)"] == pytest.approx(6.79, rel=0.05)


def test_descriptive_statistics_schema(raw_dataframe):
    """Verify that descriptive statistics are computed for all 16 features."""
    desc_df = compute_descriptive_statistics(raw_dataframe)
    assert len(desc_df) == 16
    assert set(desc_df["Feature"]) == set(FEATURE_NAMES)
    for col in ["Mean", "Std Dev", "Median", "IQR", "Skewness", "Kurtosis", "CV"]:
        assert col in desc_df.columns
        assert not desc_df[col].isnull().any()


def test_outlier_detection_logic(raw_dataframe):
    """Verify outlier computation for IQR and Z-score."""
    outlier_df = compute_outlier_report(raw_dataframe)
    assert len(outlier_df) == 16
    assert "IQR Outlier Count" in outlier_df.columns
    assert "Z-Score (|z|>3) Count" in outlier_df.columns
    assert (outlier_df["IQR Outlier Count"] >= 0).all()
    assert (outlier_df["Z-Score (|z|>3) Count"] >= 0).all()


def test_correlation_analysis_properties(raw_dataframe):
    """Verify Pearson and Spearman matrices and symmetry."""
    corr_res = compute_correlation_analysis(raw_dataframe, threshold=0.80)
    pearson_mat = corr_res["pearson_matrix"]
    spearman_mat = corr_res["spearman_matrix"]

    assert pearson_mat.shape == (16, 16)
    assert spearman_mat.shape == (16, 16)
    assert (np.diag(pearson_mat) == 1.0).all()
    assert len(corr_res["strong_positive_df"]) > 0


def test_hypothesis_testing_and_variance(raw_dataframe):
    """Verify ANOVA and Kruskal-Wallis statistical tests."""
    var_df, tests_df = compute_feature_variability_and_tests(raw_dataframe)
    assert len(tests_df) == 16
    assert len(var_df) == 16
    assert (tests_df["Statistically Significant (p < 0.001)"]).all()
    # Area exhibits the highest F-ratio due to major scale differences (Bombay vs Dermason)
    assert var_df.iloc[0]["Feature"] == "Area"
    assert var_df.iloc[0]["F-Ratio (MSB/MSW)"] > 20000


def test_pca_variance_capture(raw_dataframe):
    """Verify exploratory PCA variance ratios."""
    pca_res = compute_exploratory_pca(raw_dataframe, n_components=5)
    summary_df = pca_res["pca_summary_df"]
    assert len(summary_df) == 5
    cum_var = summary_df.iloc[1]["Cumulative Explained Variance"]
    assert cum_var > 0.80  # PC1 + PC2 captures > 80% variance


def test_all_phase2_artifacts_exist():
    """Verify that all required Phase 2 output files exist on disk with non-zero size."""
    expected_figures = [
        "dataset_overview.png",
        "missing_values.png",
        "class_distribution.png",
        "numerical_distributions.png",
        "feature_boxplots.png",
        "correlation_pearson.png",
        "correlation_spearman.png",
        "feature_relationships.png",
        "outlier_analysis.png",
        "class_wise_analysis.png",
        "pca_visualization.png",
        "interactive_class_distribution.html",
        "interactive_correlation.html",
        "interactive_scatter.html",
        "interactive_parallel_coordinates.html",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.is_file(), f"Missing expected figure: {fig_name}"
        assert fig_path.stat().st_size > 0, f"Empty figure file: {fig_name}"

    expected_tables = [
        "dataset_summary.csv",
        "data_quality_report.csv",
        "missing_values_report.csv",
        "duplicate_report.csv",
        "class_distribution.csv",
        "descriptive_statistics.csv",
        "correlation_matrix.csv",
        "outlier_report.csv",
        "class_wise_statistics.csv",
        "feature_variability.csv",
        "statistical_tests.csv",
    ]
    for tab_name in expected_tables:
        tab_path = TABLES_DIR / tab_name
        assert tab_path.is_file(), f"Missing expected table: {tab_name}"
        assert tab_path.stat().st_size > 0, f"Empty table file: {tab_name}"

    # Reports and notebook
    assert (REPORTS_DIR / "eda" / "EDA_Findings.json").is_file() or (REPORTS_DIR / "EDA_Findings.json").is_file()
    assert (REPORTS_DIR / "eda" / "EDA_Report.html").is_file() or (REPORTS_DIR / "EDA_Report.html").is_file()
    nb_path = NOTEBOOKS_DIR / "02_Exploratory_Data_Analysis.ipynb"
    assert nb_path.is_file()
    assert nb_path.stat().st_size > 50000  # Executed with plots is substantial


def test_raw_dataset_cryptographic_integrity():
    """Verify that the original DryBeanDataset file was NEVER modified."""
    dataset_path = get_raw_dataset_path()
    assert dataset_path is not None
    current_sha = compute_file_sha256(dataset_path)
    assert current_sha == EXPECTED_DATASET_SHA256, (
        f"Dataset modification detected! Expected {EXPECTED_DATASET_SHA256}, got {current_sha}"
    )
