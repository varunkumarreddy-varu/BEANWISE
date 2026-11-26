"""
Orchestration Script to Generate All Phase 2 EDA Figures, Tables, and Structured Artifacts.
Runs non-destructively on the raw Dry Bean dataset.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    get_raw_dataset_path,
    TARGET_COLUMN,
    FEATURE_NAMES,
)
from src.data.eda_analyzer import (
    compute_dataset_summary,
    compute_data_quality_report,
    compute_class_distribution,
    compute_descriptive_statistics,
    compute_outlier_report,
    compute_correlation_analysis,
    compute_class_wise_statistics,
    compute_feature_variability_and_tests,
    compute_exploratory_pca,
)
from src.visualization.eda_visualizer import (
    plot_dataset_overview,
    plot_missing_values,
    plot_class_distribution,
    plot_numerical_distributions,
    plot_feature_boxplots,
    plot_correlation_matrices,
    plot_bivariate_relationships,
    plot_outlier_summary,
    plot_class_wise_violins,
    plot_exploratory_pca,
    plot_parallel_coordinates,
)


def run_eda_generation():
    print("Starting Phase 2 Artifact Generation...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = get_raw_dataset_path()
    if not dataset_path or not dataset_path.is_file():
        raise FileNotFoundError(f"Raw dataset could not be located at {dataset_path}")

    print(f"Loading raw dataset from {dataset_path}...")
    df = pd.read_excel(dataset_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 1. Structural Summary & Data Quality
    print("1/11 Computing dataset summary and data quality reports...")
    summary_df = compute_dataset_summary(df)
    summary_df.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    quality_dict = compute_data_quality_report(df)
    quality_df = pd.DataFrame([{
        "Metric": "Total Records", "Value": quality_dict["total_records"]
    }, {
        "Metric": "Total Features (Input)", "Value": quality_dict["total_features"]
    }, {
        "Metric": "Total Cells", "Value": quality_dict["total_cells"]
    }, {
        "Metric": "Missing Cells", "Value": quality_dict["total_missing_cells"]
    }, {
        "Metric": "Missing Percentage", "Value": f"{quality_dict['missing_percentage']}%"
    }, {
        "Metric": "Exact Duplicate Records", "Value": quality_dict["exact_duplicates"]
    }, {
        "Metric": "Duplicate Percentage", "Value": f"{quality_dict['duplicate_percentage']}%"
    }, {
        "Metric": "Unique Records", "Value": quality_dict["unique_records"]
    }, {
        "Metric": "Constant Features", "Value": len(quality_dict["constant_features"])
    }, {
        "Metric": "Low Variance Features", "Value": len(quality_dict["low_variance_features"])
    }])
    quality_df.to_csv(TABLES_DIR / "data_quality_report.csv", index=False)

    # Missing values report
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().sum().values / len(df) * 100).round(4),
    })
    missing_df.to_csv(TABLES_DIR / "missing_values_report.csv", index=False)

    # Duplicate report
    dup_indices = df[df.duplicated(keep=False)].index.tolist()
    dup_df = pd.DataFrame({
        "Total Records": [len(df)],
        "Exact Duplicates": [quality_dict["exact_duplicates"]],
        "Duplicate Pct": [quality_dict["duplicate_percentage"]],
        "Unique Records": [quality_dict["unique_records"]],
        "Note": ["Duplicates preserved in Phase 2; not deleted"],
    })
    dup_df.to_csv(TABLES_DIR / "duplicate_report.csv", index=False)

    # 2. Class Distribution
    print("2/11 Computing class distributions...")
    class_df = compute_class_distribution(df)
    class_df.to_csv(TABLES_DIR / "class_distribution.csv", index=False)

    # 3. Descriptive Statistics
    print("3/11 Computing descriptive statistics...")
    desc_df = compute_descriptive_statistics(df)
    desc_df.to_csv(TABLES_DIR / "descriptive_statistics.csv", index=False)

    # 4. Outlier Analysis
    print("4/11 Computing outlier reports...")
    outlier_df = compute_outlier_report(df)
    outlier_df.to_csv(TABLES_DIR / "outlier_report.csv", index=False)

    # 5. Correlation Analysis
    print("5/11 Computing correlation matrices...")
    corr_results = compute_correlation_analysis(df, threshold=0.80)
    corr_results["pearson_matrix"].to_csv(TABLES_DIR / "correlation_matrix.csv")

    # 6. Class-Wise Statistics
    print("6/11 Computing class-wise statistics...")
    class_wise_df = compute_class_wise_statistics(df)
    class_wise_df.to_csv(TABLES_DIR / "class_wise_statistics.csv", index=False)

    # 7. Feature Variability and Statistical Tests (ANOVA & Kruskal-Wallis)
    print("7/11 Computing variance decomposition and ANOVA/Kruskal-Wallis tests...")
    var_df, tests_df = compute_feature_variability_and_tests(df)
    var_df.to_csv(TABLES_DIR / "feature_variability.csv", index=False)
    tests_df.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)

    # 8. Exploratory PCA
    print("8/11 Computing exploratory PCA...")
    pca_results = compute_exploratory_pca(df)

    # 9. Generating Visualizations
    print("9/11 Generating static and interactive plots...")
    plot_dataset_overview(df, FIGURES_DIR)
    plot_missing_values(df, FIGURES_DIR)
    plot_class_distribution(df, FIGURES_DIR)
    plot_numerical_distributions(df, FIGURES_DIR)
    plot_feature_boxplots(df, FIGURES_DIR)
    plot_correlation_matrices(corr_results["pearson_matrix"], corr_results["spearman_matrix"], FIGURES_DIR)
    plot_bivariate_relationships(df, FIGURES_DIR)
    plot_outlier_summary(outlier_df, FIGURES_DIR)
    plot_class_wise_violins(df, FIGURES_DIR)
    plot_exploratory_pca(
        pca_results["X_pca_2d"], df, pca_results["var_pc1"], pca_results["var_pc2"], FIGURES_DIR
    )
    plot_parallel_coordinates(df, FIGURES_DIR)

    # 10. Generating Structured JSON Findings
    print("10/11 Saving structured EDA findings JSON...")
    findings = {
        "dataset_metadata": {
            "filename": dataset_path.name,
            "total_records": len(df),
            "total_columns": len(df.columns),
            "numerical_features_count": len(FEATURE_NAMES),
            "target_column": TARGET_COLUMN,
            "target_classes_count": int(df[TARGET_COLUMN].nunique()),
            "classes": sorted(df[TARGET_COLUMN].unique().tolist()),
        },
        "data_quality": {
            "total_missing_cells": quality_dict["total_missing_cells"],
            "missing_percentage": quality_dict["missing_percentage"],
            "exact_duplicate_count": quality_dict["exact_duplicates"],
            "exact_duplicate_percentage": quality_dict["duplicate_percentage"],
            "constant_features": quality_dict["constant_features"],
            "low_variance_features": quality_dict["low_variance_features"],
        },
        "class_imbalance": {
            "majority_class": class_df.iloc[0]["Class"],
            "majority_count": int(class_df.iloc[0]["Sample Count"]),
            "minority_class": class_df.iloc[-1]["Class"],
            "minority_count": int(class_df.iloc[-1]["Sample Count"]),
            "imbalance_ratio": float(class_df.iloc[-1]["Ratio to Majority (1:N)"]),
        },
        "strong_correlations": {
            "strong_positive_pairs_count": len(corr_results["strong_positive_df"]),
            "strong_negative_pairs_count": len(corr_results["strong_negative_df"]),
            "top_positive_pairs": corr_results["strong_positive_df"].head(5).to_dict(orient="records"),
            "top_negative_pairs": corr_results["strong_negative_df"].head(5).to_dict(orient="records"),
        },
        "pca_exploratory_variance": {
            "pc1_variance_pct": pca_results["var_pc1"],
            "pc2_variance_pct": pca_results["var_pc2"],
            "cumulative_2pc_variance_pct": round(pca_results["var_pc1"] + pca_results["var_pc2"], 2),
        },
        "top_discriminative_features_by_f_ratio": var_df.head(5)[["Feature", "F-Ratio (MSB/MSW)", "Eta-Squared (Variance Explained)"]].to_dict(orient="records"),
    }

    with open(REPORTS_DIR / "EDA_Findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print("Phase 2 Artifact Generation Completed Successfully.")


if __name__ == "__main__":
    run_eda_generation()
