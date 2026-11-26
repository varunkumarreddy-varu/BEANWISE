"""
Exploratory Data Analysis and Statistical Computation Utilities for BEANWISE.

Provides modular functions for data quality audits, descriptive statistics,
outlier detection (IQR and Z-score), correlation analysis (Pearson and Spearman),
class-wise comparisons, variance decomposition (between vs within class),
and inferential statistical tests (ANOVA and Kruskal-Wallis with Bonferroni correction).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import FEATURE_NAMES, TARGET_COLUMN, TARGET_CLASSES


def compute_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes structural summary for each column in the dataset.
    """
    summary_data = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notnull().sum())
        missing = int(series.isnull().sum())
        unique = int(series.nunique())
        dtype_str = str(series.dtype)
        memory_bytes = int(series.memory_usage(deep=True))

        summary_data.append({
            "Column Name": col,
            "Data Type": dtype_str,
            "Non-Null Count": non_null,
            "Missing Count": missing,
            "Missing Pct": round((missing / len(df)) * 100, 4),
            "Unique Count": unique,
            "Memory Usage (KB)": round(memory_bytes / 1024, 2),
        })

    return pd.DataFrame(summary_data)


def compute_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data quality audit including missingness,
    duplicates, infinite/NaN values, and non-positive numerical values.
    """
    total_records = len(df)
    total_cells = df.size
    total_missing = int(df.isnull().sum().sum())
    missing_pct = (total_missing / total_cells) * 100

    exact_duplicates = int(df.duplicated().sum())
    duplicate_pct = (exact_duplicates / total_records) * 100
    unique_records = total_records - exact_duplicates

    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]

    # Check for invalid values
    infinite_counts = {}
    negative_counts = {}
    zero_counts = {}
    for col in numerical_cols:
        series = df[col]
        infinite_counts[col] = int(np.isinf(series).sum())
        negative_counts[col] = int((series < 0).sum())
        zero_counts[col] = int((series == 0).sum())

    # Check for constant or near-constant features (variance < 1e-5)
    variances = {col: float(df[col].var()) for col in numerical_cols}
    constant_features = [col for col, var in variances.items() if var == 0]
    low_variance_features = [col for col, var in variances.items() if 0 < var < 1e-4]

    return {
        "total_records": total_records,
        "total_features": len(numerical_cols),
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "missing_percentage": round(missing_pct, 4),
        "exact_duplicates": exact_duplicates,
        "duplicate_percentage": round(duplicate_pct, 4),
        "unique_records": unique_records,
        "infinite_values_per_col": infinite_counts,
        "negative_values_per_col": negative_counts,
        "zero_values_per_col": zero_counts,
        "constant_features": constant_features,
        "low_variance_features": low_variance_features,
    }


def compute_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates absolute counts, percentages, and imbalance metrics for target classes.
    """
    counts = df[TARGET_COLUMN].value_counts()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100

    class_df = pd.DataFrame({
        "Class": counts.index,
        "Sample Count": counts.values,
        "Percentage": percentages.round(2).values,
    })

    # Add imbalance ratios relative to majority class
    majority_count = class_df["Sample Count"].max()
    class_df["Ratio to Majority (1:N)"] = (majority_count / class_df["Sample Count"]).round(2)

    return class_df


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes extended statistical measures for all numerical features:
    Central tendency, dispersion, percentiles, skewness, kurtosis, and coefficient of variation.
    """
    stats_rows = []
    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]

    for col in numerical_cols:
        s = df[col].dropna()
        n = len(s)
        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1))
        median_val = float(s.median())

        # Mode calculation
        mode_res = stats.mode(s, keepdims=True)
        mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else np.nan

        min_val = float(s.min())
        max_val = float(s.max())
        range_val = max_val - min_val

        p10 = float(np.percentile(s, 10))
        q1 = float(np.percentile(s, 25))
        q2 = median_val
        q3 = float(np.percentile(s, 75))
        p90 = float(np.percentile(s, 90))
        iqr = q3 - q1

        variance_val = float(s.var(ddof=1))
        skew_val = float(stats.skew(s, bias=False))
        kurt_val = float(stats.kurtosis(s, bias=False))  # Excess kurtosis (Fisher)
        cv = (std_val / mean_val) if mean_val != 0 else np.nan

        stats_rows.append({
            "Feature": col,
            "Count": n,
            "Mean": round(mean_val, 4),
            "Std Dev": round(std_val, 4),
            "Variance": round(variance_val, 4),
            "Median": round(median_val, 4),
            "Mode": round(mode_val, 4),
            "Min": round(min_val, 4),
            "P10": round(p10, 4),
            "Q1 (P25)": round(q1, 4),
            "Q2 (Median)": round(q2, 4),
            "Q3 (P75)": round(q3, 4),
            "P90": round(p90, 4),
            "Max": round(max_val, 4),
            "Range": round(range_val, 4),
            "IQR": round(iqr, 4),
            "Skewness": round(skew_val, 4),
            "Kurtosis": round(kurt_val, 4),
            "CV": round(cv, 4),
        })

    return pd.DataFrame(stats_rows)


def compute_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical outlier counts using both IQR and Z-score methods.
    Does not delete or modify any records.
    """
    rows = []
    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]

    for col in numerical_cols:
        s = df[col]
        q1 = float(np.percentile(s, 25))
        q3 = float(np.percentile(s, 75))
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr

        iqr_outliers = int(((s < lower_iqr) | (s > upper_iqr)).sum())
        iqr_pct = round((iqr_outliers / len(s)) * 100, 2)

        # Z-score method
        mean_val = s.mean()
        std_val = s.std(ddof=1)
        z_scores = np.abs((s - mean_val) / std_val) if std_val > 0 else np.zeros_like(s)
        z_outliers = int((z_scores > 3.0).sum())
        z_pct = round((z_outliers / len(s)) * 100, 2)

        rows.append({
            "Feature": col,
            "IQR Lower Bound": round(lower_iqr, 4),
            "IQR Upper Bound": round(upper_iqr, 4),
            "IQR Outlier Count": iqr_outliers,
            "IQR Outlier Pct": iqr_pct,
            "Z-Score (|z|>3) Count": z_outliers,
            "Z-Score Outlier Pct": z_pct,
        })

    return pd.DataFrame(rows)


def compute_correlation_analysis(df: pd.DataFrame, threshold: float = 0.80) -> Dict[str, Any]:
    """
    Calculates Pearson and Spearman correlation matrices and identifies
    strongly correlated feature pairs (|r| >= threshold).
    """
    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]
    pearson_mat = df[numerical_cols].corr(method="pearson")
    spearman_mat = df[numerical_cols].corr(method="spearman")

    strong_positive = []
    strong_negative = []

    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            feat1 = numerical_cols[i]
            feat2 = numerical_cols[j]
            p_corr = float(pearson_mat.loc[feat1, feat2])
            s_corr = float(spearman_mat.loc[feat1, feat2])

            if p_corr >= threshold:
                strong_positive.append({
                    "Feature 1": feat1,
                    "Feature 2": feat2,
                    "Pearson r": round(p_corr, 4),
                    "Spearman rho": round(s_corr, 4),
                })
            elif p_corr <= -threshold:
                strong_negative.append({
                    "Feature 1": feat1,
                    "Feature 2": feat2,
                    "Pearson r": round(p_corr, 4),
                    "Spearman rho": round(s_corr, 4),
                })

    return {
        "pearson_matrix": pearson_mat,
        "spearman_matrix": spearman_mat,
        "strong_positive_df": pd.DataFrame(strong_positive).sort_values(by="Pearson r", ascending=False),
        "strong_negative_df": pd.DataFrame(strong_negative).sort_values(by="Pearson r", ascending=True),
    }


def compute_class_wise_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates summary statistics (mean, median, std, min, max) for each feature
    stratified across the seven dry bean varieties.
    """
    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]
    rows = []

    for cls in sorted(df[TARGET_COLUMN].unique()):
        sub_df = df[df[TARGET_COLUMN] == cls]
        for col in numerical_cols:
            s = sub_df[col]
            rows.append({
                "Class": cls,
                "Feature": col,
                "Mean": round(float(s.mean()), 4),
                "Median": round(float(s.median()), 4),
                "Std Dev": round(float(s.std(ddof=1)), 4),
                "Min": round(float(s.min()), 4),
                "Max": round(float(s.max()), 4),
            })

    return pd.DataFrame(rows)


def compute_feature_variability_and_tests(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs One-way ANOVA and Kruskal-Wallis non-parametric tests across bean varieties
    with Bonferroni multiple-comparison correction.
    Also computes within-class, between-class, and total variance for each feature.
    """
    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]
    classes = sorted(df[TARGET_COLUMN].unique())
    num_tests = len(numerical_cols)

    test_rows = []
    var_rows = []

    for col in numerical_cols:
        grouped_data = [df[df[TARGET_COLUMN] == c][col].values for c in classes]

        # One-way ANOVA
        f_stat, anova_p = stats.f_oneway(*grouped_data)

        # Kruskal-Wallis test (non-parametric alternative)
        kw_stat, kw_p = stats.kruskal(*grouped_data)

        # Bonferroni adjusted p-values
        anova_p_adj = min(1.0, anova_p * num_tests)
        kw_p_adj = min(1.0, kw_p * num_tests)

        test_rows.append({
            "Feature": col,
            "ANOVA F-Statistic": round(float(f_stat), 4),
            "ANOVA p-value": anova_p,
            "ANOVA p-adj (Bonferroni)": anova_p_adj,
            "Kruskal-Wallis H": round(float(kw_stat), 4),
            "KW p-value": kw_p,
            "KW p-adj (Bonferroni)": kw_p_adj,
            "Statistically Significant (p < 0.001)": (anova_p_adj < 0.001 and kw_p_adj < 0.001),
        })

        # Variance decomposition
        grand_mean = df[col].mean()
        total_ss = np.sum((df[col] - grand_mean) ** 2)
        total_var = float(df[col].var(ddof=1))

        # Between-class sum of squares (SSB)
        ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in grouped_data)
        df_between = len(classes) - 1
        msb = ssb / df_between if df_between > 0 else 0

        # Within-class sum of squares (SSW)
        ssw = sum(np.sum((g - np.mean(g)) ** 2) for g in grouped_data)
        df_within = len(df) - len(classes)
        msw = ssw / df_within if df_within > 0 else 1

        f_ratio = msb / msw if msw > 0 else 0
        eta_squared = ssb / total_ss if total_ss > 0 else 0

        var_rows.append({
            "Feature": col,
            "Total Variance": round(total_var, 4),
            "Between-Class Variance (MSB)": round(float(msb), 4),
            "Within-Class Variance (MSW)": round(float(msw), 4),
            "F-Ratio (MSB/MSW)": round(float(f_ratio), 4),
            "Eta-Squared (Variance Explained)": round(float(eta_squared), 4),
        })

    tests_df = pd.DataFrame(test_rows).sort_values(by="ANOVA F-Statistic", ascending=False)
    var_df = pd.DataFrame(var_rows).sort_values(by="F-Ratio (MSB/MSW)", ascending=False)
    return var_df, tests_df


def compute_exploratory_pca(df: pd.DataFrame, n_components: int = 5) -> Dict[str, Any]:
    """
    Computes temporary standardized PCA for exploratory visualization.
    Explains variance ratios without mutating original dataset.
    """
    numerical_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[numerical_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_ratios = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratios)

    pca_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(n_components)],
        "Explained Variance Ratio": [round(float(v), 4) for v in var_ratios],
        "Cumulative Explained Variance": [round(float(c), 4) for c in cum_var],
    })

    loadings_df = pd.DataFrame(
        pca.components_[:2].T,
        index=numerical_cols,
        columns=["PC1 Loading", "PC2 Loading"],
    ).round(4)

    return {
        "pca_summary_df": pca_df,
        "loadings_df": loadings_df,
        "X_pca_2d": X_pca[:, :2],
        "var_pc1": round(float(var_ratios[0]) * 100, 2),
        "var_pc2": round(float(var_ratios[1]) * 100, 2),
    }
