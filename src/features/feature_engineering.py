"""
Feature Engineering Module for BEANWISE Phase 4.

Computes domain-derived morphological indicators, handles edge cases (division by zero),
and exports metadata definitions without data leakage.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from config.settings import FEATURE_NAMES, TARGET_COLUMN

# Definitions of candidate engineered features
ENGINEERED_FEATURE_METADATA = [
    {
        "name": "Area_to_Perimeter",
        "formula": "Area / Perimeter",
        "purpose": "Scale-dependent boundary efficiency ratio",
        "source_features": ["Area", "Perimeter"],
    },
    {
        "name": "Axis_Ratio",
        "formula": "MajorAxisLength / MinorAxisLength",
        "purpose": "Direct elongation factor cross-checking AspectRation",
        "source_features": ["MajorAxisLength", "MinorAxisLength"],
    },
    {
        "name": "Axis_Difference",
        "formula": "MajorAxisLength - MinorAxisLength",
        "purpose": "Absolute dimensional difference between major and minor axes (pixels)",
        "source_features": ["MajorAxisLength", "MinorAxisLength"],
    },
    {
        "name": "Convex_Area_Ratio",
        "formula": "ConvexArea / Area",
        "purpose": "Inverse solidity measuring boundary concavity",
        "source_features": ["ConvexArea", "Area"],
    },
    {
        "name": "Compactness_Roundness_Ratio",
        "formula": "Compactness / roundness",
        "purpose": "Comparative ratio between radial compactness and circular boundary roundness",
        "source_features": ["Compactness", "roundness"],
    },
    {
        "name": "EquivDiameter_to_MajorAxis",
        "formula": "EquivDiameter / MajorAxisLength",
        "purpose": "Ratio of equivalent circular diameter to maximal seed length",
        "source_features": ["EquivDiameter", "MajorAxisLength"],
    },
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all candidate engineered features non-destructively.
    Handles division by zero and guarantees finite values.
    """
    df_out = df.copy()

    # 1. Area to Perimeter
    denom_perim = np.where(df_out["Perimeter"] == 0, np.nan, df_out["Perimeter"])
    df_out["Area_to_Perimeter"] = df_out["Area"] / denom_perim

    # 2. Axis Ratio
    denom_minor = np.where(df_out["MinorAxisLength"] == 0, np.nan, df_out["MinorAxisLength"])
    df_out["Axis_Ratio"] = df_out["MajorAxisLength"] / denom_minor

    # 3. Axis Difference
    df_out["Axis_Difference"] = df_out["MajorAxisLength"] - df_out["MinorAxisLength"]

    # 4. Convex Area Ratio
    denom_area = np.where(df_out["Area"] == 0, np.nan, df_out["Area"])
    df_out["Convex_Area_Ratio"] = df_out["ConvexArea"] / denom_area

    # 5. Compactness to Roundness Ratio
    denom_round = np.where(df_out["roundness"] == 0, np.nan, df_out["roundness"])
    df_out["Compactness_Roundness_Ratio"] = df_out["Compactness"] / denom_round

    # 6. EquivDiameter to MajorAxis
    denom_major = np.where(df_out["MajorAxisLength"] == 0, np.nan, df_out["MajorAxisLength"])
    df_out["EquivDiameter_to_MajorAxis"] = df_out["EquivDiameter"] / denom_major

    # Check for any created NaN or infinite values
    eng_cols = [m["name"] for m in ENGINEERED_FEATURE_METADATA]
    for col in eng_cols:
        if df_out[col].isnull().any() or np.isinf(df_out[col]).any():
            raise ValueError(f"Engineered feature '{col}' generated non-finite values.")

    return df_out


def get_engineered_feature_names() -> List[str]:
    """Returns the names of all candidate engineered features."""
    return [m["name"] for m in ENGINEERED_FEATURE_METADATA]
