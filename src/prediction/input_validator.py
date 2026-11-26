"""
Input Validation Module for BEANWISE Prediction Engine.

Validates single and batch inputs against physical geometric boundaries and
the empirical training distribution, differentiating invalid inputs from
out-of-distribution advisory warnings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RANGES_FILE = PROJECT_ROOT / "artifacts" / "metadata" / "training_feature_ranges.json"

EXPECTED_16_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


def load_training_ranges() -> Dict[str, Dict[str, float]]:
    """Loads training feature ranges metadata."""
    if RANGES_FILE.is_file():
        with open(RANGES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def validate_single_input(raw_input: Dict[str, Any]) -> Tuple[bool, List[str], List[str], Dict[str, float]]:
    """
    Validates a dictionary of 16 morphological inputs.

    Returns:
        (is_valid, errors, warnings, sanitized_dict)
    """
    errors: List[str] = []
    warnings: List[str] = []
    sanitized: Dict[str, float] = {}

    ranges = load_training_ranges()

    # 1. Completeness and Numeric Parsing
    for feat in EXPECTED_16_FEATURES:
        if feat not in raw_input or raw_input[feat] is None or str(raw_input[feat]).strip() == "":
            errors.append(f"Missing required measurement: '{feat}'.")
            continue

        val_raw = raw_input[feat]
        try:
            val_float = float(val_raw)
        except (ValueError, TypeError):
            errors.append(f"Field '{feat}' must be a valid numerical value. Received: '{val_raw}'.")
            continue

        if np.isnan(val_float) or np.isinf(val_float):
            errors.append(f"Field '{feat}' contains invalid non-finite value (NaN or Infinity).")
            continue

        if val_float <= 0.0:
            errors.append(f"Physical measurement '{feat}' must be strictly positive (> 0). Received: {val_float}.")
            continue

        sanitized[feat] = val_float

    # Stop if basic numeric checks failed
    if errors:
        return False, errors, warnings, sanitized

    # 2. Geometric Physical Consistency Rules
    if sanitized["MajorAxisLength"] < sanitized["MinorAxisLength"]:
        errors.append(
            f"Physical inconsistency: Major Axis Length ({sanitized['MajorAxisLength']:.2f}) "
            f"cannot be smaller than Minor Axis Length ({sanitized['MinorAxisLength']:.2f})."
        )

    if sanitized["ConvexArea"] < sanitized["Area"]:
        errors.append(
            f"Physical inconsistency: Convex Area ({sanitized['ConvexArea']:.2f}) "
            f"cannot be smaller than Area ({sanitized['Area']:.2f})."
        )

    if sanitized["Solidity"] > 1.0001:
        errors.append(f"Solidity (Area / ConvexArea) cannot exceed 1.0. Received: {sanitized['Solidity']:.4f}.")

    if sanitized["roundness"] > 1.05:
        errors.append(f"Roundness cannot exceed theoretical limit of 1.0. Received: {sanitized['roundness']:.4f}.")

    if sanitized["Compactness"] > 1.05:
        errors.append(f"Compactness cannot exceed theoretical limit of 1.0. Received: {sanitized['Compactness']:.4f}.")

    if sanitized["Eccentricity"] >= 1.0:
        errors.append(f"Eccentricity for an ellipse must be strictly less than 1.0. Received: {sanitized['Eccentricity']:.4f}.")

    if sanitized["Extent"] > 1.0:
        errors.append(f"Extent (Area / BoundingBox Area) cannot exceed 1.0. Received: {sanitized['Extent']:.4f}.")

    if errors:
        return False, errors, warnings, sanitized

    # 3. Empirical Distribution Range Warnings (Out-of-Distribution Advisories)
    if ranges:
        for feat, val in sanitized.items():
            if feat in ranges:
                f_min = ranges[feat]["min"]
                f_max = ranges[feat]["max"]
                q01 = ranges[feat]["q01"]
                q99 = ranges[feat]["q99"]

                if val < f_min or val > f_max:
                    warnings.append(
                        f"Field '{feat}' value ({val:.4f}) is outside observed training bounds [{f_min:.4f}, {f_max:.4f}]. "
                        "Model prediction reliability may be reduced."
                    )
                elif val < q01 or val > q99:
                    warnings.append(
                        f"Field '{feat}' value ({val:.4f}) falls in extreme 1% tail of the dataset distribution."
                    )

    return True, errors, warnings, sanitized


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[pd.DataFrame], List[str], List[str]]:
    """
    Validates an uploaded DataFrame for batch prediction.

    Returns:
        (is_valid, cleaned_df, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        errors.append("Uploaded file contains zero records.")
        return False, None, errors, warnings

    missing_cols = [c for c in EXPECTED_16_FEATURES if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in CSV: {missing_cols}.")
        return False, None, errors, warnings

    # Subset to the 16 features in exact expected order
    clean_df = df[EXPECTED_16_FEATURES].copy()

    # Check for nulls
    null_counts = clean_df.isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        errors.append(f"Dataset contains null or missing values: {cols_with_nulls}.")
        return False, None, errors, warnings

    # Convert to numeric
    for col in EXPECTED_16_FEATURES:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        if clean_df[col].isnull().any():
            errors.append(f"Column '{col}' contains non-numeric values.")
            return False, None, errors, warnings

        if (clean_df[col] <= 0).any():
            errors.append(f"Column '{col}' contains non-positive values (<= 0).")
            return False, None, errors, warnings

    if (clean_df["MajorAxisLength"] < clean_df["MinorAxisLength"]).any():
        errors.append("Inconsistency found: MajorAxisLength is less than MinorAxisLength in one or more rows.")
        return False, None, errors, warnings

    if (clean_df["ConvexArea"] < clean_df["Area"]).any():
        errors.append("Inconsistency found: ConvexArea is less than Area in one or more rows.")
        return False, None, errors, warnings

    return True, clean_df, errors, warnings
