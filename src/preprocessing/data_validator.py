"""
Data Validation Module for BEANWISE.

Provides programmatic verification of dataset schema, feature column presence,
numerical ranges, absence of infinite/missing values, and valid target cultivar labels.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from config.settings import FEATURE_NAMES, TARGET_COLUMN, TARGET_CLASSES


class DataValidationError(ValueError):
    """Exception raised when a dataset violates integrity or schema constraints."""
    pass


def validate_dataset_schema(
    df: pd.DataFrame,
    expected_features: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates that the DataFrame contains all expected feature columns and the target column.
    """
    feats = expected_features or FEATURE_NAMES
    target = target_column or TARGET_COLUMN

    errors = []
    missing_feats = [f for f in feats if f not in df.columns]
    if missing_feats:
        errors.append(f"Missing expected feature columns: {missing_feats}")

    if target not in df.columns:
        errors.append(f"Missing target column: '{target}'")

    return len(errors) == 0, errors


def validate_numerical_integrity(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates that all numerical features are finite, non-null, non-negative, and float/int types.
    """
    cols = feature_cols or [c for c in FEATURE_NAMES if c in df.columns]
    errors = []

    for col in cols:
        series = df[col]
        # Check dtype
        if not np.issubdtype(series.dtype, np.number):
            errors.append(f"Column '{col}' has non-numerical dtype '{series.dtype}'")
            continue

        # Check missing values
        null_count = series.isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null values")

        # Check infinite values
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            errors.append(f"Column '{col}' contains {inf_count} infinite values")

        # Check non-negative constraint
        neg_count = (series < 0).sum()
        if neg_count > 0:
            errors.append(f"Column '{col}' contains {neg_count} negative values")

    return len(errors) == 0, errors


def validate_target_classes(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    expected_classes: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates that the target column has only expected classes and that all classes are present.
    """
    target = target_col or TARGET_COLUMN
    expected = set(expected_classes or TARGET_CLASSES)
    errors = []

    if target not in df.columns:
        return False, [f"Target column '{target}' not in DataFrame"]

    found_classes = set(df[target].dropna().unique())
    unexpected = found_classes - expected
    if unexpected:
        errors.append(f"Found unexpected class labels: {unexpected}")

    missing_classes = expected - found_classes
    if missing_classes:
        errors.append(f"Missing expected class labels: {missing_classes}")

    return len(errors) == 0, errors


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Runs complete validation suite on the DataFrame. Raises DataValidationError if invalid.
    """
    schema_ok, schema_errs = validate_dataset_schema(df)
    num_ok, num_errs = validate_numerical_integrity(df)
    target_ok, target_errs = validate_target_classes(df)

    all_errors = schema_errs + num_errs + target_errs
    is_valid = len(all_errors) == 0

    if not is_valid:
        raise DataValidationError(f"Dataset validation failed with {len(all_errors)} errors: {all_errors}")

    return {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "target_classes_present": sorted(df[TARGET_COLUMN].unique().tolist()),
    }
