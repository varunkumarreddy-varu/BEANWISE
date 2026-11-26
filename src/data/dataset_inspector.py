"""
Dataset Inspection and Validation Utility for BEANWISE.

Inspects raw dataset files (Excel / CSV) non-destructively, verifies feature schemas,
computes cryptographic checksums (SHA-256), and validates target class distributions
against the UCI Dry Bean Dataset specifications.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import (
    FEATURE_NAMES,
    TARGET_CLASSES,
    TARGET_COLUMN,
    get_raw_dataset_path,
)
from src.utils.logger import get_logger

logger = get_logger("dataset_inspector")


def compute_file_sha256(file_path: Path) -> str:
    """
    Computes SHA-256 checksum of a file without reading the entire file into memory at once.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DatasetInspectionResult:
    """Structured data container holding dataset inspection findings."""
    exists: bool = False
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_extension: Optional[str] = None
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    sha256: Optional[str] = None
    is_readable: bool = False
    row_count: int = 0
    column_count: int = 0
    columns_found: List[str] = field(default_factory=list)
    missing_expected_features: List[str] = field(default_factory=list)
    has_target_column: bool = False
    target_column_name: str = TARGET_COLUMN
    unique_classes_found: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    all_expected_classes_present: bool = False
    missing_classes: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts inspection result to a clean dictionary."""
        return {
            "exists": self.exists,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_mb, 2),
            "sha256": self.sha256,
            "is_readable": self.is_readable,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns_found": self.columns_found,
            "missing_expected_features": self.missing_expected_features,
            "has_target_column": self.has_target_column,
            "unique_classes_found": self.unique_classes_found,
            "class_distribution": self.class_distribution,
            "all_expected_classes_present": self.all_expected_classes_present,
            "missing_classes": self.missing_classes,
            "error_message": self.error_message,
        }


def inspect_dataset(custom_path: Optional[Path] = None) -> DatasetInspectionResult:
    """
    Performs a non-destructive audit of the Dry Bean dataset.
    Never alters, modifies, or overwrites the dataset file.

    Args:
        custom_path: Optional specific file path. If None, resolves via get_raw_dataset_path().

    Returns:
        DatasetInspectionResult containing comprehensive inspection metrics.
    """
    result = DatasetInspectionResult()

    target_path = custom_path or get_raw_dataset_path()
    if not target_path or not target_path.is_file():
        result.exists = False
        result.error_message = (
            f"Dataset not found at specified or candidate locations. "
            f"Checked DryBeanDataset/ and dataset/raw/."
        )
        logger.warning(result.error_message)
        return result

    result.exists = True
    result.file_path = str(target_path.resolve())
    result.file_name = target_path.name
    result.file_extension = target_path.suffix.lower()

    try:
        stat_info = target_path.stat()
        result.file_size_bytes = stat_info.st_size
        result.file_size_mb = stat_info.st_size / (1024 * 1024)
        result.sha256 = compute_file_sha256(target_path)
    except Exception as exc:
        result.error_message = f"Error reading file stats or checksum: {exc}"
        logger.error(result.error_message)
        return result

    # Read dataframe non-destructively
    try:
        if result.file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(target_path, engine="openpyxl")
        elif result.file_extension == ".csv":
            df = pd.read_csv(target_path)
        else:
            result.error_message = f"Unsupported dataset extension: {result.file_extension}"
            logger.error(result.error_message)
            return result

        result.is_readable = True
        result.row_count = int(df.shape[0])
        result.column_count = int(df.shape[1])
        result.columns_found = [str(c) for c in df.columns]

        # Check feature presence (handling case variants gracefully)
        cols_lower_map = {col.lower(): col for col in df.columns}
        missing_features = []
        for feat in FEATURE_NAMES:
            if feat not in df.columns and feat.lower() not in cols_lower_map:
                missing_features.append(feat)
        result.missing_expected_features = missing_features

        # Check target column
        if TARGET_COLUMN in df.columns:
            result.has_target_column = True
            target_series = df[TARGET_COLUMN].dropna().astype(str).str.strip()
            unique_classes = sorted(target_series.unique().tolist())
            result.unique_classes_found = unique_classes
            result.class_distribution = {
                str(k): int(v) for k, v in target_series.value_counts().items()
            }

            # Check expected classes
            missing_classes = [c for c in TARGET_CLASSES if c not in unique_classes]
            result.missing_classes = missing_classes
            result.all_expected_classes_present = len(missing_classes) == 0
        elif TARGET_COLUMN.lower() in cols_lower_map:
            actual_col = cols_lower_map[TARGET_COLUMN.lower()]
            result.has_target_column = True
            result.target_column_name = actual_col
            target_series = df[actual_col].dropna().astype(str).str.strip()
            unique_classes = sorted(target_series.unique().tolist())
            result.unique_classes_found = unique_classes
            result.class_distribution = {
                str(k): int(v) for k, v in target_series.value_counts().items()
            }
            missing_classes = [c for c in TARGET_CLASSES if c not in unique_classes]
            result.missing_classes = missing_classes
            result.all_expected_classes_present = len(missing_classes) == 0
        else:
            result.has_target_column = False
            result.all_expected_classes_present = False
            result.missing_classes = list(TARGET_CLASSES)

        logger.info(
            f"Dataset audit successful: {result.file_name} "
            f"({result.row_count} rows, {result.column_count} columns, "
            f"Classes present: {result.all_expected_classes_present})"
        )

    except Exception as exc:
        result.is_readable = False
        result.error_message = f"Failed to read dataset file: {exc}"
        logger.error(result.error_message)

    return result
