"""
Data Cleaning and Preprocessing Pipeline for BEANWISE Phase 3.

Provides systematic ingestion, auditing, and duplicate record handling.
Outputs cleaned working dataset to dataset/processed/cleaned_dry_beans.csv
while strictly preserving the read-only raw dataset in DryBeanDataset/.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from config.settings import (
    PROCESSED_DATASET_DIR,
    FEATURE_NAMES,
    TARGET_COLUMN,
    TARGET_CLASSES,
    get_raw_dataset_path,
)
from src.data.dataset_inspector import compute_file_sha256
from src.preprocessing.data_validator import validate_dataset


@dataclass
class CleaningReport:
    """Structured container holding Phase 3 cleaning and preprocessing metrics."""
    raw_file_name: str = ""
    raw_sha256: str = ""
    raw_row_count: int = 0
    raw_column_count: int = 0
    missing_cells_count: int = 0
    exact_duplicates_detected: int = 0
    exact_duplicates_removed: int = 0
    cleaned_row_count: int = 0
    cleaned_column_count: int = 0
    cleaned_file_path: str = ""
    class_distribution_before: Dict[str, int] = field(default_factory=dict)
    class_distribution_after: Dict[str, int] = field(default_factory=dict)
    outlier_handling_decision: str = ""
    is_validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_file_name": self.raw_file_name,
            "raw_sha256": self.raw_sha256,
            "raw_row_count": self.raw_row_count,
            "raw_column_count": self.raw_column_count,
            "missing_cells_count": self.missing_cells_count,
            "exact_duplicates_detected": self.exact_duplicates_detected,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "cleaned_row_count": self.cleaned_row_count,
            "cleaned_column_count": self.cleaned_column_count,
            "cleaned_file_path": self.cleaned_file_path,
            "class_distribution_before": self.class_distribution_before,
            "class_distribution_after": self.class_distribution_after,
            "outlier_handling_decision": self.outlier_handling_decision,
            "is_validated": self.is_validated,
        }


def clean_dry_beans_dataset(
    output_path: Optional[Path] = None,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Executes Phase 3 cleaning pipeline:
    1. Verifies raw dataset integrity.
    2. Audits missing values, duplicates, and invalid numbers.
    3. Drops exact duplicates (if enabled) with scientific justification.
    4. Validates cleaned schema and value integrity.
    5. Saves output to dataset/processed/cleaned_dry_beans.csv without index.

    Returns:
        (cleaned_df, cleaning_report)
    """
    PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    target_output = output_path or (PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv")

    raw_path = get_raw_dataset_path()
    if not raw_path or not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset could not be located at {raw_path}")

    raw_sha = compute_file_sha256(raw_path)
    df_raw = pd.read_excel(raw_path)

    report = CleaningReport(
        raw_file_name=raw_path.name,
        raw_sha256=raw_sha,
        raw_row_count=len(df_raw),
        raw_column_count=len(df_raw.columns),
        missing_cells_count=int(df_raw.isnull().sum().sum()),
        class_distribution_before=df_raw[TARGET_COLUMN].value_counts().to_dict(),
    )

    # 1. Missing Values
    # Raw dataset has 0 missing cells. Documenting that no imputation is necessary.

    # 2. Duplicate Handling
    exact_dups = int(df_raw.duplicated().sum())
    report.exact_duplicates_detected = exact_dups

    if drop_duplicates and exact_dups > 0:
        # Justification: In digital image feature extraction for seed lots,
        # exact duplicate rows represent repeated captures or duplicate logging.
        # Dropping duplicates prevents data leakage across train/validation splits.
        df_cleaned = df_raw.drop_duplicates(keep="first").copy()
        report.exact_duplicates_removed = exact_dups
    else:
        df_cleaned = df_raw.copy()
        report.exact_duplicates_removed = 0

    # 3. Outlier Decision
    # Extreme values in Area / MinorAxisLength correspond to genuine Bombay cultivar.
    # We retain all valid biological variations; no records are artificially truncated.
    report.outlier_handling_decision = (
        "Retain all biological extremes. Extreme observations correspond to true cultivar "
        "morphology (e.g. Bombay large area). Robust scaling will be employed in model pipelines."
    )

    # 4. Data Validation
    validate_dataset(df_cleaned)
    report.is_validated = True

    # 5. Save Cleaned Working Dataset
    df_cleaned.to_csv(target_output, index=False)
    report.cleaned_file_path = str(target_output.resolve())
    report.cleaned_row_count = len(df_cleaned)
    report.cleaned_column_count = len(df_cleaned.columns)
    report.class_distribution_after = df_cleaned[TARGET_COLUMN].value_counts().to_dict()

    return df_cleaned, report
