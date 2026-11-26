"""
Data Ingestion and Inspection Package for BEANWISE.
Provides utilities to verify raw datasets, inspect integrity, and load data safely.
"""

from src.data.dataset_inspector import (
    DatasetInspectionResult,
    inspect_dataset,
    compute_file_sha256,
)

__all__ = [
    "DatasetInspectionResult",
    "inspect_dataset",
    "compute_file_sha256",
]
