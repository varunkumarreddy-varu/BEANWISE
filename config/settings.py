"""
Centralized Project Settings for BEANWISE.

Manages all relative directory paths, dataset targets, 16 morphological feature definitions,
and directory assurance routines using pathlib.Path.
Standardized for industry data science project architecture.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


# PROJECT METADATA

PROJECT_NAME: str = "BEANWISE"
PROJECT_TITLE: str = "AI-Powered Dry Bean Classification"
PROJECT_TYPE: str = "Individual Data Science and Machine Learning Project"
PYTHON_VERSION_TARGET: str = "3.12"

# BASE DIRECTORIES (RESOLVED DYNAMICALLY VIA PATHLIB)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data Subdirectories (Standardized data/ hierarchy)
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_DIR: Path = DATA_DIR  # Alias for backward compatibility
RAW_DATASET_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATASET_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATASET_DIR: Path = DATA_DIR / "external"

# Dedicated preservation path for the raw dataset
EXISTING_DATASET_DIR: Path = RAW_DATASET_DIR

# Artifacts Subdirectories
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "models"
SCALER_DIR: Path = ARTIFACTS_DIR / "scalers"
ENCODER_DIR: Path = ARTIFACTS_DIR / "encoders"
METADATA_DIR: Path = ARTIFACTS_DIR / "metadata"

# Assets Subdirectories
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
BEAN_IMAGES_DIR: Path = ASSETS_DIR / "bean_images"
TEMPLATES_DIR: Path = ASSETS_DIR / "templates"

# Reports & Documentation Subdirectories
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EDA_REPORTS_DIR: Path = REPORTS_DIR / "eda"
PREPROCESSING_REPORTS_DIR: Path = REPORTS_DIR / "preprocessing"
FEATURE_ENG_REPORTS_DIR: Path = REPORTS_DIR / "feature_engineering"
MODEL_TRAINING_REPORTS_DIR: Path = REPORTS_DIR / "model_training"
MODEL_EVAL_REPORTS_DIR: Path = REPORTS_DIR / "model_evaluation"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
FINAL_REPORT_DIR: Path = REPORTS_DIR / "final_report"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Source & Runtime Subdirectories
SOURCE_DIR: Path = PROJECT_ROOT / "src"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# Map of all core directories for systematic auditing and creation
ALL_DIRECTORIES: Dict[str, Path] = {
    "PROJECT_ROOT": PROJECT_ROOT,
    "DATA_DIR": DATA_DIR,
    "DATASET_DIR": DATASET_DIR,
    "RAW_DATASET_DIR": RAW_DATASET_DIR,
    "PROCESSED_DATASET_DIR": PROCESSED_DATASET_DIR,
    "EXTERNAL_DATASET_DIR": EXTERNAL_DATASET_DIR,
    "ARTIFACTS_DIR": ARTIFACTS_DIR,
    "MODEL_ARTIFACTS_DIR": MODEL_ARTIFACTS_DIR,
    "SCALER_DIR": SCALER_DIR,
    "ENCODER_DIR": ENCODER_DIR,
    "METADATA_DIR": METADATA_DIR,
    "ASSETS_DIR": ASSETS_DIR,
    "BEAN_IMAGES_DIR": BEAN_IMAGES_DIR,
    "TEMPLATES_DIR": TEMPLATES_DIR,
    "REPORTS_DIR": REPORTS_DIR,
    "EDA_REPORTS_DIR": EDA_REPORTS_DIR,
    "PREPROCESSING_REPORTS_DIR": PREPROCESSING_REPORTS_DIR,
    "FEATURE_ENG_REPORTS_DIR": FEATURE_ENG_REPORTS_DIR,
    "MODEL_TRAINING_REPORTS_DIR": MODEL_TRAINING_REPORTS_DIR,
    "MODEL_EVAL_REPORTS_DIR": MODEL_EVAL_REPORTS_DIR,
    "FIGURES_DIR": FIGURES_DIR,
    "TABLES_DIR": TABLES_DIR,
    "FINAL_REPORT_DIR": FINAL_REPORT_DIR,
    "LOGS_DIR": LOGS_DIR,
    "NOTEBOOKS_DIR": NOTEBOOKS_DIR,
    "DOCS_DIR": DOCS_DIR,
    "TESTS_DIR": TESTS_DIR,
    "SCRIPTS_DIR": SCRIPTS_DIR,
}


# DATASET CONSTANTS & SPECIFICATIONS (UCI DRY BEAN DATASET)

TARGET_COLUMN: str = "Class"

TARGET_CLASSES: List[str] = [
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "HOROZ",
    "SIRA",
    "DERMASON",
]

FEATURE_NAMES: List[str] = [
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

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Area": "The area of a bean zone and the number of pixels within its boundary (pixels)",
    "Perimeter": "Bean circumference defined as the length of its border (pixels)",
    "MajorAxisLength": "The length of the longest line that can be drawn from the bean (pixels)",
    "MinorAxisLength": "The longest line that can be drawn from the bean perpendicular to the major axis (pixels)",
    "AspectRation": "Defines the eccentricity and relationship between Major and Minor axis (Aspect Ratio)",
    "Eccentricity": "Eccentricity of the ellipse having the same moments as the bean region (0 to 1)",
    "ConvexArea": "Number of pixels in the smallest convex polygon that can contain the area of the bean",
    "EquivDiameter": "The diameter of a circle with the same area as the bean area",
    "Extent": "The ratio of the pixels in the bounding box to the bean area",
    "Solidity": "Ratio of the pixels in the convex shell to those found in beans (ConvexArea ratio)",
    "roundness": "Calculated with (4 * pi * Area) / (Perimeter^2); indicates circularity",
    "Compactness": "Measures the roundness of the object: EquivDiameter / MajorAxisLength",
    "ShapeFactor1": "First morphological shape factor (MajorAxisLength / Area)",
    "ShapeFactor2": "Second morphological shape factor (MinorAxisLength / Area)",
    "ShapeFactor3": "Third morphological shape factor (Area / (MajorAxisLength/2)^2 * pi)",
    "ShapeFactor4": "Fourth morphological shape factor (Area / (MajorAxisLength/2 * MinorAxisLength/2 * pi))",
}


# REUSABLE DIRECTORY MANAGEMENT FUNCTIONS

def ensure_project_directories(base_path: Optional[Path] = None) -> Dict[str, bool]:
    root = base_path or PROJECT_ROOT
    results: Dict[str, bool] = {}

    target_dirs = [
        root / ".streamlit",
        root / "config",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "external",
        root / "notebooks",
        root / "src" / "data",
        root / "src" / "preprocessing",
        root / "src" / "features",
        root / "src" / "visualization",
        root / "src" / "models",
        root / "src" / "evaluation",
        root / "src" / "prediction",
        root / "src" / "utils",
        root / "artifacts" / "models",
        root / "artifacts" / "scalers",
        root / "artifacts" / "encoders",
        root / "artifacts" / "metadata",
        root / "assets" / "bean_images",
        root / "assets" / "templates",
        root / "reports" / "eda",
        root / "reports" / "preprocessing",
        root / "reports" / "feature_engineering",
        root / "reports" / "model_training",
        root / "reports" / "model_evaluation",
        root / "reports" / "figures",
        root / "reports" / "tables",
        root / "reports" / "final_report",
        root / "logs",
        root / "tests",
        root / "scripts",
        root / "docs",
    ]

    for directory in target_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            results[str(directory.relative_to(root))] = True
        except Exception:
            results[str(directory)] = False

    return results


def verify_required_directories(base_path: Optional[Path] = None) -> Tuple[bool, List[str]]:
    root = base_path or PROJECT_ROOT
    required_dirs = [
        root / ".streamlit",
        root / "config",
        root / "data",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "external",
        root / "notebooks",
        root / "src",
        root / "src" / "data",
        root / "src" / "preprocessing",
        root / "src" / "features",
        root / "src" / "visualization",
        root / "src" / "models",
        root / "src" / "evaluation",
        root / "src" / "utils",
        root / "artifacts",
        root / "artifacts" / "models",
        root / "artifacts" / "scalers",
        root / "artifacts" / "encoders",
        root / "reports",
        root / "reports" / "figures",
        root / "reports" / "tables",
        root / "reports" / "final_report",
        root / "logs",
        root / "tests",
        root / "docs",
    ]

    missing = [str(d.relative_to(root)) for d in required_dirs if not d.is_dir()]
    return len(missing) == 0, missing


def get_raw_dataset_path(custom_path: Optional[Path] = None) -> Optional[Path]:
    if custom_path and Path(custom_path).is_file():
        return Path(custom_path).resolve()

    candidate_locations = [
        RAW_DATASET_DIR / "Dry_Bean_Dataset.xlsx",
        RAW_DATASET_DIR / "Dry_Bean_Dataset.csv",
        DATA_DIR / "Dry_Bean_Dataset.xlsx",
        DATA_DIR / "Dry_Bean_Dataset.csv",
        PROJECT_ROOT / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx",
        PROJECT_ROOT / "Dry_Bean_Dataset.xlsx",
    ]

    for candidate in candidate_locations:
        if candidate.is_file():
            return candidate.resolve()

    return None
