"""
Model Persistence Module for BEANWISE.

Manages saving and loading of trained Scikit-learn pipelines, label encoders,
and model metadata JSON records using relative project paths.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    MODEL_ARTIFACTS_DIR,
    ENCODER_DIR,
    ARTIFACTS_DIR,
)

METADATA_DIR = ARTIFACTS_DIR / "metadata"


def ensure_persistence_dirs():
    """Ensure artifacts directories exist."""
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def save_model_pipeline(
    pipeline: Pipeline,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    directory: Optional[Path] = None,
) -> Path:
    """
    Saves a trained Pipeline to disk via joblib and exports metadata JSON.

    Parameters:
        pipeline: Trained Scikit-learn Pipeline.
        model_name: Base filename (without extension).
        metadata: Optional dictionary with performance metrics and configurations.
        directory: Optional custom directory. Defaults to MODEL_ARTIFACTS_DIR.

    Returns:
        Path to the saved pipeline artifact.
    """
    ensure_persistence_dirs()
    save_dir = directory or MODEL_ARTIFACTS_DIR
    model_file = save_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_file)

    if metadata is not None:
        meta_to_save = {
            "model_name": model_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_steps": [name for name, _ in pipeline.steps],
            **metadata,
        }
        meta_file = METADATA_DIR / f"{model_name}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2)

    return model_file


def load_model_pipeline(model_name: str, directory: Optional[Path] = None) -> Pipeline:
    """
    Loads a persisted pipeline artifact from disk.

    Parameters:
        model_name: Base filename (without extension) or file path.
        directory: Optional custom directory.

    Returns:
        Loaded Pipeline.
    """
    save_dir = directory or MODEL_ARTIFACTS_DIR
    if str(model_name).endswith(".joblib"):
        path = Path(model_name) if Path(model_name).is_file() else save_dir / model_name
    else:
        path = save_dir / f"{model_name}.joblib"

    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    return joblib.load(path)


def save_label_encoder(encoder: LabelEncoder, name: str = "target_label_encoder") -> Path:
    """
    Saves a fitted LabelEncoder.
    """
    ensure_persistence_dirs()
    encoder_file = ENCODER_DIR / f"{name}.joblib"
    joblib.dump(encoder, encoder_file)
    return encoder_file


def load_label_encoder(name: str = "target_label_encoder") -> LabelEncoder:
    """
    Loads a persisted LabelEncoder.
    """
    encoder_file = ENCODER_DIR / f"{name}.joblib"
    if not encoder_file.is_file():
        raise FileNotFoundError(f"Label encoder not found at: {encoder_file}")
    return joblib.load(encoder_file)


def load_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Loads model metadata JSON.
    """
    meta_file = METADATA_DIR / f"{model_name}_metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Metadata file not found at: {meta_file}")
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)
