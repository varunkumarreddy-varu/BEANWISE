"""Evaluation package for BEANWISE."""
from .classification_metrics import compute_comprehensive_metrics
from .confusion_analysis import compute_confusion_matrices, find_top_confused_pairs
from .prediction_analysis import extract_misclassified_samples, analyze_confidence_by_correctness
from .final_evaluator import run_independent_test_evaluation

__all__ = [
    "compute_comprehensive_metrics",
    "compute_confusion_matrices",
    "find_top_confused_pairs",
    "extract_misclassified_samples",
    "analyze_confidence_by_correctness",
    "run_independent_test_evaluation",
]
