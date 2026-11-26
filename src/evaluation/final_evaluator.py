"""
Final Evaluator Module for BEANWISE.

Orchestrates independent evaluation of the selected final model pipeline
on the strictly preserved held-out test partition (2,032 samples).
Generates Phase 6 tables, figures, HTML report, and JSON summaries.
"""

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATASET_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from src.features.feature_engineering import add_engineered_features
from src.features.feature_selection import split_dataset
from src.models.model_persistence import load_model_pipeline, load_label_encoder
from src.evaluation.classification_metrics import compute_comprehensive_metrics
from src.evaluation.confusion_analysis import compute_confusion_matrices, find_top_confused_pairs
from src.evaluation.prediction_analysis import extract_misclassified_samples, analyze_confidence_by_correctness

PHASE6_DIR = REPORTS_DIR / "phase6"
PHASE6_FIGS = PHASE6_DIR / "figures"
PHASE6_DIR.mkdir(parents=True, exist_ok=True)
PHASE6_FIGS.mkdir(parents=True, exist_ok=True)


def run_independent_test_evaluation(model_name: str = "best_model") -> Dict[str, Any]:
    """
    Executes the final independent model evaluation on the held-out test partition.
    """
    print("================================================================")
    print("Executing BEANWISE Phase 6 Final Independent Model Evaluation")
    print("================================================================")

    # 1. Load Model Pipeline and Label Encoder
    pipeline = load_model_pipeline(model_name)
    encoder = load_label_encoder("target_label_encoder")
    class_names = list(encoder.classes_)
    print(f"Loaded Final Model Pipeline: '{model_name}'")
    print(f"Pipeline Architecture Steps: {[s[0] for s in pipeline.steps]}")
    print(f"Target Varieties ({len(class_names)}): {class_names}")

    # 2. Ingest Dataset and Retrieve the Strictly Held-Out Test Set
    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    raw_df = pd.read_csv(cleaned_path)
    df_all = add_engineered_features(raw_df)

    # Load split metadata
    meta_path = PROCESSED_DATASET_DIR / "split_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    test_indices = split_meta["test_indices"]
    test_df = df_all.iloc[test_indices].copy()
    print(f"Ingested Held-Out Test Partition: {len(test_df):,} samples (15.0% of dataset).")

    feature_cols = [c for c in test_df.columns if c != TARGET_COLUMN]
    X_test = test_df[feature_cols]
    y_test = encoder.transform(test_df[TARGET_COLUMN])

    # 3. Model Inference and Latency Measurement
    print("\nRunning inference on the independent test partition...")
    t_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_duration = time.perf_counter() - t_start
    per_sample_ms = (inference_duration / len(X_test)) * 1000.0

    print(f"Inference Time for {len(X_test):,} samples: {inference_duration:.4f}s ({per_sample_ms:.4f} ms/sample)")

    # Probabilities / Margin Estimation
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            pass

    if y_proba is None and hasattr(pipeline, "decision_function"):
        raw_dec = pipeline.decision_function(X_test)
        y_proba = softmax(raw_dec, axis=1)

    # 4. Metric Calculations
    metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba, class_names=class_names)
    print(f"\nFinal Independent Test Metrics:")
    print(f"- Overall Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print(f"- Balanced Accuracy:         {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"- Macro Precision:           {metrics['macro_precision']:.4f}")
    print(f"- Macro Recall:              {metrics['macro_recall']:.4f}")
    print(f"- Macro F1-Score:            {metrics['macro_f1']:.4f}")
    print(f"- Weighted F1-Score:         {metrics['weighted_f1']:.4f}")
    print(f"- Matthews Corr Coeff (MCC): {metrics['matthews_corrcoef']:.4f}")
    print(f"- Cohen's Kappa:             {metrics['cohens_kappa']:.4f}")
    if metrics["roc_auc_ovr"]:
        print(f"- Multiclass ROC-AUC (OvR):  {metrics['roc_auc_ovr']:.4f}")

    # 5. Confusion Matrices & Error Investigation
    cm_dict = compute_confusion_matrices(y_test, y_pred, class_names=class_names)
    cm_raw = cm_dict["raw_confusion_matrix"]
    cm_norm = cm_dict["normalized_confusion_matrix"]

    top_confused = find_top_confused_pairs(y_test, y_pred, class_names=class_names)
    print("\nTop Confused Cultivar Pairs:")
    print(top_confused)

    misclassified_df = extract_misclassified_samples(X_test, y_test, y_pred, class_names, y_proba)
    print(f"Total Misclassified Samples: {len(misclassified_df)} / {len(X_test)} ({len(misclassified_df)/len(X_test)*100:.2f}%)")

    conf_analysis = {}
    if y_proba is not None:
        conf_analysis = analyze_confidence_by_correctness(y_test, y_pred, y_proba)
        print(f"Confidence Analysis: Correct Mean={conf_analysis['mean_confidence_correct']}, Error Mean={conf_analysis['mean_confidence_error']}")

    # 6. Save Reports and Tables
    per_class_df = pd.DataFrame(metrics["per_class_metrics"])
    per_class_df.to_csv(PHASE6_DIR / "per_class_metrics.csv", index=False)
    per_class_df.to_csv(TABLES_DIR / "phase6_per_class_metrics.csv", index=False)

    cm_raw.to_csv(PHASE6_DIR / "confusion_matrix.csv")
    cm_norm.to_csv(PHASE6_DIR / "normalized_confusion_matrix.csv")
    cm_raw.to_csv(TABLES_DIR / "phase6_confusion_matrix.csv")

    misclassified_df.to_csv(PHASE6_DIR / "misclassified_samples.csv", index=False)
    misclassified_df.to_csv(TABLES_DIR / "phase6_misclassified_samples.csv", index=False)

    top_confused.to_csv(PHASE6_DIR / "top_confused_pairs.csv", index=False)

    # Classification report table
    rep_df = pd.DataFrame(metrics["classification_report_dict"]).T.round(4)
    rep_df.to_csv(PHASE6_DIR / "classification_report.csv")

    # Evaluation Summary JSON
    summary_data = {
        "model_evaluated": model_name,
        "test_sample_count": len(test_df),
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "matthews_corrcoef": metrics["matthews_corrcoef"],
        "cohens_kappa": metrics["cohens_kappa"],
        "roc_auc_ovr": metrics["roc_auc_ovr"],
        "total_misclassifications": len(misclassified_df),
        "inference_duration_seconds": round(inference_duration, 4),
        "latency_per_sample_ms": round(per_sample_ms, 4),
        "confidence_stats": conf_analysis,
    }
    with open(PHASE6_DIR / "final_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(PHASE6_DIR / "evaluation_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_indices_count": len(test_indices),
            "feature_names": feature_cols,
            "classes": class_names,
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **summary_data,
        }, f, indent=2)

    # 7. Generate Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Raw Confusion Matrix
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_title("Independent Test Set: Raw Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Cultivar")
    ax.set_ylabel("True Cultivar")
    plt.tight_layout()
    fig.savefig(PHASE6_FIGS / "raw_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Figure 2: Normalized Confusion Matrix
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax)
    ax.set_title("Independent Test Set: Row-Normalized Confusion Matrix (Recall)", fontweight="bold")
    ax.set_xlabel("Predicted Cultivar")
    ax.set_ylabel("True Cultivar")
    plt.tight_layout()
    fig.savefig(PHASE6_FIGS / "normalized_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Figure 3: Per-Class Performance Comparison (Precision, Recall, F1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(per_class_df))
    width = 0.25
    ax.bar(x - width, per_class_df["Precision"], width, label="Precision", color="#2b5c8f")
    ax.bar(x, per_class_df["Recall"], width, label="Recall", color="#2ca02c")
    ax.bar(x + width, per_class_df["F1-Score"], width, label="F1-Score", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels(per_class_df["Cultivar"], rotation=20, ha="right")
    ax.set_ylim([0.80, 1.05])
    ax.set_ylabel("Score")
    ax.set_title("Per-Cultivar Test Set Performance Metrics", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(PHASE6_FIGS / "per_class_performance.png", dpi=300)
    plt.close(fig)

    # Figure 4: Misclassification Error Breakdown
    fig, ax = plt.subplots(figsize=(9, 5))
    err_counts = misclassified_df["True_Class"].value_counts().reindex(class_names, fill_value=0)
    err_counts.plot(kind="bar", color="#c51b7d", ax=ax)
    ax.set_title("Test Set Misclassifications by True Cultivar", fontweight="bold")
    ax.set_ylabel("Number of Misclassified Samples")
    ax.set_xlabel("Dry Bean Cultivar")
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    for i, v in enumerate(err_counts):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(PHASE6_FIGS / "error_distribution_by_class.png", dpi=300)
    plt.close(fig)

    # Figure 5: Confidence Distribution
    if y_proba is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        max_probs = np.max(y_proba, axis=1)
        correct_mask = (y_test == y_pred)
        ax.hist(max_probs[correct_mask], bins=25, alpha=0.7, label=f"Correct (Mean={conf_analysis['mean_confidence_correct']})", color="#2ca02c")
        ax.hist(max_probs[~correct_mask], bins=25, alpha=0.7, label=f"Misclassified (Mean={conf_analysis['mean_confidence_error']})", color="#d95f02")
        ax.set_title("Prediction Confidence Distribution: Correct vs Misclassified", fontweight="bold")
        ax.set_xlabel("Model-Estimated Prediction Probability")
        ax.set_ylabel("Sample Frequency")
        ax.legend()
        plt.tight_layout()
        fig.savefig(PHASE6_FIGS / "confidence_distribution.png", dpi=300)
        plt.close(fig)

    # 8. HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BEANWISE — Phase 6 Final Independent Model Evaluation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #1e293b; max-width: 1050px; margin: 0 auto; padding: 25px; }}
        h1, h2, h3 {{ color: #0f172a; }}
        table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 9px 12px; text-align: left; }}
        th {{ background-color: #f8fafc; color: #334155; }}
        tr:nth-child(even) {{ background-color: #f1f5f9; }}
        .metric-card {{ background: #f8fafc; border-left: 4px solid #2563eb; padding: 14px 18px; margin: 16px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>BEANWISE — Phase 6 Final Model Independent Evaluation</h1>
    <p><strong>Evaluation Target:</strong> Held-Out Independent Test Partition (N = {len(test_df):,} observations, 15.0% of dataset)</p>
    
    <div class="metric-card">
        <h3>Independent Test Metrics Summary</h3>
        <p><strong>Selected Model Architecture:</strong> Support Vector Classifier (RBF Kernel)</p>
        <p><strong>Test Accuracy:</strong> {metrics['accuracy'] * 100:.2f}% | <strong>Balanced Accuracy:</strong> {metrics['balanced_accuracy'] * 100:.2f}%</p>
        <p><strong>Macro F1-Score:</strong> {metrics['macro_f1']:.4f} | <strong>Weighted F1-Score:</strong> {metrics['weighted_f1']:.4f}</p>
        <p><strong>Matthews Correlation Coefficient (MCC):</strong> {metrics['matthews_corrcoef']:.4f} | <strong>Cohen's Kappa:</strong> {metrics['cohens_kappa']:.4f}</p>
        <p><strong>Inference Latency:</strong> {per_sample_ms:.4f} ms per sample ({inference_duration:.4f}s total)</p>
    </div>

    <h2>1. Per-Cultivar Performance Breakdown</h2>
    {per_class_df.to_html(classes="table", index=False)}

    <h2>2. Raw Confusion Matrix</h2>
    {cm_raw.to_html(classes="table")}

    <h2>3. Top Misclassification Confusion Pairs</h2>
    {top_confused.to_html(classes="table", index=False)}

    <footer style="margin-top: 40px; padding-top: 15px; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 0.9em;">
        BEANWISE Machine Learning Platform — Independent Evaluation Completed.
    </footer>
</body>
</html>
"""
    with open(PHASE6_DIR / "final_model_evaluation.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\nPhase 6 Final Independent Evaluation Completed Successfully.")
    return summary_data


if __name__ == "__main__":
    run_independent_test_evaluation()
