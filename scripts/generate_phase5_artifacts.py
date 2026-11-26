"""
Automated Phase 5 Artifacts and Visualizations Generator for BEANWISE.

Executes:
1. 5-Fold Stratified Cross-Validation on training data across 9 machine learning algorithms.
2. Systematic hyperparameter tuning on top candidate models.
3. Validation set evaluation (strictly keeping test data untouched).
4. Model persistence (pipelines and metadata JSON).
5. 8 Phase 5 tables, 7 Phase 5 figures, and HTML/JSON comprehensive reports.
"""

import sys
import json
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    PROCESSED_DATASET_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    MODEL_ARTIFACTS_DIR,
    ENCODER_DIR,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from src.features.feature_engineering import add_engineered_features
from src.features.feature_selection import split_dataset
from src.models.model_factory import (
    get_all_model_pipelines,
    get_hyperparameter_grids,
)
from src.models.model_trainer import (
    cross_validate_pipelines,
    tune_pipelines,
    evaluate_on_validation,
)
from src.models.model_persistence import (
    save_model_pipeline,
    save_label_encoder,
)

PHASE5_REPORT_DIR = REPORTS_DIR / "phase5"
PHASE5_REPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def run_phase5_pipeline():
    print("================================================================")
    print("Starting BEANWISE Phase 5 Model Development and Training")
    print("================================================================")

    # 1. Load Data and Apply Features
    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    raw_df = pd.read_csv(cleaned_path)
    df_engineered = add_engineered_features(raw_df)

    # 2. Split Dataset (Train=70%, Val=15%, Test=15%)
    train_df, val_df, test_df, split_meta = split_dataset(df_engineered, random_state=42)
    print(f"Dataset split: Train={len(train_df):,}, Val={len(val_df):,}, Held-out Test={len(test_df):,}")
    print("STRICT DATA LEAKAGE RULE: Held-out Test partition will NOT be accessed during Phase 5.")

    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    X_train = train_df[feature_cols]
    y_train_raw = train_df[TARGET_COLUMN]

    X_val = val_df[feature_cols]
    y_val_raw = val_df[TARGET_COLUMN]

    # Target Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    save_label_encoder(le, "target_label_encoder")
    class_names = list(le.classes_)

    # 3. 5-Fold Stratified Cross-Validation on 9 Baseline Models
    print("\n--- Step 1/5: 5-Fold Stratified Cross-Validation on Baseline Models ---")
    base_pipelines = get_all_model_pipelines(random_state=42)
    cv_summary_df, raw_cv_dict = cross_validate_pipelines(
        base_pipelines,
        X_train,
        pd.Series(y_train),
        cv=5,
        random_state=42,
    )
    cv_summary_df.to_csv(TABLES_DIR / "phase5_baseline_cv_results.csv", index=False)
    print(cv_summary_df[["Model", "CV Mean Accuracy", "CV Mean Macro F1", "Mean Fit Time (s)"]])

    # 4. Hyperparameter Tuning for Top Candidate Models
    print("\n--- Step 2/5: Hyperparameter Tuning on Candidate Models (Random Forest, SVM, XGBoost) ---")
    param_grids = get_hyperparameter_grids()
    candidate_pipelines = {k: base_pipelines[k] for k in ["random_forest", "svm_rbf", "xgboost"]}
    tuned_models, tuning_df = tune_pipelines(
        candidate_pipelines,
        param_grids,
        X_train,
        pd.Series(y_train),
        cv=5,
        random_state=42,
    )
    tuning_df.to_csv(TABLES_DIR / "phase5_tuned_model_comparison.csv", index=False)
    print(tuning_df)

    # Save Hyperparameter Configurations Table
    configs_list = []
    for model_name, grid in param_grids.items():
        configs_list.append({
            "Model": model_name,
            "Parameters Evaluated": str(grid),
            "Search Method": "GridSearchCV (Stratified 5-Fold)",
            "Optimization Metric": "Macro F1",
        })
    pd.DataFrame(configs_list).to_csv(TABLES_DIR / "phase5_hyperparameter_configurations.csv", index=False)

    # 5. Validation Set Evaluation
    print("\n--- Step 3/5: Validation Set Evaluation (Clean Unseen Validation Partition) ---")
    all_models_to_evaluate = {**base_pipelines, **tuned_models}
    val_records = []
    val_eval_details = {}
    timing_records = []
    per_class_f1_matrix = {}

    for name, pipeline in all_models_to_evaluate.items():
        res = evaluate_on_validation(
            pipeline,
            X_train,
            pd.Series(y_train),
            X_val,
            pd.Series(y_val),
            class_names=class_names,
        )
        val_eval_details[name] = res

        val_records.append({
            "Model": name,
            "Validation Accuracy": res["accuracy"],
            "Validation Macro Precision": res["macro_precision"],
            "Validation Macro Recall": res["macro_recall"],
            "Validation Macro F1": res["macro_f1"],
            "Validation Weighted F1": res["weighted_f1"],
            "Fit Time (s)": res["fit_time"],
            "Inference Time (s)": res["prediction_time"],
        })

        timing_records.append({
            "Model": name,
            "Training Time (s)": res["fit_time"],
            "Inference Time (s)": res["prediction_time"],
            "Inference Time per Sample (ms)": round((res["prediction_time"] / len(X_val)) * 1000, 4),
        })

        # Extract per-class F1
        rep = res["classification_report"]
        per_class_f1_matrix[name] = {cls: round(rep[cls]["f1-score"], 4) for cls in class_names if cls in rep}

        # Persist Model Pipeline & Metadata
        save_model_pipeline(
            pipeline,
            name,
            metadata={
                "features": feature_cols,
                "classes": class_names,
                "validation_accuracy": res["accuracy"],
                "validation_macro_f1": res["macro_f1"],
                "validation_weighted_f1": res["weighted_f1"],
                "fit_time_seconds": res["fit_time"],
            },
        )

    val_df_results = pd.DataFrame(val_records).sort_values("Validation Macro F1", ascending=False).reset_index(drop=True)
    val_df_results.to_csv(TABLES_DIR / "phase5_validation_metrics.csv", index=False)

    timing_df = pd.DataFrame(timing_records).sort_values("Training Time (s)").reset_index(drop=True)
    timing_df.to_csv(TABLES_DIR / "phase5_training_times.csv", index=False)

    # Model Rankings
    val_df_results["Rank"] = range(1, len(val_df_results) + 1)
    val_df_results[["Rank", "Model", "Validation Macro F1", "Validation Accuracy", "Validation Weighted F1", "Fit Time (s)"]].to_csv(
        TABLES_DIR / "phase5_model_rankings.csv", index=False
    )

    # Best Model Selection
    best_model_name = val_df_results.iloc[0]["Model"]
    best_pipeline = all_models_to_evaluate[best_model_name]
    best_res = val_eval_details[best_model_name]
    print(f"\nBest Performing Model: {best_model_name}")
    print(f"Validation Macro F1: {best_res['macro_f1']}, Accuracy: {best_res['accuracy']}")

    # Save as primary best_model
    save_model_pipeline(
        best_pipeline,
        "best_model",
        metadata={
            "original_model_name": best_model_name,
            "features": feature_cols,
            "classes": class_names,
            "validation_accuracy": best_res["accuracy"],
            "validation_macro_f1": best_res["macro_f1"],
            "validation_weighted_f1": best_res["weighted_f1"],
            "fit_time_seconds": best_res["fit_time"],
        },
    )

    # Best Model Confusion Matrix Table
    cm_labels = [class_names[i] for i in best_res["confusion_matrix_labels"]]
    cm_df = pd.DataFrame(best_res["confusion_matrix"], index=cm_labels, columns=cm_labels)
    cm_df.to_csv(TABLES_DIR / "phase5_confusion_matrix.csv")

    # Best Model Per-Class Metrics Table
    rep = best_res["classification_report"]
    per_class_rows = []
    for cls in class_names:
        if cls in rep:
            per_class_rows.append({
                "Class": cls,
                "Precision": round(rep[cls]["precision"], 4),
                "Recall": round(rep[cls]["recall"], 4),
                "F1-Score": round(rep[cls]["f1-score"], 4),
                "Support": int(rep[cls]["support"]),
            })
    pd.DataFrame(per_class_rows).to_csv(TABLES_DIR / "phase5_per_class_metrics.csv", index=False)

    # 6. Generate Figures
    print("\n--- Step 4/5: Generating Phase 5 Visualization Figures ---")

    # Figure 1: Accuracy Comparison (CV vs Validation)
    fig, ax = plt.subplots(figsize=(12, 6))
    comp_df = pd.merge(
        cv_summary_df[["Model", "CV Mean Accuracy"]],
        val_df_results[["Model", "Validation Accuracy"]],
        on="Model",
        how="inner",
    ).sort_values("Validation Accuracy", ascending=False)
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width/2, comp_df["CV Mean Accuracy"], width, label="CV Mean Accuracy", color="#2b5c8f")
    ax.bar(x + width/2, comp_df["Validation Accuracy"], width, label="Validation Accuracy", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Model"], rotation=30, ha="right")
    ax.set_ylim([0.88, 0.95])
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Cross-Validation vs Validation Accuracy Across Models", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_model_accuracy_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 2: Macro F1 Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    val_sorted_f1 = val_df_results.sort_values("Validation Macro F1", ascending=True)
    colors = ["#2b5c8f" if m != best_model_name else "#d95f02" for m in val_sorted_f1["Model"]]
    ax.barh(val_sorted_f1["Model"], val_sorted_f1["Validation Macro F1"], color=colors)
    ax.set_xlim([0.88, 0.96])
    ax.set_xlabel("Macro F1-Score")
    ax.set_title("Validation Macro F1-Score Across Evaluated Models", fontweight="bold")
    for i, v in enumerate(val_sorted_f1["Validation Macro F1"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_model_f1_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 3: CV Variance Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = [raw_cv_dict[m]["test_macro_f1"] for m in cv_summary_df["Model"]]
    ax.boxplot(box_data, patch_artist=True,
               boxprops=dict(facecolor="#c6dbef", color="#2b5c8f"),
               medianprops=dict(color="#d95f02", linewidth=2))
    ax.set_xticks(range(1, len(cv_summary_df) + 1))
    ax.set_xticklabels(cv_summary_df["Model"], rotation=30, ha="right")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("5-Fold Cross-Validation Macro F1 Variance Across Baseline Models", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_cv_variance_boxplot.png", dpi=300)
    plt.close(fig)

    # Figure 4: Validation Confusion Matrix for Best Model
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(f"Validation Confusion Matrix ({best_model_name})", fontweight="bold")
    ax.set_xlabel("Predicted Cultivar Label")
    ax.set_ylabel("True Cultivar Label")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_validation_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Figure 5: Training Time vs Macro F1 Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in val_df_results.iterrows():
        is_best = (row["Model"] == best_model_name)
        ax.scatter(row["Fit Time (s)"], row["Validation Macro F1"],
                   s=140 if is_best else 90,
                   color="#d95f02" if is_best else "#2b5c8f",
                   zorder=5)
        ax.annotate(row["Model"], (row["Fit Time (s)"], row["Validation Macro F1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Validation Macro F1-Score")
    ax.set_title("Computational Cost vs Predictive Performance Trade-Off", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_training_time_vs_f1.png", dpi=300)
    plt.close(fig)

    # Figure 6: Per-Class F1 Heatmap Across Baseline Models
    fig, ax = plt.subplots(figsize=(12, 7))
    f1_df = pd.DataFrame(per_class_f1_matrix).T
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_title("Per-Cultivar F1-Score Heatmap Across Evaluated Models", fontweight="bold")
    ax.set_xlabel("Dry Bean Cultivar")
    ax.set_ylabel("Model Architecture")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_per_class_f1_heatmap.png", dpi=300)
    plt.close(fig)

    # Figure 7: Feature Importance for Best Model (or RandomForest/XGBoost)
    fig, ax = plt.subplots(figsize=(10, 7))
    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif "random_forest" in all_models_to_evaluate:
        importances = all_models_to_evaluate["random_forest"].named_steps["classifier"].feature_importances_
    else:
        importances = np.ones(len(feature_cols))

    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    feat_imp.plot(kind="barh", ax=ax, color="#2b5c8f")
    ax.set_title(f"Feature Importance Profile ({best_model_name})", fontweight="bold")
    ax.set_xlabel("Gini / Gain Importance")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "phase5_feature_importance_best_model.png", dpi=300)
    plt.close(fig)

    # 7. Generate Reports & JSON Summaries
    print("\n--- Step 5/5: Exporting Phase 5 Reports and Registry ---")
    training_summary = {
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "held_out_test_samples": len(test_df),
        "total_features": len(feature_cols),
        "evaluated_models_count": len(all_models_to_evaluate),
        "best_model": best_model_name,
        "best_model_validation_accuracy": best_res["accuracy"],
        "best_model_validation_macro_f1": best_res["macro_f1"],
        "best_model_validation_weighted_f1": best_res["weighted_f1"],
        "best_model_fit_time_seconds": best_res["fit_time"],
        "cross_validation_strategy": "5-Fold StratifiedKFold",
        "models_evaluated": list(all_models_to_evaluate.keys()),
    }
    with open(PHASE5_REPORT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    model_registry = {
        "registered_models": val_df_results.to_dict(orient="records"),
        "best_model": best_model_name,
        "tuning_summary": tuning_df.to_dict(orient="records"),
    }
    with open(PHASE5_REPORT_DIR / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(model_registry, f, indent=2)

    # HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BEANWISE — Phase 5 Model Training and Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #222; max-width: 1100px; margin: 0 auto; padding: 25px; }}
        h1, h2, h3 {{ color: #1a365d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #cbd5e0; padding: 10px 14px; text-align: left; }}
        th {{ background-color: #f7fafc; color: #2d3748; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85em; background-color: #ebf8ff; color: #2b6cb0; }}
        .metric-card {{ background: #f7fafc; border-left: 4px solid #3182ce; padding: 15px; margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>BEANWISE — Machine Learning Model Development & Validation</h1>
    <p><strong>Phase:</strong> Phase 5 — Model Development and Training | <strong>Dataset:</strong> UCI Dry Bean Dataset</p>
    
    <div class="metric-card">
        <h3>Model Evaluation Summary</h3>
        <p><strong>Training Samples:</strong> {len(train_df):,} | <strong>Validation Samples:</strong> {len(val_df):,} | <strong>Held-out Test Samples:</strong> {len(test_df):,} (Kept 100% untouched)</p>
        <p><strong>Best Selected Architecture:</strong> <span class="badge">{best_model_name}</span></p>
        <p><strong>Validation Accuracy:</strong> {best_res['accuracy'] * 100:.2f}% | <strong>Validation Macro F1:</strong> {best_res['macro_f1']:.4f}</p>
    </div>

    <h2>1. 5-Fold Stratified Cross-Validation Baseline Results</h2>
    {cv_summary_df.to_html(classes="table", index=False)}

    <h2>2. Validation Set Performance Rankings</h2>
    {val_df_results.to_html(classes="table", index=False)}

    <h2>3. Best Model Per-Class Validation Performance</h2>
    {pd.DataFrame(per_class_rows).to_html(classes="table", index=False)}

    <h2>4. Hyperparameter Tuning Summary</h2>
    {tuning_df.to_html(classes="table", index=False)}

    <footer style="margin-top: 40px; padding-top: 15px; border-top: 1px solid #e2e8f0; color: #718096; font-size: 0.9em;">
        BEANWISE Machine Learning System — Phase 5 Completed.
    </footer>
</body>
</html>
"""
    with open(PHASE5_REPORT_DIR / "model_training_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\nPhase 5 Artifact Generation Completed Successfully.")


if __name__ == "__main__":
    run_phase5_pipeline()
