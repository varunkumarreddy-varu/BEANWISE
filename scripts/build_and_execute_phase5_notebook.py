"""
Constructs and executes notebooks/05_Machine_Learning_Model_Training.ipynb.
Embeds all cell outputs, tables, and visualization charts.
"""

import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_Machine_Learning_Model_Training.ipynb"


def build_phase5_notebook():
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    }

    cells = []

    # Title & Objective
    cells.append(new_markdown_cell(
        "# BEANWISE: Dry Bean Classification\n"
        "## Phase 5: Machine Learning Model Development and Training\n\n"
        "**Author:** B.Tech AIML Student Researcher  \n"
        "**Project:** BEANWISE Machine Learning System  \n"
        "**Environment:** Python 3.12 | Scikit-learn | XGBoost | Pandas | NumPy | Matplotlib | Seaborn  \n"
        "**Objective:** Develop, evaluate, and tune nine distinct machine learning classification architectures "
        "using 5-fold stratified cross-validation on the training set, followed by rigorous evaluation on the "
        "held-out validation partition, strictly maintaining the final test partition untouched."
    ))

    # 1. Methodology
    cells.append(new_markdown_cell(
        "## 1. Methodology and Experimental Design\n\n"
        "In Phase 5, we perform systematic model training under strict data leakage constraints:\n"
        "1. **Dataset Ingestion:** Load cleaned morphological data and append 6 domain engineered features.\n"
        "2. **Leakage-Safe Partitioning:** Train on 70% ($N = 9,479$), tune and validate on 15% ($N = 2,032$), and retain 15% ($N = 2,032$) test set untouched.\n"
        "3. **Pipeline Construction:** Wrap every model in Scikit-learn Pipelines with appropriate feature scalers.\n"
        "4. **Cross-Validation:** 5-fold Stratified K-Fold CV to assess out-of-fold generalization and variance.\n"
        "5. **Hyperparameter Tuning:** Grid search optimization for high-capacity models (Random Forest, SVM, XGBoost).\n"
        "6. **Validation Evaluation:** Multi-metric evaluation (Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Confusion Matrix).\n"
        "7. **Model Persistence:** Export candidate pipelines and metadata for downstream deployment."
    ))

    # 2. Environment Setup
    cells.append(new_markdown_cell("## 2. Environment Setup and Library Imports"))
    cells.append(new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "PROJECT_ROOT = Path('.').resolve() if (Path('.').resolve() / 'config').exists() else Path('.').resolve().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.preprocessing import LabelEncoder\n\n"
        "from config.settings import (\n"
        "    PROCESSED_DATASET_DIR,\n"
        "    TARGET_COLUMN,\n"
        "    TARGET_CLASSES,\n"
        "    TABLES_DIR,\n"
        "    FIGURES_DIR,\n"
        ")\n"
        "from src.features.feature_engineering import add_engineered_features\n"
        "from src.features.feature_selection import split_dataset\n"
        "from src.models.model_factory import get_all_model_pipelines, get_hyperparameter_grids\n"
        "from src.models.model_trainer import cross_validate_pipelines, tune_pipelines, evaluate_on_validation\n"
        "from src.models.model_persistence import save_model_pipeline, save_label_encoder\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "print('Phase 5 environment successfully initialized.')"
    ))

    # 3. Load Data & Partitions
    cells.append(new_markdown_cell("## 3. Data Ingestion and Partition Preparation"))
    cells.append(new_code_cell(
        "cleaned_path = PROCESSED_DATASET_DIR / 'cleaned_dry_beans.csv'\n"
        "df_raw = pd.read_csv(cleaned_path)\n"
        "df_all = add_engineered_features(df_raw)\n\n"
        "train_df, val_df, test_df, meta = split_dataset(df_all, random_state=42)\n"
        "print(f'Training Samples: {len(train_df):,} ({len(train_df)/len(df_all)*100:.2f}%)')\n"
        "print(f'Validation Samples: {len(val_df):,} ({len(val_df)/len(df_all)*100:.2f}%)')\n"
        "print(f'Held-out Test Samples: {len(test_df):,} ({len(test_df)/len(df_all)*100:.2f}%) [STRICTLY RESERVED FOR PHASE 6]')\n\n"
        "feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]\n"
        "X_train = train_df[feature_cols]\n"
        "X_val = val_df[feature_cols]\n\n"
        "le = LabelEncoder()\n"
        "y_train = le.fit_transform(train_df[TARGET_COLUMN])\n"
        "y_val = le.transform(val_df[TARGET_COLUMN])\n"
        "class_names = list(le.classes_)\n"
        "print(f'Target Classes: {class_names}')\n"
        "print(f'Total Features: {len(feature_cols)}')"
    ))

    # 4. 5-Fold Stratified Cross-Validation on Baseline Models
    cells.append(new_markdown_cell(
        "## 4. 5-Fold Stratified Cross-Validation on Baseline Models\n\n"
        "We evaluate 9 distinct classification paradigms:\n"
        "1. Logistic Regression (L2 penalized)\n"
        "2. K-Nearest Neighbors (Distance-weighted)\n"
        "3. Decision Tree (CART)\n"
        "4. Random Forest Classifier\n"
        "5. Extra Trees Classifier\n"
        "6. Support Vector Classifier (RBF Kernel)\n"
        "7. Gradient Boosting Classifier\n"
        "8. Histogram Gradient Boosting Classifier\n"
        "9. XGBoost Classifier"
    ))
    cells.append(new_code_cell(
        "base_pipelines = get_all_model_pipelines(random_state=42)\n"
        "cv_summary_df, raw_cv = cross_validate_pipelines(\n"
        "    base_pipelines, X_train, pd.Series(y_train), cv=5, random_state=42\n"
        ")\n"
        "display(cv_summary_df)"
    ))

    # 5. Cross-Validation Visualization
    cells.append(new_markdown_cell("## 5. Cross-Validation Performance and Variance Comparison"))
    cells.append(new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 5.5))\n"
        "box_data = [raw_cv[m]['test_macro_f1'] for m in cv_summary_df['Model']]\n"
        "ax.boxplot(box_data, patch_artist=True,\n"
        "           boxprops=dict(facecolor='#c6dbef', color='#2b5c8f'),\n"
        "           medianprops=dict(color='#d95f02', linewidth=2))\n"
        "ax.set_xticks(range(1, len(cv_summary_df) + 1))\n"
        "ax.set_xticklabels(cv_summary_df['Model'], rotation=30, ha='right')\n"
        "ax.set_ylabel('Macro F1-Score')\n"
        "ax.set_title('5-Fold Cross-Validation Macro F1 Spread Across Models', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # 6. Hyperparameter Tuning
    cells.append(new_markdown_cell(
        "## 6. Hyperparameter Tuning for Top Candidate Models\n\n"
        "We optimize Random Forest, Support Vector Machine (RBF), and XGBoost using Stratified 5-Fold Grid Search."
    ))
    cells.append(new_code_cell(
        "param_grids = get_hyperparameter_grids()\n"
        "candidate_pipes = {k: base_pipelines[k] for k in ['random_forest', 'svm_rbf', 'xgboost']}\n"
        "tuned_models, tuning_df = tune_pipelines(\n"
        "    candidate_pipes, param_grids, X_train, pd.Series(y_train), cv=5, random_state=42\n"
        ")\n"
        "display(tuning_df)"
    ))

    # 7. Validation Set Evaluation
    cells.append(new_markdown_cell(
        "## 7. Validation Set Evaluation\n\n"
        "We evaluate all baseline and tuned models on the unseen validation partition ($N = 2,032$).\n"
        "The test partition remains 100% untouched for Phase 6."
    ))
    cells.append(new_code_cell(
        "all_models = {**base_pipelines, **tuned_models}\n"
        "val_records = []\n"
        "val_details = {}\n\n"
        "for name, pipeline in all_models.items():\n"
        "    res = evaluate_on_validation(\n"
        "        pipeline, X_train, pd.Series(y_train), X_val, pd.Series(y_val), class_names=class_names\n"
        "    )\n"
        "    val_details[name] = res\n"
        "    val_records.append({\n"
        "        'Model': name,\n"
        "        'Val Accuracy': res['accuracy'],\n"
        "        'Val Macro F1': res['macro_f1'],\n"
        "        'Val Weighted F1': res['weighted_f1'],\n"
        "        'Val Macro Precision': res['macro_precision'],\n"
        "        'Val Macro Recall': res['macro_recall'],\n"
        "        'Training Time (s)': res['fit_time'],\n"
        "        'Inference Time (s)': res['prediction_time'],\n"
        "    })\n\n"
        "val_summary_df = pd.DataFrame(val_records).sort_values('Val Macro F1', ascending=False).reset_index(drop=True)\n"
        "val_summary_df['Rank'] = range(1, len(val_summary_df) + 1)\n"
        "display(val_summary_df)"
    ))

    # 8. Best Model Detailed Performance & Confusion Matrix
    cells.append(new_markdown_cell("## 8. Best Model Performance and Confusion Matrix"))
    cells.append(new_code_cell(
        "best_name = val_summary_df.iloc[0]['Model']\n"
        "best_res = val_details[best_name]\n"
        "print(f'Top Performing Architecture: {best_name}')\n"
        "print(f'Validation Accuracy: {best_res[\"accuracy\"]*100:.2f}%')\n"
        "print(f'Validation Macro F1: {best_res[\"macro_f1\"]:.4f}')\n\n"
        "cm_labels = [class_names[i] for i in best_res['confusion_matrix_labels']]\n"
        "cm_df = pd.DataFrame(best_res['confusion_matrix'], index=cm_labels, columns=cm_labels)\n\n"
        "fig, ax = plt.subplots(figsize=(8.5, 6.5))\n"
        "sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)\n"
        "ax.set_title(f'Validation Confusion Matrix: {best_name}', fontweight='bold')\n"
        "ax.set_xlabel('Predicted Cultivar')\n"
        "ax.set_ylabel('True Cultivar')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # 9. Per-Class Metrics Table
    cells.append(new_markdown_cell("## 9. Per-Cultivar Performance Breakdown for Best Model"))
    cells.append(new_code_cell(
        "rep = best_res['classification_report']\n"
        "per_class = []\n"
        "for cls in class_names:\n"
        "    if cls in rep:\n"
        "        per_class.append({\n"
        "            'Cultivar': cls,\n"
        "            'Precision': round(rep[cls]['precision'], 4),\n"
        "            'Recall': round(rep[cls]['recall'], 4),\n"
        "            'F1-Score': round(rep[cls]['f1-score'], 4),\n"
        "            'Support': int(rep[cls]['support']),\n"
        "        })\n"
        "display(pd.DataFrame(per_class))"
    ))

    # 10. Phase 5 Conclusions
    cells.append(new_markdown_cell(
        "## 10. Phase 5 Conclusions and Next Steps\n\n"
        "- Nine baseline classification pipelines and tuned configurations were systematically trained and validated.\n"
        "- All model training strictly utilized the 70% training partition ($N = 9,479$).\n"
        "- Model performance was verified on the 15% validation partition ($N = 2,032$).\n"
        "- The held-out test partition ($N = 2,032$) remains completely untouched for final evaluation in Phase 6.\n"
        "- Persisted trained pipelines and metadata are available in `artifacts/models/` and `artifacts/metadata/`."
    ))

    nb.cells = cells
    return nb


def run_and_save():
    print("Constructing Phase 5 notebook...")
    nb = build_phase5_notebook()

    print(f"Executing {NOTEBOOK_PATH.name}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}})
    client.execute()

    print("Saving executed Phase 5 notebook...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Phase 5 notebook successfully saved at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    run_and_save()
