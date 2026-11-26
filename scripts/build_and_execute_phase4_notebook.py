"""
Constructs and executes notebooks/04_Feature_Engineering_Selection.ipynb.
Embeds all cell outputs, tables, and visualization charts.
"""

import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_Feature_Engineering_Selection.ipynb"


def build_phase4_notebook():
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
        "## Phase 4: Feature Engineering and Feature Selection\n\n"
        "**Author:** B.Tech AIML Student Researcher  \n"
        "**Project:** BEANWISE Machine Learning System  \n"
        "**Environment:** Python 3.12 | Scikit-learn | Pandas | NumPy | Matplotlib | Seaborn  \n"
        "**Objective:** Formulate domain-derived morphological indicators, establish leakage-safe "
        "stratified data splits, and evaluate multi-criteria feature selection algorithms on training data."
    ))

    # 1. Introduction
    cells.append(new_markdown_cell(
        "## 1. Introduction and Methodology\n\n"
        "In Phase 4, we examine whether mathematical combinations of physical features improve cultivar separation, "
        "and systematically rank feature utility using multiple selection paradigms:\n"
        "1. **Load Cleaned Dataset:** Ingest `dataset/processed/cleaned_dry_beans.csv` (13,543 records).\n"
        "2. **Feature Engineering:** Derive 6 domain-relevant morphological ratios.\n"
        "3. **Data Leakage Prevention:** Stratified 3-way split into 70% Train, 15% Validation, and 15% Test partitions.\n"
        "4. **Feature Evaluation:** Evaluate Mutual Information, ANOVA F-test, Random Forest Gini Importance, and RFE strictly on the training partition.\n"
        "5. **Consolidated Ranking:** Compare multi-method feature utility and identify optimal feature subsets for Phase 5 modeling."
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
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "from config.settings import (\n"
        "    FEATURE_NAMES,\n"
        "    TARGET_COLUMN,\n"
        "    TARGET_CLASSES,\n"
        "    PROCESSED_DATASET_DIR,\n"
        ")\n"
        "from src.features.feature_engineering import (\n"
        "    add_engineered_features,\n"
        "    get_engineered_feature_names,\n"
        "    ENGINEERED_FEATURE_METADATA,\n"
        ")\n"
        "from src.features.feature_selection import (\n"
        "    split_dataset,\n"
        "    evaluate_feature_selection_methods,\n"
        ")\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "print('Phase 4 environment initialized.')"
    ))

    # 3. Load Cleaned Dataset
    cells.append(new_markdown_cell("## 3. Load Cleaned Dataset"))
    cells.append(new_code_cell(
        "cleaned_path = PROCESSED_DATASET_DIR / 'cleaned_dry_beans.csv'\n"
        "df = pd.read_csv(cleaned_path)\n"
        "print(f'Loaded Cleaned Observations: {len(df):,}')\n"
        "print(f'Columns: {list(df.columns)}')\n"
        "df.head()"
    ))

    # 4. Feature Engineering
    cells.append(new_markdown_cell(
        "## 4. Morphological Feature Engineering\n\n"
        "We compute 6 domain-relevant ratios:\n"
        "- `Area_to_Perimeter`: `Area / Perimeter`\n"
        "- `Axis_Ratio`: `MajorAxisLength / MinorAxisLength`\n"
        "- `Axis_Difference`: `MajorAxisLength - MinorAxisLength`\n"
        "- `Convex_Area_Ratio`: `ConvexArea / Area`\n"
        "- `Compactness_Roundness_Ratio`: `Compactness / roundness`\n"
        "- `EquivDiameter_to_MajorAxis`: `EquivDiameter / MajorAxisLength`"
    ))
    cells.append(new_code_cell(
        "df_engineered = add_engineered_features(df)\n"
        "eng_cols = get_engineered_feature_names()\n"
        "print(f'New features computed: {eng_cols}')\n"
        "print(f'Total features available: {len(FEATURE_NAMES) + len(eng_cols)}')\n"
        "display(df_engineered[eng_cols].describe().round(4))"
    ))

    # 5. Data Leakage Prevention (Stratified Split)
    cells.append(new_markdown_cell(
        "## 5. Data Leakage Prevention: Stratified Partitioning\n\n"
        "To strictly prevent data snooping and information leakage:\n"
        "- Training Partition: 70.0% (9,480 samples)\n"
        "- Validation Partition: 15.0% (2,031 samples)\n"
        "- Final Held-Out Test Partition: 15.0% (2,032 samples)\n"
        "All feature scaling, selection, and model tuning will strictly utilize only the training partition."
    ))
    cells.append(new_code_cell(
        "train_df, val_df, test_df, split_meta = split_dataset(df_engineered, random_state=42)\n"
        "print(f'Train Samples: {len(train_df):,} ({split_meta[\"train_pct\"]}%)')\n"
        "print(f'Validation Samples: {len(val_df):,} ({split_meta[\"val_pct\"]}%)')\n"
        "print(f'Test Samples: {len(test_df):,} ({split_meta[\"test_pct\"]}%)')\n\n"
        "# Verify class balance in each split\n"
        "split_comp = pd.DataFrame({\n"
        "    'Class': sorted(df[TARGET_COLUMN].unique()),\n"
        "    'Train Count': [split_meta['train_class_distribution'][c] for c in sorted(df[TARGET_COLUMN].unique())],\n"
        "    'Val Count': [split_meta['val_class_distribution'][c] for c in sorted(df[TARGET_COLUMN].unique())],\n"
        "    'Test Count': [split_meta['test_class_distribution'][c] for c in sorted(df[TARGET_COLUMN].unique())],\n"
        "})\n"
        "split_comp['Train %'] = (split_comp['Train Count'] / len(train_df) * 100).round(2)\n"
        "split_comp['Val %'] = (split_comp['Val Count'] / len(val_df) * 100).round(2)\n"
        "display(split_comp)"
    ))

    # 6. Feature Selection on Training Set
    cells.append(new_markdown_cell(
        "## 6. Multi-Criteria Feature Selection (Training Set Only)\n\n"
        "We evaluate Mutual Information, ANOVA F-test, Random Forest Gini Importance, and Recursive Feature Elimination."
    ))
    cells.append(new_code_cell(
        "all_feature_names = FEATURE_NAMES + eng_cols\n"
        "X_train = train_df[all_feature_names]\n"
        "y_train = train_df[TARGET_COLUMN]\n\n"
        "sel_results = evaluate_feature_selection_methods(train_df, y_train, all_feature_names, random_state=42)\n"
        "rankings_df = sel_results['consolidated_rankings_df']\n"
        "print('Consolidated Multi-Criteria Feature Rankings (Sorted by Mean Rank):')\n"
        "display(rankings_df)"
    ))

    # 7. Correlation Heatmap on Training Features
    cells.append(new_markdown_cell("## 7. Correlation Analysis on Training Features"))
    cells.append(new_code_cell(
        "corr_mat = X_train.corr(method='pearson')\n"
        "fig, ax = plt.subplots(figsize=(13, 10))\n"
        "sns.heatmap(corr_mat, cmap='coolwarm', center=0, annot=False, ax=ax)\n"
        "ax.set_title('Training Set Correlation Matrix (22 Features)', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # 8. PCA on Scaled Training Features
    cells.append(new_markdown_cell("## 8. Principal Component Analysis on Training Set"))
    cells.append(new_code_cell(
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train[FEATURE_NAMES])\n"
        "pca = PCA(n_components=6, random_state=42)\n"
        "X_train_pca = pca.fit_transform(X_train_scaled)\n\n"
        "pca_df = pd.DataFrame({\n"
        "    'Component': [f'PC{i+1}' for i in range(6)],\n"
        "    'Explained Variance Ratio': pca.explained_variance_ratio_.round(4),\n"
        "    'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_).round(4)\n"
        "})\n"
        "display(pca_df)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "ax.bar(range(1, 7), pca.explained_variance_ratio_ * 100, color='#2b5c8f', label='Individual')\n"
        "ax.plot(range(1, 7), np.cumsum(pca.explained_variance_ratio_) * 100, color='#d95f02', marker='o', label='Cumulative')\n"
        "ax.set_xlabel('Principal Component')\n"
        "ax.set_ylabel('Variance Explained (%)')\n"
        "ax.set_title('Training Set Scree Plot', fontweight='bold')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # 9. Phase 4 Conclusion
    cells.append(new_markdown_cell(
        "## 9. Phase 4 Conclusion and Preprocessing Strategy for Modeling\n\n"
        "- 6 engineered features were successfully formulated and added.\n"
        "- Stratified 3-way partition established (Train = 9,480, Val = 2,031, Test = 2,032).\n"
        "- Multi-criteria feature selection identified top discriminative features: `ShapeFactor1`, `Perimeter`, "
        "`MajorAxisLength`, `Area`, `Compactness`, `AspectRation`, and `Axis_Ratio`.\n"
        "- Test set remains 100% untouched for Phase 6 evaluation.\n"
        "- Transitioning directly to Phase 5 model development."
    ))

    nb.cells = cells
    return nb


def run_and_save():
    print("Constructing Phase 4 notebook...")
    nb = build_phase4_notebook()

    print(f"Executing {NOTEBOOK_PATH.name}...")
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()

    print("Saving executed Phase 4 notebook...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Phase 4 notebook successfully saved at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    run_and_save()
