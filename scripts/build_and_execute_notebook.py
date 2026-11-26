"""
Script to Construct and Execute notebooks/02_Exploratory_Data_Analysis.ipynb.
Uses nbformat and nbclient to programmatically create and run all cells,
saving fully populated execution outputs and inline plots.
"""

import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_Exploratory_Data_Analysis.ipynb"


def build_notebook():
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

    # Title & Metadata
    cells.append(new_markdown_cell(
        "# BEANWISE: Dry Bean Classification\n"
        "## Phase 2: Exploratory Data Analysis and Data Quality Audit\n\n"
        "**Author:** B.Tech AIML Student Researcher  \n"
        "**Project:** BEANWISE Machine Learning System  \n"
        "**Environment:** Python 3.12 | Scikit-learn | Pandas | Matplotlib | Seaborn | Plotly  \n"
        "**Objective:** Systematic investigation of dataset schema, missingness, duplicates, "
        "distributions, correlations, statistical outliers, and class-discriminative variance."
    ))

    # Section 1: Introduction
    cells.append(new_markdown_cell(
        "## 1. Introduction\n\n"
        "The Dry Bean Dataset represents high-resolution computer vision measurements of seven commercial "
        "dry bean (*Phaseolus vulgaris L.*) cultivars: Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira. "
        "Seed classification is crucial in seed certification and agricultural processing, where visual similarity "
        "makes manual grading error-prone.\n\n"
        "The objective of this exploratory data analysis (EDA) is to:\n"
        "1. Understand the structural properties and data types of the dataset.\n"
        "2. Audit data quality (missing values, exact duplicates, invalid values, and near-constant features).\n"
        "3. Evaluate univariate statistical distributions, central tendencies, and dispersions.\n"
        "4. Examine class representation and measure cultivar imbalance.\n"
        "5. Investigate bivariate and multivariate correlations, identifying potential multicollinearity.\n"
        "6. Quantify statistical outliers across features without modifying the raw data.\n"
        "7. Perform hypothesis testing (ANOVA and Kruskal-Wallis) to identify features with high discriminative power.\n"
        "8. Synthesize findings to guide future preprocessing and model selection."
    ))

    # Section 2: Environment Setup and Library Imports
    cells.append(new_markdown_cell(
        "## 2. Environment Setup and Library Imports\n\n"
        "We import core data science, visualization, and statistical libraries with fixed random states "
        "and standard plotting parameters."
    ))

    cells.append(new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# Ensure project root is in path\n"
        "PROJECT_ROOT = Path('.').resolve().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from scipy import stats\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        "from config.settings import (\n"
        "    PROJECT_NAME,\n"
        "    TARGET_COLUMN,\n"
        "    TARGET_CLASSES,\n"
        "    FEATURE_NAMES,\n"
        "    FEATURE_DESCRIPTIONS,\n"
        "    get_raw_dataset_path,\n"
        ")\n"
        "from src.data.eda_analyzer import (\n"
        "    compute_dataset_summary,\n"
        "    compute_data_quality_report,\n"
        "    compute_class_distribution,\n"
        "    compute_descriptive_statistics,\n"
        "    compute_outlier_report,\n"
        "    compute_correlation_analysis,\n"
        "    compute_class_wise_statistics,\n"
        "    compute_feature_variability_and_tests,\n"
        "    compute_exploratory_pca,\n"
        ")\n\n"
        "# Configure default plotting parameters\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n"
        "plt.rcParams['figure.dpi'] = 120\n"
        "pd.set_option('display.max_columns', 20)\n"
        "pd.set_option('display.width', 1000)\n"
        "print('Environment and libraries initialized successfully.')"
    ))

    # Section 3: Data Loading
    cells.append(new_markdown_cell(
        "## 3. Data Loading and Initial Inspection\n\n"
        "The dataset is loaded directly from the original `DryBeanDataset/Dry_Bean_Dataset.xlsx` file. "
        "We verify existence, dimensions, columns, and inspect initial records."
    ))

    cells.append(new_code_cell(
        "dataset_path = get_raw_dataset_path()\n"
        "print(f'Loading raw dataset from: {dataset_path}')\n"
        "df = pd.read_excel(dataset_path)\n"
        "print(f'Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns')\n"
        "print(f'Target Column Present: {TARGET_COLUMN in df.columns}')\n"
        "df.head(10)"
    ))

    cells.append(new_code_cell(
        "print('Last 5 records:')\n"
        "display(df.tail(5))\n\n"
        "print('Random sample of 5 records (reproducible seed 42):')\n"
        "display(df.sample(5, random_state=42))"
    ))

    # Section 4: Dataset Structure and Schema
    cells.append(new_markdown_cell(
        "## 4. Dataset Structure and Attribute Types\n\n"
        "We inspect the data types, non-null counts, and unique value counts for all 17 attributes."
    ))

    cells.append(new_code_cell(
        "summary_df = compute_dataset_summary(df)\n"
        "display(summary_df)"
    ))

    cells.append(new_code_cell(
        "print('Dataset Memory and Dtype Overview:')\n"
        "df.info()"
    ))

    # Section 5: Data Quality Analysis
    cells.append(new_markdown_cell(
        "## 5. Data Quality Analysis\n\n"
        "A rigorous data quality audit investigates:\n"
        "- Missing values per column and total missingness percentage.\n"
        "- Exact duplicate records.\n"
        "- Invalid values (negative, zero, NaN, infinite).\n"
        "- Constant or near-constant features."
    ))

    cells.append(new_code_cell(
        "quality_dict = compute_data_quality_report(df)\n"
        "print('--- DATA QUALITY AUDIT RESULTS ---')\n"
        "print(f'Total observations: {quality_dict[\"total_records\"]:,}')\n"
        "print(f'Total input features: {quality_dict[\"total_features\"]}')\n"
        "print(f'Total missing cells: {quality_dict[\"total_missing_cells\"]} ({quality_dict[\"missing_percentage\"]}%)')\n"
        "print(f'Exact duplicate records: {quality_dict[\"exact_duplicates\"]} ({quality_dict[\"duplicate_percentage\"]}%)')\n"
        "print(f'Unique records: {quality_dict[\"unique_records\"]:,}')\n"
        "print(f'Constant features: {quality_dict[\"constant_features\"]}')\n"
        "print(f'Low variance features: {quality_dict[\"low_variance_features\"]}')\n"
        "print(f'Negative numerical values: {sum(quality_dict[\"negative_values_per_col\"].values())}')\n"
        "print(f'Zero values in features: {sum(quality_dict[\"zero_values_per_col\"].values())}')\n"
        "print(f'Infinite values in features: {sum(quality_dict[\"infinite_values_per_col\"].values())}')"
    ))

    cells.append(new_markdown_cell(
        "**Observation on Duplicates:**  \n"
        "There are 68 exact duplicate records (0.50% of the dataset). In accordance with Phase 2 rules, "
        "no records are removed in this phase. Whether these duplicates reflect true biological twin samples "
        "or image acquisition artifact will be evaluated in Phase 3 preprocessing."
    ))

    # Section 6: Target Class Distribution
    cells.append(new_markdown_cell(
        "## 6. Target Class Distribution and Imbalance Evaluation\n\n"
        "We evaluate the distribution across the seven dry bean varieties."
    ))

    cells.append(new_code_cell(
        "class_df = compute_class_distribution(df)\n"
        "display(class_df)\n\n"
        "# Class distribution plot\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n"
        "sns.barplot(data=class_df, x='Class', y='Sample Count', ax=ax1, palette='Blues_r')\n"
        "ax1.set_title('Target Class Frequencies', fontweight='bold')\n"
        "ax1.set_ylabel('Sample Count')\n"
        "ax1.tick_params(axis='x', rotation=30)\n"
        "for p in ax1.patches:\n"
        "    h = p.get_height()\n"
        "    ax1.annotate(f'{int(h):,}', (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom', fontsize=8)\n\n"
        "ax2.pie(class_df['Sample Count'], labels=class_df['Class'], autopct='%1.1f%%', startangle=140,\n"
        "        colors=sns.color_palette('tab10', 7))\n"
        "ax2.set_title('Class Proportions', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(new_markdown_cell(
        "**Observation on Class Imbalance:**  \n"
        "The dataset exhibits moderate imbalance: Dermason constitutes 26.05% (3,546 samples) as the majority class, "
        "while Bombay constitutes only 3.84% (522 samples) as the minority class. The imbalance ratio is 6.79:1. "
        "Stratified cross-validation and balanced evaluation metrics (Macro F1) will be essential during modeling."
    ))

    # Section 7: Descriptive Statistical Analysis
    cells.append(new_markdown_cell(
        "## 7. Descriptive Statistical Analysis\n\n"
        "Detailed computation of central tendency, spread, quartiles, skewness, and kurtosis for all 16 features."
    ))

    cells.append(new_code_cell(
        "stats_df = compute_descriptive_statistics(df)\n"
        "display(stats_df)"
    ))

    # Section 8: Univariate Distributions and Boxplots
    cells.append(new_markdown_cell(
        "## 8. Univariate Feature Distributions and Boxplots\n\n"
        "Visualizing frequency histograms, kernel density estimates, and boxplots across all features."
    ))

    cells.append(new_code_cell(
        "fig, axes = plt.subplots(4, 4, figsize=(15, 11))\n"
        "axes = axes.flatten()\n"
        "for idx, feat in enumerate(FEATURE_NAMES):\n"
        "    sns.histplot(df[feat], kde=True, ax=axes[idx], color='#1f77b4', stat='density', alpha=0.5)\n"
        "    axes[idx].set_title(feat, fontsize=9, fontweight='bold')\n"
        "    axes[idx].set_xlabel('')\n"
        "    axes[idx].tick_params(labelsize=8)\n"
        "plt.suptitle('Univariate Distributions across 16 Features', fontsize=12, fontweight='bold', y=0.995)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(new_code_cell(
        "fig, axes = plt.subplots(4, 4, figsize=(15, 11))\n"
        "axes = axes.flatten()\n"
        "for idx, feat in enumerate(FEATURE_NAMES):\n"
        "    sns.boxplot(y=df[feat], ax=axes[idx], color='#6baed6', fliersize=2)\n"
        "    axes[idx].set_title(feat, fontsize=9, fontweight='bold')\n"
        "    axes[idx].set_ylabel('')\n"
        "    axes[idx].tick_params(labelsize=8)\n"
        "plt.suptitle('Feature Dispersion and Statistical Outlier Boxplots', fontsize=12, fontweight='bold', y=0.995)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Section 9: Correlation Analysis
    cells.append(new_markdown_cell(
        "## 9. Correlation Analysis (Pearson & Spearman)\n\n"
        "We evaluate linear (Pearson) and monotonic (Spearman) relationships to diagnose multicollinearity."
    ))

    cells.append(new_code_cell(
        "corr_results = compute_correlation_analysis(df, threshold=0.80)\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))\n"
        "sns.heatmap(corr_results['pearson_matrix'], cmap='coolwarm', center=0, ax=ax1, cbar_kws={'label': 'r'})\n"
        "ax1.set_title('Pearson Correlation Matrix', fontweight='bold')\n"
        "sns.heatmap(corr_results['spearman_matrix'], cmap='coolwarm', center=0, ax=ax2, cbar_kws={'label': 'rho'})\n"
        "ax2.set_title('Spearman Rank Correlation Matrix', fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print('Top Strong Positive Correlations (|r| >= 0.80):')\n"
        "display(corr_results['strong_positive_df'].head(8))\n\n"
        "print('Top Strong Negative Correlations (|r| <= -0.80):')\n"
        "display(corr_results['strong_negative_df'].head(8))"
    ))

    cells.append(new_markdown_cell(
        "**Observation on Multicollinearity:**  \n"
        "There is extreme collinearity among dimensional attributes:\n"
        "- `Area` and `ConvexArea`: $r = 1.00$\n"
        "- `Area` and `EquivDiameter`: $r = 0.99$\n"
        "- `Perimeter` and `MajorAxisLength`: $r = 0.98$\n"
        "- `AspectRation` and `Eccentricity`: $r = 0.92$\n"
        "Tree-based models (such as Random Forest and XGBoost) handle collinearity naturally, whereas linear models "
        "will benefit from regularization or dimensionality reduction."
    ))

    # Section 10: Outlier Analysis
    cells.append(new_markdown_cell(
        "## 10. Statistical Outlier Audit (IQR vs. Z-Score)\n\n"
        "We quantify potential statistical outliers without modifying any raw measurements."
    ))

    cells.append(new_code_cell(
        "outlier_df = compute_outlier_report(df)\n"
        "display(outlier_df)"
    ))

    # Section 11: Class-Stratified Feature Analysis & Hypothesis Testing
    cells.append(new_markdown_cell(
        "## 11. Class-Stratified Comparisons & Hypothesis Testing\n\n"
        "We test whether feature means and medians differ significantly across the seven varieties using "
        "One-way ANOVA and the non-parametric Kruskal-Wallis test, applying Bonferroni correction for 16 comparisons."
    ))

    cells.append(new_code_cell(
        "var_df, tests_df = compute_feature_variability_and_tests(df)\n"
        "print('Hypothesis Testing Results (ANOVA & Kruskal-Wallis):')\n"
        "display(tests_df)\n\n"
        "print('Feature Variance Decomposition (Between vs Within Class):')\n"
        "display(var_df)"
    ))

    cells.append(new_markdown_cell(
        "**Observation on Discriminative Power:**  \n"
        "All 16 features exhibit statistically significant differences across cultivars ($p < 0.001$ after Bonferroni correction). "
        "The highest F-ratios occur in `ShapeFactor1` ($F = 9,331.1$, $\\eta^2 = 0.804$), `Compactness` ($F = 7,613.2$), "
        "and `ShapeFactor3` ($F = 6,864.8$), indicating that morphological shape factors provide the strongest "
        "differentiation among varieties."
    ))

    # Section 12: Exploratory PCA
    cells.append(new_markdown_cell(
        "## 12. Exploratory Principal Component Analysis (PCA)\n\n"
        "We temporarily standardize the 16 features to inspect low-dimensional clustering. "
        "This is an exploratory visualization only and does not alter the raw data."
    ))

    cells.append(new_code_cell(
        "pca_res = compute_exploratory_pca(df, n_components=5)\n"
        "display(pca_res['pca_summary_df'])\n\n"
        "fig, ax = plt.subplots(figsize=(9, 6.5))\n"
        "pca_df = pd.DataFrame({\n"
        "    'PC1': pca_res['X_pca_2d'][:, 0],\n"
        "    'PC2': pca_res['X_pca_2d'][:, 1],\n"
        "    'Class': df[TARGET_COLUMN].values\n"
        "})\n"
        "sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Class', palette='tab10', alpha=0.6, s=15, ax=ax)\n"
        "ax.set_title(f'Exploratory PCA: PC1 vs PC2 (Explained Variance: {pca_res[\"var_pc1\"] + pca_res[\"var_pc2\"]:.1f}%)', fontweight='bold')\n"
        "ax.set_xlabel(f'PC1 ({pca_res[\"var_pc1\"]}% Variance)')\n"
        "ax.set_ylabel(f'PC2 ({pca_res[\"var_pc2\"]}% Variance)')\n"
        "ax.legend(title='Cultivar', bbox_to_anchor=(1.02, 1), loc='upper left')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Section 13: Summary of Findings and Phase 3 Recommendations
    cells.append(new_markdown_cell(
        "## 13. Summary of Findings and Preprocessing Recommendations\n\n"
        "### Key Findings\n"
        "1. **Completeness:** The dataset has zero missing values across all 13,611 records and 17 columns.\n"
        "2. **Duplicates:** 68 exact duplicate records (0.50%) were identified and preserved intact.\n"
        "3. **Data Types:** All 16 features are strictly numerical (float64 and int64); target `Class` is categorical.\n"
        "4. **Class Imbalance:** Moderately imbalanced (Dermason 26.05% vs Bombay 3.84%, ratio 6.79:1).\n"
        "5. **Multicollinearity:** Extreme linear correlation between Area, ConvexArea, Perimeter, and EquivDiameter ($r > 0.98$).\n"
        "6. **Discriminative Attributes:** `ShapeFactor1`, `Compactness`, and `ShapeFactor3` demonstrate the highest "
        "between-class variance ratios ($F > 6,800$), whereas size features separate Bombay and Cali.\n"
        "7. **Dimensionality:** Two principal components capture 85.3% of total standardized variance.\n\n"
        "### Recommendations for Phase 3\n"
        "- Apply feature scaling (`StandardScaler` or `RobustScaler`) to handle variable magnitude differences (Area ~40,000 vs ShapeFactor ~0.005).\n"
        "- Use stratified sampling for train/validation/test splits to preserve class proportions.\n"
        "- Benchmark both linear regularized models and non-linear ensembles (Random Forest, XGBoost)."
    ))

    nb.cells = cells
    return nb


def run_and_save_notebook():
    print("Constructing notebook structure...", flush=True)
    nb = build_notebook()

    print(f"Executing notebook {NOTEBOOK_PATH.name}...", flush=True)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    print("Saving executed notebook with cell outputs...", flush=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook successfully saved at: {NOTEBOOK_PATH}", flush=True)


if __name__ == "__main__":
    run_and_save_notebook()
