"""
Constructs and executes notebooks/03_Data_Cleaning_Preprocessing.ipynb.
Embeds all cell outputs, tables, and visualization charts.
"""

import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_Data_Cleaning_Preprocessing.ipynb"


def build_phase3_notebook():
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
        "## Phase 3: Data Cleaning and Preprocessing\n\n"
        "**Author:** B.Tech AIML Student Researcher  \n"
        "**Project:** BEANWISE Machine Learning System  \n"
        "**Environment:** Python 3.12 | Pandas | NumPy | Matplotlib | Seaborn  \n"
        "**Objective:** Transform the investigated raw Dry Bean Dataset into a validated, analysis-ready "
        "working dataset, resolving duplicate records while strictly preserving the raw source."
    ))

    # 1. Introduction & Methodology
    cells.append(new_markdown_cell(
        "## 1. Introduction and Methodology\n\n"
        "Following Phase 2 exploratory data analysis, this phase executes systematic data cleaning:\n"
        "1. **Verification of Raw Source Data:** Verify cryptographic integrity (SHA-256) and dimensions.\n"
        "2. **Data Quality Reassessment:** Confirm missing values, duplicates, and invalid numbers.\n"
        "3. **Missing Value Treatment:** Verify zero-missing status and document imputation policy.\n"
        "4. **Duplicate Record Handling:** Investigate 68 exact duplicate rows and remove them with scientific justification to eliminate data leakage.\n"
        "5. **Outlier Treatment Evaluation:** Document the decision to retain biological extremes.\n"
        "6. **Cleaned Dataset Creation:** Export cleaned data to `dataset/processed/cleaned_dry_beans.csv`.\n"
        "7. **Validation:** Execute programmatic schema and range assertions."
    ))

    # 2. Environment Setup
    cells.append(new_markdown_cell("## 2. Environment Setup and Library Imports"))
    cells.append(new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "PROJECT_ROOT = Path('.').resolve().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from config.settings import (\n"
        "    FEATURE_NAMES,\n"
        "    TARGET_COLUMN,\n"
        "    TARGET_CLASSES,\n"
        "    get_raw_dataset_path,\n"
        "    PROCESSED_DATASET_DIR,\n"
        ")\n"
        "from src.data.dataset_inspector import compute_file_sha256\n"
        "from src.preprocessing.data_cleaner import clean_dry_beans_dataset\n"
        "from src.preprocessing.data_validator import validate_dataset\n\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "print('Phase 3 environment initialized.')"
    ))

    # 3. Source Data Verification
    cells.append(new_markdown_cell("## 3. Source Data Verification"))
    cells.append(new_code_cell(
        "raw_path = get_raw_dataset_path()\n"
        "raw_sha = compute_file_sha256(raw_path)\n"
        "df_raw = pd.read_excel(raw_path)\n\n"
        "print(f'Raw Dataset Path: {raw_path}')\n"
        "print(f'SHA-256 Checksum: {raw_sha}')\n"
        "print(f'Total Raw Observations: {len(df_raw):,}')\n"
        "print(f'Total Columns: {len(df_raw.columns)}')\n"
        "print(f'Target Column: {TARGET_COLUMN} (Present: {TARGET_COLUMN in df_raw.columns})')\n"
        "print(f'Unique Classes: {sorted(df_raw[TARGET_COLUMN].unique().tolist())}')"
    ))

    # 4. Data Quality Reassessment
    cells.append(new_markdown_cell("## 4. Data Quality Reassessment"))
    cells.append(new_code_cell(
        "missing_count = int(df_raw.isnull().sum().sum())\n"
        "exact_dups = int(df_raw.duplicated().sum())\n"
        "negative_vals = int((df_raw[FEATURE_NAMES] < 0).sum().sum())\n"
        "inf_vals = int(np.isinf(df_raw[FEATURE_NAMES]).sum().sum())\n\n"
        "quality_df = pd.DataFrame([\n"
        "    {'Metric': 'Missing Cells', 'Count': missing_count, 'Pct': 0.0},\n"
        "    {'Metric': 'Exact Duplicate Rows', 'Count': exact_dups, 'Pct': round(exact_dups / len(df_raw) * 100, 3)},\n"
        "    {'Metric': 'Negative Feature Values', 'Count': negative_vals, 'Pct': 0.0},\n"
        "    {'Metric': 'Infinite Feature Values', 'Count': inf_vals, 'Pct': 0.0},\n"
        "])\n"
        "display(quality_df)"
    ))

    # 5. Duplicate Record Investigation & Handling
    cells.append(new_markdown_cell(
        "## 5. Duplicate Record Investigation and Treatment\n\n"
        "**Investigation:** Exactly 68 rows (0.50% of the dataset) have identical feature values across all 16 morphological columns. "
        "In digital seed image analysis, these represent duplicate logging or repeated photography of the same seed sample. "
        "Retaining duplicates creates risk of data leakage (identical seed appearing in both training and evaluation folds).  \n"
        "**Treatment:** Exact duplicates are removed, retaining the first occurrence."
    ))

    cells.append(new_code_cell(
        "cleaned_df, report = clean_dry_beans_dataset()\n"
        "print(f'Original Rows: {report.raw_row_count:,}')\n"
        "print(f'Duplicates Removed: {report.exact_duplicates_removed}')\n"
        "print(f'Cleaned Working Rows: {report.cleaned_row_count:,}')\n"
        "print(f'Cleaned File Saved: {report.cleaned_file_path}')"
    ))

    # 6. Class Distribution Before vs After Cleaning
    cells.append(new_markdown_cell("## 6. Class Distribution Impact Evaluation"))
    cells.append(new_code_cell(
        "classes = sorted(report.class_distribution_before.keys())\n"
        "comp_df = pd.DataFrame({\n"
        "    'Class': classes,\n"
        "    'Raw Count': [report.class_distribution_before[c] for c in classes],\n"
        "    'Cleaned Count': [report.class_distribution_after[c] for c in classes],\n"
        "    'Duplicates Removed': [report.class_distribution_before[c] - report.class_distribution_after[c] for c in classes],\n"
        "})\n"
        "comp_df['Raw Pct'] = (comp_df['Raw Count'] / report.raw_row_count * 100).round(2)\n"
        "comp_df['Cleaned Pct'] = (comp_df['Cleaned Count'] / report.cleaned_row_count * 100).round(2)\n"
        "display(comp_df)"
    ))

    # 7. Outlier Strategy Documentation
    cells.append(new_markdown_cell(
        "## 7. Outlier Treatment Strategy\n\n"
        "Statistical outliers in Area and MinorAxisLength correspond to genuine biological properties of large seed cultivars "
        "(Bombay variety mean Area = 173,500 pixels). Artificial truncation or removal would destroy legitimate cultivar variance. "
        "All valid biological extremes are retained. Robust scaling or tree-based algorithms will be employed in modeling."
    ))

    # 8. Data Validation
    cells.append(new_markdown_cell("## 8. Cleaned Dataset Validation"))
    cells.append(new_code_cell(
        "validation_res = validate_dataset(cleaned_df)\n"
        "print('Validation Result:')\n"
        "for k, v in validation_res.items():\n"
        "    print(f' - {k}: {v}')"
    ))

    # 9. Conclusion
    cells.append(new_markdown_cell(
        "## 9. Phase 3 Conclusion and Transition to Phase 4\n\n"
        "- Cleaned dataset created at `dataset/processed/cleaned_dry_beans.csv` (13,543 observations, 17 columns).\n"
        "- Data integrity verified: 0 missing cells, 0 duplicates, 0 invalid values.\n"
        "- All seven target cultivars are fully represented with unaltered relative proportions.\n"
        "- Ready for Phase 4 feature engineering and selection."
    ))

    nb.cells = cells
    return nb


def run_and_save():
    print("Constructing Phase 3 notebook...")
    nb = build_phase3_notebook()

    print(f"Executing {NOTEBOOK_PATH.name}...")
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()

    print("Saving executed Phase 3 notebook...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Phase 3 notebook successfully saved at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    run_and_save()
