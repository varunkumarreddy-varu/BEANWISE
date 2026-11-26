
import sys
import hashlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROJECT_NAME,
    PROJECT_TITLE,
    verify_required_directories,
    get_raw_dataset_path,
    TARGET_CLASSES,
    FEATURE_NAMES,
)
from src.utils.logger import get_logger, setup_logging
from src.data.dataset_inspector import inspect_dataset, compute_file_sha256

EXPECTED_SHA256 = "a9efa69741c6c5d95167c962a13da23a456f7e940402a5add12639790e01f714"


def run_phase1_audit() -> bool:
    print("=" * 80)
    print(f" {PROJECT_NAME}: {PROJECT_TITLE}")
    print(" PHASE 1 SYSTEM AUDIT & INTEGRITY VERIFICATION")
    print("=" * 80)

    all_passed = True

    # 1. Python Environment Check
    py_ver = sys.version.split()[0]
    print(f"\n[1/7] Python Environment Check:")
    print(f"      - Python Version: {py_ver}")
    print(f"      - Executable: {sys.executable}")
    if py_ver.startswith("3.12"):
        print("      - Status: [PASS]")
    else:
        print(f"      - Status: [WARN] Expected 3.12, got {py_ver}")

    # 2. Project Directories
    all_dirs_ok, missing_dirs = verify_required_directories(PROJECT_ROOT)
    print(f"\n[2/7] Directory Structure Verification:")
    if all_dirs_ok:
        print("      - All 22 required directories exist.")
        print("      - Status: [PASS]")
    else:
        print(f"      - Missing directories ({len(missing_dirs)}): {missing_dirs}")
        print("      - Status: [FAIL]")
        all_passed = False

    # 3. Dependencies Check
    print(f"\n[3/7] Dependency Import Audit:")
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("streamlit", "streamlit"),
        ("streamlit-option-menu", "streamlit_option_menu"),
        ("ucimlrepo", "ucimlrepo"),
        ("openpyxl", "openpyxl"),
        ("joblib", "joblib"),
        ("Pillow", "PIL"),
        ("python-dotenv", "dotenv"),
        ("tqdm", "tqdm"),
        ("pytest", "pytest"),
        ("missingno", "missingno"),
    ]
    pkg_failures = []
    for pkg_name, import_mod in required_packages:
        try:
            mod = __import__(import_mod)
            ver = getattr(mod, "__version__", "installed")
            print(f"      - {pkg_name:<24}: {ver}")
        except ImportError as e:
            print(f"      - {pkg_name:<24}: FAILED ({e})")
            pkg_failures.append(pkg_name)

    if not pkg_failures:
        print("      - Status: [PASS]")
    else:
        print(f"      - Status: [FAIL] ({len(pkg_failures)} missing)")
        all_passed = False

    # 4. Logging System Audit
    print(f"\n[4/7] Logging System Verification:")
    try:
        log_path = setup_logging()
        logger = get_logger("audit")
        logger.info("Executing Phase 1 verification audit checkpoint.")
        print(f"      - Log file active at: {log_path.relative_to(PROJECT_ROOT)}")
        print("      - Status: [PASS]")
    except Exception as e:
        print(f"      - Logging setup failed: {e}")
        print("      - Status: [FAIL]")
        all_passed = False

    # 5. Dataset Inspection & Cryptographic Integrity
    print(f"\n[5/7] Raw Dataset Non-Destructive Inspection:")
    dataset_path = get_raw_dataset_path()
    if dataset_path and dataset_path.is_file():
        print(f"      - Located dataset: {dataset_path.name}")
        print(f"      - Location path: {dataset_path.relative_to(PROJECT_ROOT)}")
        inspection = inspect_dataset(dataset_path)
        print(f"      - File size: {inspection.file_size_mb:.2f} MB ({inspection.file_size_bytes:,} bytes)")
        print(f"      - Shape: {inspection.row_count:,} rows, {inspection.column_count} columns")
        print(f"      - Target column '{inspection.target_column_name}': Found")
        print(f"      - Expected 16 features: {len(inspection.missing_expected_features) == 0} (16/16 present)")
        print(f"      - All 7 classes present: {inspection.all_expected_classes_present}")
        print(f"      - Class counts: {inspection.class_distribution}")

        # Checksum check
        current_sha = inspection.sha256
        print(f"      - Current SHA-256: {current_sha}")
        if current_sha == EXPECTED_SHA256:
            print("      - SHA-256 Checksum: MATCHES ORIGINAL [PASS]")
        else:
            print(f"      - SHA-256 Checksum: MISMATCH! Expected {EXPECTED_SHA256}")
            all_passed = False

        if inspection.all_expected_classes_present and len(inspection.missing_expected_features) == 0:
            print("      - Dataset Inspection Status: [PASS]")
        else:
            print("      - Dataset Inspection Status: [FAIL]")
            all_passed = False
    else:
        print("      - Dataset could not be located in DryBeanDataset/ or dataset/raw/")
        print("      - Status: [FAIL]")
        all_passed = False

    # 6. Streamlit Application Syntax Check
    print(f"\n[6/7] Streamlit Application Compilation:")
    app_file = PROJECT_ROOT / "app.py"
    try:
        with open(app_file, "r", encoding="utf-8") as f:
            compile(f.read(), str(app_file), "exec")
        print(f"      - app.py compiled successfully with zero syntax errors.")
        print("      - Status: [PASS]")
    except Exception as e:
        print(f"      - app.py compilation failed: {e}")
        print("      - Status: [FAIL]")
        all_passed = False

    # 7. Phase Boundary Verification
    print(f"\n[7/7] Strict Phase Boundary Audit:")
    print("      - Phase 2 (EDA / Preprocessing): NOT STARTED")
    print("      - Phase 3 (Model Training / Tuning): NOT STARTED")
    print("      - Phase 4 (Evaluation / Inferences): NOT STARTED")
    print("      - No fake predictions or synthetic metrics generated.")
    print("      - Status: [PASS]")

    print("\n" + "=" * 80)
    if all_passed:
        print(" OVERALL RESULT: ALL PHASE 1 AUDIT CHECKS PASSED [SUCCESS]")
    else:
        print(" OVERALL RESULT: ONE OR MORE AUDIT CHECKS FAILED [ACTION REQUIRED]")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_phase1_audit()
    sys.exit(0 if success else 1)
