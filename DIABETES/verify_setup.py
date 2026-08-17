#!/usr/bin/env python3
"""
Setup Verification Script for Diabetes Dataset Extension

This script verifies that all files are in place and dependencies are installed
before running the experiments.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path
import importlib.util

def check_mark(condition):
    """Return check mark or X based on condition."""
    return "✅" if condition else "❌"

def check_file_exists(filepath):
    """Check if a file exists."""
    return Path(filepath).exists()

def check_module(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None

def main():
    print("="*70)
    print("DIABETES DATASET EXTENSION - SETUP VERIFICATION")
    print("="*70)

    all_checks_passed = True

    # 1. Check dataset file
    print("\n[1] DATASET FILES")
    print("-" * 70)

    dataset_file = "data/BRFSS2015.csv"
    dataset_exists = check_file_exists(dataset_file)
    print(f"{check_mark(dataset_exists)} Dataset file: {dataset_file}")

    if not dataset_exists:
        print("   ⚠️  Dataset file not found!")
        print("   Please ensure BRFSS2015.csv is in the data/ directory")
        all_checks_passed = False

    # 2. Check notebook files
    print("\n[2] JUPYTER NOTEBOOKS")
    print("-" * 70)

    notebooks = [
        "data_preprocessor.ipynb",
        "RandomForest/STD_RF.ipynb",
        "RandomForest/DP_RF.ipynb",
        "LR/STD_LR.ipynb",
        "LR/DP_LR.ipynb",
        "GaussianNB/STD_GNB.ipynb",
        "GaussianNB/DP_GNB.ipynb",
        "SVM/STD_SVM.ipynb",
        "SVM/DP_SVM.ipynb",
        "DNN/STD_DNN.ipynb",
        "DNN/DP_DNN.ipynb",
    ]

    missing_notebooks = []
    for notebook in notebooks:
        exists = check_file_exists(notebook)
        status = check_mark(exists)
        print(f"{status} {notebook}")
        if not exists:
            missing_notebooks.append(notebook)
            all_checks_passed = False

    if missing_notebooks:
        print(f"\n   ⚠️  {len(missing_notebooks)} notebook(s) missing!")

    # 3. Check documentation files
    print("\n[3] DOCUMENTATION")
    print("-" * 70)

    docs = [
        "README.md",
        "EXECUTION_GUIDE.md",
        "DNN/README.md",
        "DNN/QUICKSTART.md",
    ]

    for doc in docs:
        exists = check_file_exists(doc)
        print(f"{check_mark(exists)} {doc}")
        if not exists:
            all_checks_passed = False

    # 4. Check analysis scripts
    print("\n[4] ANALYSIS SCRIPTS")
    print("-" * 70)

    scripts = [
        "Result/compare_all_models.py",
        "verify_setup.py",
    ]

    for script in scripts:
        exists = check_file_exists(script)
        print(f"{check_mark(exists)} {script}")
        if not exists:
            all_checks_passed = False

    # 5. Check Python dependencies
    print("\n[5] PYTHON DEPENDENCIES")
    print("-" * 70)

    required_modules = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "diffprivlib": "IBM Differential Privacy Library",
        "tensorflow": "TensorFlow",
        "tensorflow_privacy": "TensorFlow Privacy",
    }

    missing_modules = []
    for module, name in required_modules.items():
        installed = check_module(module)
        print(f"{check_mark(installed)} {name} ({module})")
        if not installed:
            missing_modules.append(module)
            all_checks_passed = False

    if missing_modules:
        print(f"\n   ⚠️  {len(missing_modules)} module(s) not installed!")
        print("   Install with: pip install " + " ".join(missing_modules))

    # 6. Check directory structure
    print("\n[6] DIRECTORY STRUCTURE")
    print("-" * 70)

    directories = [
        "data",
        "RandomForest",
        "LR",
        "GaussianNB",
        "SVM",
        "DNN",
        "Result",
    ]

    for directory in directories:
        exists = Path(directory).is_dir()
        print(f"{check_mark(exists)} {directory}/")
        if not exists:
            all_checks_passed = False

    # 7. Check Python version
    print("\n[7] PYTHON VERSION")
    print("-" * 70)

    python_version = sys.version_info
    version_ok = python_version >= (3, 8)
    print(f"{check_mark(version_ok)} Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    if not version_ok:
        print("   ⚠️  Python 3.8+ required!")
        all_checks_passed = False

    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou are ready to run the experiments.")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook data_preprocessor.ipynb")
        print("  2. Then run baseline models (STD_*.ipynb)")
        print("  3. Finally run DP models (DP_*.ipynb)")
        print("\nFor detailed instructions, see EXECUTION_GUIDE.md")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
        print("See EXECUTION_GUIDE.md for detailed setup instructions.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
