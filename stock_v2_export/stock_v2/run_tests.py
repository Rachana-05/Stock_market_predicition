#!/usr/bin/env python3
"""
run_tests.py — Run all pytest tests + pylint code quality check
===============================================================
Usage:
    python run_tests.py

Shows:
    - pytest results (unit tests, integration tests, accuracy validation)
    - pylint score (code quality)
    - Combined summary
"""
import os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("  NSE Stock Prediction — Full Test Suite")
print("="*60)

# ── PYTEST
print("\n[1/2] Running pytest...\n")
pytest_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_suite.py",
     "-v", "--tb=short", "--no-header"],
    cwd=BASE
)

# ── PYLINT
print("\n[2/2] Running pylint...\n")
modules = ["data/fetch.py", "data/features.py",
           "models/ml_models.py", "models/dl_models.py",
           "models/qml_models.py", "models/qnn_models.py",
           "models/ensemble.py", "tracking/mlflow_manager.py",
           "api/main.py"]
existing = [m for m in modules if os.path.exists(os.path.join(BASE, m))]

pylint_result = subprocess.run(
    [sys.executable, "-m", "pylint", "--rcfile=.pylintrc"] + existing,
    cwd=BASE
)

# ── SUMMARY
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  pytest:  {'PASSED' if pytest_result.returncode == 0 else 'SOME FAILURES'}")
print(f"  pylint:  {'OK' if pylint_result.returncode == 0 else 'See score above'}")
print("="*60)
