"""
Master Script: Run All RQs for Bitcoin Price Prediction
=======================================================

This script runs all 7 research question scripts sequentially.
Each RQ script loads the raw Bitcoin data, performs analysis, 
and saves outputs to its respective figures/ and tables/ folders.

Usage on Kaggle:
1. Upload the Bitcoin dataset to /kaggle/input/bitcoin-historical-data/
2. Run each script individually or use this master script
3. Outputs will be saved in RQ{1-7}/figures/ and RQ{1-7}/tables/

Dataset: Bitcoin Historical Data from Kaggle
Target: Next-day Bitcoin closing price (regression)
"""

import subprocess
import sys

rqs = [
    "RQ1/rq1_baseline_performance.py",
    "RQ2/rq2_model_comparison.py",
    "RQ3/rq3_preprocessing_effect.py",
    "RQ4/rq4_feature_importance.py",
    "RQ5/rq5_metric_sensitivity.py",
    "RQ6/rq6_robustness_generalization.py",
    "RQ7/rq7_practical_recommendation.py"
]

for rq in rqs:
    print(f"\n{'='*60}")
    print(f"Running: {rq}")
    print('='*60)
    result = subprocess.run([sys.executable, rq], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {rq} failed with return code {result.returncode}")
    else:
        print(f"SUCCESS: {rq} completed")

print("\n" + "="*60)
print("ALL RQs COMPLETED!")
print("="*60)
