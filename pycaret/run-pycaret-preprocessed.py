# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "pycaret>=3.3",
#     "pandas<2.2",
#     "scikit-learn<1.5",
# ]
# ///
"""
Phase 3: PyCaret on user-preprocessed data.

Run with: uv run pycaret/run-pycaret-preprocessed.py

Uses the manually cleaned, feature-engineered, encoded, and scaled
train/test split from the Phase 1 pipeline. PyCaret preprocessing is
disabled to avoid double-processing.
"""

import pandas as pd
from pycaret.regression import setup, compare_models, pull, predict_model, save_model
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# --- Load preprocessed train/test split ---
X_train = pd.read_csv("data/x-train.csv")
X_test = pd.read_csv("data/x-test.csv")
y_train = pd.read_csv("data/y-train.csv")
y_test = pd.read_csv("data/y-test.csv")

train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# --- PyCaret setup (preprocessing disabled) ---
print("\n" + "=" * 55)
print("PYCARET SETUP (PREPROCESSED DATA)")
print("=" * 55)

reg = setup(
    data=train_df,
    test_data=test_df,
    target="price",
    index=False,
    session_id=42,
    preprocess=False,
    verbose=False,
)

# --- Compare models ---
print("\n" + "=" * 55)
print("COMPARING MODELS")
print("=" * 55)

best_model = compare_models(n_select=1)
results = pull()
print(results.to_string())

# Save full results to CSV
os.makedirs("pycaret", exist_ok=True)
results.to_csv("pycaret/pycaret-preprocessed-results.csv", index=False)
print("\nFull results saved to pycaret/pycaret-preprocessed-results.csv")

# Save best model
save_model(best_model, "pycaret/pycaret-preprocessed-best")
print(f"Best model saved to pycaret/pycaret-preprocessed-best.pkl")

# --- Predict on test set ---
print("\n" + "=" * 55)
print("TEST SET EVALUATION")
print("=" * 55)

preds = predict_model(best_model, data=test_df)
pred_col = "prediction_label" if "prediction_label" in preds.columns else "Label"

mae = mean_absolute_error(y_test["price"], preds[pred_col])
r2 = r2_score(y_test["price"], preds[pred_col])
rmse = np.sqrt(((y_test["price"].values - preds[pred_col].values) ** 2).mean())

print(f"\nBest PyCaret model: {type(best_model).__name__}")
print(f"  MAE:  ${mae:,.0f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  R2:   {r2:.3f}")

# --- Comparison with manual and Phase 2 models ---
print("\n" + "=" * 55)
print("ALL PHASES COMPARISON")
print("=" * 55)

previous_results = {
    "Phase 1: Linear Regression": {"MAE": 2197, "RMSE": 3335, "R2": 0.859},
    "Phase 1: Random Forest": {"MAE": 1533, "RMSE": 2195, "R2": 0.939},
    "Phase 1: Gradient Boosting": {"MAE": 1676, "RMSE": 2466, "R2": 0.923},
    "Phase 2: LightGBM (raw)": {"MAE": 2177, "RMSE": 3558, "R2": 0.840},
}

print(f"\n{'Model':<35} {'MAE':>10} {'RMSE':>10} {'R2':>8}")
print("-" * 65)
for name, metrics in previous_results.items():
    print(f"{name:<35} ${metrics['MAE']:>8,} ${metrics['RMSE']:>8,} {metrics['R2']:>8.3f}")
print("-" * 65)
print(f"{'Phase 3: ' + type(best_model).__name__ + ' (preprocessed)':<35} ${mae:>8,.0f} ${rmse:>8,.0f} {r2:>8.3f}")
print("=" * 65)
