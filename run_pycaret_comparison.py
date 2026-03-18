# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "pycaret>=3.3",
#     "pandas<2.2",
#     "scikit-learn<1.5",
# ]
# ///
"""
PyCaret model comparison for car price prediction.

Run with: uv run run_pycaret_comparison.py

Uses the featured dataset (before manual encoding/scaling) because
PyCaret handles its own preprocessing internally. Giving it already-
encoded data would double-process it.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, pull, predict_model, save_model
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os

# --- Load featured data (before our preprocessing) ---
df = pd.read_csv("data/car-price-featured.csv")
print(f"Dataset: {df.shape}")

# Same split as manual pipeline for fair comparison
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# --- PyCaret setup ---
print("\n" + "=" * 55)
print("PYCARET SETUP")
print("=" * 55)

reg = setup(
    data=train_df,
    target="price",
    session_id=42,
    verbose=False,
)

# --- Compare 20 models ---
print("\n" + "=" * 55)
print("COMPARING 20 MODELS")
print("=" * 55)

best_model = compare_models(n_select=1)
results = pull()
print(results.to_string())

# Save full results to CSV
os.makedirs("models", exist_ok=True)
results.to_csv("data/pycaret-results.csv", index=False)
print("\nFull results saved to data/pycaret-results.csv")

# Save best model
save_model(best_model, "models/pycaret_best")
print(f"Best model saved to models/pycaret_best.pkl")

# --- Predict on test set ---
print("\n" + "=" * 55)
print("TEST SET EVALUATION")
print("=" * 55)

preds = predict_model(best_model, data=test_df)
pred_col = "prediction_label" if "prediction_label" in preds.columns else "Label"

mae = mean_absolute_error(test_df["price"], preds[pred_col])
r2 = r2_score(test_df["price"], preds[pred_col])
rmse = np.sqrt(((test_df["price"] - preds[pred_col]) ** 2).mean())

print(f"\nBest PyCaret model: {type(best_model).__name__}")
print(f"  MAE:  ${mae:,.0f}")
print(f"  RMSE: ${rmse:,.0f}")
print(f"  R2:   {r2:.3f}")

# --- Comparison with manual models ---
print("\n" + "=" * 55)
print("MANUAL vs PYCARET COMPARISON")
print("=" * 55)

manual_results = {
    "Linear Regression": {"MAE": 2197, "RMSE": 3335, "R2": 0.859},
    "Random Forest": {"MAE": 1533, "RMSE": 2195, "R2": 0.939},
    "Gradient Boosting": {"MAE": 1676, "RMSE": 2466, "R2": 0.923},
}

print(f"\n{'Model':<30} {'MAE':>10} {'RMSE':>10} {'R2':>8}")
print("-" * 60)
for name, metrics in manual_results.items():
    print(f"{name:<30} ${metrics['MAE']:>8,} ${metrics['RMSE']:>8,} {metrics['R2']:>8.3f}")
print("-" * 60)
print(f"{'PyCaret: ' + type(best_model).__name__:<30} ${mae:>8,.0f} ${rmse:>8,.0f} {r2:>8.3f}")
print("=" * 60)
