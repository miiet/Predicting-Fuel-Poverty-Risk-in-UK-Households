import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load train/test
# -----------------------------
train_path = DATA_DIR / "need_train.csv"
test_path = DATA_DIR / "need_test.csv"

print(f"Loading:\n- {train_path}\n- {test_path}")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "fuel_poverty_risk"

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].astype(int)

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].astype(int)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# -----------------------------
# Identify leakage columns to DROP
# (we remove energy-consumption-derived features)
# -----------------------------
LEAKAGE_COLS = [
    "Gcons_recent_avg",
    "Econs_recent_avg",
]

leakage_present = [c for c in LEAKAGE_COLS if c in X_train.columns]
if leakage_present:
    print("Dropping leakage features:", leakage_present)
    X_train = X_train.drop(columns=leakage_present)
    X_test = X_test.drop(columns=leakage_present)
else:
    print("No leakage columns found to drop (ok).")


# -----------------------------
# Preprocessing: One-hot encode categoricals
# -----------------------------
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# -----------------------------
# Model
# -----------------------------
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight="balanced",
)

clf = Pipeline(steps=[("prep", preprocess), ("model", model)])

print("Training LightGBM (no leakage)...")
clf.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.6).astype(int)  # keep your earlier good threshold default

roc_auc = roc_auc_score(y_test, proba)
report = classification_report(y_test, pred, output_dict=True)
cm = confusion_matrix(y_test, pred)

print(f"ROC-AUC: {roc_auc:.6f}")
print("Confusion matrix:\n", cm)

# Save ROC curve plot
fpr, tpr, thresholds = roc_curve(y_test, proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — LightGBM (no leakage)")
roc_path = OUT_FIGURES / "roc_lightgbm_no_leakage.png"
plt.savefig(roc_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved ROC plot -> {roc_path}")

# -----------------------------
# Save model + metrics
# -----------------------------
model_path = OUT_MODELS / "lightgbm_no_leakage.pkl"
joblib.dump(clf, model_path)
print(f"Saved model -> {model_path}")

metrics = {
    "model": "LightGBM_no_leakage",
    "threshold": 0.6,
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features_used": int(X_train.shape[1]),
    "dropped_leakage_cols": leakage_present,
}

metrics_path = OUT_METRICS / "lightgbm_no_leakage_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved metrics -> {metrics_path}")
print("✅ Done.")