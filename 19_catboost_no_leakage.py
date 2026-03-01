from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

train_df = pd.read_csv(DATA_DIR / "need_train.csv")
test_df = pd.read_csv(DATA_DIR / "need_test.csv")

target_col = "fuel_poverty_risk"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].astype(int)
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col].astype(int)

LEAKAGE_COLS = ["Gcons_recent_avg", "Econs_recent_avg"]
drop_cols = [c for c in LEAKAGE_COLS if c in X_train.columns]
if drop_cols:
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

print("Categorical columns:", cat_cols)

# CatBoost can handle imbalance via class_weights
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
class_weights = [1.0, neg / max(pos, 1)]

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
    class_weights=class_weights,
)

print("Training CatBoost (no leakage)...")
model.fit(X_train, y_train, cat_features=cat_idx)

proba = model.predict_proba(X_test)[:, 1]

threshold = 0.6
pred = (proba >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, proba)
print(f"ROC-AUC: {roc_auc:.6f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred))