from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


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
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Handle class imbalance (roughly 90/10)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / max(pos, 1)

model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

clf = Pipeline(steps=[("prep", preprocess), ("model", model)])

print("Training XGBoost (no leakage)...")
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]

# Use threshold 0.6 like your LightGBM final
threshold = 0.6
pred = (proba >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, proba)
print(f"ROC-AUC: {roc_auc:.6f}")
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred))