import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    out_dir = project_root / "outputs" / "shap_xgboost_no_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "need_train.csv")
    test_df = pd.read_csv(data_dir / "need_test.csv")

    target = "fuel_poverty_risk"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    # Drop leakage features
    leak_cols = ["Gcons_recent_avg", "Econs_recent_avg"]
    drop_cols = [c for c in leak_cols if c in X_train.columns]
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    )

    # class imbalance handling
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
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

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    print("Training XGBoost (no leakage)...")
    pipe.fit(X_train, y_train)

    # Transform data for SHAP (work on the encoded feature matrix)
    X_train_enc = pipe.named_steps["prep"].transform(X_train)
    X_test_enc = pipe.named_steps["prep"].transform(X_test)

    feature_names = pipe.named_steps["prep"].get_feature_names_out()

    # Convert to dense for plotting convenience (safe: feature count is modest)
    X_test_dense = X_test_enc.toarray() if hasattr(X_test_enc, "toarray") else X_test_enc
    X_test_df = pd.DataFrame(X_test_dense, columns=feature_names)

    # Sample for speed
    sample_n = 2000
    X_shap = X_test_df.sample(n=min(sample_n, len(X_test_df)), random_state=42)

    print("Building SHAP TreeExplainer for XGBoost...")
    xgb_model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(X_shap)

    print("Saving SHAP summary (beeswarm)...")
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close()

    print("Saving SHAP global bar plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=200)
    plt.close()

    print("Saving SHAP waterfall example...")
    row = X_shap.iloc[[0]]
    exp = explainer(row)
    plt.figure()
    shap.plots.waterfall(exp[0], show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_waterfall_example.png", dpi=200)
    plt.close()

    print("\n✅ Saved SHAP outputs to:", out_dir)
    print(" - shap_summary.png")
    print(" - shap_global_bar.png")
    print(" - shap_waterfall_example.png")


if __name__ == "__main__":
    main()