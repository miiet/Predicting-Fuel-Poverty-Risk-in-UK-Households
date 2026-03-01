import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    out_dir = project_root / "outputs" / "shap_catboost_no_leakage"
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
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    # imbalance handling
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
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

    # Sample for SHAP speed
    sample_n = 2000
    X_shap = X_test.sample(n=min(sample_n, len(X_test)), random_state=42).copy()

    pool = Pool(X_shap, cat_features=cat_idx)

    print("Computing CatBoost SHAP values...")
    # Returns array shape: (n_samples, n_features + 1)
    # last column is expected value (base value)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    base_values = shap_vals[:, -1]
    shap_values = shap_vals[:, :-1]

    # Build SHAP Explanation object (for nicer plots)
    exp = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_shap.values,
        feature_names=list(X_shap.columns),
    )

    print("Saving SHAP summary (beeswarm)...")
    plt.figure()
    shap.plots.beeswarm(exp, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close()

    print("Saving SHAP global bar plot...")
    plt.figure()
    shap.plots.bar(exp, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=200)
    plt.close()

    print("Saving SHAP waterfall example...")
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