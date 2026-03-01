import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

TRAIN_PATH = "data/processed/need_train.csv"
TEST_PATH = "data/processed/need_test.csv"

OUT_DIR = "outputs/shap_no_leakage"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "fuel_poverty_risk"
    leak_cols = ["EPC", "IMD_BAND", "Gcons_recent_avg", "Econs_recent_avg"]

    X_train = train_df.drop(columns=[target] + leak_cols)
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target] + leak_cols)
    y_test = test_df[target]

    categorical_features = ["PROP_TYPE", "COUNCIL_TAX_BAND", "REGION"]
    numeric_features = [
        "PROP_AGE_BAND",
        "FLOOR_AREA_BAND",
        "CONSERVATORY_FLAG",
        "LI_FLAG",
        "CWI_FLAG",
        "PV_FLAG",
        "MAIN_HEAT_FUEL",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    print("Fitting preprocessor...")
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # Get feature names after encoding
    feature_names = preprocessor.get_feature_names_out()

    # Convert to dense DataFrames for SHAP convenience (safe here: only ~31 features after encoding)
    X_train_enc_df = pd.DataFrame(
        X_train_enc.toarray() if hasattr(X_train_enc, "toarray") else X_train_enc,
        columns=feature_names
    )
    X_test_enc_df = pd.DataFrame(
        X_test_enc.toarray() if hasattr(X_test_enc, "toarray") else X_test_enc,
        columns=feature_names
    )

    print("Training LightGBM (no leakage) for SHAP...")
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        force_row_wise=True,
    )
    model.fit(X_train_enc_df, y_train)

    print("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed (SHAP can be slow on 10k+ rows)
    sample_n = 2000
    X_shap = X_test_enc_df.sample(n=min(sample_n, len(X_test_enc_df)), random_state=42)

    shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    print("Saving SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_pos, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=200)
    plt.close()

    print("Saving SHAP bar (global importance) plot...")
    plt.figure()
    shap.summary_plot(shap_values_pos, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_global_bar.png"), dpi=200)
    plt.close()

    # Save one local explanation (waterfall) for a single case
    print("Saving SHAP waterfall plot for one example...")
    idx = X_shap.index[0]
    row = X_shap.loc[idx:idx]

    # shap.Explanation object (preferred)
    try:
        exp = explainer(row)
        exp_pos = exp[..., 1] if exp.values.ndim == 3 else exp  # handle binary outputs
        plt.figure()
        shap.plots.waterfall(exp_pos[0], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_waterfall_example.png"), dpi=200)
        plt.close()
    except Exception:
        # Fallback: older shap versions
        base = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        vals = explainer.shap_values(row)
        vals_pos = vals[1] if isinstance(vals, list) else vals
        shap.force_plot(base, vals_pos[0], row.iloc[0, :], matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "shap_force_example.png"), dpi=200)
        plt.close()

    print("\n✅ SHAP outputs saved to:", OUT_DIR)
    print("Files:")
    print(" - shap_summary.png")
    print(" - shap_global_bar.png")
    print(" - shap_waterfall_example.png (or shap_force_example.png)")

if __name__ == "__main__":
    main()