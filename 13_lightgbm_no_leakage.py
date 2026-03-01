import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TRAIN_PATH = "data/processed/need_train.csv"
TEST_PATH = "data/processed/need_test.csv"

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "fuel_poverty_risk"

    # Drop the "ingredients" used to define the target to avoid leakage:
    # - EPC (used in poor_efficiency)
    # - IMD_BAND (used in deprivation)
    # - Energy usage averages (used in high_energy_use)
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
        ]
    )

    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        force_row_wise=True,  # avoids the row-wise overhead message
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    print("Training LightGBM (no leakage features)...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.6).astype(int)  # use your chosen screening threshold

    roc_auc = roc_auc_score(y_test, y_proba)
    print("\nROC-AUC:", roc_auc)

    print("\nClassification Report (threshold=0.6):")
    print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    main()