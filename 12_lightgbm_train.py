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

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    categorical_features = ["PROP_TYPE", "COUNCIL_TAX_BAND", "REGION", "EPC"]

    numeric_features = [
        "PROP_AGE_BAND",
        "FLOOR_AREA_BAND",
        "CONSERVATORY_FLAG",
        "LI_FLAG",
        "CWI_FLAG",
        "PV_FLAG",
        "MAIN_HEAT_FUEL",
        "IMD_BAND",
        "Gcons_recent_avg",
        "Econs_recent_avg",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    print("Training LightGBM...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    print("\nROC-AUC:", roc_auc)

    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    main()