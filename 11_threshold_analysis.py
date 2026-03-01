import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

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
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", roc_auc)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print("\n" + "="*60)
        print(f"Threshold = {thresh}")
        y_pred = (y_proba >= thresh).astype(int)
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    main()