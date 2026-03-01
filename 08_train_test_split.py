import pandas as pd
from sklearn.model_selection import train_test_split
import os

IN_PATH = "data/processed/need_final_with_target.csv"
TRAIN_PATH = "data/processed/need_train.csv"
TEST_PATH = "data/processed/need_test.csv"

def main():
    df = pd.read_csv(IN_PATH)

    target = "fuel_poverty_risk"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # important: keep same class balance
    )

    train_df = X_train.copy()
    train_df[target] = y_train.values

    test_df = X_test.copy()
    test_df[target] = y_test.values

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("✅ Train shape:", train_df.shape)
    print("✅ Test shape:", test_df.shape)

    print("\nClass balance check:")
    print("Overall:\n", y.value_counts(normalize=True))
    print("Train:\n", train_df[target].value_counts(normalize=True))
    print("Test:\n", test_df[target].value_counts(normalize=True))

if __name__ == "__main__":
    main()