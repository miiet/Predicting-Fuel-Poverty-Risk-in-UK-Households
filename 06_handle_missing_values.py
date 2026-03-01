import pandas as pd
import os

IN_PATH = "data/processed/need_clean_features.csv"
OUT_PATH = "data/processed/need_clean_features_filled.csv"

def main():
    df = pd.read_csv(IN_PATH)

    print("Before filling missing values:")
    print(df.isna().mean().sort_values(ascending=False))

    # 1) Fill CONSERVATORY_FLAG with 0
    df["CONSERVATORY_FLAG"] = df["CONSERVATORY_FLAG"].fillna(0)

    # 2) Fill Gcons_recent_avg with 0 (means no gas / not using gas)
    df["Gcons_recent_avg"] = df["Gcons_recent_avg"].fillna(0)

    # 3) Fill Econs_recent_avg with median
    elec_median = df["Econs_recent_avg"].median()
    df["Econs_recent_avg"] = df["Econs_recent_avg"].fillna(elec_median)

    print("\nAfter filling missing values:")
    print(df.isna().mean().sort_values(ascending=False))

    # Save filled dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\n✅ Filled clean dataset saved to:", OUT_PATH)
    print("Shape:", df.shape)

if __name__ == "__main__":
    main()