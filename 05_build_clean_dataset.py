import pandas as pd
import os

IN_PATH = "data/processed/need_with_energy_features.csv"
OUT_PATH = "data/processed/need_clean_features.csv"

def main():
    df = pd.read_csv(IN_PATH)

    # Merge IMD columns into one
    df["IMD_BAND"] = df["IMD_BAND_ENG"].combine_first(df["IMD_BAND_WALES"])

    # Columns we want to keep
    keep_cols = [
        "PROP_TYPE",
        "PROP_AGE_BAND",
        "FLOOR_AREA_BAND",
        "COUNCIL_TAX_BAND",
        "CONSERVATORY_FLAG",
        "REGION",
        "EPC",
        "LI_FLAG",
        "CWI_FLAG",
        "PV_FLAG",
        "MAIN_HEAT_FUEL",
        "IMD_BAND",
        "Gcons_recent_avg",
        "Econs_recent_avg",
    ]

    df_clean = df[keep_cols].copy()

    # Save clean dataset
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)

    print("✅ Clean dataset saved to:", OUT_PATH)
    print("Shape:", df_clean.shape)
    print("\nMissing values fraction:")
    print(df_clean.isna().mean().sort_values(ascending=False))

if __name__ == "__main__":
    main()