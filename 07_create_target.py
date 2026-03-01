import pandas as pd
import os

IN_PATH = "data/processed/need_clean_features_filled.csv"
OUT_PATH = "data/processed/need_final_with_target.csv"

def main():
    df = pd.read_csv(IN_PATH)

    # 1) Poor efficiency: EPC in D/E/F/G or No EPC
    poor_set = {"D", "E", "F", "G", "No EPC"}
    df["poor_efficiency"] = df["EPC"].isin(poor_set)

    # 2) High energy use: above 75th percentile
    gas_q75 = df["Gcons_recent_avg"].quantile(0.75)
    elec_q75 = df["Econs_recent_avg"].quantile(0.75)

    df["high_gas_use"] = df["Gcons_recent_avg"] > gas_q75
    df["high_elec_use"] = df["Econs_recent_avg"] > elec_q75

    df["high_energy_use"] = df["high_gas_use"] | df["high_elec_use"]

    # 3) High deprivation: IMD band 1 or 2
    df["high_deprivation"] = df["IMD_BAND"].isin([1, 2])

    # 4) Combine into target
    df["fuel_poverty_risk"] = (
        df["poor_efficiency"] &
        df["high_energy_use"] &
        df["high_deprivation"]
    ).astype(int)

    print("Target distribution:")
    print(df["fuel_poverty_risk"].value_counts(normalize=True))

    # Keep only features + target
    feature_cols = [
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

    df_final = df[feature_cols + ["fuel_poverty_risk"]].copy()

    os.makedirs("data/processed", exist_ok=True)
    df_final.to_csv(OUT_PATH, index=False)

    print("\n✅ Final dataset with target saved to:", OUT_PATH)
    print("Shape:", df_final.shape)

if __name__ == "__main__":
    main()