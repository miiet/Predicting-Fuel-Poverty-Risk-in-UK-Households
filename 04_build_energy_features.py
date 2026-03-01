import pandas as pd

DATA_PATH = "data/raw/anon2025_50k.csv"
OUT_PATH = "data/processed/need_with_energy_features.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    gas_cols = [f"Gcons{y}" for y in range(2019, 2024)]
    elec_cols = [f"Econs{y}" for y in range(2019, 2024)]

    # Create averages (ignore NaN automatically)
    df["Gcons_recent_avg"] = df[gas_cols].mean(axis=1)
    df["Econs_recent_avg"] = df[elec_cols].mean(axis=1)

    print("Created features:")
    print(df[["Gcons_recent_avg", "Econs_recent_avg"]].describe())

    # Save new dataset
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\nSaved to:", OUT_PATH)

if __name__ == "__main__":
    main()