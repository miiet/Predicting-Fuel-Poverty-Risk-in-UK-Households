import pandas as pd

DATA_PATH = "data/raw/anon2025_50k.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Pick recent years only (last 5 years)
    gas_cols = [f"Gcons{y}" for y in range(2019, 2024)]
    elec_cols = [f"Econs{y}" for y in range(2019, 2024)]

    print("Gas columns:", gas_cols)
    print("Electricity columns:", elec_cols)

    print("\n=== GAS CONSUMPTION (recent years) ===")
    print(df[gas_cols].describe())

    print("\nMissing fraction per GAS column:")
    print(df[gas_cols].isna().mean())

    print("\n=== ELECTRICITY CONSUMPTION (recent years) ===")
    print(df[elec_cols].describe())

    print("\nMissing fraction per ELEC column:")
    print(df[elec_cols].isna().mean())

if __name__ == "__main__":
    main()