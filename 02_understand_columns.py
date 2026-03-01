import pandas as pd

DATA_PATH = "data/raw/anon2025_50k.csv"

GROUPS = {
    "A_structure": [
        "PROP_TYPE", "PROP_AGE_BAND", "FLOOR_AREA_BAND",
        "COUNCIL_TAX_BAND", "CONSERVATORY_FLAG"
    ],
    "B_area": ["IMD_BAND_ENG", "IMD_BAND_WALES", "REGION"],
    "C_efficiency_upgrades": ["EPC", "LI_FLAG", "LI_DATE", "CWI_FLAG", "CWI_DATE", "PV_FLAG", "PV_DATE"],
    "D_heating": ["MAIN_HEAT_FUEL"],
}

def describe_group(df: pd.DataFrame, cols: list[str], title: str) -> None:
    print("\n" + "=" * 80)
    print(f"{title}  (columns: {len(cols)})")
    print("=" * 80)

    for c in cols:
        s = df[c]
        missing = s.isna().mean()
        print(f"\n[{c}]  dtype={s.dtype}  missing={missing:.3f}")

        # If it's categorical/text -> show top values
        if s.dtype == "object" or s.dtype.name == "string":
            print("Top values:")
            print(s.value_counts(dropna=False).head(10))
        else:
            # numeric -> show summary
            print("Summary:")
            print(s.describe())

def main():
    df = pd.read_csv(DATA_PATH)

    # only show the small groups first
    for name, cols in GROUPS.items():
        describe_group(df, cols, name)

if __name__ == "__main__":
    main()
