import pandas as pd

# Path to raw dataset
DATA_PATH = "data/raw/anon2025_50k.csv"  # change name if your file is different

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\nShape of dataset:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

if __name__ == "__main__":
    main()