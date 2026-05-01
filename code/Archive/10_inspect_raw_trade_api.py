import pandas as pd
from pathlib import Path

PROJECT_DIR = Path("/")
RAW_DIR = PROJECT_DIR / "data_raw" / "trade_api"

print("Looking in:", RAW_DIR)

if not RAW_DIR.exists():
    print("trade_api folder not found.")
    raise SystemExit

files = sorted(list(RAW_DIR.glob("*.csv")))

print(f"\nNumber of CSV files found: {len(files)}")

if len(files) == 0:
    print("No CSV files found in data_raw/trade_api.")
    raise SystemExit

for file_path in files[:20]:
    print("\n" + "=" * 80)
    print(file_path.name)
    print("=" * 80)

    try:
        df = pd.read_csv(file_path, nrows=5)
        print("Columns:")
        print(list(df.columns))
        print("\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"Could not read file: {e}")