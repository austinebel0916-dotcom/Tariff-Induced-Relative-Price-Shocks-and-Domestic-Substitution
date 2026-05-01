import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path("/")
DATA_CLEAN = PROJECT_DIR / "data_clean"

files_to_check = [
    "imports_hts8_country_2015_2020.csv",
    "exports_hts8_country_2015_2020.csv",
    "industry_trade_panel.csv",
    "panel_dataset.csv",
    "import_shares_pre_2015_2017_full.csv",
    "list1_tariffs.csv",
]

for filename in files_to_check:
    file_path = DATA_CLEAN / filename

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    if not file_path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(file_path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values by column:")
    print(df.isna().sum())