from pathlib import Path
import pandas as pd
import requests
import time

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("../..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

output_path = DATA_CLEAN / "exports_hts8_country_2015_2020.csv"

# -----------------------------
# API Setup
# -----------------------------
BASE_URL = "https://api.census.gov/data/timeseries/intltrade/exports/hs"

YEARS = list(range(2015, 2021))

# -----------------------------
# Loop over years
# -----------------------------
dfs = []

for year in YEARS:
    print(f"Downloading exports for {year}...")

    params = {
        "get": "ALL_VAL_MO,CTY_CODE,CTY_NAME,E_COMMODITY",
        "YEAR": str(year),
        "MONTH": "12",
        "COMM_LVL": "HS10"
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"Failed for {year}: {response.status_code}")
        print(response.text[:500])
        continue

    data = response.json()

    if len(data) <= 1:
        print(f"No data returned for {year}")
        continue

    df = pd.DataFrame(data[1:], columns=data[0])

    df = df.rename(columns={
        "E_COMMODITY": "hts_code",
        "ALL_VAL_MO": "export_value",
        "CTY_CODE": "cty_code",
        "CTY_NAME": "cty_name"
    })

    df["year"] = year
    df["export_value"] = pd.to_numeric(df["export_value"], errors="coerce")

    dfs.append(df)

    time.sleep(1)

# -----------------------------
# Combine
# -----------------------------
if not dfs:
    raise RuntimeError("No export data downloaded. Check API parameters.")

exports = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Save
# -----------------------------
exports.to_csv(output_path, index=False)

# -----------------------------
# Diagnostics
# -----------------------------
print("Saved exports to:", output_path)
print("\nPreview:")
print(exports.head())
print("\nRows:", len(exports))
print("Years:", sorted(exports["year"].unique()))
print("Unique HTS codes:", exports["hts_code"].nunique())