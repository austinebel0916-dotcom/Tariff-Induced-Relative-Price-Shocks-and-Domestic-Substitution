import requests
import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path("..")
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_CLEAN = PROJECT_ROOT / "data_clean"

NAICS_RAW_DIR = DATA_RAW / "naics_trade"
NAICS_RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/naicsimport"

years = [2018, 2019, 2020]

def fetch_naics_imports_year(year):
    params = {
        "get": "NAME,NAICS,NAICS_LABEL,CTY_CODE,CON_VAL_YR,GEN_VAL_YR,CAL_DUT_YR,DUT_VAL_YR,YEAR,MONTH",
        "for": "world:*",
        "time": f"{year}-12"
    }

    r = requests.get(BASE_URL, params=params, timeout=60)

    print("URL:", r.url)
    print("Status:", r.status_code)

    if r.status_code != 200:
        print(r.text[:1000])
        r.raise_for_status()

    data = r.json()

    if len(data) <= 1:
        return pd.DataFrame(columns=[
            "NAICS", "NAICS_SDESC", "CTY_CODE",
            "CON_VAL_YR", "GEN_VAL_YR", "CAL_DUT_YR", "DUT_VAL_YR",
            "YEAR", "MONTH"
        ])

    df = pd.DataFrame(data[1:], columns=data[0])
    df["query_year"] = year
    return df

all_years = []

for year in years:
    print(f"\nDownloading NAICS imports year {year}...")
    df_year = fetch_naics_imports_year(year)
    all_years.append(df_year)
    time.sleep(0.5)

imports = pd.concat(all_years, ignore_index=True)

imports = imports.loc[:, ~imports.columns.duplicated()]

imports = imports.rename(columns={
    "NAME": "geo_name",
    "NAICS": "naics",
    "NAICS_LABEL": "naics_desc",
    "CTY_CODE": "cty_code",
    "CON_VAL_YR": "imports_consumption",
    "GEN_VAL_YR": "imports_general",
    "CAL_DUT_YR": "calculated_duty",
    "DUT_VAL_YR": "dutiable_value",
    "query_year": "year"
})

imports = imports.loc[:, ~imports.columns.duplicated()]

for col in ["imports_consumption", "imports_general", "calculated_duty", "dutiable_value"]:
    imports[col] = pd.to_numeric(imports[col], errors="coerce").fillna(0)

imports["year"] = pd.to_numeric(imports["year"], errors="coerce")
imports["naics"] = imports["naics"].astype(str).str.strip()
imports["cty_code"] = imports["cty_code"].astype(str).str.strip()
imports["naics_desc"] = imports["naics_desc"].astype(str).str.strip()

imports["naics4"] = imports["naics"].str[:4]

imports_naics4 = (
    imports
    .groupby(["naics4", "year"], as_index=False)
    .agg(
        imports_consumption=("imports_consumption", "sum"),
        imports_general=("imports_general", "sum"),
        calculated_duty=("calculated_duty", "sum"),
        dutiable_value=("dutiable_value", "sum")
    )
)

raw_out = NAICS_RAW_DIR / "naics_imports_2018_2020_raw.csv"
clean_out = DATA_CLEAN / "naics4_imports_2018_2020.csv"

imports.to_csv(raw_out, index=False)
imports_naics4.to_csv(clean_out, index=False)

print("\nSaved raw NAICS imports to:", raw_out)
print("Saved clean NAICS4 imports to:", clean_out)

# -----------------------------
# Inspect
# -----------------------------
print("\nRaw shape:")
print(imports.shape)

print("\nNAICS4 shape:")
print(imports_naics4.shape)

print("\nYears:")
print(sorted(imports_naics4["year"].dropna().unique()))

print("\nNAICS4 examples:")
print(imports_naics4.head(30))

print("\nTop NAICS4 by imports_consumption:")
print(
    imports_naics4
    .sort_values("imports_consumption", ascending=False)
    .head(20)
)

print("\nMissing values:")
print(imports_naics4.isna().sum())