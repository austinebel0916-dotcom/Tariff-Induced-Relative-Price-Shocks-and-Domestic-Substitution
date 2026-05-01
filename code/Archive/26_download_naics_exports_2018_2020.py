import requests
import pandas as pd
from pathlib import Path
import time

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("../..")
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_CLEAN = PROJECT_ROOT / "data_clean"

NAICS_RAW_DIR = DATA_RAW / "naics_trade"
NAICS_RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# API setup
# -----------------------------
BASE_URL = "https://api.census.gov/data/timeseries/intltrade/exports/naics"

years = [2018, 2019, 2020]

def fetch_naics_exports_year(year):
    params = {
        "get": "NAICS,NAICS_SDESC,CTY_CODE,CTY_NAME,ALL_VAL_YR,YEAR,MONTH,COMM_LVL",
        "COMM_LVL": "NA4",
        "time": f"{year}-12"
    }

    r = requests.get(BASE_URL, params=params, timeout=180)

    print("URL:", r.url)
    print("Status:", r.status_code)

    if r.status_code != 200:
        print(r.text[:1000])
        r.raise_for_status()

    data = r.json()

    if len(data) <= 1:
        return pd.DataFrame(columns=[
            "NAICS", "NAICS_SDESC", "CTY_CODE", "CTY_NAME",
            "ALL_VAL_YR", "YEAR", "MONTH", "COMM_LVL", "time"
        ])

    df = pd.DataFrame(data[1:], columns=data[0])
    df["query_year"] = year
    return df

# -----------------------------
# Download loop
# -----------------------------
all_years = []

for year in years:
    print(f"\nDownloading NAICS exports year {year}...")
    df_year = fetch_naics_exports_year(year)
    all_years.append(df_year)
    time.sleep(0.5)

exports = pd.concat(all_years, ignore_index=True)

# Remove duplicate columns just in case
exports = exports.loc[:, ~exports.columns.duplicated()]

# -----------------------------
# Clean
# -----------------------------
exports = exports.rename(columns={
    "NAICS": "naics",
    "NAICS_SDESC": "naics_desc",
    "CTY_CODE": "cty_code",
    "CTY_NAME": "cty_name",
    "ALL_VAL_YR": "exports_total",
    "query_year": "year"
})

exports = exports.loc[:, ~exports.columns.duplicated()]

exports["exports_total"] = pd.to_numeric(exports["exports_total"], errors="coerce").fillna(0)
exports["year"] = pd.to_numeric(exports["year"], errors="coerce")
exports["naics"] = exports["naics"].astype(str).str.strip()
exports["cty_code"] = exports["cty_code"].astype(str).str.strip()
exports["cty_name"] = exports["cty_name"].astype(str).str.strip()
exports["naics_desc"] = exports["naics_desc"].astype(str).str.strip()

# -----------------------------
# Build valid numeric NAICS4 manufacturing codes
# -----------------------------
exports["naics4"] = exports["naics"].str[:4]

# Keep numeric four-digit manufacturing industries only:
# NAICS manufacturing begins with 31, 32, or 33.
exports_naics4 = exports[
    exports["naics4"].str.match(r"^\d{4}$", na=False) &
    exports["naics4"].str.startswith(("31", "32", "33"))
].copy()

exports_naics4 = (
    exports_naics4
    .groupby(["naics4", "year"], as_index=False)
    .agg(exports_total=("exports_total", "sum"))
)

# -----------------------------
# Save
# -----------------------------
raw_out = NAICS_RAW_DIR / "naics_exports_2018_2020_raw.csv"
clean_out = DATA_CLEAN / "naics4_exports_2018_2020.csv"

exports.to_csv(raw_out, index=False)
exports_naics4.to_csv(clean_out, index=False)

print("\nSaved raw NAICS exports to:", raw_out)
print("Saved clean NAICS4 exports to:", clean_out)

# -----------------------------
# Inspect
# -----------------------------
print("\nRaw shape:")
print(exports.shape)

print("\nNAICS4 shape:")
print(exports_naics4.shape)

print("\nYears:")
print(sorted(exports_naics4["year"].dropna().unique()))

print("\nNAICS4 examples:")
print(exports_naics4.head(30))

print("\nTop NAICS4 by exports_total:")
print(
    exports_naics4
    .sort_values("exports_total", ascending=False)
    .head(20)
)

print("\nMissing values:")
print(exports_naics4.isna().sum())