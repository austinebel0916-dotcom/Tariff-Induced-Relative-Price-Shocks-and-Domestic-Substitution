import requests
import pandas as pd
from pathlib import Path
import time

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("..")
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_CLEAN = PROJECT_ROOT / "data_clean"

NAICS_RAW_DIR = DATA_RAW / "naics_trade"
NAICS_RAW_DIR.mkdir(parents=True, exist_ok=True)

asm_path = DATA_CLEAN / "asm_output_2018_2020.csv"

# -----------------------------
# Get NAICS4 manufacturing codes from ASM
# -----------------------------
asm = pd.read_csv(asm_path, dtype={"naics2017": str, "indlevel": str})

naics4_codes = (
    asm[asm["indlevel"] == "4"]["naics2017"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print("Number of NAICS4 codes from ASM:", len(naics4_codes))
print("First 20 NAICS4 codes:", naics4_codes[:20])

# -----------------------------
# API setup
# -----------------------------
BASE_URL = "https://api.census.gov/data/timeseries/intltrade/exports/naics"
years = [2018, 2019, 2020]

def fetch_one_naics_year(naics4, year):
    params = {
        "get": "NAICS,NAICS_SDESC,CTY_CODE,CTY_NAME,ALL_VAL_YR,YEAR,MONTH,COMM_LVL",
        "NAICS": naics4,
        "time": f"{year}-12"
    }

    r = requests.get(BASE_URL, params=params, timeout=120)

    if r.status_code != 200:
        print("\nFAILED")
        print("NAICS:", naics4, "year:", year)
        print("URL:", r.url)
        print("Status:", r.status_code)
        print(r.text[:500])
        return pd.DataFrame()

    data = r.json()

    if len(data) <= 1:
        return pd.DataFrame()

    df = pd.DataFrame(data[1:], columns=data[0])
    df["query_naics4"] = naics4
    df["query_year"] = year
    return df

# -----------------------------
# Download loop
# -----------------------------
chunks = []

for year in years:
    for i, naics4 in enumerate(naics4_codes, start=1):
        print(f"Downloading year={year}, NAICS4={naics4} ({i}/{len(naics4_codes)})...")
        df_chunk = fetch_one_naics_year(naics4, year)

        if not df_chunk.empty:
            chunks.append(df_chunk)

        time.sleep(0.2)

if len(chunks) == 0:
    print("No export data downloaded.")
    raise SystemExit

exports = pd.concat(chunks, ignore_index=True)

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
    "query_year": "year",
    "query_naics4": "naics4"
})

exports = exports.loc[:, ~exports.columns.duplicated()]

exports["exports_total"] = pd.to_numeric(exports["exports_total"], errors="coerce").fillna(0)
exports["year"] = pd.to_numeric(exports["year"], errors="coerce")
exports["naics"] = exports["naics"].astype(str).str.strip()
exports["naics4"] = exports["naics4"].astype(str).str.strip()
exports["cty_code"] = exports["cty_code"].astype(str).str.strip()
exports["cty_name"] = exports["cty_name"].astype(str).str.strip()
exports["naics_desc"] = exports["naics_desc"].astype(str).str.strip()

# -----------------------------
# Aggregate across countries/destinations
# -----------------------------
exports_naics4 = (
    exports
    .groupby(["naics4", "year"], as_index=False)
    .agg(exports_total=("exports_total", "sum"))
)

# -----------------------------
# Save
# -----------------------------
raw_out = NAICS_RAW_DIR / "naics4_exports_2018_2020_raw_loop.csv"
clean_out = DATA_CLEAN / "naics4_exports_2018_2020.csv"

exports.to_csv(raw_out, index=False)
exports_naics4.to_csv(clean_out, index=False)

print("\nSaved raw NAICS4 exports to:", raw_out)
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