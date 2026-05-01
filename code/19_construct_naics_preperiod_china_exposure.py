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
BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/naicsimport"
years = [2015, 2016, 2017]

def fetch_one_naics_year(naics4, year, geo_name, geo_for):
    params = {
        "get": "NAME,NAICS,NAICS_LABEL,CON_VAL_YR,GEN_VAL_YR,YEAR,MONTH",
        "NAICS": naics4,
        "for": geo_for,
        "time": f"{year}-12"
    }

    r = requests.get(BASE_URL, params=params, timeout=120)

    if r.status_code != 200:
        print("\nFAILED")
        print("Geo:", geo_name)
        print("NAICS:", naics4, "year:", year)
        print("URL:", r.url)
        print("Status:", r.status_code)
        print(r.text[:500])
        return pd.DataFrame()

    data = r.json()

    if len(data) <= 1:
        return pd.DataFrame()

    df = pd.DataFrame(data[1:], columns=data[0])

    # Drop duplicate columns returned by API, if any
    df = df.loc[:, ~df.columns.duplicated()]

    df["query_naics4"] = naics4
    df["query_year"] = year
    df["geo_name_query"] = geo_name

    # Final safety check: make all column names unique
    df = df.loc[:, ~df.columns.duplicated()]

    return df

# -----------------------------
# Download loop
# -----------------------------
geos = [
    ("world_total", "world:*"),
    ("china", "usitc standard countries and areas:5700"),
]

chunks = []

for geo_name, geo_for in geos:
    for year in years:
        for i, naics4 in enumerate(naics4_codes, start=1):
            print(f"Downloading geo={geo_name}, year={year}, NAICS4={naics4} ({i}/{len(naics4_codes)})...")
            df_chunk = fetch_one_naics_year(naics4, year, geo_name, geo_for)

            if not df_chunk.empty:
                chunks.append(df_chunk)

            time.sleep(0.2)

if len(chunks) == 0:
    print("No import data downloaded.")
    raise SystemExit

# Make sure every chunk has unique column names before concatenation
clean_chunks = []
for chunk in chunks:
    chunk = chunk.loc[:, ~chunk.columns.duplicated()].copy()
    clean_chunks.append(chunk)

chunks = clean_chunks

imports = pd.concat(chunks, ignore_index=True)
imports = imports.loc[:, ~imports.columns.duplicated()]

# -----------------------------
# Clean
# -----------------------------
imports = imports.rename(columns={
    "NAME": "geo_name",
    "NAICS": "naics",
    "NAICS_LABEL": "naics_desc",
    "CON_VAL_YR": "imports_consumption",
    "GEN_VAL_YR": "imports_general",
    "query_year": "year",
    "query_naics4": "naics4"
})

imports = imports.loc[:, ~imports.columns.duplicated()]

imports["imports_consumption"] = pd.to_numeric(imports["imports_consumption"], errors="coerce").fillna(0)
imports["imports_general"] = pd.to_numeric(imports["imports_general"], errors="coerce").fillna(0)
imports["year"] = pd.to_numeric(imports["year"], errors="coerce")

imports["naics"] = imports["naics"].astype(str).str.strip()
imports["naics4"] = imports["naics4"].astype(str).str.strip()
imports["geo_name"] = imports["geo_name"].astype(str).str.strip()
imports["geo_name_query"] = imports["geo_name_query"].astype(str).str.strip()

# -----------------------------
# Save raw country/geo-level data
# -----------------------------
raw_out = NAICS_RAW_DIR / "naics4_imports_world_china_2015_2017_raw.csv"
imports.to_csv(raw_out, index=False)

print("\nSaved raw NAICS4 world/China imports to:", raw_out)

# -----------------------------
# Build pre-period China share
# -----------------------------
total_imports = imports[imports["geo_name_query"] == "world_total"].copy()
china_imports = imports[imports["geo_name_query"] == "china"].copy()

total_pre = (
    total_imports
    .groupby("naics4", as_index=False)
    .agg(total_imports_pre=("imports_consumption", "sum"))
)

china_pre = (
    china_imports
    .groupby("naics4", as_index=False)
    .agg(china_imports_pre=("imports_consumption", "sum"))
)

pre_shares = total_pre.merge(
    china_pre,
    on="naics4",
    how="left"
)

pre_shares["china_imports_pre"] = pre_shares["china_imports_pre"].fillna(0)

pre_shares["china_share_pre_naics4"] = (
    pre_shares["china_imports_pre"] / pre_shares["total_imports_pre"]
)

# -----------------------------
# Save clean pre-period exposure
# -----------------------------
clean_out = DATA_CLEAN / "naics4_china_share_pre_2015_2017.csv"
pre_shares.to_csv(clean_out, index=False)

print("Saved clean NAICS4 pre-period China shares to:", clean_out)

# -----------------------------
# Inspect
# -----------------------------
print("\nRaw imports shape:")
print(imports.shape)

print("\nRows by geo query:")
print(imports["geo_name_query"].value_counts())

print("\nTotal pre-period shape:")
print(total_pre.shape)

print("\nChina pre-period shape:")
print(china_pre.shape)

print("\nPre-period share shape:")
print(pre_shares.shape)

print("\nChina share summary:")
print(pre_shares["china_share_pre_naics4"].describe())

print("\nTop China-exposed NAICS4 industries:")
print(
    pre_shares
    .sort_values("china_share_pre_naics4", ascending=False)
    .head(20)
)

print("\nMissing values:")
print(pre_shares.isna().sum())