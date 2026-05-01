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

ASM_RAW_DIR = DATA_RAW / "asm"
ASM_RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# API setup
# -----------------------------
BASE_URL = "https://api.census.gov/data/timeseries/asm/area2017"

years = [2018, 2019, 2020]

# Variables:
# NAICS2017 = NAICS industry code
# INDLEVEL = industry aggregation level
# RCPTOT = sales, value of shipments, or revenue ($1,000)
# EMP = employment, useful as a check/control later
# PAYANN = annual payroll, useful as a check/control later

def fetch_asm_year(year):
    params = {
        "get": "NAICS2017,INDLEVEL,RCPTOT,EMP,PAYANN",
        "for": "us:*",
        "YEAR": str(year)
    }

    r = requests.get(BASE_URL, params=params, timeout=60)
    print("URL:", r.url)
    print("Status:", r.status_code)

    if r.status_code != 200:
        print(r.text[:1000])
        r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["query_year"] = year
    return df

# -----------------------------
# Download loop
# -----------------------------
all_years = []

for year in years:
    print(f"\nDownloading ASM year {year}...")
    df_year = fetch_asm_year(year)
    all_years.append(df_year)
    time.sleep(0.5)

asm = pd.concat(all_years, ignore_index=True)

# -----------------------------
# Clean
# -----------------------------
asm = asm.rename(columns={
    "NAICS2017": "naics2017",
    "INDLEVEL": "indlevel",
    "RCPTOT": "shipments_revenue_1000",
    "EMP": "employment",
    "PAYANN": "annual_payroll_1000",
    "YEAR": "year"
})

for col in ["shipments_revenue_1000", "employment", "annual_payroll_1000"]:
    asm[col] = pd.to_numeric(asm[col], errors="coerce")

asm["year"] = pd.to_numeric(asm["year"], errors="coerce")
asm["naics2017"] = asm["naics2017"].astype(str).str.strip()
asm["indlevel"] = asm["indlevel"].astype(str).str.strip()

# Convert $1,000 units to dollars
asm["domestic_output"] = asm["shipments_revenue_1000"] * 1000

# -----------------------------
# Save
# -----------------------------
raw_out = ASM_RAW_DIR / "asm_output_2018_2020_raw.csv"
clean_out = DATA_CLEAN / "asm_output_2018_2020.csv"

asm.to_csv(raw_out, index=False)
asm.to_csv(clean_out, index=False)

print("\nSaved raw ASM file to:", raw_out)
print("Saved clean ASM file to:", clean_out)

# -----------------------------
# Inspect
# -----------------------------
print("\nShape:")
print(asm.shape)

print("\nColumns:")
print(list(asm.columns))

print("\nYears:")
print(sorted(asm["year"].dropna().unique()))

print("\nINDLEVEL counts:")
print(asm["indlevel"].value_counts(dropna=False).head(20))

print("\nNAICS examples:")
print(asm[["year", "naics2017", "indlevel", "shipments_revenue_1000", "domestic_output"]].head(30))

print("\nMissing values:")
print(asm.isna().sum())