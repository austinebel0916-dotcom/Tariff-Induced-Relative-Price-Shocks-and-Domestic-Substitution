from pathlib import Path
import time
import requests
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("../..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
DATA_RAW = PROJECT_ROOT / "data_raw"
TRADE_RAW_DIR = DATA_RAW / "trade_api"
TRADE_RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Inputs
# -----------------------------
tariff_path = DATA_CLEAN / "list1_tariffs.csv"
tariffs = pd.read_csv(tariff_path, dtype={"hts_code": str})

# Keep unique 8-digit tariffed HTS codes
tariffed_hts8 = set(tariffs["hts_code"].str.replace(".", "", regex=False).str.zfill(8))

# We will query by 2-digit prefix to reduce the number of calls
prefixes = sorted({code[:2] for code in tariffed_hts8})

# Pre-period years for annual import shares
years = [2015, 2016, 2017]

# Census International Trade API endpoint for imports HS
BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"

# -----------------------------
# Helper function
# -----------------------------
def fetch_trade_chunk(year: int, prefix: str) -> pd.DataFrame:
    """
    Pull December year-to-date import values for all HS10 products beginning
    with a given 2-digit prefix. We use GEN_VAL_YR from December as the full-year
    general imports value.
    """
    params = {
        "get": "I_COMMODITY,CTY_CODE,CTY_NAME,GEN_VAL_YR",
        "YEAR": str(year),
        "MONTH": "12",
        "COMM_LVL": "HS10",
        "I_COMMODITY": f"{prefix}*"
    }

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    if len(data) <= 1:
        return pd.DataFrame(columns=["I_COMMODITY", "CTY_CODE", "CTY_NAME", "GEN_VAL_YR", "YEAR", "MONTH"])

    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# -----------------------------
# Download loop
# -----------------------------
all_chunks = []

for year in years:
    for prefix in prefixes:
        print(f"Downloading year={year}, prefix={prefix}...")
        chunk = fetch_trade_chunk(year, prefix)
        chunk["query_year"] = year
        chunk["query_prefix"] = prefix
        all_chunks.append(chunk)

        # polite pause
        time.sleep(0.5)

raw_trade = pd.concat(all_chunks, ignore_index=True)

# -----------------------------
# Clean and standardize
# -----------------------------
raw_trade = raw_trade.rename(columns={
    "I_COMMODITY": "hts10",
    "CTY_CODE": "cty_code",
    "CTY_NAME": "cty_name",
    "GEN_VAL_YR": "import_value"
})
print("Columns after rename:", raw_trade.columns.tolist())

# If duplicate column names came back from the API, keep only first occurrence
raw_trade = raw_trade.loc[:, ~raw_trade.columns.duplicated()]

# Force Series selection safely
raw_trade["hts10"] = pd.Series(raw_trade["hts10"]).astype(str).str.replace(".", "", regex=False).str.strip()
raw_trade["cty_code"] = pd.Series(raw_trade["cty_code"]).astype(str).str.strip()
raw_trade["cty_name"] = pd.Series(raw_trade["cty_name"]).astype(str).str.strip()

# Import value to numeric
raw_trade["import_value"] = pd.to_numeric(raw_trade["import_value"], errors="coerce").fillna(0)

# Build 8-digit code from HS10
raw_trade["hts_code"] = raw_trade["hts10"].str[:8]

# Keep only tariffed 8-digit products from your List 1
trade_filtered = raw_trade[raw_trade["hts_code"].isin(tariffed_hts8)].copy()

# Aggregate HS10 -> HTS8 within country-year
trade_hts8 = (
    trade_filtered
    .groupby(["hts_code", "cty_code", "cty_name", "query_year"], as_index=False)["import_value"]
    .sum()
    .rename(columns={"query_year": "year"})
)

# -----------------------------
# Save outputs
# -----------------------------
raw_out = TRADE_RAW_DIR / "imports_hs10_2015_2017_raw.csv"
filtered_out = DATA_CLEAN / "imports_hts8_country_2015_2017.csv"

raw_trade.to_csv(raw_out, index=False)
trade_hts8.to_csv(filtered_out, index=False)

print("\nSaved raw trade file to:", raw_out)
print("Saved filtered HTS8 country-year trade file to:", filtered_out)

print("\nFiltered file preview:")
print(trade_hts8.head())

print("\nRows in filtered file:", len(trade_hts8))
print("Unique HTS8 codes:", trade_hts8["hts_code"].nunique())
print("Unique countries:", trade_hts8["cty_code"].nunique())
print("Years:", sorted(trade_hts8["year"].unique()))