from pathlib import Path
import time
import requests
import pandas as pd

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
DATA_RAW = PROJECT_ROOT / "data_raw"
TRADE_RAW_DIR = DATA_RAW / "trade_api"
TRADE_RAW_DIR.mkdir(parents=True, exist_ok=True)

tariff_path = DATA_CLEAN / "list1_tariffs.csv"
tariffs = pd.read_csv(tariff_path, dtype={"hts_code": str})

tariffed_hts8 = set(tariffs["hts_code"].str.replace(".", "", regex=False).str.zfill(8))
prefixes = sorted({code[:2] for code in tariffed_hts8})

years = [2015, 2016, 2017, 2018, 2019, 2020]

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"

def fetch_trade_chunk(year: int, prefix: str) -> pd.DataFrame:
    """
    Pull December year-to-date import values and quantities for all HS10 products
    beginning with a given 2-digit prefix.

    GEN_VAL_YR = full-year general import value
    GEN_QY1_YR = full-year general import quantity 1
    GEN_QY2_YR = full-year general import quantity 2
    UNIT_QY1   = unit for quantity 1
    UNIT_QY2   = unit for quantity 2
    """
    params = {
        "get": "I_COMMODITY,CTY_CODE,CTY_NAME,GEN_VAL_YR,GEN_QY1_YR,GEN_QY2_YR,UNIT_QY1,UNIT_QY2",
        "YEAR": str(year),
        "MONTH": "12",
        "COMM_LVL": "HS10",
        "I_COMMODITY": f"{prefix}*"
    }

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()

    if len(data) <= 1:
        return pd.DataFrame(columns=[
            "I_COMMODITY", "CTY_CODE", "CTY_NAME",
            "GEN_VAL_YR", "GEN_QY1_YR", "GEN_QY2_YR",
            "UNIT_QY1", "UNIT_QY2", "YEAR", "MONTH"
        ])

    df = pd.DataFrame(data[1:], columns=data[0])
    return df

all_chunks = []

for year in years:
    for prefix in prefixes:
        print(f"Downloading year={year}, prefix={prefix}...")
        chunk = fetch_trade_chunk(year, prefix)
        chunk["query_year"] = year
        chunk["query_prefix"] = prefix
        all_chunks.append(chunk)
        time.sleep(0.5)

raw_trade = pd.concat(all_chunks, ignore_index=True)

raw_trade = raw_trade.rename(columns={
    "I_COMMODITY": "hts10",
    "CTY_CODE": "cty_code",
    "CTY_NAME": "cty_name",
    "GEN_VAL_YR": "import_value",
    "GEN_QY1_YR": "import_qty1",
    "GEN_QY2_YR": "import_qty2",
    "UNIT_QY1": "unit_qty1",
    "UNIT_QY2": "unit_qty2"
})

print("Columns after rename:", raw_trade.columns.tolist())

raw_trade = raw_trade.loc[:, ~raw_trade.columns.duplicated()]

raw_trade["hts10"] = pd.Series(raw_trade["hts10"]).astype(str).str.replace(".", "", regex=False).str.strip()
raw_trade["cty_code"] = pd.Series(raw_trade["cty_code"]).astype(str).str.strip()
raw_trade["cty_name"] = pd.Series(raw_trade["cty_name"]).astype(str).str.strip()
raw_trade["unit_qty1"] = pd.Series(raw_trade["unit_qty1"]).astype(str).str.strip()
raw_trade["unit_qty2"] = pd.Series(raw_trade["unit_qty2"]).astype(str).str.strip()

for col in ["import_value", "import_qty1", "import_qty2"]:
    raw_trade[col] = pd.to_numeric(raw_trade[col], errors="coerce").fillna(0)

raw_trade["hts_code"] = raw_trade["hts10"].str[:8]

trade_filtered = raw_trade[raw_trade["hts_code"].isin(tariffed_hts8)].copy()

trade_hts8 = (
    trade_filtered
    .groupby(["hts_code", "cty_code", "cty_name", "query_year"], as_index=False)
    .agg(
        import_value=("import_value", "sum"),
        import_qty1=("import_qty1", "sum"),
        import_qty2=("import_qty2", "sum"),
        unit_qty1=("unit_qty1", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
        unit_qty2=("unit_qty2", lambda x: ",".join(sorted(set(x.dropna().astype(str)))))
    )
    .rename(columns={"query_year": "year"})
)

raw_out = TRADE_RAW_DIR / "imports_hs10_2015_2020_raw_with_quantity.csv"
filtered_out = DATA_CLEAN / "imports_hts8_country_2015_2020_with_quantity.csv"

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

print("\nQuantity checks:")
print("Rows with import_qty1 > 0:", (trade_hts8["import_qty1"] > 0).sum())
print("Rows with import_qty2 > 0:", (trade_hts8["import_qty2"] > 0).sum())
print("Rows with import_value > 0:", (trade_hts8["import_value"] > 0).sum())

print("\nMost common quantity 1 units:")
print(trade_hts8["unit_qty1"].value_counts().head(20))

print("\nMost common quantity 2 units:")
print(trade_hts8["unit_qty2"].value_counts().head(20))