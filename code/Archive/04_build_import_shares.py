from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("../..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

input_path = DATA_CLEAN / "imports_hts8_country_2015_2017.csv"
output_path = DATA_CLEAN / "import_shares_pre_2015_2017.csv"

# -----------------------------
# Load trade data
# -----------------------------
df = pd.read_csv(input_path, dtype={"hts_code": str, "cty_code": str})

# -----------------------------
# Drop aggregates / non-country entries
# -----------------------------
df["cty_name"] = df["cty_name"].astype(str).str.strip()
df["cty_code"] = df["cty_code"].astype(str).str.strip()

# obvious non-country codes
df = df[df["cty_code"] != "-"].copy()
df = df[df["cty_code"].str.fullmatch(r"\d+")].copy()

# drop known aggregate / regional names by pattern
aggregate_patterns = [
    "TOTAL",
    "EUROPEAN UNION",
    "EURO AREA",
    "PACIFIC RIM",
    "USMCA",
    "NAFTA",
    "LATIN AMERICAN",
    "OECD",
    "OPEC",
    "ASEAN",
    "APEC",
    "EFTA",
    "NATO",
    "CACM",
    "CARICOM",
    "ANDean",
    "MERCOSUR",
    "WORLD",
    "OTHER",
]

pattern = "|".join(aggregate_patterns)
df = df[~df["cty_name"].str.upper().str.contains(pattern, na=False, regex=True)].copy()

# -----------------------------
# Average import value over pre-period
# -----------------------------
pre_avg = (
    df.groupby(["hts_code", "cty_code", "cty_name"], as_index=False)["import_value"]
      .mean()
      .rename(columns={"import_value": "avg_import_value_pre"})
)

# -----------------------------
# Compute total imports by HTS code
# -----------------------------
totals = (
    pre_avg.groupby("hts_code", as_index=False)["avg_import_value_pre"]
           .sum()
           .rename(columns={"avg_import_value_pre": "total_imports_pre"})
)

pre_avg = pre_avg.merge(totals, on="hts_code", how="left")

# -----------------------------
# Construct import share
# -----------------------------
pre_avg["import_share_pre"] = (
    pre_avg["avg_import_value_pre"] / pre_avg["total_imports_pre"]
)

# -----------------------------
# Keep clean output
# -----------------------------
shares = pre_avg[[
    "hts_code",
    "cty_code",
    "cty_name",
    "avg_import_value_pre",
    "total_imports_pre",
    "import_share_pre"
]].copy()

shares.to_csv(output_path, index=False)

# -----------------------------
# Diagnostics
# -----------------------------
check = shares.groupby("hts_code", as_index=False)["import_share_pre"].sum()
check["diff_from_one"] = check["import_share_pre"] - 1

print("Saved import shares to:", output_path)
print("\nPreview:")
print(shares.head())

print("\nRows:", len(shares))
print("Unique HTS codes:", shares["hts_code"].nunique())
print("Unique countries:", shares["cty_code"].nunique())

print("\nShare-sum check (should be very close to 1 within each HTS code):")
print(check["diff_from_one"].describe())

china = shares[shares["cty_name"].str.upper() == "CHINA"]
print("\nChina rows:", len(china))
print(china.head())