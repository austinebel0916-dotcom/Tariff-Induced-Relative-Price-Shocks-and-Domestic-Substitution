from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

input_path = DATA_CLEAN / "imports_hts8_country_2015_2020.csv"
output_path = DATA_CLEAN / "import_shares_pre_2015_2017_full.csv"

# -----------------------------
# Load full panel
# -----------------------------
df = pd.read_csv(input_path, dtype={"hts_code": str, "cty_code": str})

# -----------------------------
# Restrict to pre-period ONLY
# -----------------------------
df = df[df["year"].isin([2015, 2016, 2017])].copy()

# -----------------------------
# Clean country fields
# -----------------------------
df["cty_name"] = df["cty_name"].astype(str).str.strip()
df["cty_code"] = df["cty_code"].astype(str).str.strip()

# Drop obvious non-country rows
df = df[df["cty_code"] != "-"].copy()
df = df[df["cty_code"].str.fullmatch(r"\d+")].copy()

# Drop aggregates via pattern
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
    "WORLD",
    "OTHER"
]

pattern = "|".join(aggregate_patterns)
df = df[~df["cty_name"].str.upper().str.contains(pattern, na=False, regex=True)].copy()

# -----------------------------
# Compute pre-period averages
# -----------------------------
pre_avg = (
    df.groupby(["hts_code", "cty_code", "cty_name"], as_index=False)["import_value"]
      .mean()
      .rename(columns={"import_value": "avg_import_value_pre"})
)

# -----------------------------
# Total imports per product
# -----------------------------
totals = (
    pre_avg.groupby("hts_code", as_index=False)["avg_import_value_pre"]
           .sum()
           .rename(columns={"avg_import_value_pre": "total_imports_pre"})
)

pre_avg = pre_avg.merge(totals, on="hts_code", how="left")

# -----------------------------
# Construct shares
# -----------------------------
pre_avg["import_share_pre"] = (
    pre_avg["avg_import_value_pre"] / pre_avg["total_imports_pre"]
)

# -----------------------------
# Save
# -----------------------------
pre_avg.to_csv(output_path, index=False)

# -----------------------------
# Diagnostics
# -----------------------------
check = pre_avg.groupby("hts_code")["import_share_pre"].sum()

print("Saved to:", output_path)
print("\nRows:", len(pre_avg))
print("Unique HTS codes:", pre_avg["hts_code"].nunique())
print("Unique countries:", pre_avg["cty_code"].nunique())

print("\nShare sum check:")
print(check.describe())

china = pre_avg[pre_avg["cty_name"].str.upper() == "CHINA"]
print("\nChina rows:", len(china))
print(china.head())