import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("..")
DATA_RAW = PROJECT_ROOT / "data_raw" / "naics_trade"
DATA_CLEAN = PROJECT_ROOT / "data_clean"

raw_path = DATA_RAW / "naics4_exports_2018_2020_raw_loop.csv"
out_path = DATA_CLEAN / "naics4_exports_2018_2020.csv"

# -----------------------------
# Load raw export data
# -----------------------------
exports = pd.read_csv(
    raw_path,
    dtype={
        "naics": str,
        "naics4": str,
        "cty_code": str,
        "cty_name": str
    }
)

exports["cty_code"] = exports["cty_code"].astype(str).str.strip()
exports["cty_name"] = exports["cty_name"].astype(str).str.strip()
exports["naics4"] = exports["naics4"].astype(str).str.strip()
exports["year"] = pd.to_numeric(exports["year"], errors="coerce")
exports["exports_total"] = pd.to_numeric(exports["exports_total"], errors="coerce").fillna(0)

# -----------------------------
# Keep only total-for-all-countries rows
# -----------------------------
exports_total_only = exports[
    (exports["cty_code"] == "-") |
    (exports["cty_name"].str.upper() == "TOTAL FOR ALL COUNTRIES")
].copy()

print("Raw exports shape:", exports.shape)
print("Total-only rows shape:", exports_total_only.shape)

print("\nCountry names in total-only sample:")
print(exports_total_only["cty_name"].value_counts(dropna=False))

# -----------------------------
# Collapse to NAICS4-year
# -----------------------------
exports_naics4 = (
    exports_total_only
    .groupby(["naics4", "year"], as_index=False)
    .agg(exports_total=("exports_total", "sum"))
)

# -----------------------------
# Save corrected clean file
# -----------------------------
exports_naics4.to_csv(out_path, index=False)

print("\nSaved corrected clean NAICS4 exports to:", out_path)

# -----------------------------
# Inspect
# -----------------------------
print("\nNAICS4 shape:")
print(exports_naics4.shape)

print("\nYears:")
print(sorted(exports_naics4["year"].dropna().unique()))

print("\nExamples:")
print(exports_naics4.head(30))

print("\nTop NAICS4 by exports_total:")
print(
    exports_naics4
    .sort_values("exports_total", ascending=False)
    .head(20)
)

print("\nCheck NAICS4 3364:")
print(exports_naics4[exports_naics4["naics4"] == "3364"])

print("\nMissing values:")
print(exports_naics4.isna().sum())