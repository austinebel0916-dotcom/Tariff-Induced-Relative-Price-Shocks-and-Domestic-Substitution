import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("../..")
DATA_RAW = PROJECT_ROOT / "data_raw" / "naics_trade"

raw_path = DATA_RAW / "naics4_exports_2018_2020_raw_loop.csv"

exports = pd.read_csv(raw_path, dtype={"naics4": str, "cty_code": str, "cty_name": str})

print("Raw exports shape:", exports.shape)

print("\nColumns:")
print(list(exports.columns))

print("\nCountry code counts:")
print(exports["cty_code"].value_counts(dropna=False).head(30))

print("\nCountry name counts:")
print(exports["cty_name"].value_counts(dropna=False).head(30))

print("\nRows where country name suggests total:")
total_like = exports[
    exports["cty_name"].astype(str).str.upper().str.contains("TOTAL|WORLD|ALL", na=False) |
    exports["cty_code"].astype(str).str.contains("-", na=False)
].copy()

print(total_like[["naics4", "year", "cty_code", "cty_name", "exports_total"]].head(50))
print("\nTotal-like shape:", total_like.shape)

print("\nExample rows for NAICS4 3364 in 2018:")
example = exports[
    (exports["naics4"] == "3364") &
    (exports["year"] == 2018)
].copy()

print(example[["naics4", "year", "cty_code", "cty_name", "exports_total"]].head(100))

print("\nSum across all rows for 3364, 2018:")
print(example["exports_total"].sum())

print("\nTop rows for 3364, 2018:")
print(example.sort_values("exports_total", ascending=False)[
    ["naics4", "year", "cty_code", "cty_name", "exports_total"]
].head(20))