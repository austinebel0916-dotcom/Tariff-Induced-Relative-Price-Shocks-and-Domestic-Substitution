import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

panel_path = DATA_CLEAN / "panel_dataset_with_unit_values_post_shock.csv"
qty_imports_path = DATA_CLEAN / "imports_hts8_country_2015_2020_with_quantity.csv"

out_path = DATA_CLEAN / "panel_dataset_with_china_unit_values.csv"

# -----------------------------
# Load data
# -----------------------------
panel = pd.read_csv(panel_path, dtype={"hts_code": str})
imports_qty = pd.read_csv(qty_imports_path, dtype={"hts_code": str, "cty_code": str})

panel["hts_code"] = panel["hts_code"].astype(str).str.zfill(8)
imports_qty["hts_code"] = imports_qty["hts_code"].astype(str).str.zfill(8)

# -----------------------------
# Keep China rows only
# -----------------------------
# Your prior export preview showed CHINA as cty_name and code 5700.
china = imports_qty[
    (imports_qty["cty_name"].astype(str).str.upper().str.strip() == "CHINA") |
    (imports_qty["cty_code"].astype(str).str.strip().str.zfill(4) == "5700")
].copy()

print("China rows:", china.shape)
print("Unique HTS8 codes in China rows:", china["hts_code"].nunique())
print("Years in China rows:", sorted(china["year"].unique()))

# -----------------------------
# Clean quantity/value
# -----------------------------
for col in ["import_value", "import_qty1", "import_qty2"]:
    china[col] = pd.to_numeric(china[col], errors="coerce")

china["unit_qty1"] = china["unit_qty1"].astype(str).str.strip()
china["ambiguous_unit_qty1"] = china["unit_qty1"].str.contains(",", regex=False)

# Use quantity 1 when positive and non-ambiguous
china_price = china[
    (china["import_value"] > 0) &
    (china["import_qty1"] > 0) &
    (~china["ambiguous_unit_qty1"])
].copy()

china_price["china_unit_value"] = china_price["import_value"] / china_price["import_qty1"]
china_price["ln_china_unit_value"] = np.log(china_price["china_unit_value"])

china_price = china_price[
    [
        "hts_code",
        "year",
        "import_value",
        "import_qty1",
        "unit_qty1",
        "china_unit_value",
        "ln_china_unit_value"
    ]
].rename(columns={
    "import_value": "china_import_value_for_uv",
    "import_qty1": "china_import_qty1_for_uv",
    "unit_qty1": "china_unit_qty1"
})

# -----------------------------
# Merge onto panel
# -----------------------------
merged = panel.merge(
    china_price,
    on=["hts_code", "year"],
    how="left",
    validate="one_to_one"
)

# -----------------------------
# Checks
# -----------------------------
print("\nOriginal panel shape:", panel.shape)
print("Merged panel shape:", merged.shape)

print("\nMissing ln_china_unit_value:")
print(merged["ln_china_unit_value"].isna().sum())

print("\nNon-missing ln_china_unit_value:")
print(merged["ln_china_unit_value"].notna().sum())

print("\nShare of panel with non-missing ln_china_unit_value:")
print(merged["ln_china_unit_value"].notna().mean())

print("\nNon-missing China unit values by year:")
print(merged.groupby("year")["ln_china_unit_value"].apply(lambda x: x.notna().sum()))

print("\nChina unit labels:")
print(merged["china_unit_qty1"].value_counts(dropna=False).head(20))

print("\nPreview:")
print(merged[[
    "hts_code",
    "year",
    "imports_china",
    "china_share",
    "pred_tariff_shock_post_pp",
    "ln_import_unit_value",
    "ln_china_unit_value"
]].head(20))

# -----------------------------
# Save
# -----------------------------
merged.to_csv(out_path, index=False)

print("\nSaved:", out_path)