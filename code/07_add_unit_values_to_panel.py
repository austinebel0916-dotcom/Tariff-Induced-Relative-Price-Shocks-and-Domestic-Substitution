import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

panel_path = DATA_CLEAN / "panel_dataset.csv"
qty_imports_path = DATA_CLEAN / "imports_hts8_country_2015_2020_with_quantity.csv"

out_path = DATA_CLEAN / "panel_dataset_with_unit_values.csv"

panel = pd.read_csv(panel_path, dtype={"hts_code": str})
imports_qty = pd.read_csv(qty_imports_path, dtype={"hts_code": str, "cty_code": str})

panel["hts_code"] = panel["hts_code"].astype(str).str.zfill(8)
imports_qty["hts_code"] = imports_qty["hts_code"].astype(str).str.zfill(8)

imports_total = imports_qty[imports_qty["cty_code"].astype(str).str.strip() == "-"].copy()

print("Total-country rows:", imports_total.shape)
print("Unique HTS8 codes in total rows:", imports_total["hts_code"].nunique())
print("Years in total rows:", sorted(imports_total["year"].unique()))

for col in ["import_value", "import_qty1", "import_qty2"]:
    imports_total[col] = pd.to_numeric(imports_total[col], errors="coerce")

imports_total["unit_qty1"] = imports_total["unit_qty1"].astype(str).str.strip()
imports_total["ambiguous_unit_qty1"] = imports_total["unit_qty1"].str.contains(",", regex=False)

price_proxy = imports_total[
    (imports_total["import_value"] > 0) &
    (imports_total["import_qty1"] > 0) &
    (~imports_total["ambiguous_unit_qty1"])
].copy()

price_proxy["unit_value"] = price_proxy["import_value"] / price_proxy["import_qty1"]
price_proxy["ln_import_unit_value"] = np.log(price_proxy["unit_value"])

price_proxy = price_proxy[
    [
        "hts_code",
        "year",
        "import_value",
        "import_qty1",
        "unit_qty1",
        "unit_value",
        "ln_import_unit_value"
    ]
].copy()

merged = panel.merge(
    price_proxy,
    on=["hts_code", "year"],
    how="left",
    validate="one_to_one"
)

print("\nOriginal panel shape:", panel.shape)
print("Merged panel shape:", merged.shape)

print("\nMissing ln_import_unit_value:")
print(merged["ln_import_unit_value"].isna().sum())

print("\nNon-missing ln_import_unit_value:")
print(merged["ln_import_unit_value"].notna().sum())

print("\nShare of panel with non-missing ln_import_unit_value:")
print(merged["ln_import_unit_value"].notna().mean())

print("\nUnit labels in usable price proxy:")
print(merged["unit_qty1"].value_counts(dropna=False).head(20))

print("\nPreview:")
print(merged.head())

merged.to_csv(out_path, index=False)

print("\nSaved:", out_path)