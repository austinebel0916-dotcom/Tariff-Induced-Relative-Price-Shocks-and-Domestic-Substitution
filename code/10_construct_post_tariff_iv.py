import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

in_path = DATA_CLEAN / "panel_dataset_with_unit_values.csv"
out_path = DATA_CLEAN / "panel_dataset_with_unit_values_post_shock.csv"

df = pd.read_csv(in_path, dtype={"hts_code": str})
df["hts_code"] = df["hts_code"].astype(str).str.zfill(8)

df["post_2018"] = (df["year"] >= 2018).astype(int)

df["tariff_change_post"] = 0.25 * df["post_2018"]

df["pred_tariff_shock_post"] = (
    df["import_share_pre"] * df["tariff_change_post"]
)

# Optional scaled version: percentage-point tariff shock.
# This is easier to read in regression tables.
df["pred_tariff_shock_post_pp"] = df["pred_tariff_shock_post"] * 100

print("Panel shape:", df.shape)

print("\nYears:")
print(sorted(df["year"].unique()))

print("\nMissing values:")
print(df[[
    "import_share_pre",
    "pred_tariff_shock",
    "pred_tariff_shock_post",
    "pred_tariff_shock_post_pp",
    "ln_import_unit_value"
]].isna().sum())

print("\nMean old pred_tariff_shock by year:")
print(df.groupby("year")["pred_tariff_shock"].mean())

print("\nMean NEW pred_tariff_shock_post by year:")
print(df.groupby("year")["pred_tariff_shock_post"].mean())

print("\nMean NEW pred_tariff_shock_post_pp by year:")
print(df.groupby("year")["pred_tariff_shock_post_pp"].mean())

print("\nPreview:")
print(df[[
    "hts_code",
    "year",
    "import_share_pre",
    "post_2018",
    "tariff_change_post",
    "pred_tariff_shock",
    "pred_tariff_shock_post",
    "pred_tariff_shock_post_pp",
    "ln_import_unit_value"
]].head(12))

df.to_csv(out_path, index=False)

print("\nSaved:", out_path)