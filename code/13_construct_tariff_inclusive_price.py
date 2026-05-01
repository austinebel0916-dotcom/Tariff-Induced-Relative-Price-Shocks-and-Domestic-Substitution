import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

in_path = DATA_CLEAN / "panel_dataset_with_china_unit_values.csv"
out_path = DATA_CLEAN / "panel_dataset_with_tariff_inclusive_price.csv"

df = pd.read_csv(in_path, dtype={"hts_code": str})
df["hts_code"] = df["hts_code"].astype(str).str.zfill(8)

df["post_2018"] = (df["year"] >= 2018).astype(int)
df["section301_tariff_rate"] = 0.25 * df["post_2018"]

# Tariff-inclusive price proxy:
# China unit value multiplied by one plus the tariff rate.
df["china_unit_value_tariff_inclusive"] = (
    df["china_unit_value"] * (1 + df["section301_tariff_rate"])
)

df["ln_china_unit_value_tariff_inclusive"] = np.log(
    df["china_unit_value_tariff_inclusive"]
)

print("Panel shape:", df.shape)

print("\nNon-missing ordinary China unit value:")
print(df["ln_china_unit_value"].notna().sum())

print("\nNon-missing tariff-inclusive China unit value:")
print(df["ln_china_unit_value_tariff_inclusive"].notna().sum())

print("\nMean ordinary China unit value log by year:")
print(df.groupby("year")["ln_china_unit_value"].mean())

print("\nMean tariff-inclusive China unit value log by year:")
print(df.groupby("year")["ln_china_unit_value_tariff_inclusive"].mean())

print("\nPreview:")
print(df[[
    "hts_code",
    "year",
    "pred_tariff_shock_post_pp",
    "section301_tariff_rate",
    "ln_china_unit_value",
    "ln_china_unit_value_tariff_inclusive"
]].head(20))

df.to_csv(out_path, index=False)

print("\nSaved:", out_path)