import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

domestic_path = DATA_CLEAN / "naics4_domestic_absorption_panel_2018_2020.csv"
china_share_path = DATA_CLEAN / "naics4_china_share_pre_2015_2017.csv"

out_path = DATA_CLEAN / "naics4_iv_panel_2018_2020.csv"

domestic = pd.read_csv(domestic_path, dtype={"naics4": str})
china_share = pd.read_csv(china_share_path, dtype={"naics4": str})

domestic["naics4"] = domestic["naics4"].astype(str).str.strip()
china_share["naics4"] = china_share["naics4"].astype(str).str.strip()

panel = domestic.merge(
    china_share,
    on="naics4",
    how="left",
    validate="many_to_one"
)

panel["valid_domestic_share"] = (
    (panel["domestic_absorption"] > 0) &
    (panel["domestic_share"] >= 0) &
    (panel["domestic_share"] <= 1)
)

panel["post_2018"] = (panel["year"] >= 2018).astype(int)

panel["pred_tariff_shock_naics4"] = (
    panel["china_share_pre_naics4"] * 0.25 * panel["post_2018"]
)

panel["pred_tariff_shock_naics4_pp"] = panel["pred_tariff_shock_naics4"] * 100

panel["effective_duty_rate_pp"] = panel["effective_duty_rate"] * 100

panel["domestic_share_pct"] = panel["domestic_share"] * 100
panel["import_share_absorption_pct"] = panel["import_share_absorption"] * 100

print("Domestic panel shape:", domestic.shape)
print("China share shape:", china_share.shape)
print("Merged IV panel shape:", panel.shape)

print("\nYears:")
print(sorted(panel["year"].dropna().unique()))

print("\nUnique NAICS4 industries:")
print(panel["naics4"].nunique())

print("\nMissing values in key variables:")
key_vars = [
    "domestic_share",
    "valid_domestic_share",
    "effective_duty_rate",
    "effective_duty_rate_pp",
    "china_share_pre_naics4",
    "pred_tariff_shock_naics4",
    "pred_tariff_shock_naics4_pp"
]
print(panel[key_vars].isna().sum())

print("\nValid domestic share counts:")
print(panel["valid_domestic_share"].value_counts(dropna=False))

print("\nPredicted tariff shock summary:")
print(panel["pred_tariff_shock_naics4_pp"].describe())

print("\nEffective duty rate summary:")
print(panel["effective_duty_rate_pp"].describe())

print("\nDomestic share summary, valid rows only:")
print(panel.loc[panel["valid_domestic_share"], "domestic_share"].describe())

print("\nTop predicted exposure industries:")
print(
    panel[["naics4", "year", "china_share_pre_naics4", "pred_tariff_shock_naics4_pp", "effective_duty_rate_pp", "domestic_share"]]
    .drop_duplicates()
    .sort_values("pred_tariff_shock_naics4_pp", ascending=False)
    .head(30)
)

print("\nRows missing pre-period China exposure:")
print(
    panel[panel["china_share_pre_naics4"].isna()][
        ["naics4", "year", "domestic_share", "effective_duty_rate"]
    ]
)

panel.to_csv(out_path, index=False)

print("\nSaved:", out_path)