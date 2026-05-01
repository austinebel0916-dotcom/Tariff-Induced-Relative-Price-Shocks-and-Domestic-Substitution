from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

trade_path = DATA_CLEAN / "imports_hts8_country_2015_2020.csv"
shares_path = DATA_CLEAN / "import_shares_pre_2015_2017_full.csv"
tariff_path = DATA_CLEAN / "list1_tariffs.csv"

output_path = DATA_CLEAN / "panel_dataset.csv"

trade = pd.read_csv(trade_path, dtype={"hts_code": str})
shares = pd.read_csv(shares_path, dtype={"hts_code": str})
tariffs = pd.read_csv(tariff_path, dtype={"hts_code": str})

trade["cty_name"] = trade["cty_name"].astype(str).str.strip()
trade["cty_code"] = trade["cty_code"].astype(str).str.strip()

trade = trade[trade["cty_code"] != "-"].copy()
trade = trade[trade["cty_code"].str.fullmatch(r"\d+")].copy()

aggregate_patterns = [
    "TOTAL", "EUROPEAN UNION", "EURO AREA", "PACIFIC RIM",
    "USMCA", "NAFTA", "LATIN AMERICAN", "OECD", "OPEC",
    "ASEAN", "APEC", "EFTA", "NATO", "WORLD", "OTHER"
]

pattern = "|".join(aggregate_patterns)
trade = trade[~trade["cty_name"].str.upper().str.contains(pattern, na=False, regex=True)].copy()

total_imports = (
    trade.groupby(["hts_code", "year"], as_index=False)["import_value"]
    .sum()
    .rename(columns={"import_value": "imports_total"})
)

china_imports = (
    trade[trade["cty_name"].str.upper() == "CHINA"]
    .groupby(["hts_code", "year"], as_index=False)["import_value"]
    .sum()
    .rename(columns={"import_value": "imports_china"})
)

panel = total_imports.merge(china_imports, on=["hts_code", "year"], how="left")
panel["imports_china"] = panel["imports_china"].fillna(0)

panel["china_share"] = panel["imports_china"] / panel["imports_total"]

china_pre = shares[shares["cty_name"].str.upper() == "CHINA"].copy()
china_pre = china_pre[["hts_code", "import_share_pre"]]

panel = panel.merge(china_pre, on="hts_code", how="left")
panel = panel.merge(tariffs[["hts_code", "tariff_change"]], on="hts_code", how="left")

panel["pred_tariff_shock"] = panel["import_share_pre"] * panel["tariff_change"]

panel["ln_imports_total"] = np.log(panel["imports_total"] + 1)

panel.to_csv(output_path, index=False)

print("Saved panel dataset to:", output_path)
print("\nPreview:")
print(panel.head())

print("\nRows:", len(panel))
print("Years:", sorted(panel["year"].unique()))
print("HTS codes:", panel["hts_code"].nunique())