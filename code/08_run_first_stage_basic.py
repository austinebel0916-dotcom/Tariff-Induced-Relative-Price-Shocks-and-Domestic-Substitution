from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT_FIG = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABLE = PROJECT_ROOT / "output" / "tables"

data_path = DATA_CLEAN / "panel_dataset.csv"

fig_path = OUTPUT_FIG / "first_stage_regression.png"
table_path = OUTPUT_TABLE / "first_stage_results.csv"

OUTPUT_FIG.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path, dtype={"hts_code": str})

df = df.dropna(subset=["pred_tariff_shock", "china_share"]).copy()

print("Rows in regression sample:", len(df))
print("Unique HTS8 products:", df["hts_code"].nunique())
print("Years:", sorted(df["year"].unique()))

X = sm.add_constant(df["pred_tariff_shock"])
y = df["china_share"]

model = sm.OLS(y, X).fit(cov_type="HC1")

print("\nPreliminary first-stage / exposure validation regression:")
print(model.summary())

results = pd.DataFrame({
    "term": model.params.index,
    "coefficient": model.params.values,
    "std_error": model.bse.values,
    "t_stat": model.tvalues.values,
    "p_value": model.pvalues.values
})

results.to_csv(table_path, index=False)
print("\nSaved regression table to:", table_path)