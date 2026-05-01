import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("../..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT = PROJECT_ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

panel_path = DATA_CLEAN / "panel_dataset_with_china_unit_values.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(panel_path, dtype={"hts_code": str})
df["hts_code"] = df["hts_code"].astype(str).str.zfill(8)

# -----------------------------
# Keep usable first-stage sample
# -----------------------------
reg = df[
    df["ln_china_unit_value"].notna() &
    df["pred_tariff_shock_post_pp"].notna()
].copy()

# Drop infinite values just in case
reg = reg.replace([np.inf, -np.inf], np.nan)
reg = reg[
    reg["ln_china_unit_value"].notna() &
    reg["pred_tariff_shock_post_pp"].notna()
].copy()

print("China first-stage sample shape:", reg.shape)
print("Unique HTS8 products:", reg["hts_code"].nunique())
print("Years:", sorted(reg["year"].unique()))

print("\nMean corrected instrument by year:")
print(reg.groupby("year")["pred_tariff_shock_post_pp"].mean())

print("\nNon-missing observations by year:")
print(reg.groupby("year")["ln_china_unit_value"].count())

# -----------------------------
# Two-way fixed effect demeaning
# -----------------------------
# This estimates the same logic as:
# ln_china_unit_value ~ pred_tariff_shock_post_pp + product FE + year FE
# but avoids building hundreds of dummy variables.

y = "ln_china_unit_value"
x = "pred_tariff_shock_post_pp"

grand_y = reg[y].mean()
grand_x = reg[x].mean()

reg["y_twfe"] = (
    reg[y]
    - reg.groupby("hts_code")[y].transform("mean")
    - reg.groupby("year")[y].transform("mean")
    + grand_y
)

reg["x_twfe"] = (
    reg[x]
    - reg.groupby("hts_code")[x].transform("mean")
    - reg.groupby("year")[x].transform("mean")
    + grand_x
)

# Check variation after fixed effects
print("\nTWFE residualized variation:")
print(reg[["y_twfe", "x_twfe"]].describe())

# -----------------------------
# First-stage regression on residualized variables
# -----------------------------
X = sm.add_constant(reg["x_twfe"])
Y = reg["y_twfe"]

model = sm.OLS(Y, X).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["hts_code"]}
)

print(model.summary())

# -----------------------------
# Save results
# -----------------------------
out_path = OUTPUT / "first_stage_china_unit_value_demeaned_results.txt"

with open(out_path, "w") as f:
    f.write(model.summary().as_text())

print("\nSaved first-stage results to:", out_path)

# -----------------------------
# Print key coefficient only
# -----------------------------
coef = model.params.get("x_twfe")
se = model.bse.get("x_twfe")
pval = model.pvalues.get("x_twfe")
tval = model.tvalues.get("x_twfe")

print("\nKey China first-stage result, TWFE demeaned:")
print(f"Coefficient on pred_tariff_shock_post_pp: {coef}")
print(f"Clustered SE: {se}")
print(f"t-stat: {tval}")
print(f"p-value: {pval}")

if tval is not None:
    print(f"Approx. first-stage F-stat: {tval ** 2}")