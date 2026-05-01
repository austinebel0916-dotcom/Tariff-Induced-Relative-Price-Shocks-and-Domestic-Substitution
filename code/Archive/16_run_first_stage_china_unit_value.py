import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

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

print("China first-stage sample shape:", reg.shape)
print("Unique HTS8 products:", reg["hts_code"].nunique())
print("Years:", sorted(reg["year"].unique()))

print("\nMean corrected instrument by year:")
print(reg.groupby("year")["pred_tariff_shock_post_pp"].mean())

print("\nNon-missing observations by year:")
print(reg.groupby("year")["ln_china_unit_value"].count())

# -----------------------------
# First-stage regression
# -----------------------------
# Outcome: log China import unit value
# Instrument: predicted tariff shock, in percentage points
# Fixed effects: HTS8 product and year
# SEs clustered by HTS8 product
model = smf.ols(
    "ln_china_unit_value ~ pred_tariff_shock_post_pp + C(hts_code) + C(year)",
    data=reg
).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["hts_code"]}
)

# -----------------------------
# Save full results
# -----------------------------
out_path = OUTPUT / "first_stage_china_unit_value_results.txt"

with open(out_path, "w") as f:
    f.write(model.summary().as_text())

print("\nSaved first-stage results to:", out_path)

# -----------------------------
# Print key coefficient only
# -----------------------------
coef = model.params.get("pred_tariff_shock_post_pp")
se = model.bse.get("pred_tariff_shock_post_pp")
pval = model.pvalues.get("pred_tariff_shock_post_pp")
tval = model.tvalues.get("pred_tariff_shock_post_pp")

print("\nKey China first-stage result:")
print(f"Coefficient on pred_tariff_shock_post_pp: {coef}")
print(f"Clustered SE: {se}")
print(f"t-stat: {tval}")
print(f"p-value: {pval}")

if tval is not None:
    print(f"Approx. first-stage F-stat: {tval ** 2}")