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

panel_path = DATA_CLEAN / "panel_dataset_with_unit_values.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(panel_path, dtype={"hts_code": str})
df["hts_code"] = df["hts_code"].astype(str).str.zfill(8)

# -----------------------------
# Keep usable first-stage sample
# -----------------------------
reg = df[
    df["ln_import_unit_value"].notna() &
    df["pred_tariff_shock"].notna()
].copy()

print("First-stage sample shape:", reg.shape)
print("Unique HTS8 products:", reg["hts_code"].nunique())
print("Years:", sorted(reg["year"].unique()))

# -----------------------------
# First-stage regression
# -----------------------------
# Product FE: C(hts_code)
# Year FE: C(year)
# Cluster SEs at product level
model = smf.ols(
    "ln_import_unit_value ~ pred_tariff_shock + C(hts_code) + C(year)",
    data=reg
).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["hts_code"]}
)

print(model.summary())

# -----------------------------
# Save results
# -----------------------------
out_path = OUTPUT / "first_stage_unit_value_results.txt"

with open(out_path, "w") as f:
    f.write(model.summary().as_text())

print("\nSaved first-stage results to:", out_path)

# -----------------------------
# Print key coefficient only
# -----------------------------
coef = model.params.get("pred_tariff_shock")
se = model.bse.get("pred_tariff_shock")
pval = model.pvalues.get("pred_tariff_shock")
tval = model.tvalues.get("pred_tariff_shock")

print("\nKey first-stage result:")
print(f"Coefficient on pred_tariff_shock: {coef}")
print(f"Clustered SE: {se}")
print(f"t-stat: {tval}")
print(f"p-value: {pval}")