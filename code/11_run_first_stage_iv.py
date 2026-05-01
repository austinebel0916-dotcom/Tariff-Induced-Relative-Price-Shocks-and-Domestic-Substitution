import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT = PROJECT_ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

panel_path = DATA_CLEAN / "panel_dataset_with_unit_values_post_shock.csv"

df = pd.read_csv(panel_path, dtype={"hts_code": str})
df["hts_code"] = df["hts_code"].astype(str).str.zfill(8)

reg = df[
    df["ln_import_unit_value"].notna() &
    df["pred_tariff_shock_post_pp"].notna()
].copy()

print("First-stage sample shape:", reg.shape)
print("Unique HTS8 products:", reg["hts_code"].nunique())
print("Years:", sorted(reg["year"].unique()))

print("\nMean corrected instrument by year:")
print(reg.groupby("year")["pred_tariff_shock_post_pp"].mean())

model = smf.ols(
    "ln_import_unit_value ~ pred_tariff_shock_post_pp + C(hts_code) + C(year)",
    data=reg
).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["hts_code"]}
)

out_path = OUTPUT / "first_stage_post_shock_results.txt"

with open(out_path, "w") as f:
    f.write(model.summary().as_text())

print("\nSaved first-stage results to:", out_path)

coef = model.params.get("pred_tariff_shock_post_pp")
se = model.bse.get("pred_tariff_shock_post_pp")
pval = model.pvalues.get("pred_tariff_shock_post_pp")
tval = model.tvalues.get("pred_tariff_shock_post_pp")

print("\nKey first-stage result:")
print(f"Coefficient on pred_tariff_shock_post_pp: {coef}")
print(f"Clustered SE: {se}")
print(f"t-stat: {tval}")
print(f"p-value: {pval}")

if tval is not None:
    print(f"Approx. first-stage F-stat: {tval ** 2}")