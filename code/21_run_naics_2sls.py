import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT = PROJECT_ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

panel_path = DATA_CLEAN / "naics4_iv_panel_2018_2020.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(panel_path, dtype={"naics4": str})
df["naics4"] = df["naics4"].astype(str).str.strip()

# -----------------------------
# Main regression sample
# -----------------------------
reg = df[
    (df["valid_domestic_share"] == True) &
    df["domestic_share"].notna() &
    df["effective_duty_rate_pp"].notna() &
    df["pred_tariff_shock_naics4_pp"].notna()
].copy()

# Drop infinite values
reg = reg.replace([np.inf, -np.inf], np.nan)
reg = reg.dropna(subset=[
    "domestic_share",
    "effective_duty_rate_pp",
    "pred_tariff_shock_naics4_pp",
    "year",
    "naics4"
]).copy()

print("2SLS sample shape:", reg.shape)
print("Unique NAICS4 industries:", reg["naics4"].nunique())
print("Years:", sorted(reg["year"].unique()))

print("\nObservations by year:")
print(reg.groupby("year").size())

print("\nKey variable summaries:")
print(reg[[
    "domestic_share",
    "domestic_share_pct",
    "effective_duty_rate_pp",
    "pred_tariff_shock_naics4_pp",
    "china_share_pre_naics4"
]].describe())

# -----------------------------
# Year fixed effects
# -----------------------------
year_dummies = pd.get_dummies(reg["year"].astype(int), prefix="year", drop_first=True)
year_dummies = year_dummies.astype(float)

# -----------------------------
# First stage
# effective_duty_rate_pp = pi * predicted_tariff_shock + year FE
# -----------------------------
X_first = pd.concat([
    reg[["pred_tariff_shock_naics4_pp"]].astype(float),
    year_dummies
], axis=1)

X_first = sm.add_constant(X_first)
y_first = reg["effective_duty_rate_pp"].astype(float)

first_stage = sm.OLS(y_first, X_first).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["naics4"]}
)

reg["effective_duty_hat_pp"] = first_stage.fittedvalues

# -----------------------------
# Second stage
# domestic_share_pct = beta * predicted effective duty rate + year FE
# -----------------------------
X_second = pd.concat([
    reg[["effective_duty_hat_pp"]].astype(float),
    year_dummies
], axis=1)

X_second = sm.add_constant(X_second)
y_second = reg["domestic_share_pct"].astype(float)

second_stage = sm.OLS(y_second, X_second).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["naics4"]}
)

# -----------------------------
# Also run reduced form
# domestic_share_pct = rho * predicted_tariff_shock + year FE
# -----------------------------
X_rf = pd.concat([
    reg[["pred_tariff_shock_naics4_pp"]].astype(float),
    year_dummies
], axis=1)

X_rf = sm.add_constant(X_rf)
y_rf = reg["domestic_share_pct"].astype(float)

reduced_form = sm.OLS(y_rf, X_rf).fit(
    cov_type="cluster",
    cov_kwds={"groups": reg["naics4"]}
)

# -----------------------------
# Print results
# -----------------------------
print("\n" + "=" * 80)
print("FIRST STAGE")
print("=" * 80)
print(first_stage.summary())

print("\n" + "=" * 80)
print("SECOND STAGE")
print("=" * 80)
print(second_stage.summary())

print("\n" + "=" * 80)
print("REDUCED FORM")
print("=" * 80)
print(reduced_form.summary())

# -----------------------------
# Key coefficients
# -----------------------------
fs_coef = first_stage.params.get("pred_tariff_shock_naics4_pp")
fs_se = first_stage.bse.get("pred_tariff_shock_naics4_pp")
fs_t = first_stage.tvalues.get("pred_tariff_shock_naics4_pp")
fs_p = first_stage.pvalues.get("pred_tariff_shock_naics4_pp")
fs_f = fs_t ** 2 if fs_t is not None else np.nan

ss_coef = second_stage.params.get("effective_duty_hat_pp")
ss_se = second_stage.bse.get("effective_duty_hat_pp")
ss_t = second_stage.tvalues.get("effective_duty_hat_pp")
ss_p = second_stage.pvalues.get("effective_duty_hat_pp")

rf_coef = reduced_form.params.get("pred_tariff_shock_naics4_pp")
rf_se = reduced_form.bse.get("pred_tariff_shock_naics4_pp")
rf_t = reduced_form.tvalues.get("pred_tariff_shock_naics4_pp")
rf_p = reduced_form.pvalues.get("pred_tariff_shock_naics4_pp")

print("\n" + "=" * 80)
print("KEY RESULTS")
print("=" * 80)

print("\nFirst stage:")
print(f"Coefficient on pred_tariff_shock_naics4_pp: {fs_coef}")
print(f"Clustered SE: {fs_se}")
print(f"t-stat: {fs_t}")
print(f"p-value: {fs_p}")
print(f"Approx. first-stage F-stat: {fs_f}")

print("\nSecond stage:")
print(f"Coefficient on effective_duty_hat_pp: {ss_coef}")
print(f"Clustered SE: {ss_se}")
print(f"t-stat: {ss_t}")
print(f"p-value: {ss_p}")

print("\nReduced form:")
print(f"Coefficient on pred_tariff_shock_naics4_pp: {rf_coef}")
print(f"Clustered SE: {rf_se}")
print(f"t-stat: {rf_t}")
print(f"p-value: {rf_p}")

# -----------------------------
# Save outputs
# -----------------------------
out_txt = OUTPUT / "naics4_preliminary_2sls_results.txt"

with open(out_txt, "w") as f:
    f.write("FIRST STAGE\n")
    f.write(first_stage.summary().as_text())
    f.write("\n\nSECOND STAGE\n")
    f.write(second_stage.summary().as_text())
    f.write("\n\nREDUCED FORM\n")
    f.write(reduced_form.summary().as_text())
    f.write("\n\nKEY RESULTS\n")
    f.write(f"First-stage coefficient: {fs_coef}\n")
    f.write(f"First-stage clustered SE: {fs_se}\n")
    f.write(f"First-stage t-stat: {fs_t}\n")
    f.write(f"First-stage p-value: {fs_p}\n")
    f.write(f"Approx. first-stage F-stat: {fs_f}\n")
    f.write(f"Second-stage coefficient: {ss_coef}\n")
    f.write(f"Second-stage clustered SE: {ss_se}\n")
    f.write(f"Second-stage t-stat: {ss_t}\n")
    f.write(f"Second-stage p-value: {ss_p}\n")
    f.write(f"Reduced-form coefficient: {rf_coef}\n")
    f.write(f"Reduced-form clustered SE: {rf_se}\n")
    f.write(f"Reduced-form t-stat: {rf_t}\n")
    f.write(f"Reduced-form p-value: {rf_p}\n")

print("\nSaved results to:", out_txt)

# Save regression sample with fitted values
sample_out = DATA_CLEAN / "naics4_preliminary_2sls_sample.csv"
reg.to_csv(sample_out, index=False)

print("Saved regression sample to:", sample_out)