from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("../..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT_TABLE = PROJECT_ROOT / "output" / "tables"

data_path = DATA_CLEAN / "panel_dataset.csv"
table_path = OUTPUT_TABLE / "preliminary_effects.csv"

OUTPUT_TABLE.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(data_path, dtype={"hts_code": str})

# -----------------------------
# Clean sample
# -----------------------------
df = df.dropna(subset=[
    "pred_tariff_shock",
    "china_share",
    "ln_imports_total"
]).copy()

# -----------------------------
# Construct variables
# -----------------------------
df["post"] = (df["year"] >= 2018).astype(int)
df["tariff_post"] = df["pred_tariff_shock"] * df["post"]
df["non_china_share"] = 1 - df["china_share"]

# -----------------------------
# Models
# -----------------------------
models = {
    "China share": "china_share ~ tariff_post + C(hts_code) + C(year)",
    "Non-China share": "non_china_share ~ tariff_post + C(hts_code) + C(year)",
    "Log imports": "ln_imports_total ~ tariff_post + C(hts_code) + C(year)"
}

results_list = []

for name, formula in models.items():
    print(f"\n--- {name} ---")

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["hts_code"]}
    )

    print(model.summary())

    results_list.append({
        "outcome": name,
        "coef": model.params["tariff_post"],
        "std_err": model.bse["tariff_post"],
        "t_stat": model.tvalues["tariff_post"],
        "p_value": model.pvalues["tariff_post"],
        "n_obs": int(model.nobs),
        "r2": model.rsquared
    })

# -----------------------------
# Save results
# -----------------------------
results = pd.DataFrame(results_list)
results.to_csv(table_path, index=False)

print("\nSaved results to:", table_path)
print(results)