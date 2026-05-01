import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

asm_path = DATA_CLEAN / "asm_output_2018_2020.csv"
imports_path = DATA_CLEAN / "naics4_imports_2018_2020.csv"
exports_path = DATA_CLEAN / "naics4_exports_2018_2020.csv"

out_path = DATA_CLEAN / "naics4_domestic_absorption_panel_2018_2020.csv"

# -----------------------------
# Load data
# -----------------------------
asm = pd.read_csv(asm_path, dtype={"naics2017": str, "indlevel": str})
imports = pd.read_csv(imports_path, dtype={"naics4": str})
exports = pd.read_csv(exports_path, dtype={"naics4": str})

# -----------------------------
# Keep ASM NAICS4 industries
# -----------------------------
asm4 = asm[asm["indlevel"].astype(str).str.strip() == "4"].copy()

asm4 = asm4.rename(columns={
    "naics2017": "naics4"
})

asm4["naics4"] = asm4["naics4"].astype(str).str.strip()
asm4["year"] = pd.to_numeric(asm4["year"], errors="coerce")

asm4 = asm4[
    asm4["naics4"].str.match(r"^\d{4}$", na=False) &
    asm4["naics4"].str.startswith(("31", "32", "33"))
].copy()

asm4 = asm4[[
    "naics4",
    "year",
    "shipments_revenue_1000",
    "domestic_output",
    "employment",
    "annual_payroll_1000"
]].copy()

# -----------------------------
# Clean imports
# -----------------------------
imports["naics4"] = imports["naics4"].astype(str).str.strip()
imports["year"] = pd.to_numeric(imports["year"], errors="coerce")

imports = imports[
    imports["naics4"].str.match(r"^\d{4}$", na=False) &
    imports["naics4"].str.startswith(("31", "32", "33"))
].copy()

for col in ["imports_consumption", "imports_general", "calculated_duty", "dutiable_value"]:
    imports[col] = pd.to_numeric(imports[col], errors="coerce").fillna(0)

# If the earlier imports file contains any duplicates after filtering, collapse safely.
imports4 = (
    imports
    .groupby(["naics4", "year"], as_index=False)
    .agg(
        imports_consumption=("imports_consumption", "sum"),
        imports_general=("imports_general", "sum"),
        calculated_duty=("calculated_duty", "sum"),
        dutiable_value=("dutiable_value", "sum")
    )
)

# -----------------------------
# Clean exports
# -----------------------------
exports["naics4"] = exports["naics4"].astype(str).str.strip()
exports["year"] = pd.to_numeric(exports["year"], errors="coerce")
exports["exports_total"] = pd.to_numeric(exports["exports_total"], errors="coerce").fillna(0)

exports = exports[
    exports["naics4"].str.match(r"^\d{4}$", na=False) &
    exports["naics4"].str.startswith(("31", "32", "33"))
].copy()

exports4 = (
    exports
    .groupby(["naics4", "year"], as_index=False)
    .agg(exports_total=("exports_total", "sum"))
)

# -----------------------------
# Merge
# -----------------------------
panel = asm4.merge(
    imports4,
    on=["naics4", "year"],
    how="left",
    validate="one_to_one"
)

panel = panel.merge(
    exports4,
    on=["naics4", "year"],
    how="left",
    validate="one_to_one"
)

# Fill missing trade values with zero only after merge
for col in ["imports_consumption", "imports_general", "calculated_duty", "dutiable_value", "exports_total"]:
    panel[col] = panel[col].fillna(0)

# -----------------------------
# Construct domestic absorption outcome
# -----------------------------
panel["domestic_supply_for_us"] = panel["domestic_output"] - panel["exports_total"]

panel["domestic_absorption"] = (
    panel["domestic_supply_for_us"] + panel["imports_consumption"]
)

panel["domestic_share"] = (
    panel["domestic_supply_for_us"] / panel["domestic_absorption"]
)

panel["import_share_absorption"] = (
    panel["imports_consumption"] / panel["domestic_absorption"]
)

# Optional logs for later regressions
panel["ln_domestic_output"] = np.where(
    panel["domestic_output"] > 0,
    np.log(panel["domestic_output"]),
    np.nan
)

panel["ln_imports_consumption"] = np.where(
    panel["imports_consumption"] > 0,
    np.log(panel["imports_consumption"]),
    np.nan
)

# Effective duty rate, useful later as descriptive evidence
panel["effective_duty_rate"] = np.where(
    panel["dutiable_value"] > 0,
    panel["calculated_duty"] / panel["dutiable_value"],
    np.nan
)

# -----------------------------
# Checks
# -----------------------------
print("ASM4 shape:", asm4.shape)
print("Imports4 shape:", imports4.shape)
print("Exports4 shape:", exports4.shape)
print("Merged panel shape:", panel.shape)

print("\nYears:")
print(sorted(panel["year"].dropna().unique()))

print("\nUnique NAICS4 industries:")
print(panel["naics4"].nunique())

print("\nMissing values:")
print(panel.isna().sum())

print("\nDomestic share summary:")
print(panel["domestic_share"].describe())

print("\nImport share of absorption summary:")
print(panel["import_share_absorption"].describe())

print("\nEffective duty rate summary:")
print(panel["effective_duty_rate"].describe())

print("\nPotential problem rows:")
problem_rows = panel[
    (panel["domestic_supply_for_us"] < 0) |
    (panel["domestic_absorption"] <= 0) |
    (panel["domestic_share"] < 0) |
    (panel["domestic_share"] > 1)
].copy()

print(problem_rows[[
    "naics4",
    "year",
    "domestic_output",
    "exports_total",
    "imports_consumption",
    "domestic_supply_for_us",
    "domestic_absorption",
    "domestic_share"
]].head(50))

print("\nNumber of problem rows:", len(problem_rows))

print("\nPreview:")
print(panel[[
    "naics4",
    "year",
    "domestic_output",
    "exports_total",
    "imports_consumption",
    "domestic_supply_for_us",
    "domestic_absorption",
    "domestic_share",
    "effective_duty_rate"
]].head(30))

# -----------------------------
# Save
# -----------------------------
panel.to_csv(out_path, index=False)

print("\nSaved:", out_path)