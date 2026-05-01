import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("../..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"

asm_path = DATA_CLEAN / "asm_output_2018_2020.csv"

asm = pd.read_csv(asm_path, dtype={"naics2017": str, "indlevel": str})

print("ASM shape:", asm.shape)

print("\nINDLEVEL counts:")
print(asm["indlevel"].value_counts().sort_index())

for level in ["2", "3", "4", "5", "6"]:
    subset = asm[asm["indlevel"] == level].copy()

    print("\n" + "=" * 80)
    print(f"INDLEVEL {level}")
    print("=" * 80)

    print("Rows:", len(subset))
    print("Unique NAICS:", subset["naics2017"].nunique())
    print("Years:", sorted(subset["year"].unique()))

    print("\nExample NAICS codes:")
    print(subset[["year", "naics2017", "shipments_revenue_1000", "domestic_output"]].head(20))

    print("\nTotal domestic output by year:")
    print(subset.groupby("year")["domestic_output"].sum())