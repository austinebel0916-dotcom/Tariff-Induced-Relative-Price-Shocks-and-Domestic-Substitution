from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("..")
DATA_CLEAN = PROJECT_ROOT / "data_clean"
OUTPUT_FIG = PROJECT_ROOT / "output" / "figures"

input_path = DATA_CLEAN / "list1_tariffs.csv"
fig_path = OUTPUT_FIG / "tariff_distribution.png"

OUTPUT_FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path, dtype={"hts_code": str})

df["hts2"] = df["hts_code"].str[:2]

hts_labels = {
    "28": "Chemicals",
    "40": "Rubber & Plastics",
    "84": "Machinery",
    "85": "Electrical Equipment",
    "86": "Rail Transport",
    "87": "Vehicles",
    "88": "Aircraft",
    "89": "Ships",
    "90": "Precision Instruments",
    "99": "Special Classifications"
}

df["industry_name"] = df["hts2"].map(hts_labels).fillna(df["hts2"])

counts = (
    df.groupby(["hts2", "industry_name"], as_index=False)
    .size()
    .rename(columns={"size": "tariffed_products"})
    .sort_values("tariffed_products", ascending=False)
)

counts = counts[counts["tariffed_products"] > 10]

plt.figure(figsize=(9, 6))

plt.bar(
    counts["industry_name"],
    counts["tariffed_products"]
)

plt.title("Distribution of Tariffed Products Across Industries (2018)", fontsize=14)
plt.xlabel("Industry", fontsize=12)
plt.ylabel("Number of Tariffed Products", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.25)

plt.figtext(
    0.5,
    0.01,
    "Notes: Industries are defined using the first two digits of the Harmonized Tariff Schedule (HTS). "
    "Each observation is an 8-digit product category subject to Section 301 List 1 tariffs. "
    "The figure reports the number of tariffed products in each industry. "
    "Industries with very small counts are omitted for clarity. "
    "Source: U.S. Trade Representative.",
    ha="center",
    fontsize=8,
    wrap=True
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(fig_path, dpi=300)
plt.show()

print("Saved descriptive figure to:", fig_path)
print("\nCounts:")
print(counts)