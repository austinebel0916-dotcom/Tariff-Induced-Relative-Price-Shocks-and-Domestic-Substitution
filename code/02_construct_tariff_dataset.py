import pandas as pd
from pathlib import Path

# Load HTS codes from previous step
input_path = Path("../data_clean/list1_hts_codes.csv")
df = pd.read_csv(input_path)

# Add tariff change (List 1 = 25%)
df["tariff_change"] = 0.25

# Add implementation year
df["year"] = 2018

# Ensure HTS codes are strings (important for merges later)
df["hts_code"] = df["hts_code"].str.replace(".", "", regex=False)

# Save clean tariff dataset
output_path = Path("../data_clean/list1_tariffs.csv")
df.to_csv(output_path, index=False)

print("Saved tariff dataset to:", output_path)
print(df.head())