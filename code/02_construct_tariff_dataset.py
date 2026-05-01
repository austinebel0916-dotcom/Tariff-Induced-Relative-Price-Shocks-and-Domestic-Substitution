import pandas as pd
from pathlib import Path

input_path = Path("../data_clean/list1_hts_codes.csv")
df = pd.read_csv(input_path)

df["tariff_change"] = 0.25

df["year"] = 2018

df["hts_code"] = df["hts_code"].str.replace(".", "", regex=False)

output_path = Path("../data_clean/list1_tariffs.csv")
df.to_csv(output_path, index=False)

print("Saved tariff dataset to:", output_path)
print(df.head())