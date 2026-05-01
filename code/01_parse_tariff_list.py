from pathlib import Path
from pypdf import PdfReader
import re
import pandas as pd

pdf_path = Path("../data_raw/ustr_301/list1/ustr_list1_federal_register.pdf")

reader = PdfReader(str(pdf_path))

all_text = ""

for i in range(4, 9):
    text = reader.pages[i].extract_text()
    all_text += text

codes = re.findall(r"\b\d{4}\.\d{2}\.\d{2}\b", all_text)
codes = sorted(set(codes))

df = pd.DataFrame({"hts_code": codes})

output_path = Path("../data_clean/list1_hts_codes.csv")
df.to_csv(output_path, index=False)

print("Saved to:", output_path)
print(df.head())
import pandas as pd
import numpy as np


df = pd.read_csv("../data_clean/list1_hts_codes.csv")

df_full = pd.concat([df, control], ignore_index=True)

print(df_full["tariff"].value_counts())
import matplotlib.pyplot as plt

counts = df_full["tariff"].value_counts().sort_index()

plt.figure()
counts.plot(kind="bar")

plt.title("Distribution of Tariff Exposure Across Product Categories")
plt.xlabel("Tariff Exposure (0 = No Tariff, 1 = Tariff)")
plt.ylabel("Number of Product Categories")

plt.xticks(rotation=0)

df["industry"] = df["hts_code"].str[:2]


industry_counts = df["industry"].value_counts().sort_index()

import matplotlib.pyplot as plt

plt.figure()
industry_counts.plot(kind="bar")

plt.title("Concentration of U.S. Tariffs Across Industries (HTS Chapters)")

plt.xlabel("Industry (HTS Chapter Code)")

plt.ylabel("Number of Tariffed Product Categories (8-digit HTS codes)")

plt.xticks(rotation=45)

plt.savefig("../output/figures/industry_distribution.png")
plt.show()

plt.savefig("../output/figures/tariff_distribution.png")

plt.show()