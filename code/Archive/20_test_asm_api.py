import requests
import pandas as pd

# -----------------------------
# Test Census ASM 2018-2021 API
# -----------------------------
# We are checking the available variable names and whether value of shipments exists.

variables_url = "https://api.census.gov/data/timeseries/asm/area2017/variables.json"

print("Testing variables endpoint:")
print(variables_url)

r = requests.get(variables_url, timeout=60)
print("Status code:", r.status_code)

if r.status_code != 200:
    print(r.text[:1000])
    raise SystemExit

variables = r.json()["variables"]

# Search for likely shipment/output variables
keywords = ["SHIP", "VALUE", "SALES", "RCPT", "NAICS", "IND"]

matches = []

for var_name, meta in variables.items():
    label = str(meta.get("label", ""))
    concept = str(meta.get("concept", ""))
    text = f"{var_name} {label} {concept}".upper()

    if any(k in text for k in keywords):
        matches.append({
            "var_name": var_name,
            "label": label,
            "concept": concept
        })

matches_df = pd.DataFrame(matches)

print("\nPossible relevant variables:")
print(matches_df.head(100).to_string(index=False))

print("\nNumber of matches:", len(matches_df))