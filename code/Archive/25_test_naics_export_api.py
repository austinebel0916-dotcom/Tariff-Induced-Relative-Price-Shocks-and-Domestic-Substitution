import requests
import pandas as pd

variables_url = "https://api.census.gov/data/timeseries/intltrade/exports/naics/variables.json"

print("Testing variables endpoint:")
print(variables_url)

r = requests.get(variables_url, timeout=60)
print("Status code:", r.status_code)

if r.status_code != 200:
    print(r.text[:1000])
    raise SystemExit

variables = r.json()["variables"]

keywords = [
    "NAICS",
    "CTY",
    "COUNTRY",
    "ALL_VAL",
    "E_VAL",
    "YEAR",
    "MONTH",
    "time"
]

matches = []

for var_name, meta in variables.items():
    label = str(meta.get("label", ""))
    concept = str(meta.get("concept", ""))
    text = f"{var_name} {label} {concept}".upper()

    if any(k in text for k in keywords):
        matches.append({
            "var_name": var_name,
            "label": label,
            "concept": concept,
            "predicateType": meta.get("predicateType", ""),
            "required": meta.get("required", "")
        })

matches_df = pd.DataFrame(matches)

print("\nPossible relevant variables:")
print(matches_df.head(150).to_string(index=False))

print("\nRequired variables:")
print(
    matches_df[matches_df["required"].astype(str).str.lower() == "true"]
    .to_string(index=False)
)

print("\nNumber of matches:", len(matches_df))