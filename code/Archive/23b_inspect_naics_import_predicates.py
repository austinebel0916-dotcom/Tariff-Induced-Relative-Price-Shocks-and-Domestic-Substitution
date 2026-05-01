import requests
import pandas as pd

variables_url = "https://api.census.gov/data/timeseries/intltrade/imports/naicsimport/variables.json"

r = requests.get(variables_url, timeout=60)
print("Status code:", r.status_code)

if r.status_code != 200:
    print(r.text[:1000])
    raise SystemExit

variables = r.json()["variables"]

rows = []

for var_name, meta in variables.items():
    rows.append({
        "var_name": var_name,
        "label": meta.get("label", ""),
        "concept": meta.get("concept", ""),
        "predicateType": meta.get("predicateType", ""),
        "required": meta.get("required", ""),
        "group": meta.get("group", ""),
        "limit": meta.get("limit", ""),
        "attributes": meta.get("attributes", "")
    })

df = pd.DataFrame(rows)

print("\nRequired variables / predicates:")
print(
    df[df["required"].astype(str).str.lower() == "true"]
    .sort_values("var_name")
    .to_string(index=False)
)

print("\nLikely query predicates:")
predicates = df[
    df["predicateType"].astype(str).str.len() > 0
].copy()

print(
    predicates[
        ["var_name", "label", "predicateType", "required", "group", "limit"]
    ]
    .sort_values("var_name")
    .to_string(index=False)
)

print("\nSpecific variables we care about:")
care = df[df["var_name"].isin([
    "YEAR",
    "MONTH",
    "time",
    "NAICS",
    "CTY_CODE",
    "SUMMARY_LVL",
    "GEN_VAL_YR",
    "CON_VAL_YR",
    "CAL_DUT_YR",
    "DUT_VAL_YR"
])]
print(care.to_string(index=False))