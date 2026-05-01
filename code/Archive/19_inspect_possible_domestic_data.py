from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/")

# Search likely folders
folders_to_search = [
    PROJECT_ROOT / "data_raw",
    PROJECT_ROOT / "data_clean",
]

keywords = [
    "domestic",
    "production",
    "shipments",
    "output",
    "bea",
    "asm",
    "napcs",
    "naics",
    "industry"
]

print("Searching project for possible domestic production files...\n")

matches = []

for folder in folders_to_search:
    if not folder.exists():
        continue

    for path in folder.rglob("*"):
        if path.is_file():
            name_lower = path.name.lower()
            if any(k in name_lower for k in keywords):
                matches.append(path)

if not matches:
    print("No obvious domestic production files found.")
else:
    print(f"Found {len(matches)} possible files:\n")
    for path in matches:
        print(path)

print("\nInspecting CSV files among matches...\n")

for path in matches:
    if path.suffix.lower() == ".csv":
        print("=" * 80)
        print(path)
        print("=" * 80)
        try:
            df = pd.read_csv(path, nrows=5)
            print("Columns:")
            print(list(df.columns))
            print("\nFirst 5 rows:")
            print(df.head())
        except Exception as e:
            print(f"Could not read file: {e}")
        print()