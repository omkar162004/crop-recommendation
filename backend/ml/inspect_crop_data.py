"""Step 1 of version 2: look at the district-level crop data before building on it.

Put the downloaded CSV at backend/data/raw/crop_production.csv, then run from backend/:
    python ml/inspect_crop_data.py
"""
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "crop_production.csv"
df = pd.read_csv(PATH)
df.columns = [c.strip() for c in df.columns]

print("Shape:", df.shape)
print("\nColumns and types:")
print(df.dtypes.to_string())
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nMissing values per column:")
print(df.isna().sum().to_string())


def find(words, exclude=()):
    """Find a column whose name contains any of the words (case-insensitive)."""
    for col in df.columns:
        name = col.lower()
        if any(w in name for w in words) and not any(x in name for x in exclude):
            return col
    return None


state, district = find(["state"]), find(["district"])
year, season = find(["year"]), find(["season"])
crop = find(["crop"], exclude=["year"])  # "Crop_Year" is the year, not the crop

print("\nGuessed columns:", {"state": state, "district": district, "year": year, "season": season, "crop": crop})

if state:
    print("\nStates:", df[state].nunique())
if district:
    print("Districts:", df[[c for c in (state, district) if c]].drop_duplicates().shape[0])
if year:
    print("Years:", df[year].min(), "to", df[year].max())
if season:
    print("\nSeasons:\n" + df[season].astype(str).str.strip().value_counts().to_string())
if crop:
    print("\nNumber of crops:", df[crop].nunique())
    print("Most common crops (rows):\n" + df[crop].value_counts().head(30).to_string())