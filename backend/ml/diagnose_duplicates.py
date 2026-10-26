"""The file has several rows for the same district, year, season and crop. Why, and how do they differ?

Run from backend/:
    python ml/diagnose_duplicates.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "crop_production.csv"
df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
for col in ["State_Name", "District_Name", "Season", "Crop"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

df = df[(df["Area"] > 0) & (df["Production"] > 0)].copy()
df["row_yield"] = df["Production"] / df["Area"]

key = ["State_Name", "District_Name", "Crop_Year", "Season"]
rice = df[(df["Crop"] == "Rice") & df["Crop_Year"].between(2005, 2015)]

sizes = rice.groupby(key).size()
print("How many rice rows exist per state/district/year/season (2005-2015)?")
print(sizes.value_counts().sort_index().head(8).to_string())

# for groups of exactly two rows: compare the low-yield row with the high-yield row
ordered = rice.sort_values("row_yield")
pairs = ordered.groupby(key).agg(n=("Area", "size"),
                                 area_lo=("Area", "first"), area_hi=("Area", "last"),
                                 prod_lo=("Production", "first"), prod_hi=("Production", "last"))
pairs = pairs[pairs["n"] == 2]
print(f"\nGroups with exactly two rows: {len(pairs):,}")

if len(pairs):
    for label, hi, lo in [("area", "area_hi", "area_lo"), ("production", "prod_hi", "prod_lo")]:
        factor = np.log10(pairs[hi] / pairs[lo]).round(1)
        print(f"\nhigh-yield row vs low-yield row, {label}: log10 of the ratio (0 = same, 1 = 10x, 3 = 1000x)")
        print(factor.value_counts().head(6).to_string())

    print("\nSix example pairs:")
    show = pairs.head(6).reset_index()[key + ["area_lo", "prod_lo", "area_hi", "prod_hi"]]
    print(show.to_string(index=False))