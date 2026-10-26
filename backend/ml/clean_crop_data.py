"""Repair the crop file: remove the duplicated rows and fix the unit mix-up.

What the diagnostics showed: most records appear twice with identical area, but one copy has
production 100x larger. Rows from 2016 on appear only in the 100x form. This script keeps the
plausible copy, rescales the lone 100x rows, and drops anything still implausible.

Run from backend/:
    python ml/clean_crop_data.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "crop_production.csv"
OUT = BASE / "data" / "processed" / "crop_production_clean.csv"

KEY = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop"]
MIN_REFERENCE_ROWS = 30   # a crop needs this many trusted rows to get a reference yield
UNIT_FACTOR = 100         # the factor the diagnostics found between the two copies

df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
for col in ["State_Name", "District_Name", "Season", "Crop"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
print(f"Rows in file: {len(df):,}")

df = df[(df["Area"] > 0) & (df["Production"] > 0)].drop_duplicates(subset=["Area", "Production"] + KEY).copy()
print(f"After removing empty rows and exact duplicates: {len(df):,}")

# 1. One row per record: where several exist, keep the one with the smallest production.
keep = df.groupby(KEY)["Production"].idxmin()
repeated = df.groupby(KEY)["Area"].transform("size") > 1
trusted = df.loc[keep][repeated.loc[keep].values].copy()   # rows that had a twin, in their plausible form
df = df.loc[keep].copy()
print(f"After keeping one row per record: {len(df):,}")

# 2. Reference yield per crop, taken only from records that had a twin (the plausible copy)
trusted["row_yield"] = trusted["Production"] / trusted["Area"]
counts = trusted.groupby("Crop")["row_yield"].agg(["median", "size"])
reference = counts[counts["size"] >= MIN_REFERENCE_ROWS]["median"]

df["row_yield"] = df["Production"] / df["Area"]
df["reference"] = df["Crop"].map(reference)
no_ref = df[df["reference"].isna()]
print(f"\nCrops dropped for lack of a reference yield: {no_ref['Crop'].nunique()} "
      f"({len(no_ref):,} rows), for example: {', '.join(sorted(no_ref['Crop'].unique())[:8])}")
df = df.dropna(subset=["reference"]).copy()

# 3. Rescale lone rows that are in the 100x form: closer (in log terms) to reference*100 than to reference
too_big = df["row_yield"] > df["reference"] * UNIT_FACTOR ** 0.5
print("\nRows rescaled (production / 100), by year:")
print(df[too_big].groupby("Crop_Year").size().to_string())
df.loc[too_big, "Production"] = df.loc[too_big, "Production"] / UNIT_FACTOR
df["row_yield"] = df["Production"] / df["Area"]

# 4. Drop what is still implausible
plausible = df["row_yield"].between(df["reference"] / 20, df["reference"] * 8)
print(f"\nDropped as still implausible: {(~plausible).sum():,} rows ({(~plausible).mean():.1%})")
df = df[plausible]

OUT.parent.mkdir(parents=True, exist_ok=True)
df[KEY + ["Area", "Production"]].to_csv(OUT, index=False)
print(f"\nSaved {OUT} with {len(df):,} rows")

# 5. Re-run the reality check
print("\nNational yield 2005-2015 after cleaning (expected, tonnes per hectare: "
      "rice 2.0-2.8, wheat 2.5-3.3, maize 2.0-3.0, sugarcane 60-75, potato 17-24, groundnut 1.0-1.5)")
recent = df[df["Crop_Year"].between(2005, 2015)]
for crop in ["Rice", "Wheat", "Maize", "Sugarcane", "Potato", "Groundnut"]:
    sub = recent[recent["Crop"] == crop]
    if len(sub):
        print(f"  {crop:10} {sub['Production'].sum() / sub['Area'].sum():8.2f}")