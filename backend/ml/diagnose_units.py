"""Find out why national yields look far too big: duplicate rows? mixed units by year or state?

Run from backend/:
    python ml/diagnose_units.py
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "crop_production.csv"
df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
for col in ["State_Name", "District_Name", "Season", "Crop"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

print(f"Rows: {len(df):,}")
key = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop"]
print(f"Exact duplicate rows: {df.duplicated().sum():,}")
print(f"Rows that share the same state/district/year/season/crop with another row: "
      f"{df.duplicated(subset=key, keep=False).sum():,}")

df = df[(df["Area"] > 0) & (df["Production"] > 0)].copy()
df["row_yield"] = df["Production"] / df["Area"]

crops = ["Rice", "Wheat", "Maize", "Groundnut", "Sugarcane"]
by_year = (df[df["Crop"].isin(crops)]
           .pivot_table(index="Crop_Year", columns="Crop", values="row_yield", aggfunc="median")
           .round(2))
print("\nMedian yield per row, by year")
print("(if production is in tonnes: rice ~2, wheat ~3, maize ~2.5, groundnut ~1.2, sugarcane ~65)")
print(by_year.to_string())

rice = df[df["Crop"] == "Rice"]
top_states = rice.groupby("State_Name")["Area"].sum().sort_values(ascending=False).head(12).index
print("\nRice: median yield per row, by state (12 biggest rice states)")
print(rice[rice["State_Name"].isin(top_states)].groupby("State_Name")["row_yield"].median().round(2).to_string())

bad = rice[rice["row_yield"] > 15]  # no rice crop yields 15 tonnes per hectare
print(f"\nRice rows with yield above 15: {len(bad):,} of {len(rice):,} ({len(bad) / len(rice):.1%})")
if len(bad):
    print("Most affected states:\n" + bad["State_Name"].value_counts().head(5).to_string())
    print("Most affected years:\n" + bad["Crop_Year"].value_counts().head(5).to_string())