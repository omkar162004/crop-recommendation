"""Step 2 of version 2: turn 575k raw rows into one clean row per district, season and crop.

Run from backend/:
    python ml/build_district_table.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "processed" / "crop_production_clean.csv"
OUT = BASE / "data" / "processed" / "district_season_crop.csv"

KEEP_SEASONS = ["Kharif", "Rabi", "Summer", "Whole Year"]  # Autumn and Winter are only ~4% of rows
FIRST_YEAR = 2000          # recent years only, so they match the weather archive we will use
MIN_YEARS = 3              # a crop must appear in a district-season for at least this many years
MIN_AREA_HA = 100          # ignore tiny plots: their yields are noisy
MIN_DISTRICTS = 30         # a crop-season needs this many districts for a trustworthy national median

df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
for col in ["State_Name", "District_Name", "Season", "Crop"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)  # fixes "Other  Rabi pulses"
print(f"Raw rows: {len(df):,}")

df = df[df["Season"].isin(KEEP_SEASONS) & (df["Crop_Year"] >= FIRST_YEAR)]
print(f"After season and year filter: {len(df):,}")

df = df[(df["Area"] > 0) & (df["Production"] > 0)]
print(f"After removing zero or missing area/production: {len(df):,}")

df = df.assign(state=df["State_Name"], district=df["District_Name"].str.title())

# One row per district, season and crop, averaged over the years.
# Yield = total production / total area, which weights big harvest years properly.
table = (
    df.groupby(["state", "district", "Season", "Crop"])
    .agg(years=("Crop_Year", "nunique"), area=("Area", "mean"),
         production=("Production", "sum"), area_total=("Area", "sum"))
    .reset_index()
)
table["yield"] = table["production"] / table["area_total"]
table = table[(table["years"] >= MIN_YEARS) & (table["area"] >= MIN_AREA_HA)]
print(f"District-season-crop rows after filters: {len(table):,}")

# Crop units differ (coconut in nuts, cotton and jute in bales), so never compare yields
# between crops. Compare each district only with other districts growing the SAME crop
# in the SAME season: 1.0 = typical, 1.5 = 50% better than typical.
keys = ["Season", "Crop"]
table = table[table.groupby(keys)["yield"].transform("size") >= MIN_DISTRICTS]
table["national_median_yield"] = table.groupby(keys)["yield"].transform("median")
table["relative_yield"] = (table["yield"] / table["national_median_yield"]).clip(0, 3)

# How much of the district's cropland (in that season) this crop occupies
table["area_share"] = table["area"] / table.groupby(["state", "district", "Season"])["area"].transform("sum")

table = table.rename(columns={"Season": "season", "Crop": "crop"})
table = table[["state", "district", "season", "crop", "years", "area", "area_share",
               "yield", "national_median_yield", "relative_yield"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT, index=False)

print(f"\nSaved {OUT}")
print("Districts:", table[["state", "district"]].drop_duplicates().shape[0])
print("Crops:", table["crop"].nunique())
print("\nDistricts per season:")
print(table.groupby("season")[["state", "district"]].apply(lambda g: len(g.drop_duplicates())).to_string())
print("\nTop 25 crops by number of districts:")
print(table.groupby("crop")[["state", "district"]].apply(lambda g: len(g.drop_duplicates()))
      .sort_values(ascending=False).head(25).to_string())
print("\nRelative yield spread (1.0 = typical):")
print(table["relative_yield"].describe(percentiles=[.1, .5, .9]).round(2).to_string())