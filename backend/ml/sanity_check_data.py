"""Does the crop data look like reality? Compares national averages with well-known figures.

Run from backend/:
    python ml/sanity_check_data.py
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "crop_production.csv"
df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
for col in ["State_Name", "District_Name", "Season", "Crop"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
df = df[(df["Area"] > 0) & (df["Production"] > 0) & df["Crop_Year"].between(2005, 2015)]

# Rough national averages in tonnes per hectare, from memory: check them against official sources.
EXPECTED = {
    "Rice": (2.0, 2.8), "Wheat": (2.5, 3.3), "Maize": (2.0, 3.0),
    "Sugarcane": (60, 75), "Potato": (17, 24), "Groundnut": (1.0, 1.5),
}

print("National yield 2005-2015 (total production / total area)\n")
for crop, (low, high) in EXPECTED.items():
    sub = df[df["Crop"] == crop]
    if sub.empty:
        print(f"{crop:12} not found in the file")
        continue
    value = sub["Production"].sum() / sub["Area"].sum()
    if low * 0.7 <= value <= high * 1.3:
        verdict = "matches, so production is in tonnes"
    elif low * 7 <= value <= high * 13:
        verdict = "about 10x too big, so production looks like quintals"
    else:
        verdict = "matches neither unit: investigate"
    print(f"{crop:12} {value:8.2f}   expected {low}-{high}   {verdict}")

print("\nTop 5 states by average yearly area")
for crop in ("Rice", "Wheat", "Cotton(lint)"):
    sub = df[df["Crop"] == crop]
    if sub.empty:
        continue
    top = (sub.groupby(["State_Name", "Crop_Year"])["Area"].sum()
           .groupby("State_Name").mean().sort_values(ascending=False).head(5))
    print(f"\n{crop}:\n" + top.round(0).to_string())