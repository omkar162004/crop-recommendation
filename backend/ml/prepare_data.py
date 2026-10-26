from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]          # the backend/ folder
RAW = BASE / "data" / "raw" / "crop_recommendation.csv"
OUT = BASE / "data" / "processed" / "train.csv"

EXPECTED_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

df = pd.read_csv(RAW)

print("Shape:", df.shape)                            # expect (2200, 8)
print("Columns:", df.columns.tolist())
print(df.isna().sum().sum(), "nulls,", df.duplicated().sum(), "duplicates")
print(df["label"].value_counts())                    # expect 22 crops x 100 rows

missing = set(EXPECTED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

# Keep ALL columns: the full model uses N, P, K and the basic model ignores them
df[EXPECTED_COLUMNS].to_csv(OUT, index=False)
print("Saved", OUT, "with columns:", EXPECTED_COLUMNS)