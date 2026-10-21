import pandas as pd

df = pd.read_csv("data/raw/crop_recommendation.csv")

print(df.shape)                        # expect (2200, 8)
print(df.columns.tolist())
print(df.isna().sum().sum(), "nulls,", df.duplicated().sum(), "duplicates")
print(df["label"].value_counts())      # expect 22 crops x 100 rows

FEATURES = ["temperature", "humidity", "rainfall", "ph"]
df[FEATURES + ["label"]].to_csv("data/processed/train.csv", index=False)
print("Saved data/processed/train.csv")