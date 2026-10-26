"""Reality check: does the model pick crops that are really grown in well-known regions?

Run from the backend/ folder:
    python ml/validate_regions.py          (uses config.CLIMATE_WINDOW_DAYS)
    python ml/validate_regions.py 90       (try a different climate window)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets us import `app`

import pandas as pd  # noqa: E402

from app import config  # noqa: E402
from app.services.climate import fetch_climate  # noqa: E402
from app.services.errors import NoDataError, ServiceUnavailable  # noqa: E402
from app.services.predictor import Predictor  # noqa: E402

# (place, latitude, longitude, crops from the model's 22 that are genuinely grown there)
# These are rough agronomic judgements, not official statistics. Check and edit them.
CASES = [
    ("Kolkata", 22.57, 88.36, {"rice", "jute"}),
    ("Guwahati", 26.14, 91.74, {"rice", "jute"}),
    ("Kochi", 9.93, 76.27, {"coconut", "banana", "rice"}),
    ("Madikeri (Coorg)", 12.42, 75.74, {"coffee"}),
    ("Chikmagalur", 13.32, 75.77, {"coffee"}),
    ("Shimla", 31.10, 77.17, {"apple"}),
    ("Srinagar", 34.08, 74.80, {"apple"}),
    ("Nagpur", 21.15, 79.09, {"cotton", "orange", "pigeonpeas"}),
    ("Jalgaon", 21.01, 75.56, {"banana", "cotton"}),
    ("Nashik", 20.00, 73.79, {"grapes", "pomegranate"}),
    ("Solapur", 17.66, 75.91, {"pomegranate", "pigeonpeas", "chickpea"}),
    ("Lucknow", 26.85, 80.95, {"mango", "rice"}),
    ("Ludhiana", 30.90, 75.86, {"rice", "maize"}),
    ("Ratnagiri", 16.99, 73.31, {"mango", "coconut", "rice"}),
    ("Coimbatore", 11.02, 76.96, {"coconut", "banana", "cotton"}),
    ("Ahmedabad", 23.02, 72.57, {"cotton"}),
    ("Indore", 22.72, 75.86, {"chickpea", "maize"}),
    ("Jodhpur", 26.24, 73.02, {"mothbeans", "mungbean"}),
]


def main():
    if len(sys.argv) > 1:
        config.CLIMATE_WINDOW_DAYS = int(sys.argv[1])

    predictor = Predictor()
    typical_ph = round(float(pd.read_csv(config.TRAIN_CSV)["ph"].median()), 2)
    print(f"Climate window: {config.CLIMATE_WINDOW_DAYS} days. Soil pH fixed at {typical_ph} "
          "(the dataset median) because this test checks the climate side.\n")

    tested = top1 = top3 = 0
    for place, lat, lon, accepted in CASES:
        try:
            climate = fetch_climate(lat, lon)
        except (ServiceUnavailable, NoDataError) as exc:
            print(f"{place:18} skipped ({str(exc)[:60]})")
            continue

        result = predictor.predict({**climate, "ph": typical_ph})
        picks = [p["crop"] for p in result["predictions"]]
        tested += 1
        hit1 = bool(picks) and picks[0] in accepted
        hit3 = any(c in accepted for c in picks)
        top1 += hit1
        top3 += hit3

        shown = ", ".join(picks) if picks else "no confident match (out of range)"
        mark = "TOP-1" if hit1 else ("top-3" if hit3 else "miss ")
        print(f"{mark}  {place:18} model says: {shown:42} expected one of: {', '.join(sorted(accepted))}")

    if tested:
        print(f"\nPlaces tested: {tested}")
        print(f"Top-1 hit rate: {top1}/{tested} = {top1 / tested:.0%}")
        print(f"Top-3 hit rate: {top3}/{tested} = {top3 / tested:.0%}")


if __name__ == "__main__":
    main()