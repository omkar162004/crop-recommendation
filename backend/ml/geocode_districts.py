"""Step 3 of version 2: find a latitude and longitude for every district.

Two passes per district: Open-Meteo's place search first, then OpenStreetMap's Nominatim for
the ones it could not place. A result is only accepted if it lies in the right state.

Run from backend/ (several minutes; safe to stop and rerun, it resumes where it left off):
    python ml/geocode_districts.py
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets us import `app`

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from app import config  # noqa: E402
from app.services.errors import ServiceUnavailable  # noqa: E402
from app.services.http import get_json  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "data" / "processed" / "crop_production_clean.csv"
OUT = BASE / "data" / "processed" / "district_coordinates.csv"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "crop-recommender-hackathon/1.0 (student project)"}  # Nominatim requires this
DONE = {"state", "open-meteo", "nominatim"}  # rows with these results are kept on a rerun

# the crop file and the geocoder sometimes spell a state differently
STATE_ALIASES = {
    "orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
    "chattisgarh": "chhattisgarh", "nct of delhi": "delhi",
}
# the crop file predates these splits, so treat the pairs as the same region
REGION_GROUPS = [{"andhra pradesh", "telangana"}, {"jammu and kashmir", "ladakh"}]

# obvious misspellings or old names in the crop file (add more here if you spot them)
DISTRICT_FIXES = {
    "spsr nellore": "Nellore", "visakhapatanam": "Visakhapatnam", "kamrup metro": "Kamrup Metropolitan",
}


def norm(text) -> str:
    text = re.sub(r"[^a-z ]", " ", str(text).lower().replace("&", "and"))
    text = re.sub(r"\s+", " ", text).strip()
    return STATE_ALIASES.get(text, text)


def same_state(a, b) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return any(a in group and b in group for group in REGION_GROUPS)


def queries(district: str):
    name = DISTRICT_FIXES.get(norm(district), district.title())
    yield name
    short = re.sub(r"\b(north|south|east|west|central|urban|rural|metro|district)\b", "", name, flags=re.I)
    short = re.sub(r"\s+", " ", short).strip()
    if short and short.lower() != name.lower():
        yield short


def search(name: str) -> list:
    data = get_json(
        config.OPEN_METEO_GEOCODE_URL,
        {"name": name, "count": 10, "language": "en", "countryCode": "IN"},
        timeout=10,
    )
    return data.get("results") or []


def nominatim(query: str) -> list:
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "addressdetails": 1, "limit": 3, "countrycodes": "in"},
            headers=HEADERS, timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ServiceUnavailable(f"Nominatim failed: {exc}") from exc


def pick_open_meteo(results: list, state: str):
    """Only a result inside the right state is accepted; prefer district-level places."""
    in_state = [r for r in results if r.get("country_code") == "IN" and same_state(state, r.get("admin1", ""))]
    if not in_state:
        return None
    in_state.sort(key=lambda r: (r.get("feature_code") not in ("ADM2", "PPLA2"), -(r.get("population") or 0)))
    return in_state[0]


def pick_nominatim(results: list, state: str):
    for r in results:
        if same_state(state, r.get("address", {}).get("state", "")):
            return r
    return None


def lookup(state: str, district: str) -> dict:
    def found(lat, lon, name, region, source):
        return {"state": state, "district": district, "lat": float(lat), "lon": float(lon),
                "matched_name": name, "matched_state": region, "match": source}

    failed = False
    for query in queries(district):
        try:
            hit = pick_open_meteo(search(query), state)
        except ServiceUnavailable:
            failed = True
            continue
        if hit:
            return found(hit["latitude"], hit["longitude"], hit.get("name"), hit.get("admin1"), "open-meteo")

    name = next(queries(district))
    for query in (f"{name} district, {state}, India", f"{name}, {state}, India"):
        try:
            hit = pick_nominatim(nominatim(query), state)
        except ServiceUnavailable:
            failed = True
            hit = None
        time.sleep(1.1)  # Nominatim's policy: at most one request per second
        if hit:
            return found(hit["lat"], hit["lon"], hit.get("name"), hit.get("address", {}).get("state"), "nominatim")

    return {"state": state, "district": district, "match": "error" if failed else "none"}


def main():
    crops = pd.read_csv(SOURCE, usecols=["State_Name", "District_Name"]).drop_duplicates()
    todo = [(s, d.title()) for s, d in zip(crops["State_Name"], crops["District_Name"])]

    done = pd.DataFrame()
    if OUT.exists():
        previous = pd.read_csv(OUT)
        done = previous[previous["match"].isin(DONE)]  # everything else is retried
    finished = set(zip(done.get("state", []), done.get("district", [])))
    rows = done.to_dict("records")
    todo = [t for t in todo if t not in finished]
    print(f"Districts: {len(finished) + len(todo)}  (already done: {len(finished)}, to look up: {len(todo)})")

    for i, (state, district) in enumerate(todo, 1):
        rows.append(lookup(state, district))
        time.sleep(0.15)
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {i}/{len(todo)}")

    result = pd.DataFrame(rows)
    inside = result["lat"].between(6, 37.5) & result["lon"].between(68, 98)
    result.loc[result["lat"].notna() & ~inside, "match"] = "outside_india"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print(f"\nSaved {OUT}")
    print(result["match"].value_counts().to_string())
    review = result[~result["match"].isin(DONE)]
    if len(review):
        print("\nStill unplaced (fine to leave out, or add a fix to DISTRICT_FIXES):")
        print(review[["state", "district", "match"]].head(60).to_string(index=False))


if __name__ == "__main__":
    main()