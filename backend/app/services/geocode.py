from app import config
from app.services.errors import NoDataError
from app.services.http import get_json


def geocode(query: str) -> dict:
    data = get_json(
        config.OPEN_METEO_GEOCODE_URL,
        {"name": query, "count": 1, "language": "en"},
        timeout=10,
    )
    results = data.get("results") or []
    if not results:
        raise NoDataError(f'Could not find a place called "{query}".')
    hit = results[0]
    name = ", ".join(p for p in [hit.get("name"), hit.get("admin1"), hit.get("country")] if p)
    return {"name": name, "lat": hit["latitude"], "lon": hit["longitude"]}