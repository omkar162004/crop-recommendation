import json
import math
import time

from app import config


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load() -> dict:
    try:
        return json.loads(config.CACHE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {"climate": {}, "soil": {}}


def save(kind: str, lat: float, lon: float, value: dict) -> None:
    """Remember the last successful live reading (write-through), keyed by rounded location."""
    store = _load()
    store.setdefault(kind, {})[f"{round(lat, 2)},{round(lon, 2)}"] = {
        "lat": lat, "lon": lon, "value": value, "saved_at": time.time(),
    }
    config.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CACHE_FILE.write_text(json.dumps(store, indent=2))


def nearest(kind: str, lat: float, lon: float, max_km: float):
    """Closest saved reading within max_km, or None."""
    best = None
    for entry in _load().get(kind, {}).values():
        km = _haversine_km(lat, lon, entry["lat"], entry["lon"])
        if km <= max_km and (best is None or km < best["distance_km"]):
            best = {
                "value": entry["value"],
                "distance_km": round(km, 1),
                "age_hours": round((time.time() - entry["saved_at"]) / 3600, 1),
            }
    return best