import time
from concurrent.futures import ThreadPoolExecutor

from app import config
from app.services import cache, climate, health, soil
from app.services.errors import NoDataError, ServiceUnavailable

_MEMORY = {}  # (kind, rounded location) -> (timestamp, value)


def _key(lat, lon):
    return f"{round(lat, 2)},{round(lon, 2)}"


def _get_one(kind, fetch_fn, lat, lon, simulate_failure):
    """Return (value, source_info). Live first, then the last-known-good cache."""
    if not simulate_failure:
        hit = _MEMORY.get((kind, _key(lat, lon)))
        if hit and time.time() - hit[0] < config.CACHE_TTL_SECONDS[kind]:
            return hit[1], {"status": "live"}

        started = time.perf_counter()
        try:
            value = fetch_fn(lat, lon)
        except NoDataError:
            health.record(kind, True, (time.perf_counter() - started) * 1000, "no data at location")
            raise
        except ServiceUnavailable as exc:
            health.record(kind, False, (time.perf_counter() - started) * 1000, str(exc)[:200])
        else:
            health.record(kind, True, (time.perf_counter() - started) * 1000)
            _MEMORY[(kind, _key(lat, lon))] = (time.time(), value)
            cache.save(kind, lat, lon, value)
            return value, {"status": "live"}
    else:
        health.record(kind, False, 0, "simulated failure")

    fallback = cache.nearest(kind, lat, lon, config.FALLBACK_MAX_KM)
    if fallback is None:
        raise ServiceUnavailable(
            f"Live {kind} data is unavailable and there is no saved reading within "
            f"{config.FALLBACK_MAX_KM} km. Try again, or look up this place once while online."
        )
    return fallback["value"], {
        "status": "cached",
        "distance_km": fallback["distance_km"],
        "age_hours": fallback["age_hours"],
    }


def get_readings(lat: float, lon: float, simulate_failure: bool = False, ph_override=None) -> dict:
    """Fetch climate and soil in parallel; each one falls back independently.

    If the user typed their own soil pH (ph_override), the soil services are skipped entirely.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        climate_future = pool.submit(_get_one, "climate", climate.fetch_climate, lat, lon, simulate_failure)
        soil_future = None
        if ph_override is None:
            soil_future = pool.submit(_get_one, "soil", soil.fetch_soil, lat, lon, simulate_failure)

        climate_value, climate_source = climate_future.result()
        if soil_future is None:
            soil_value, soil_source = {"ph": float(ph_override)}, {"status": "user"}
        else:
            soil_value, soil_source = soil_future.result()

    return {
        "readings": {**climate_value, **soil_value},
        "sources": {"climate": climate_source, "soil": soil_source},
        "used_fallback": "cached" in (climate_source["status"], soil_source["status"]),
    }