from app import config
from app.services.errors import NoDataError, ServiceUnavailable
from app.services.http import get_json


def fetch_soil(lat: float, lon: float) -> dict:
    """Topsoil pH from SoilGrids, converted to the real pH scale."""
    params = [("lon", lon), ("lat", lat), ("property", "phh2o"), ("value", "mean")]
    params += [("depth", d) for d in config.SOIL_DEPTHS]

    data = get_json(config.SOILGRIDS_URL, params, timeout=config.HTTP_TIMEOUT_SOIL)

    try:
        layers = data["properties"]["layers"]
        layer = next((l for l in layers if l.get("name") == "phh2o"), None)
        if layer is None:
            raise NoDataError("SoilGrids returned no pH layer for this location.")

        # SoilGrids returns whole numbers (pH x 10); the response tells us the divisor
        d_factor = (layer.get("unit_measure") or {}).get("d_factor") or 10
        values = [
            depth.get("values", {}).get("mean")
            for depth in layer.get("depths", [])
        ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ServiceUnavailable(f"Unexpected SoilGrids response: {exc}") from exc

    values = [v for v in values if v is not None]
    if not values:
        raise NoDataError("No soil data for this location (it may be open water or built-up land).")

    ph = sum(values) / len(values) / d_factor
    if not 2.0 <= ph <= 11.0:  # sanity check catches scaling mistakes
        raise ServiceUnavailable(f"Implausible pH value {ph:.2f} from SoilGrids.")
    return {"ph": round(ph, 2)}