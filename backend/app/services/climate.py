from datetime import date, timedelta

import pandas as pd

from app import config
from app.services.errors import NoDataError, ServiceUnavailable
from app.services.http import get_json


def fetch_climate(lat: float, lon: float) -> dict:
    """Average temperature/humidity and monthly-equivalent rainfall over the recent window."""
    end = date.today() - timedelta(days=config.CLIMATE_END_LAG_DAYS)
    start = end - timedelta(days=config.CLIMATE_WINDOW_DAYS - 1)

    data = get_json(
        config.OPEN_METEO_ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation",
            "timezone": "UTC",
        },
        timeout=config.HTTP_TIMEOUT_CLIMATE,
    )

    try:
        hourly = data["hourly"]
        temp = pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce")
        hum = pd.to_numeric(pd.Series(hourly["relative_humidity_2m"]), errors="coerce")
        rain = pd.to_numeric(pd.Series(hourly["precipitation"]), errors="coerce")
    except (KeyError, TypeError) as exc:
        raise ServiceUnavailable(f"Unexpected Open-Meteo response: {exc}") from exc

    if temp.notna().sum() == 0 or hum.notna().sum() == 0 or rain.notna().sum() == 0:
        raise NoDataError("No recent climate data is available for this location.")

    # scale rainfall by the number of hours that actually have data, so gaps don't undercount
    days_with_data = rain.notna().sum() / 24
    monthly_rain = rain.sum() / days_with_data * 30

    return {
        "temperature": round(float(temp.mean()), 1),
        "humidity": round(float(hum.mean()), 1),
        "rainfall": round(float(monthly_rain), 1),
    }