import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]  # the backend/ folder
load_dotenv(BASE_DIR / ".env")

# --- files ---
MODELS_DIR = BASE_DIR / "models"
TRAIN_CSV = BASE_DIR / "data" / "processed" / "train.csv"
CACHE_FILE = BASE_DIR / "data" / "fallback" / "cached_readings.json"

# --- external APIs (all free, no key) ---
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# --- climate window ---
# The dataset's rainfall looks like a monthly total (its units are undocumented), so we
# average the last 90 days and express rainfall as "mm per 30 days".
CLIMATE_WINDOW_DAYS = 180
CLIMATE_END_LAG_DAYS = 6  # archive data lags real time by a few days
SOIL_DEPTHS = ["0-5cm"]  # averaged topsoil layers

# --- reliability ---
HTTP_TIMEOUT_CLIMATE = 15
HTTP_TIMEOUT_SOIL = 25  # SoilGrids can be slow
RETRY_ATTEMPTS = 3
CACHE_TTL_SECONDS = {"climate": 6 * 3600, "soil": 7 * 24 * 3600}
FALLBACK_MAX_KM = 50  # only reuse a cached reading taken this close to the request

# --- recommendation logic ---
LOW_CONFIDENCE_THRESHOLD = 0.5
RANGE_TOLERANCE = 0.05  # allow 5% beyond the training min/max before flagging out-of-range

# --- web ---
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]