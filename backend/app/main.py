from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config
from app.services import geocode as geocode_service
from app.services import health, readings
from app.services.errors import NoDataError, ServiceUnavailable
from app.services.predictor import Predictor

app = FastAPI(title="Groundtruth crop recommendation API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # tighten via the CORS_ORIGINS env var before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


class NPK(BaseModel):
    N: float = Field(ge=0)
    P: float = Field(ge=0)
    K: float = Field(ge=0)


class PredictRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    name: Optional[str] = None
    npk: Optional[NPK] = None  # only sent when the user ticks "I know my soil N, P, K"
    ph: Optional[float] = Field(default=None, ge=3, le=10)  # user-typed soil pH; skips the soil APIs
    simulate_failure: bool = False  # demo switch: pretend both APIs are down


@app.get("/health")
def health_check():
    predictor = get_predictor()
    return {"status": "ok", "models": predictor.info(), "external_apis": health.summary()}


@app.get("/meta")
def meta():
    """What the frontend needs to build the N-P-K inputs."""
    predictor = get_predictor()
    return {
        "npk_ranges": predictor.npk_ranges(),
        "npk_note": "Values on the training dataset's scale (its units are not documented).",
        "rainfall_basis": f"mm per 30 days, averaged over the last {config.CLIMATE_WINDOW_DAYS} days",
    }


@app.get("/geocode")
def geocode(q: str = Query(min_length=2)):
    try:
        return geocode_service.geocode(q)
    except NoDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ServiceUnavailable:
        raise HTTPException(status_code=503, detail="The place-search service is unavailable. Enter coordinates instead.")


@app.post("/predict")
def predict(req: PredictRequest):
    predictor = get_predictor()

    npk = None
    if req.npk is not None:
        npk = req.npk.model_dump()
        for key, (low, high) in predictor.npk_ranges().items():
            if not 0 <= npk[key] <= high:
                raise HTTPException(
                    status_code=422,
                    detail=f"{key} must be between 0 and {high:g} (the dataset's scale).",
                )

    try:
        data = readings.get_readings(req.lat, req.lon, req.simulate_failure, req.ph)
    except NoDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result = predictor.predict(data["readings"], npk)
    return {
        "location": {"name": req.name, "lat": req.lat, "lon": req.lon},
        "readings": data["readings"],
        "sources": data["sources"],
        "used_fallback": data["used_fallback"],
        **result,
    }