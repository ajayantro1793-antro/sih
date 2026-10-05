"""
FastAPI backend for the Live Prediction dashboard.

Wraps the EXACT same functions the Streamlit app and
notebooks/03_live_predict.py / 03b_live_predict_cloudburst.py already
use -- no model/inference logic is duplicated here, only exposed over HTTP.

Run from your project root (D:\\SIH):
    cd D:\\SIH
    venv\\Scripts\\Activate.ps1
    uvicorn backend.main:app --reload --port 8000
"""

import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from configs.config import (
    MODELS_DIR, REGION_NAME, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    LEAD_TIME_HOURS, SEQUENCE_TIMESTEPS,
)
from src.live_predict_core import run_live_prediction
from src.live_predict_core_cloudburst import run_live_cloudburst_prediction
from src.live_log import append_log as append_log_thunderstorm, load_log as load_log_thunderstorm
from src.live_log_cloudburst import append_log as append_log_cloudburst, load_log as load_log_cloudburst
from src.thresholds import get_district_names

# ---------------------------------------------------------------------------
# Per-hazard prediction cache
# ---------------------------------------------------------------------------

HAZARDS = {
    "thunderstorm": {
        "run": run_live_prediction,
        "append_log": append_log_thunderstorm,
        "load_log": load_log_thunderstorm,
        "metadata_filename": "model_metadata.json",
    },
    "cloudburst": {
        "run": run_live_cloudburst_prediction,
        "append_log": append_log_cloudburst,
        "load_log": load_log_cloudburst,
        "metadata_filename": "cloudburst_model_metadata.json",
    },
}

_PREDICTION_CACHE: dict = {}       # hazard -> last good result dict
_CACHE_TS: dict = {}               # hazard -> unix timestamp of last successful fetch
_REFRESH_LOCK = threading.Lock()   # prevents two background loops from running
BACKGROUND_INTERVAL_S = 20 * 60   # re-fetch every 20 minutes


# ---------------------------------------------------------------------------
# Background scheduler — runs the full pipeline once, stores result in cache
# ---------------------------------------------------------------------------

def _refresh_hazard(hazard: str, cfg: dict) -> None:
    """Run one prediction cycle for `hazard` and update the cache."""
    print(f"[bg] Starting refresh for '{hazard}'...")
    try:
        result = cfg["run"]()
        if result.get("error"):
            print(f"[bg] '{hazard}' prediction returned error: {result['error']}")
            return
        _PREDICTION_CACHE[hazard] = result
        _CACHE_TS[hazard] = time.time()
        print(f"[bg] '{hazard}' cache updated — {len(result.get('districts', []))} districts.")
        try:
            cfg["append_log"](districts=result["districts"], latest_time=result["latest_time"])
        except Exception as log_err:
            print(f"[bg] Log append warning for '{hazard}': {log_err}")
    except Exception as exc:
        print(f"[bg] '{hazard}' refresh failed: {exc}")


def _background_loop() -> None:
    """
    Runs once at startup (staggered by 5 s between hazards so Open-Meteo
    never sees two simultaneous 4-batch requests), then repeats every
    BACKGROUND_INTERVAL_S seconds.

    Because both hazards call fetch_live_atmospheric_data() internally,
    and that function now has a 15-minute in-memory cache + thread lock,
    the second hazard's fetch is served from cache (0 extra HTTP requests).
    """
    hazard_list = list(HAZARDS.keys())
    while True:
        for idx, hazard in enumerate(hazard_list):
            if idx > 0:
                time.sleep(5)   # stagger to avoid simultaneous Open-Meteo hits
            _refresh_hazard(hazard, HAZARDS[hazard])
        print(f"[bg] All hazards refreshed. Next cycle in {BACKGROUND_INTERVAL_S // 60} min.")
        time.sleep(BACKGROUND_INTERVAL_S)


# ---------------------------------------------------------------------------
# FastAPI lifespan — start background thread on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background refresh thread when the server starts."""
    t = threading.Thread(target=_background_loop, daemon=True, name="bg-refresh")
    t.start()
    print("[startup] Background refresh thread started.")
    yield
    print("[shutdown] Background refresh thread will stop (daemon).")


# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Weather Nowcasting Live API", lifespan=lifespan)

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://sih-ruby-five.vercel.app",
        *frontend_origins,
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hazard_cfg(hazard: str):
    cfg = HAZARDS.get(hazard)
    if cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown hazard '{hazard}'. Use 'thunderstorm' or 'cloudburst'.",
        )
    return cfg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    cache_status = {
        h: {
            "cached": h in _PREDICTION_CACHE,
            "age_seconds": int(time.time() - _CACHE_TS[h]) if h in _CACHE_TS else None,
        }
        for h in HAZARDS
    }
    return {"status": "ok", "cache": cache_status}


@app.get("/api/live/{hazard}")
def get_live_prediction(hazard: str):
    """
    Returns the most recent cached prediction for this hazard.

    The actual Open-Meteo fetch and model run happen in a background thread
    every 20 minutes — this endpoint never blocks on a live HTTP call.
    If the cache is empty (server just started and first bg fetch hasn't
    finished yet), it falls back to triggering one synchronous fetch and
    caches the result.
    """
    _get_hazard_cfg(hazard)   # validate hazard name

    if hazard in _PREDICTION_CACHE:
        result = _PREDICTION_CACHE[hazard]
        age = int(time.time() - _CACHE_TS.get(hazard, 0))
        result = dict(result)          # shallow copy so we don't mutate the cache
        result["cache_age_seconds"] = age
        return result

    # Cache miss — server just started; do a synchronous fetch once
    print(f"[api] Cache miss for '{hazard}', triggering synchronous fetch...")
    _refresh_hazard(hazard, HAZARDS[hazard])

    if hazard in _PREDICTION_CACHE:
        return dict(_PREDICTION_CACHE[hazard])

    raise HTTPException(
        status_code=503,
        detail=(
            "Prediction not yet available — the server is still warming up. "
            "Please wait ~60 seconds and try again."
        ),
    )


@app.get("/api/log/{hazard}")
def get_log(hazard: str, limit: int = 200):
    """
    Recent logged predictions for this hazard (most recent last).
    """
    cfg = _get_hazard_cfg(hazard)
    df = cfg["load_log"]()
    if len(df) == 0:
        return {"rows": []}

    # Standard JSON has no representation for NaN/Infinity, and rows from
    # earlier testing may contain them -- sanitize to None (-> JSON null)
    # rather than letting the encoder reject the whole response.
    df = df.tail(limit).replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notnull(df), None)
    return {"rows": df.to_dict(orient="records")}


@app.get("/api/metadata/{hazard}")
def get_metadata(hazard: str):
    """
    The real metadata your training script wrote -- feature list, district
    list, sequence length, lead time, sample counts, train/val/test date
    ranges, and per-district test metrics.
    """
    cfg = _get_hazard_cfg(hazard)
    path = os.path.join(MODELS_DIR, cfg["metadata_filename"])
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"{cfg['metadata_filename']} not found -- train this model first "
                   f"(02_train_model.py or 02b_train_cloudburst_model.py).",
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/config")
def get_config():
    """
    Real, static system parameters from configs/config.py.
    """
    return {
        "region_name": REGION_NAME,
        "lat_min": LAT_MIN, "lat_max": LAT_MAX,
        "lon_min": LON_MIN, "lon_max": LON_MAX,
        "lead_time_hours": LEAD_TIME_HOURS,
        "sequence_timesteps": SEQUENCE_TIMESTEPS,
        "n_districts": len(get_district_names()),
    }
