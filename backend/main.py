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

app = FastAPI(title="Weather Nowcasting Live API")

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


def _get_hazard_cfg(hazard: str):
    cfg = HAZARDS.get(hazard)
    if cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown hazard '{hazard}'. Use 'thunderstorm' or 'cloudburst'.",
        )
    return cfg


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/live/{hazard}")
def get_live_prediction(hazard: str):
    """
    Fetch live data, run the trained model, log the result, and return it.
    """
    cfg = _get_hazard_cfg(hazard)
    result = cfg["run"]()

    if result["error"]:
        raise HTTPException(status_code=502, detail=result["error"])

    cfg["append_log"](districts=result["districts"], latest_time=result["latest_time"])
    return result


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
