"""
Central configuration for the weather nowcasting MVP.
Edit these values as your project develops — everything else imports from here.
"""

import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# --- Study Region: Tamil Nadu (full state) ---
# Current study region. Covers the Tamil Nadu bounding box at
# approximately 0.25-degree ERA5/Open-Meteo resolution
# (~23 x 17 = ~391 grid points).
REGION_NAME = "TamilNadu"
LAT_MIN, LAT_MAX = 8.0, 13.5
LON_MIN, LON_MAX = 76.5, 80.5

# --- Sequence length for the LSTM ---
# With HOURLY data (see below), 6 timesteps would only be a 6-hour lookback
# -- too short for meaningful storm buildup context. Use 24 timesteps
# (24-hour lookback) instead. Update notebooks/01_build_dataset.py and
# notebooks/00_quick_start_synthetic_test.py calls to match this if you
# change it here.
SEQUENCE_TIMESTEPS = 24

# --- Data Settings ---
# NOTE: the actual levels fetched/used by the pipeline are hardcoded as
# LEVELS = [850, 500] in notebooks/download_open_meteo.py and
# src/live_fetch.py (kept hardcoded there, deliberately not read from
# here, so training and live data can never silently diverge -- see
# those files' docstrings). This list is currently unused by the
# pipeline; update it only if you also update both hardcoded copies.
PRESSURE_LEVELS = [850, 500]

# --- Labeling ---
# Prediction horizon: predict storm occurrence this many hours ahead
LEAD_TIME_HOURS = 3

# --- Model Settings ---
RANDOM_SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

# --- Alert Threshold ---
# Fallback only. After training, 02_train_model.py writes the validation-tuned
# threshold to data/processed/alert_threshold.npy. Live inference and Streamlit
# load that saved threshold automatically.
STORM_PROBABILITY_THRESHOLD = 0.50
