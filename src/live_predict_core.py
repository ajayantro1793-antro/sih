"""
Shared live-prediction logic, used by both notebooks/03_live_predict.py
(command-line) and app.py (dashboard) — keeping one implementation avoids
the two drifting out of sync.

UPDATED to support terrain features (elevation/slope/TWI). Terrain is
STATIC -- loaded from the precomputed grid (see
notebooks/01c_add_terrain_features.py) instead of being fetched live.
Falls back cleanly to "no terrain" if that script hasn't been run yet
(i.e. feature_names.npy has no terrain names in it) -- so this is safe
to drop in even before you've generated the terrain grid.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_PROCESSED_DIR, MODELS_DIR, SEQUENCE_TIMESTEPS
from src.thresholds import get_alert_thresholds, get_district_names
from src.live_fetch import fetch_live_atmospheric_data
from src.features import calculate_all_features
from src.model import load_multi_district_model


def run_live_prediction():
    """
    Fetch live data for the whole state grid, compute features, normalize
    with training statistics, and run the trained multi-district model.

    Returns:
        dict with keys:
          districts: list of per-district dicts
              {name, probability, threshold, alert}
          latest_time, n_timesteps_fetched, error (str or None)
    """
    district_names = [str(n) for n in get_district_names()]
    thresholds = get_alert_thresholds()

    result = {
        "districts": [],
        "latest_time": None,
        "n_timesteps_fetched": None,
        "features": {},
        "error": None,
    }

    try:
        live_ds = fetch_live_atmospheric_data(timesteps_needed=SEQUENCE_TIMESTEPS)
        result["n_timesteps_fetched"] = live_ds.sizes["time"]

        if live_ds.sizes["time"] < SEQUENCE_TIMESTEPS:
            result["error"] = (
                f"Only got {live_ds.sizes['time']} timesteps, need {SEQUENCE_TIMESTEPS}. "
                f"Open-Meteo may not have provided enough history."
            )
            return result

        features_ds = calculate_all_features(live_ds)
        saved_feature_names = list(np.load(
            os.path.join(DATA_PROCESSED_DIR, "feature_names.npy"), allow_pickle=True
        ))

        # Terrain features (elevation/slope/TWI) are STATIC -- loaded from
        # the precomputed grid instead of being fetched live.
        terrain_grid_path = os.path.join(DATA_PROCESSED_DIR, "terrain_grid.npy")
        terrain_names_path = os.path.join(DATA_PROCESSED_DIR, "terrain_feature_names.npy")
        terrain_names, terrain_grid = [], None
        if os.path.exists(terrain_grid_path) and os.path.exists(terrain_names_path):
            terrain_names = list(np.load(terrain_names_path, allow_pickle=True))
            terrain_grid = np.load(terrain_grid_path)  # (H, W, n_terrain)

        feature_arrays = []
        for name in saved_feature_names:
            if name in terrain_names:
                feature_arrays.append(terrain_grid[..., terrain_names.index(name)])  # (H, W), no time dim yet
            else:
                feature_arrays.append(features_ds[name].values)  # (time, H, W)

        min_time = min(arr.shape[0] for arr in feature_arrays if arr.ndim == 3)
        aligned_arrays = []
        for arr in feature_arrays:
            if arr.ndim == 3:
                aligned_arrays.append(arr[-min_time:])
            else:
                # Static terrain array -- broadcast across the same time window.
                aligned_arrays.append(np.broadcast_to(arr[np.newaxis, :, :], (min_time,) + arr.shape))
        stacked = np.stack(aligned_arrays, axis=-1)
        stacked = np.nan_to_num(stacked, nan=0.0)

        # Real current-conditions snapshot: area-mean (over the whole grid)
        # of the MOST RECENT timestep's raw (pre-normalization) values, one
        # per feature -- these are the model's actual inputs, not invented
        # telemetry. Reuses the array already computed above at ~zero extra
        # cost (no second live fetch).
        result["features"] = {
            name: float(np.nanmean(stacked[-1, ..., i]))
            for i, name in enumerate(saved_feature_names)
        }

        if stacked.shape[0] < SEQUENCE_TIMESTEPS:
            result["error"] = (
                f"Only {stacked.shape[0]} timesteps after feature engineering, need {SEQUENCE_TIMESTEPS}."
            )
            return result

        mean = np.load(os.path.join(DATA_PROCESSED_DIR, "feature_mean.npy"))
        std = np.load(os.path.join(DATA_PROCESSED_DIR, "feature_std.npy"))
        # mean/std were saved with shape (1,1,1,1,F) to broadcast against
        # 5-D BATCHED training arrays (samples,timesteps,lat,lon,features).
        # Live inference works with a single 4-D sequence
        # (timesteps,lat,lon,features) with no batch dimension yet --
        # applying the 5-D stats directly makes numpy insert a spurious
        # extra dimension (confirmed: produces a 6-D array once the
        # newaxis below is added, which model.predict() rejects). Squeeze
        # the leading axis first so shapes broadcast correctly.
        mean = np.squeeze(mean, axis=0)
        std = np.squeeze(std, axis=0)
        stacked_norm = (stacked - mean) / std

        X_live = stacked_norm[-SEQUENCE_TIMESTEPS:][np.newaxis, ...]

        district_masks = np.load(os.path.join(DATA_PROCESSED_DIR, "district_masks.npy"))
        model_path = os.path.join(MODELS_DIR, "thunderstorm_mvp.h5")
        # Weights-only file -- rebuild the architecture from district_masks.npy
        # first, then load weights into it. See src/model.py's
        # get_callbacks()/load_multi_district_model() docstrings for why.
        model = load_multi_district_model(
            weights_path=model_path,
            input_shape=X_live.shape[1:],
            district_masks=district_masks,
            name_prefix="thunderstorm",
        )
        probabilities = model.predict(X_live, verbose=0)[0]  # (n_districts,)

        if len(probabilities) != len(district_names):
            result["error"] = (
                f"Model outputs {len(probabilities)} values but "
                f"{len(district_names)} districts are configured -- was the "
                f"model trained on a different district_masks.npy? Re-run "
                f"01_build_dataset.py and 02_train_model.py together."
            )
            return result

        result["districts"] = [
            {
                "name": name,
                "probability": float(prob),
                "threshold": float(thr),
                "alert": bool(prob >= thr),
            }
            for name, prob, thr in zip(district_names, probabilities, thresholds)
        ]
        result["latest_time"] = str(live_ds.time.values[-1])

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


if __name__ == "__main__":
    result = run_live_prediction()
    print(result)
