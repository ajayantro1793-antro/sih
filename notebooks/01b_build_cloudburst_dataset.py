"""
Build the cloudburst training dataset -- MULTI-DISTRICT version.

Matches the real contract your pipeline uses (inferred from
src/model.py's build_multi_district_model(), src/districts.py, and
notebooks/02_train_model.py's load_processed_data() /
normalize_from_train()):

  - X is saved RAW / un-normalized, shape (N, T, H, W, F).
    Normalization is fit on the TRAIN split only, at train time
    (02_train_model.py's normalize_from_train()) -- doing it here
    instead would leak val/test statistics into training.
  - y is (N, n_districts): one label column per district, built from
    src.districts.build_district_masks() + a per-district rainfall
    percentile (src/labels_cloudburst.py's build_district_cloudburst_labels,
    which wraps your real build_district_storm_labels() with a
    stricter percentile).
  - feature_names.npy, sample_times.npy, district_names.npy,
    district_masks.npy are saved alongside X/y, exactly like your
    thunderstorm pipeline's artifacts.

Everything here is cloudburst_-prefixed so nothing overwrites the
thunderstorm X.npy/y.npy/district_masks.npy/etc.

ASSUMPTION FLAGGED: I still don't have your actual 01_build_dataset.py.
PRESSURE_FILE_PATTERN / PRECIP_FILE_PATTERN below are inferred (same
guess as before); the "X is raw, normalize at train time" design is
inferred from what 02_train_model.py's load_processed_data() and
normalize_from_train() expect it to receive. If your real
01_build_dataset.py does something different, let me know.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SEQUENCE_TIMESTEPS
from src.data_loader import load_multiple_files, subset_region
from src.features import calculate_all_features
from src.features_cloudburst import add_cloudburst_features
from src.labels_cloudburst import build_district_cloudburst_labels
from src.districts import build_district_masks, DISTRICT_NAMES

# ADJUST these two if your actual raw filenames differ
PRESSURE_FILE_PATTERN = "openmeteo_pressure_*.nc"
PRECIP_FILE_PATTERN = "openmeteo_precip_*.nc"


def safe_subset_region(ds):
    """
    subset_region() crops with .sel(lat=slice(lat_min, lat_max)), which
    only returns points if `lat` is stored ascending. Sorting ascending
    first makes the slice work regardless of the raw file's native
    ordering (see the earlier fix applied here for the thunderstorm-shaped
    version of this script).
    """
    ds = ds.sortby("lat").sortby("lon")
    return subset_region(ds)


def build_sliding_windows(feature_array: np.ndarray, labels: np.ndarray, seq_len: int):
    """
    X[i] = feature_array[i : i+seq_len]. `labels` is (time, n_districts);
    the window's target is labels[end-1] (all districts at once). Dropped
    if ANY district is NaN at that time index -- in practice every
    district's forward-window cutoff lands on the same source timestamps,
    so they go NaN together near the end of the record.
    """
    n_time = feature_array.shape[0]
    X, y, kept_idx = [], [], []

    for start in range(n_time - seq_len + 1):
        end = start + seq_len
        label_row = labels[end - 1]
        if np.isnan(label_row).any():
            continue
        X.append(feature_array[start:end])
        y.append(label_row)
        kept_idx.append(end - 1)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(kept_idx),
    )


def main():
    # 1. Pressure-level data -> engineered features (raw, NOT normalized)
    ds = load_multiple_files(DATA_RAW_DIR, pattern=PRESSURE_FILE_PATTERN)
    ds = safe_subset_region(ds)

    feature_ds = calculate_all_features(ds)               # your existing 6 features
    feature_ds = add_cloudburst_features(ds, feature_ds)   # + moisture_flux_convergence, humidity_500

    feature_names = list(feature_ds.data_vars)
    print(f"Cloudburst feature set ({len(feature_names)}): {feature_names}")

    # Align all features to the shortest common time length (iwv_trend
    # loses one step to .diff()).
    min_time = min(feature_ds[name].sizes["time"] for name in feature_names)
    stacked = np.stack(
        [feature_ds[name].isel(time=slice(0, min_time)).values for name in feature_names],
        axis=-1,
    )  # (time, lat, lon, n_features) -- RAW, matches your real pipeline
    stacked = np.nan_to_num(stacked, nan=0.0)

    # 2. District masks, built from THIS dataset's own lat/lon grid
    district_masks = build_district_masks(feature_ds.lat.values, feature_ds.lon.values)
    district_names = np.array(DISTRICT_NAMES)
    print(f"Districts ({len(district_names)}): {list(district_names)}")

    # 3. Precipitation data -> per-district cloudburst labels
    precip_ds = load_multiple_files(DATA_RAW_DIR, pattern=PRECIP_FILE_PATTERN)
    precip_ds = safe_subset_region(precip_ds)
    labels, thresholds_mm = build_district_cloudburst_labels(precip_ds, district_masks)
    labels = labels[:min_time]
    print("Per-district cloudburst rainfall thresholds (mm): " + str(
        dict(zip(district_names.tolist(), [round(t, 2) for t in thresholds_mm]))
    ))

    # 4. Sliding windows, using your configured sequence length
    X, y, kept_idx = build_sliding_windows(stacked, labels, SEQUENCE_TIMESTEPS)
    sample_times = ds["time"].values[:min_time][kept_idx]

    print(f"Cloudburst dataset: X {X.shape}, y {y.shape}")
    for d, name in enumerate(district_names):
        print(f"  {str(name):16s} positive={y[:, d].mean():.2%}")

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(DATA_PROCESSED_DIR, "cloudburst_X.npy"), X)
    np.save(os.path.join(DATA_PROCESSED_DIR, "cloudburst_y.npy"), y)
    np.save(os.path.join(DATA_PROCESSED_DIR, "cloudburst_feature_names.npy"), np.array(feature_names))
    np.save(os.path.join(DATA_PROCESSED_DIR, "cloudburst_sample_times.npy"), sample_times)
    np.save(os.path.join(DATA_PROCESSED_DIR, "cloudburst_district_names.npy"), district_names)
    np.save(os.path.join(DATA_PROCESSED_DIR, "cloudburst_district_masks.npy"), district_masks)

    print(f"Saved cloudburst_*.npy to {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
