"""
01_build_dataset.py

Build the real training dataset from Open-Meteo (Historical Forecast API):
    raw Open-Meteo -> features -> forward rainfall-proxy labels -> sliding windows.

Training and live prediction (src/live_fetch.py) now both draw from
Open-Meteo, using the same variable definitions/units, so there is a
single atmospheric-data source for this project end-to-end. See
notebooks/download_open_meteo.py for how the raw files are produced.

Important methodological choices:
- The target at time T uses rainfall from T+1 through T+LEAD_TIME_HOURS.
- The rainfall percentile threshold is estimated from the earliest training
  portion only, so validation/test periods do not influence the label cutoff.
- Feature normalization is deliberately NOT performed here. It is fitted on
  training samples only in 02_train_model.py and saved for live inference.

Usage:
    python notebooks/01_build_dataset.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SEQUENCE_TIMESTEPS,
    TRAIN_SPLIT, LEAD_TIME_HOURS,
)
from src.data_loader import load_multiple_files, inspect_dataset
from src.features import calculate_all_features
from src.labels import build_district_storm_labels, inspect_precip_distribution
from src.districts import build_district_masks, DISTRICT_NAMES


def load_and_merge_pressure_data():
    print("Loading pressure-level files...")
    # NOTE: pattern matches notebooks/download_open_meteo.py's output naming
    # (openmeteo_pressure_YYYY_MM.nc) -- update this if your download script's
    # naming convention changes, or files will be silently skipped.
    ds = load_multiple_files(DATA_RAW_DIR, pattern="openmeteo_pressure_*.nc")
    inspect_dataset(ds)
    return ds


def load_and_merge_precip_data():
    print("Loading precipitation files...")
    # NOTE: matches openmeteo_precip_YYYY_MM.nc from download_open_meteo.py.
    # Stored as 'tp' in METERS (ERA5 convention) so src/labels.py's mm
    # conversion (* 1000) needs no changes.
    return load_multiple_files(DATA_RAW_DIR, pattern="openmeteo_precip_*.nc")


def build_sliding_windows(
    feature_array: np.ndarray,
    label_array: np.ndarray,
    time_array: np.ndarray,
    timesteps: int = SEQUENCE_TIMESTEPS,
):
    """
    Create X[i]=features[T-timesteps+1:T], y[i]=label(s) at T.

    label_array can be 1D (n_time,) -- single-region labels -- or 2D
    (n_time, n_districts) -- per-district labels. A sample is dropped
    only if EVERY district's label is NaN at that T (a district or two
    missing data near the very edges of the record shouldn't force
    dropping all districts' samples for that timestep).
    """
    n_time = feature_array.shape[0]
    is_multi = label_array.ndim == 2
    X, y, sample_times = [], [], []

    for i in range(n_time - timesteps + 1):
        label = label_array[i + timesteps - 1]
        if is_multi:
            if np.all(np.isnan(label)):
                continue
            label = np.nan_to_num(label, nan=0.0)  # shouldn't normally hit NaN mid-array
        else:
            if np.isnan(label):
                continue
        X.append(feature_array[i:i + timesteps])
        y.append(label)
        sample_times.append(time_array[i + timesteps - 1])

    return (
        np.asarray(X, dtype="float32"),
        np.asarray(y, dtype="float32"),
        np.asarray(sample_times),
    )


def main():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    pressure_ds = load_and_merge_pressure_data()
    precip_ds = load_and_merge_precip_data()

    # Keep the two sources on their common time interval before feature/label alignment.
    common_start = max(pressure_ds.time.values.min(), precip_ds.time.values.min())
    common_end = min(pressure_ds.time.values.max(), precip_ds.time.values.max())
    pressure_ds = pressure_ds.sel(time=slice(common_start, common_end))
    precip_ds = precip_ds.sel(time=slice(common_start, common_end))

    print("\nChecking precipitation distribution...")
    inspect_precip_distribution(precip_ds)

    print("\nComputing atmospheric features...")
    features_ds = calculate_all_features(pressure_ds)
    feature_names = list(features_ds.data_vars)
    print(f"Feature variables: {feature_names}")

    print("\nBuilding per-district masks...")
    district_masks = build_district_masks(precip_ds["lat"].values, precip_ds["lon"].values)
    print(f"Districts: {DISTRICT_NAMES}")

    print("\nBuilding forward rainfall-proxy labels (per district)...")
    labels_per_time, district_thresholds_mm = build_district_storm_labels(
        precip_ds,
        district_masks,
        lead_time_hours=LEAD_TIME_HOURS,
        threshold_reference_fraction=TRAIN_SPLIT,
    )  # labels_per_time: (n_precip_time, n_districts), aligned to precip_ds.time
    for name, thr in zip(DISTRICT_NAMES, district_thresholds_mm):
        pos_rate = np.nanmean(labels_per_time[:, DISTRICT_NAMES.index(name)])
        print(f"  {name:16s} threshold={thr:7.3f} mm  positive_rate={pos_rate:.2%}")

    # Align all feature variables after feature engineering (IWV trend loses one step).
    feature_arrays = [features_ds[name].values for name in feature_names]
    min_time = min(arr.shape[0] for arr in feature_arrays)
    feature_arrays = [arr[-min_time:] for arr in feature_arrays]
    stacked = np.stack(feature_arrays, axis=-1).astype("float32")

    time_values = features_ds["time"].values[-min_time:]
    # Match each feature timestamp to the nearest precip-label timestamp.
    precip_times = precip_ds["time"].values
    nearest_idx = np.searchsorted(precip_times, time_values)
    nearest_idx = np.clip(nearest_idx, 0, len(precip_times) - 1)
    labels_aligned = labels_per_time[nearest_idx]  # (min_time, n_districts)

    # Replace invalid feature values only after all feature calculations are complete.
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nRaw feature array shape: {stacked.shape}")
    print(f"Aligned labels shape: {labels_aligned.shape}")

    X, y, sample_times = build_sliding_windows(
        stacked, labels_aligned, time_values, timesteps=SEQUENCE_TIMESTEPS
    )
    print(f"Final X shape: {X.shape}  (samples, timesteps, lat, lon, features)")
    print(f"Final y shape: {y.shape}  (samples, n_districts)")
    print(f"Overall positive rate: {y.mean():.2%}  (per-district rates printed above)")

    # X.npy intentionally contains RAW features.  Normalization is fitted only on
    # the training split in 02_train_model.py to prevent test-set leakage.
    np.save(os.path.join(DATA_PROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y.npy"), y)
    np.save(os.path.join(DATA_PROCESSED_DIR, "feature_names.npy"), np.asarray(feature_names))
    np.save(os.path.join(DATA_PROCESSED_DIR, "sample_times.npy"), sample_times)
    np.save(os.path.join(DATA_PROCESSED_DIR, "district_names.npy"), np.asarray(DISTRICT_NAMES))
    np.save(os.path.join(DATA_PROCESSED_DIR, "district_masks.npy"), district_masks)

    print(f"\nSaved raw processed arrays to {DATA_PROCESSED_DIR}/")
    print("  X.npy, y.npy (now per-district), feature_names.npy, sample_times.npy,")
    print("  district_names.npy, district_masks.npy")
    print("\nNext: run notebooks/02_train_model.py. It will fit normalization on TRAIN only.")


if __name__ == "__main__":
    main()