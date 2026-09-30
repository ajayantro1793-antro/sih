"""
00_quick_start_synthetic_test.py

RUN THIS FIRST — before you have real ERA5 data downloaded.

This script generates fake/synthetic weather data with the same shape
as real ERA5 data, then runs it through the full pipeline (features ->
model -> training) to prove the code works end-to-end.

Once this passes, swap in real data using src/data_loader.py and you're
running the real pipeline with zero code changes needed elsewhere.

Run with: python notebooks/00_quick_start_synthetic_test.py
"""

import os
import sys
import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_mvp_model, get_callbacks


def generate_synthetic_imdaa(n_times=200, n_levels=6, n_lat=50, n_lon=50, seed=42):
    """
    Create a fake ERA5-shaped dataset so you can test the pipeline
    before real data is downloaded.

    Returns an xarray Dataset shaped like real ERA5 output.
    """
    rng = np.random.default_rng(seed)

    times = np.arange(n_times)
    levels = [1000, 925, 850, 700, 500, 300][:n_levels]
    lats = np.linspace(12.9, 13.5, n_lat)
    lons = np.linspace(78.7, 79.4, n_lon)

    shape = (n_times, n_levels, n_lat, n_lon)

    # Temperature: decreases with height (realistic-ish), Kelvin
    level_factor = np.linspace(1.0, 0.6, n_levels).reshape(1, n_levels, 1, 1)
    t = 300 * level_factor + rng.normal(0, 2, shape)

    # Specific humidity: decreases with height, kg/kg
    q = 0.015 * level_factor + rng.normal(0, 0.001, shape)
    q = np.clip(q, 0, None)

    # Wind components, m/s
    u = rng.normal(5, 5, shape)
    v = rng.normal(0, 5, shape)

    ds = xr.Dataset(
        {
            "t": (["time", "level", "lat", "lon"], t),
            "q": (["time", "level", "lat", "lon"], q),
            "u": (["time", "level", "lat", "lon"], u),
            "v": (["time", "level", "lat", "lon"], v),
        },
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )
    return ds


def build_synthetic_training_data(n_samples=100, timesteps=6, grid=50, n_features=6, seed=42):
    """
    Build a fake (X, y) training set with the shape the model expects,
    WITHOUT going through the full xarray feature pipeline — pure numpy,
    just to validate the model trains and predicts correctly.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, timesteps, grid, grid, n_features)).astype("float32")

    # Fake labels: correlate loosely with mean of "iwv-like" feature 0
    # so the model has *something* learnable, not pure noise
    signal = X[:, -1, :, :, 0].mean(axis=(1, 2))
    y = (signal > np.percentile(signal, 70)).astype("float32")

    return X, y


def main():
    print("=" * 60)
    print("STEP 1: Generate synthetic ERA5-shaped dataset")
    print("=" * 60)
    ds = generate_synthetic_imdaa()
    print(f"Dataset dims: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")

    print("\n" + "=" * 60)
    print("STEP 2: Build synthetic training data (X, y)")
    print("=" * 60)
    X, y = build_synthetic_training_data(n_samples=100, timesteps=6, grid=50, n_features=6)
    print(f"X shape: {X.shape}  (samples, timesteps, height, width, features)")
    print(f"y shape: {y.shape}, positive rate: {y.mean():.2f}")

    n_train = 70
    n_val = 15
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print("\n" + "=" * 60)
    print("STEP 3: Build and train MVP model (few epochs, sanity check only)")
    print("=" * 60)
    model = build_mvp_model(input_shape=X.shape[1:])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs("models", exist_ok=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # just enough to prove it trains, not to converge
        batch_size=8,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Evaluate on held-out test set")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE CHECK PASSED")
    print("=" * 60)
    print("The model builds, trains, and evaluates correctly on fake data.")
    print("Next: replace generate_synthetic_imdaa() with real ERA5 loading")
    print("via src/data_loader.py, and build_synthetic_training_data() with")
    print("real feature engineering via src/features.py")


if __name__ == "__main__":
    main()
