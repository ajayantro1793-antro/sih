"""
notebooks/01c_add_terrain_features.py

Adds static terrain-derived features (elevation, slope, Topographic
Wetness Index) to an ALREADY-BUILT dataset for either hazard, by
augmenting the saved X.npy/feature_names.npy arrays rather than
rebuilding the whole pipeline from raw Open-Meteo files. Terrain
doesn't change over time, so each terrain feature is broadcast
identically across every timestep of every sample.

This works without needing notebooks/01_build_dataset.py's source,
since it operates on its OUTPUT arrays.

Also saves a standalone data/processed/terrain_grid.npy +
terrain_feature_names.npy (shared across both hazards, since geography
doesn't depend on hazard) -- live prediction loads THIS small file
directly instead of reprocessing the DEM on every live request; see
the updated live_predict_core*.py files.

MEMORY: your machine hit an ArrayMemoryError during training on the
full X_train array earlier, so this script never holds the full
dataset in memory twice. It memory-maps the input (mmap_mode="r") and
writes the output as a memory-mapped array too, processing in small
batches along the sample axis.

Usage:
    python notebooks/01c_add_terrain_features.py --hazard thunderstorm --dem data/raw/dem/tamilnadu_srtm30.tif
    python notebooks/01c_add_terrain_features.py --hazard cloudburst   --dem data/raw/dem/tamilnadu_srtm30.tif

By default this writes NEW files (X_terrain.npy, feature_names_terrain.npy)
alongside the originals -- nothing is silently overwritten. Pass
--overwrite to replace the originals directly (back them up first if
you might want to retrain without terrain features later).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATA_PROCESSED_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.terrain_features import build_terrain_feature_stack, TERRAIN_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Add terrain features to an already-built dataset.")
    parser.add_argument("--hazard", required=True, choices=["thunderstorm", "cloudburst"])
    parser.add_argument("--dem", required=True, help="Path to the downloaded DEM GeoTIFF")
    parser.add_argument(
        "--terrain-resolution-m", type=float, default=200.0,
        help="Working resolution (meters) for slope/flow-accumulation, before aggregating "
             "onto the model grid. Lower = more detail but slower; raise this (e.g. 500-1000) "
             "if you don't have richdem installed and the numpy fallback is too slow.",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Samples processed per chunk.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite X.npy/feature_names.npy directly instead of writing "
             "X_terrain.npy/feature_names_terrain.npy alongside them.",
    )
    parser.add_argument(
        "--allow-duplicate-terrain", action="store_true",
        help="Force appending terrain features even if the dataset already has them "
             "(normally refused -- running this script twice on an already-augmented "
             "dataset would otherwise silently duplicate elevation_m/slope_degrees/twi).",
    )
    parser.add_argument(
        "--force-recompute-terrain", action="store_true",
        help="Recompute the DEM pipeline even if data/processed/terrain_grid.npy already "
             "exists (e.g. after changing --terrain-resolution-m or the DEM file). By "
             "default, an existing terrain_grid.npy is reused as-is -- this is what makes "
             "the second hazard's run fast.",
    )
    args = parser.parse_args()

    prefix = "cloudburst_" if args.hazard == "cloudburst" else ""
    X_path = os.path.join(DATA_PROCESSED_DIR, f"{prefix}X.npy")
    names_path = os.path.join(DATA_PROCESSED_DIR, f"{prefix}feature_names.npy")

    X = np.load(X_path, mmap_mode="r")  # (N, T, H, W, F) -- do NOT load fully into RAM
    feature_names = list(np.load(names_path, allow_pickle=True))
    n_samples, n_timesteps, h, w, n_features = X.shape
    print(f"Loaded {X_path}: {X.shape} ({n_features} existing features: {feature_names})")

    already_has_terrain = all(name in feature_names for name in TERRAIN_FEATURE_NAMES)
    if already_has_terrain and not args.allow_duplicate_terrain:
        print(f"\nERROR: {names_path} already contains {TERRAIN_FEATURE_NAMES} -- this dataset "
              f"looks already terrain-augmented. Refusing to append a second, duplicate copy "
              f"(this is exactly what silently produced a 12-channel dataset with "
              f"elevation_m/slope_degrees/twi listed twice previously).\n"
              f"If you have duplicates to clean up, use "
              f"notebooks/01d_dedupe_terrain_features.py --hazard {args.hazard} instead.\n"
              f"If you genuinely want to force a duplicate append, pass --allow-duplicate-terrain.")
        return

    lat_values = np.linspace(LAT_MIN, LAT_MAX, h)
    lon_values = np.linspace(LON_MIN, LON_MAX, w)

    terrain_grid_path = os.path.join(DATA_PROCESSED_DIR, "terrain_grid.npy")
    terrain_names_path = os.path.join(DATA_PROCESSED_DIR, "terrain_feature_names.npy")

    if os.path.exists(terrain_grid_path) and os.path.exists(terrain_names_path) and not args.force_recompute_terrain:
        cached_names = list(np.load(terrain_names_path, allow_pickle=True))
        if cached_names == TERRAIN_FEATURE_NAMES:
            print(f"Reusing cached {terrain_grid_path} (pass --force-recompute-terrain to rebuild it, "
                  f"e.g. after changing --terrain-resolution-m or the DEM file).")
            terrain_stack = np.load(terrain_grid_path).astype(np.float32)
        else:
            print(f"Cached {terrain_grid_path} has different feature names {cached_names} than "
                  f"expected {TERRAIN_FEATURE_NAMES} -- recomputing.")
            terrain_stack = None
    else:
        terrain_stack = None

    if terrain_stack is None:
        print(f"Building terrain features from {args.dem} "
              f"(working resolution {args.terrain_resolution_m:.0f}m)...")
        terrain = build_terrain_feature_stack(
            args.dem, lat_values, lon_values, terrain_resolution_m=args.terrain_resolution_m
        )
        terrain_stack = np.stack([terrain[name] for name in TERRAIN_FEATURE_NAMES], axis=-1)  # (H, W, 3)
        terrain_stack = np.nan_to_num(terrain_stack, nan=0.0).astype(np.float32)

        # Standalone terrain grid -- shared across hazards, loaded directly by
        # live prediction instead of reprocessing the DEM on every request.
        np.save(terrain_grid_path, terrain_stack)
        np.save(terrain_names_path, np.array(TERRAIN_FEATURE_NAMES))
        print(f"Saved {terrain_grid_path} (shared across hazards)")

    for i, name in enumerate(TERRAIN_FEATURE_NAMES):
        vals = terrain_stack[..., i]
        print(f"  {name}: min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f}")

    n_terrain = len(TERRAIN_FEATURE_NAMES)
    feature_names_augmented = feature_names + TERRAIN_FEATURE_NAMES

    out_X_path = X_path if args.overwrite else os.path.join(DATA_PROCESSED_DIR, f"{prefix}X_terrain.npy")
    out_names_path = names_path if args.overwrite else os.path.join(DATA_PROCESSED_DIR, f"{prefix}feature_names_terrain.npy")

    if args.overwrite and out_X_path == X_path:
        # Can't safely memmap-write over a file we're simultaneously
        # memmap-reading from -- write to a temp path, then swap in.
        tmp_X_path = out_X_path + ".tmp"
    else:
        tmp_X_path = out_X_path

    out = np.lib.format.open_memmap(
        tmp_X_path, mode="w+", dtype=np.float32,
        shape=(n_samples, n_timesteps, h, w, n_features + n_terrain),
    )

    batch_size = args.batch_size
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        out[start:end, :, :, :, :n_features] = X[start:end]
        out[start:end, :, :, :, n_features:] = terrain_stack[np.newaxis, np.newaxis, :, :, :]
        print(f"  ...{end}/{n_samples} samples")
    out.flush()
    del out

    if tmp_X_path != out_X_path:
        del X  # release the mmap on X_path before replacing it
        os.replace(tmp_X_path, out_X_path)

    np.save(out_names_path, np.array(feature_names_augmented))

    print(f"\nSaved {out_X_path}: (..., {n_features + n_terrain} features)")
    print(f"Saved {out_names_path}: {feature_names_augmented}")
    if not args.overwrite:
        print(f"\nOriginals ({X_path}, {names_path}) left untouched. To train on the "
              f"terrain-augmented dataset, either re-run this with --overwrite (back up the "
              f"originals first), or manually rename the _terrain files over them.")


if __name__ == "__main__":
    main()
