"""
notebooks/01d_dedupe_terrain_features.py

One-off fix: notebooks/01c_add_terrain_features.py --overwrite was
apparently run twice in a row for the thunderstorm dataset, appending
elevation_m/slope_degrees/twi TWICE -- X.npy ended up with 12 channels
(6 atmospheric + terrain + terrain again), feature_names.npy listing
elevation_m/slope_degrees/twi twice.

Because 01c always reuses the SAME cached terrain_grid.npy, the two
terrain appends are numerically identical -- nothing was corrupted, just
3 wasted/duplicate channels. This script removes the duplicate columns
by keeping only the FIRST occurrence of each feature name, safely
(memory-mapped, processed in batches -- same care as 01c itself, since
a naive np.load() of a 12-channel array here would be ~7.9GB).

01c itself now refuses to double-append (see its --allow-duplicate-terrain
guard), so this should only ever be needed once, for data created before
that guard existed.

IMPORTANT: after running this, you MUST retrain (python
notebooks/02_train_model.py or 02b_train_cloudburst_model.py) -- the
model you already trained was built for the larger, duplicated channel
count and will not load correctly against the deduplicated data.

Usage:
    python notebooks/01d_dedupe_terrain_features.py --hazard thunderstorm
    python notebooks/01d_dedupe_terrain_features.py --hazard cloudburst
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate feature columns from a built dataset.")
    parser.add_argument("--hazard", required=True, choices=["thunderstorm", "cloudburst"])
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    prefix = "cloudburst_" if args.hazard == "cloudburst" else ""
    X_path = os.path.join(DATA_PROCESSED_DIR, f"{prefix}X.npy")
    names_path = os.path.join(DATA_PROCESSED_DIR, f"{prefix}feature_names.npy")

    feature_names = list(np.load(names_path, allow_pickle=True))
    print(f"Current features ({len(feature_names)}): {feature_names}")

    # Keep only the FIRST occurrence of each name, in original order.
    seen = set()
    keep_indices = []
    for i, name in enumerate(feature_names):
        if name not in seen:
            seen.add(name)
            keep_indices.append(i)

    if len(keep_indices) == len(feature_names):
        print("No duplicate feature names found -- nothing to fix.")
        return

    deduped_names = [feature_names[i] for i in keep_indices]
    dropped_idx = [i for i in range(len(feature_names)) if i not in keep_indices]
    print(f"Deduplicated features ({len(deduped_names)}): {deduped_names}")
    print(f"Dropping {len(dropped_idx)} duplicate channel(s) at index(es) {dropped_idx}")

    X = np.load(X_path, mmap_mode="r")
    n_samples, n_timesteps, h, w, n_features = X.shape
    assert n_features == len(feature_names), (
        f"{X_path} has {n_features} channels but {names_path} lists {len(feature_names)} names -- "
        f"these are out of sync, stop and check manually before proceeding."
    )

    tmp_path = X_path + ".dedup_tmp"
    out = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32,
        shape=(n_samples, n_timesteps, h, w, len(keep_indices)),
    )
    for start in range(0, n_samples, args.batch_size):
        end = min(start + args.batch_size, n_samples)
        out[start:end] = X[start:end][..., keep_indices]
        print(f"  ...{end}/{n_samples} samples")
    out.flush()
    del out
    del X  # release the mmap on X_path before replacing it

    os.replace(tmp_path, X_path)
    np.save(names_path, np.array(deduped_names))

    print(f"\nFixed {X_path}: now (..., {len(deduped_names)} features)")
    print(f"Fixed {names_path}: {deduped_names}")
    print(f"\nIMPORTANT: retrain now -- the previously saved model was built for "
          f"{n_features} channels and will not match this {len(deduped_names)}-channel data.")


if __name__ == "__main__":
    main()
