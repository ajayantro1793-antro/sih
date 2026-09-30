"""
Load the tuned PER-DISTRICT cloudburst alert thresholds saved by
notebooks/02b_train_cloudburst_model.py.

Mirrors src/thresholds.py exactly, just pointed at the cloudburst_
-prefixed files -- kept as a separate module (rather than adding a
hazard parameter to thresholds.py) so live prediction can never
accidentally load the thunderstorm model's tuned thresholds by mistake.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_PROCESSED_DIR

THRESHOLD_PATH = os.path.join(DATA_PROCESSED_DIR, "cloudburst_alert_threshold.npy")
DISTRICT_NAMES_PATH = os.path.join(DATA_PROCESSED_DIR, "cloudburst_district_names.npy")


def get_cloudburst_alert_thresholds(default: float = 0.5) -> np.ndarray:
    """
    Load the per-district cloudburst alert thresholds tuned during
    training (see notebooks/02b_train_cloudburst_model.py's
    find_best_threshold_per_district()). Falls back to an array of
    `default`, sized to match cloudburst_district_names.npy, with a
    warning if the threshold file doesn't exist yet.
    """
    n_districts = None
    if os.path.exists(DISTRICT_NAMES_PATH):
        n_districts = len(np.load(DISTRICT_NAMES_PATH, allow_pickle=True))

    if not os.path.exists(THRESHOLD_PATH):
        print(f"WARNING: {THRESHOLD_PATH} not found -- has "
              f"notebooks/02b_train_cloudburst_model.py been run yet? "
              f"Falling back to default threshold {default} for all districts.")
        return np.full(n_districts or 1, default, dtype="float32")

    return np.load(THRESHOLD_PATH).astype("float32")


def get_cloudburst_district_names() -> np.ndarray:
    if not os.path.exists(DISTRICT_NAMES_PATH):
        raise FileNotFoundError(
            f"{DISTRICT_NAMES_PATH} not found -- run "
            f"notebooks/01b_build_cloudburst_dataset.py first."
        )
    return np.load(DISTRICT_NAMES_PATH, allow_pickle=True)


if __name__ == "__main__":
    names = get_cloudburst_district_names()
    thresholds = get_cloudburst_alert_thresholds()
    for n, t in zip(names, thresholds):
        print(f"{n:16s}: {t:.3f}")
