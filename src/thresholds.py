"""
Central place to load the tuned PER-DISTRICT alert thresholds saved by
notebooks/02_train_model.py, so live prediction and the dashboard both
use the same actual tuned values rather than hand-copied numbers that
can silently go stale after retraining.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_PROCESSED_DIR

THRESHOLD_PATH = os.path.join(DATA_PROCESSED_DIR, "alert_threshold.npy")
DISTRICT_NAMES_PATH = os.path.join(DATA_PROCESSED_DIR, "district_names.npy")


def get_alert_thresholds(default: float = 0.5) -> np.ndarray:
    """
    Load the per-district alert thresholds tuned during training (see
    notebooks/02_train_model.py's find_best_threshold_per_district()).
    Falls back to an array of `default`, sized to match district_names.npy,
    with a warning if the threshold file doesn't exist yet.
    """
    n_districts = None
    if os.path.exists(DISTRICT_NAMES_PATH):
        n_districts = len(np.load(DISTRICT_NAMES_PATH, allow_pickle=True))

    if not os.path.exists(THRESHOLD_PATH):
        print(f"WARNING: {THRESHOLD_PATH} not found -- has "
              f"notebooks/02_train_model.py been run yet? Falling back to "
              f"default threshold {default} for all districts.")
        return np.full(n_districts or 1, default, dtype="float32")

    return np.load(THRESHOLD_PATH).astype("float32")


def get_district_names() -> np.ndarray:
    if not os.path.exists(DISTRICT_NAMES_PATH):
        raise FileNotFoundError(
            f"{DISTRICT_NAMES_PATH} not found -- run notebooks/01_build_dataset.py first."
        )
    return np.load(DISTRICT_NAMES_PATH, allow_pickle=True)


if __name__ == "__main__":
    names = get_district_names()
    thresholds = get_alert_thresholds()
    for n, t in zip(names, thresholds):
        print(f"{n:16s}: {t:.3f}")
