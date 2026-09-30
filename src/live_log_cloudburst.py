"""
Logs every live CLOUDBURST prediction to its own CSV, mirroring
src/live_log.py exactly (same manual-annotation workflow: log now, verify
'actual_outcome' later) but written to a separate file so thunderstorm and
cloudburst prediction rows never mix in one log with no way to tell them
apart.
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_PROCESSED_DIR

LOG_PATH = os.path.join(DATA_PROCESSED_DIR, "cloudburst_predictions_log.csv")

COLUMNS = [
    "logged_at_utc", "latest_data_time", "district", "probability", "threshold",
    "alert", "actual_outcome", "notes",
]


def append_log(districts: list, latest_time: str, notes: str = ""):
    """
    Append one row PER DISTRICT for this live cloudburst prediction run
    (creates the CSV if missing). `districts` is the list of per-district
    dicts returned by
    src.live_predict_core_cloudburst.run_live_cloudburst_prediction()['districts'].
    """
    logged_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = [
        {
            "logged_at_utc": logged_at,
            "latest_data_time": latest_time,
            "district": d["name"],
            "probability": round(d["probability"], 4),
            "threshold": d["threshold"],
            "alert": d["alert"],
            "actual_outcome": "",  # fill in later: "cloudburst" / "no_cloudburst" / leave blank
            "notes": notes,
        }
        for d in districts
    ]

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
    else:
        df = pd.DataFrame(records)

    df.to_csv(LOG_PATH, index=False)
    return df


def load_log() -> pd.DataFrame:
    """Load the log as a DataFrame, or an empty one with the right columns if none exists yet."""
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        for col in ["actual_outcome", "notes"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df
    return pd.DataFrame(columns=COLUMNS)


def save_log(df: pd.DataFrame):
    """Save an edited log DataFrame back to disk (e.g. after annotating actual outcomes)."""
    df.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    df = load_log()
    print(f"Log has {len(df)} entries at {LOG_PATH}")
    if len(df) > 0:
        print(df.to_string(index=False))
