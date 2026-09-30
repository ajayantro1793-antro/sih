"""
03b_live_predict_cloudburst.py

Fetches LIVE atmospheric data (via Open-Meteo) for the whole state grid
and runs the trained multi-district CLOUDBURST model, printing a "right
now" cloudburst risk for EACH Tamil Nadu district separately. Logs one
row per district to data/processed/cloudburst_predictions_log.csv (a
separate log from the thunderstorm one) so you can build a track record
over time and later fill in 'actual_outcome' per district/timestamp.

IMPORTANT CAVEATS (read before trusting this for anything real):
1. Training and live prediction both draw from Open-Meteo's operational
   forecast models -- one consistent data source end-to-end -- but it's
   still a short historical record and a proof-of-concept integration,
   not a validated operational forecast.
2. Cloudburst labels are the same rainfall-extremity proxy your
   thunderstorm model uses, at a stricter percentile -- not a confirmed
   cloudburst event record. True cloudbursts (IMD: ~100 mm/hr over
   ~20-30 sq km) are sub-grid-scale, sub-hourly events that this
   hourly, 0.25-deg pipeline cannot resolve directly.
3. District boundaries are rectangular approximations (see
   src/districts.py), not real administrative polygons.
4. Treat this as demonstrating the ARCHITECTURE end-to-end, not a
   production weather forecast.

Usage:
    python notebooks/03b_live_predict_cloudburst.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.live_predict_core_cloudburst import run_live_cloudburst_prediction
from src.live_log_cloudburst import append_log, load_log


def main():
    print("=" * 60)
    print("LIVE CLOUDBURST RISK PREDICTION -- PER DISTRICT")
    print("=" * 60)
    print("NOTE: uses Open-Meteo operational forecast data (same source used")
    print("for training) -- see this script's docstring for important caveats.\n")

    result = run_live_cloudburst_prediction()

    if result["error"]:
        print(f"\nERROR: {result['error']}")
        return

    print(f"Latest data timestamp: {result['latest_time']}\n")
    print(f"{'District':16s}{'Probability':>14s}{'Threshold':>12s}{'Alert':>10s}")
    n_alerts = 0
    for d in sorted(result["districts"], key=lambda x: -x["probability"]):
        alert_str = "ALERT" if d["alert"] else "-"
        n_alerts += int(d["alert"])
        print(f"{d['name']:16s}{d['probability']:13.1%} {d['threshold']:11.1%} {alert_str:>10s}")

    print("\n" + "=" * 60)
    if n_alerts:
        print(f"{n_alerts} district(s) with elevated cloudburst risk")
    else:
        print("No district currently shows elevated cloudburst risk")
    print("=" * 60)

    append_log(districts=result["districts"], latest_time=result["latest_time"])
    history = load_log()
    print(f"\nLogged. Track record so far: {len(history)} district-prediction row(s) recorded.")
    print("Run this script again later to build up more history.")
    print("\nOnce you know what actually happened in a district, open")
    print("data/processed/cloudburst_predictions_log.csv and fill in the")
    print("'actual_outcome' column for that row.")
    print("\nRemember: this is a proof-of-concept live integration. Do not use")
    print("for actual safety decisions -- see the caveats above.")


if __name__ == "__main__":
    main()
