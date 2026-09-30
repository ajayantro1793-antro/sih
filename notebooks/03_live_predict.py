"""
03_live_predict.py

Fetches LIVE atmospheric data (via Open-Meteo) for the whole state grid
and runs the trained multi-district model, printing a "right now" storm
risk for EACH Tamil Nadu district separately. Also logs one row per
district to data/processed/live_predictions_log.csv so you can build a
track record over time (run this periodically) and later fill in
'actual_outcome' per district/timestamp.

IMPORTANT CAVEATS (read before trusting this for anything real):
1. Training (notebooks/download_open_meteo.py) and live prediction
   (this script) both draw from Open-Meteo's operational forecast
   models, so there's a single consistent data source end-to-end --
   but it's still a short historical record and a proof-of-concept
   integration, not a validated operational forecast.
2. District boundaries are rectangular approximations (see
   src/districts.py), not real administrative polygons.
3. Treat this as demonstrating the ARCHITECTURE end-to-end, not a
   production weather forecast.

Usage:
    python notebooks/03_live_predict.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.live_predict_core import run_live_prediction
from src.live_log import append_log, load_log


def main():
    print("=" * 60)
    print("LIVE THUNDERSTORM RISK PREDICTION -- PER DISTRICT")
    print("=" * 60)
    print("NOTE: uses Open-Meteo operational forecast data (same source used")
    print("for training) -- see this script's docstring for important caveats.\n")

    result = run_live_prediction()

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
        print(f"{n_alerts} district(s) with elevated thunderstorm risk")
    else:
        print("No district currently shows elevated thunderstorm risk")
    print("=" * 60)

    append_log(districts=result["districts"], latest_time=result["latest_time"])
    history = load_log()
    print(f"\nLogged. Track record so far: {len(history)} district-prediction row(s) recorded.")
    print("Run this script again later to build up more history -- view the")
    print("accumulating track record in the Streamlit app (app.py).")
    print("\nOnce you know what actually happened in a district, open")
    print("data/processed/live_predictions_log.csv and fill in the")
    print("'actual_outcome' column for that row -- or do this in the app's Live tab.")
    print("\nRemember: this is a proof-of-concept live integration. Do not use")
    print("for actual safety decisions -- see the caveats above.")


if __name__ == "__main__":
    main()
