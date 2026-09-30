"""
Open-Meteo downloader for SIH Weather Nowcasting -- TRAINING data.

Replaces download_era5.py / the CDS API. This project now uses Open-Meteo
for BOTH training and live prediction, so the two no longer draw from two
different atmospheric models with different biases (see CHANGES.md).

Data source: Open-Meteo's Historical Forecast API
(https://open-meteo.com/en/docs/historical-forecast-api), hosted at
historical-forecast-api.open-meteo.com. This is NOT the same endpoint as
the Historical Weather API (archive-api.open-meteo.com) -- that one is
ERA5-based and does NOT expose pressure-level variables. The Historical
Forecast API instead archives the same operational forecast models used
by the live /v1/forecast endpoint (Best Match = blended ECMWF/GFS/etc by
default, same as src/live_fetch.py uses for live prediction), so training
and live inference are drawn from the same underlying model family and
use IDENTICAL variable names/units. Coverage starts around 2021-2022
depending on model -- see the docs page for exact per-model start dates.

Downloads one month at a time (like the old ERA5 script) so that:
- requests remain manageable
- failed months can be resumed
- existing files are skipped
- one year or multiple years can be downloaded easily

Fetches, in ONE request per grid-point batch per month:
    Pressure levels (850, 500 hPa): temperature, relative humidity,
        wind speed, wind direction -- same variables/levels as
        src/live_fetch.py and src/features.py expect.
    Surface: precipitation (mm, preceding-hour sum) -- used only for
        building training labels (src/labels.py). Stored as ERA5-style
        'tp' in METERS (divided by 1000) so src/labels.py's existing
        "* 1000" mm-conversion logic needs no changes.

Output files (in data/raw/), matching the naming convention
notebooks/01_build_dataset.py expects:
    openmeteo_pressure_YYYY_MM.nc  (vars: t, r, u, v; dims: time, level, lat, lon)
    openmeteo_precip_YYYY_MM.nc    (var: tp; dims: time, lat, lon)

Usage:
    python notebooks/download_open_meteo.py
"""

import os
import sys
import time
import calendar
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import xarray as xr
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Historical Forecast API coverage begins ~2021-2022 depending on model
# (see docs) -- pick a start date safely inside that window. Adjust to
# taste; a longer/more-recent range gives the model more monsoon cycles
# to learn from.
START_YEAR = 2022
START_MONTH = 1

END_YEAR = 2023
END_MONTH = 12

# Must match src/live_fetch.py exactly, or training/live features will
# be computed on different grids/levels.
GRID_STEP = 0.25
LEVELS = [850, 500]
BATCH_SIZE = 25  # grid points per HTTP request, same rationale as live_fetch.py


# ============================================================
# GRID (identical to src/live_fetch.py's build_grid_points())
# ============================================================

def build_grid_points():
    lats = np.arange(LAT_MAX, LAT_MIN - 1e-6, -GRID_STEP)  # descending
    lons = np.arange(LON_MIN, LON_MAX + 1e-6, GRID_STEP)   # ascending
    return lats, lons


def wind_speed_dir_to_uv(speed_ms, direction_deg):
    """Same convention as src/live_fetch.py -- kept in sync intentionally."""
    dir_rad = np.radians(direction_deg)
    u = -speed_ms * np.sin(dir_rad)
    v = -speed_ms * np.cos(dir_rad)
    return u, v


# ============================================================
# MONTH RANGE
# ============================================================

def generate_months(start_year, start_month, end_year, end_month):
    year, month = start_year, start_month
    while True:
        yield year, month
        if year == end_year and month == end_month:
            break
        month += 1
        if month > 12:
            month = 1
            year += 1


# ============================================================
# FETCH
# ============================================================

def _fetch_batch(lat_list, lon_list, start_date, end_date,
                  max_retries=3, initial_backoff_seconds=5, max_backoff_seconds=60):
    """
    Fetch pressure-level + precipitation variables for a BATCH of
    lat/lon points in one request. Retries a few times with short
    exponential backoff for transient errors (brief bursts, momentary
    5xx blips).

    This is DELIBERATELY short -- it's not meant to outlast a real
    hourly/daily quota exhaustion (those reset at fixed clock
    boundaries, not after some arbitrary delay). If Open-Meteo's
    free-tier quota (see https://open-meteo.com/en/terms) is actually
    exhausted, this will keep failing no matter how long it backs off,
    so the caller (fetch_month, driven by main()'s wall-clock-aware
    retry) is responsible for waiting for the real reset instead.

    Returns:
        list of per-location result dicts, in the SAME ORDER as the
        input lat_list/lon_list.
    """
    variables = ["precipitation"]
    for lvl in LEVELS:
        variables += [f"temperature_{lvl}hPa", f"relative_humidity_{lvl}hPa",
                      f"wind_speed_{lvl}hPa", f"wind_direction_{lvl}hPa"]

    params = {
        "latitude": ",".join(str(lat) for lat in lat_list),
        "longitude": ",".join(str(lon) for lon in lon_list),
        "hourly": ",".join(variables),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    backoff = initial_backoff_seconds
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=120)
            is_retryable = resp.status_code == 429 or resp.status_code >= 500
            if is_retryable:
                reason = ("rate limited (429)" if resp.status_code == 429
                          else f"server error ({resp.status_code})")
                if attempt == max_retries:
                    raise requests.exceptions.RequestException(reason)
                retry_after = resp.headers.get("Retry-After")
                wait_s = min(float(retry_after), max_backoff_seconds) if retry_after else backoff
                print(f"    Request failed ({reason}) -- waiting {wait_s:.0f}s "
                      f"before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_s)
                backoff = min(backoff * 2, max_backoff_seconds)
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data = [data]
            return data
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise
            print(f"    Request failed ({e}) -- waiting {backoff}s before retry "
                  f"{attempt + 1}/{max_retries}...")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_seconds)


def fetch_month(year, month):
    """
    Fetch one calendar month of pressure-level + precipitation data for
    the full study-region grid.

    Returns:
        (pressure_ds, precip_ds): two xr.Datasets, dims
        (time, level, lat, lon) and (time, lat, lon) respectively.
    """
    start_date = date(year, month, 1).isoformat()
    days_in_month = calendar.monthrange(year, month)[1]
    end_date = date(year, month, days_in_month).isoformat()

    lats, lons = build_grid_points()
    n_lat, n_lon = len(lats), len(lons)
    n_level = len(LEVELS)

    flat_points = [(i, j, lat, lon) for i, lat in enumerate(lats) for j, lon in enumerate(lons)]
    n_points = len(flat_points)
    n_batches = (n_points + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Fetching {n_lat}x{n_lon} grid ({n_points} points) for {start_date}..{end_date} "
          f"in {n_batches} batched requests...")

    t_grid = r_grid = u_grid = v_grid = tp_grid = None
    time_index = None

    for batch_idx in range(n_batches):
        batch = flat_points[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
        batch_lats = [p[2] for p in batch]
        batch_lons = [p[3] for p in batch]

        print(f"    Batch {batch_idx + 1}/{n_batches} ({len(batch)} points)...")
        results = _fetch_batch(batch_lats, batch_lons, start_date, end_date)

        if batch_idx < n_batches - 1:
            time.sleep(1.5)  # avoid rate limiting, same pacing as live_fetch.py

        for point_idx, (i, j, lat, lon) in enumerate(batch):
            hourly = results[point_idx]["hourly"]
            times = np.array(hourly["time"], dtype="datetime64[ns]")

            if time_index is None:
                time_index = times
                n_time = len(time_index)
                t_grid = np.full((n_time, n_level, n_lat, n_lon), np.nan, dtype="float32")
                r_grid = np.full((n_time, n_level, n_lat, n_lon), np.nan, dtype="float32")
                u_grid = np.full((n_time, n_level, n_lat, n_lon), np.nan, dtype="float32")
                v_grid = np.full((n_time, n_level, n_lat, n_lon), np.nan, dtype="float32")
                tp_grid = np.full((n_time, n_lat, n_lon), np.nan, dtype="float32")

            n_here = min(len(times), t_grid.shape[0])

            precip_mm = np.array(hourly["precipitation"])
            tp_grid[:n_here, i, j] = precip_mm[:n_here] / 1000.0  # mm -> meters (ERA5 'tp' convention)

            for k, lvl in enumerate(LEVELS):
                temp_c = np.array(hourly[f"temperature_{lvl}hPa"])
                rh = np.array(hourly[f"relative_humidity_{lvl}hPa"])
                speed = np.array(hourly[f"wind_speed_{lvl}hPa"])
                direction = np.array(hourly[f"wind_direction_{lvl}hPa"])
                u, v = wind_speed_dir_to_uv(speed, direction)

                t_grid[:n_here, k, i, j] = temp_c[:n_here] + 273.15  # -> Kelvin
                r_grid[:n_here, k, i, j] = rh[:n_here]
                u_grid[:n_here, k, i, j] = u[:n_here]
                v_grid[:n_here, k, i, j] = v[:n_here]

    pressure_ds = xr.Dataset(
        {
            "t": (["time", "level", "lat", "lon"], t_grid),
            "r": (["time", "level", "lat", "lon"], r_grid),
            "u": (["time", "level", "lat", "lon"], u_grid),
            "v": (["time", "level", "lat", "lon"], v_grid),
        },
        coords={"time": time_index, "level": LEVELS, "lat": lats, "lon": lons},
    )
    precip_ds = xr.Dataset(
        {"tp": (["time", "lat", "lon"], tp_grid)},
        coords={"time": time_index, "lat": lats, "lon": lons},
    )
    return pressure_ds, precip_ds


# ============================================================
# DOWNLOAD
# ============================================================

# If a month keeps failing even after _fetch_batch's own short retries,
# it almost always means Open-Meteo's HOURLY or DAILY free-tier quota
# (see https://open-meteo.com/en/terms) is genuinely exhausted -- and
# those only reset at fixed UTC clock boundaries (top of the hour /
# midnight), not after some arbitrary backoff delay. So instead of
# guessing a wait time, we compute the actual time remaining until the
# next boundary and sleep until just past it.
def _seconds_until_next_utc_boundary(period, buffer_seconds=90):
    """period: 'hour' or 'day'. Returns seconds until just after the next
    UTC hour/day boundary, so we retry right after Open-Meteo's own
    counters would have reset (plus a small safety buffer)."""
    now = datetime.now(timezone.utc)
    if period == "hour":
        nxt = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    else:
        nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    return (nxt - now).total_seconds() + buffer_seconds


def main():
    for year, month in generate_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        month_str = f"{month:02d}"
        pressure_file = OUTPUT_DIR / f"openmeteo_pressure_{year}_{month_str}.nc"
        precip_file = OUTPUT_DIR / f"openmeteo_precip_{year}_{month_str}.nc"

        print("\n" + "=" * 75)
        print(f"OPEN-METEO DOWNLOAD: {year}-{month_str}")
        print("=" * 75)

        if pressure_file.exists() and precip_file.exists():
            print(f"[SKIP] Both files already exist for {year}-{month_str}")
            continue

        # Attempt 1: try immediately. If it fails, that means _fetch_batch's
        # own short retries were also exhausted, so a real quota wall is
        # the likely cause -- wait for the next HOURLY reset (attempt 2).
        # If it fails again right after that, the DAILY quota is probably
        # exhausted -- wait for the next UTC midnight (attempt 3). If it
        # still fails after that, something else is wrong; stop and let
        # the person investigate rather than looping forever.
        for attempt in (1, 2, 3):
            try:
                pressure_ds, precip_ds = fetch_month(year, month)
                break
            except requests.exceptions.RequestException as e:
                if attempt == 3:
                    print(f"[GIVE UP] {year}-{month_str} still failing after "
                          f"waiting through an hourly AND a daily reset ({e}). "
                          f"This is no longer a normal quota issue -- check "
                          f"your network/firewall, or that Open-Meteo isn't "
                          f"blocking this IP. Re-run this script later to "
                          f"resume from here (completed months are skipped "
                          f"automatically).")
                    return
                wait_s = _seconds_until_next_utc_boundary(
                    "hour" if attempt == 1 else "day")
                reset_label = "next hourly quota reset" if attempt == 1 else \
                    "next daily quota reset (UTC midnight)"
                resume_at = datetime.now(timezone.utc) + timedelta(seconds=wait_s)
                print(f"[RATE LIMITED] {year}-{month_str} ({e}). Waiting for "
                      f"{reset_label} -- ~{wait_s / 60:.0f} min, resuming "
                      f"around {resume_at.strftime('%H:%M:%S')} UTC. "
                      f"(Safe to leave this running; Ctrl+C and re-run later "
                      f"also works, completed months are skipped.)")
                time.sleep(wait_s)

        if not pressure_file.exists():
            pressure_ds.to_netcdf(pressure_file)
            print(f"[DONE] {pressure_file.name}")
        if not precip_file.exists():
            precip_ds.to_netcdf(precip_file)
            print(f"[DONE] {precip_file.name}")

    print("\n" + "=" * 75)
    print("ALL OPEN-METEO DOWNLOADS COMPLETE")
    print("=" * 75)
    print(f"Period: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
    print(f"Region: {LAT_MAX}\N{DEGREE SIGN}N -> {LAT_MIN}\N{DEGREE SIGN}N, "
          f"{LON_MIN}\N{DEGREE SIGN}E -> {LON_MAX}\N{DEGREE SIGN}E")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()