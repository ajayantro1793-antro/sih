"""
Fetch LIVE atmospheric data from Open-Meteo (free, no API key) and convert
it into the same (time, level, lat, lon) xarray format used for training
data -- so src/features.py works unchanged on live data.

SINGLE DATA SOURCE: training (notebooks/download_open_meteo.py, via
Open-Meteo's Historical Forecast API) and live prediction (this module,
via Open-Meteo's regular /v1/forecast API) both draw from Open-Meteo's
operational forecast models (Best Match = blended ECMWF/GFS/etc by
default), using the SAME variable names and units. This avoids the
train/live distribution mismatch that comes from training on one
atmospheric model (e.g. ERA5 reanalysis) and predicting live from a
different one. The two endpoints differ only in how far back they can
look: the Historical Forecast API archives multi-year history for
training, while this module's /v1/forecast endpoint supplies a short
recent window ('past_days') for near-real-time inference.

VARIABLES: this project's download_open_meteo.py fetches relative
humidity ('r'), not specific humidity ('q'), and only 850/500 hPa levels
-- see src/features.py, which derives specific humidity internally from
'r' and 't'. This module fetches raw relative humidity from Open-Meteo
and outputs it as 'r' UNCONVERTED, so features.py's conversion logic is
the single source of truth (avoids two different RH->q implementations
silently drifting apart between training and live inference).

PERFORMANCE NOTE: with the full Tamil Nadu grid (~391 points), fetching
one point per HTTP request would be far too slow for a "live" dashboard.
This module batches multiple grid points into each request using
Open-Meteo's comma-separated multi-location support.

Usage:
    from src.live_fetch import fetch_live_atmospheric_data
    ds = fetch_live_atmospheric_data()
"""

import os
import sys
import time
import numpy as np
import xarray as xr
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GRID_STEP = 0.25  # must match the resolution used when downloading ERA5
BATCH_SIZE = 25   # grid points per HTTP request -- keeps URL length and
                   # response size reasonable while cutting request count
                   # by ~25x compared to one-point-per-request

# Hardcoded (not imported from configs.config) to guarantee this always
# matches notebooks/download_open_meteo.py's LEVELS exactly -- that
# script has its own hardcoded level list rather than reading from
# config, so importing a possibly-stale config constant here risked
# live data silently using different levels than training data.
LEVELS = [850, 500]


def build_grid_points():
    """Recreate the exact same lat/lon grid used for training data (see
    notebooks/download_open_meteo.py's build_grid_points(), kept in sync)."""
    lats = np.arange(LAT_MAX, LAT_MIN - 1e-6, -GRID_STEP)  # descending
    lons = np.arange(LON_MIN, LON_MAX + 1e-6, GRID_STEP)   # ascending
    return lats, lons


def wind_speed_dir_to_uv(speed_ms, direction_deg):
    """
    Convert meteorological wind speed + direction (direction = where wind
    comes FROM, degrees) into U/V components, matching the training
    data's convention (see notebooks/download_open_meteo.py).
    """
    dir_rad = np.radians(direction_deg)
    u = -speed_ms * np.sin(dir_rad)
    v = -speed_ms * np.cos(dir_rad)
    return u, v


def _fetch_batch(lat_list, lon_list, levels, past_days=2, forecast_days=1,
                  max_retries=4, initial_backoff_seconds=5):
    """
    Fetch pressure-level variables for a BATCH of lat/lon points in one
    request, using Open-Meteo's comma-separated multi-location support.

    Retries with exponential backoff on HTTP 429 (rate limited) --
    Open-Meteo's free tier can reject requests if too many arrive in
    quick succession, which happens routinely with ~16 back-to-back
    batched requests for a full-state grid.

    Returns:
        list of per-location result dicts, in the SAME ORDER as the
        input lat_list/lon_list (Open-Meteo preserves request order for
        multi-location responses).
    """
    variables = []
    for lvl in levels:
        variables += [f"temperature_{lvl}hPa", f"relative_humidity_{lvl}hPa",
                      f"wind_speed_{lvl}hPa", f"wind_direction_{lvl}hPa"]

    params = {
        "latitude": ",".join(str(lat) for lat in lat_list),
        "longitude": ",".join(str(lon) for lon in lon_list),
        "hourly": ",".join(variables),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }

    backoff = initial_backoff_seconds
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            is_retryable = resp.status_code == 429 or resp.status_code >= 500
            if is_retryable:
                reason = ("rate limited (429)" if resp.status_code == 429
                          else f"server error ({resp.status_code})")
                raise requests.exceptions.RequestException(reason)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data = [data]
            return data
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt == max_retries:
                raise  # exhausted retries, propagate the real error
            print(f"    Request failed ({e}) -- waiting {backoff}s before retry "
                  f"{attempt + 1}/{max_retries}...")
            time.sleep(backoff)
            backoff *= 2  # exponential backoff


def fetch_live_atmospheric_data(levels=None, timesteps_needed=24):
    """
    Fetch live data for the full study-region grid and assemble it into an
    xarray Dataset with dims (time, level, lat, lon) and variables
    t, r, u, v -- matching src/features.py's expectations exactly (r =
    relative humidity, unconverted; features.py derives specific humidity
    internally).

    Returns:
        xr.Dataset with the most recent `timesteps_needed` hourly steps.
    """
    if levels is None:
        levels = LEVELS

    lats, lons = build_grid_points()
    n_lat, n_lon = len(lats), len(lons)
    n_level = len(levels)

    # Flatten grid into a list of (i, j, lat, lon) so we can batch requests
    # while remembering where each point belongs in the final grid.
    flat_points = [(i, j, lat, lon) for i, lat in enumerate(lats) for j, lon in enumerate(lons)]
    n_points = len(flat_points)
    n_batches = (n_points + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Fetching live data for {n_lat}x{n_lon} grid ({n_points} points) "
          f"in {n_batches} batched requests of up to {BATCH_SIZE} points each...")

    t_grid, r_grid, u_grid, v_grid = None, None, None, None
    time_index = None

    for batch_idx in range(n_batches):
        batch = flat_points[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_lats = [p[2] for p in batch]
        batch_lons = [p[3] for p in batch]

        print(f"  Batch {batch_idx + 1}/{n_batches} ({len(batch)} points)...")
        results = _fetch_batch(batch_lats, batch_lons, levels)

        if batch_idx < n_batches - 1:
            time.sleep(1.5)  # small pause between requests to avoid rate limiting

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

            for k, lvl in enumerate(levels):
                temp_c = np.array(hourly[f"temperature_{lvl}hPa"])
                rh = np.array(hourly[f"relative_humidity_{lvl}hPa"])
                speed = np.array(hourly[f"wind_speed_{lvl}hPa"])
                direction = np.array(hourly[f"wind_direction_{lvl}hPa"])

                u, v = wind_speed_dir_to_uv(speed, direction)

                n_here = min(len(temp_c), t_grid.shape[0])
                t_grid[:n_here, k, i, j] = temp_c[:n_here] + 273.15  # -> Kelvin, matches training data
                r_grid[:n_here, k, i, j] = rh[:n_here]  # raw %, NOT converted here
                u_grid[:n_here, k, i, j] = u[:n_here]
                v_grid[:n_here, k, i, j] = v[:n_here]

    ds = xr.Dataset(
        {
            "t": (["time", "level", "lat", "lon"], t_grid),
            "r": (["time", "level", "lat", "lon"], r_grid),
            "u": (["time", "level", "lat", "lon"], u_grid),
            "v": (["time", "level", "lat", "lon"], v_grid),
        },
        coords={"time": time_index, "level": levels, "lat": lats, "lon": lons},
    )

    # Keep only the most recent `timesteps_needed` steps
    ds = ds.isel(time=slice(-timesteps_needed, None))
    print(f"Fetched {ds.sizes['time']} recent hourly timesteps, "
          f"latest: {str(ds.time.values[-1])}")
    return ds


if __name__ == "__main__":
    ds = fetch_live_atmospheric_data()
    print(ds)