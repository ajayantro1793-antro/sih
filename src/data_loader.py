"""
Data loading utilities for training NetCDF files (produced by
notebooks/download_open_meteo.py). Also tolerates ERA5-style
dimension/coordinate names for anyone still working with older ERA5
downloads (see _standardize_dim_names()).

Usage:
    from src.data_loader import load_era5_data, subset_region

    ds = load_era5_data("data/raw/openmeteo_pressure_2022_07.nc")
    ds_region = subset_region(ds)
"""

import os
import sys
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


def load_era5_data(filepath: str) -> xr.Dataset:
    """
    Load a single training NetCDF file and standardize dimension names so
    downstream code (src/features.py) always sees the same names
    regardless of which export format produced the file. Function name
    kept for backward compatibility; works on Open-Meteo files (the
    current default, see notebooks/download_open_meteo.py) as well as
    older ERA5 exports.

    Some ERA5 exports use: valid_time, pressure_level, latitude, longitude
    Our feature code expects: time, level, lat, lon
    (Open-Meteo files from download_open_meteo.py already use time/level/lat/lon.)

    Args:
        filepath: path to .nc file

    Returns:
        xarray Dataset with dimensions standardized to (time, level, lat, lon)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Download data first via notebooks/download_open_meteo.py"
        )
    ds = xr.open_dataset(filepath)
    ds = _standardize_dim_names(ds)
    return ds


def _standardize_dim_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename ERA5's alternate dimension/coordinate names to the names the
    rest of the codebase (src/features.py) expects. Safe to call on data
    that's already standardized — only renames what's present.
    """
    rename_map = {
        "valid_time": "time",
        "pressure_level": "level",
        "latitude": "lat",
        "longitude": "lon",
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    if existing_renames:
        ds = ds.rename(existing_renames)
    return ds


def load_multiple_files(directory: str, pattern: str = "*.nc") -> xr.Dataset:
    """
    Load and merge multiple ERA5 files (e.g. multiple months or day-chunks)
    into one dataset, with dimension names standardized.

    Uses manual open + concat instead of xr.open_mfdataset to avoid
    requiring the 'dask' package — fine for small MVP-sized datasets
    like ours (a year or so of data, one regional grid).

    Automatically drops duplicate timestamps (keeping the first
    occurrence). This guards against a real, recurring failure mode in
    this project: leftover files from an earlier download convention
    (e.g. old day-chunked filenames) matching the same glob pattern as
    newer whole-month files, silently double-counting the same dates and
    crashing later feature-engineering steps (e.g. .diff(dim="time"))
    with a cryptic "duplicate index values" error instead of a clear one.

    Args:
        directory: folder containing .nc files
        pattern: glob pattern to match files
                 (e.g. "era5_pressure_*.nc" for pressure-level files)

    Returns:
        Combined xarray Dataset, concatenated along 'time', sorted
        chronologically, with duplicate timestamps removed
    """
    import glob

    filepaths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not filepaths:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' in {directory}\n"
            f"Run notebooks/download_open_meteo.py first."
        )

    print(f"Matched {len(filepaths)} file(s) for pattern '{pattern}':")
    for fp in filepaths:
        print(f"  {os.path.basename(fp)}")

    datasets = [_standardize_dim_names(xr.open_dataset(fp)) for fp in filepaths]
    ds = xr.concat(datasets, dim="time")
    ds = ds.sortby("time")

    n_before = ds.sizes["time"]
    is_duplicate = ds.get_index("time").duplicated(keep="first")
    if is_duplicate.any():
        n_dupes = int(is_duplicate.sum())
        print(f"WARNING: found {n_dupes} duplicate timestamp(s) across the "
              f"matched files (likely overlapping/leftover files from an "
              f"earlier download convention) — keeping the first "
              f"occurrence of each and dropping the rest.")
        ds = ds.isel(time=~is_duplicate)

    n_after = ds.sizes["time"]
    if n_after != n_before:
        print(f"Time dimension: {n_before} -> {n_after} after deduplication")

    return ds


def subset_region(
    ds: xr.Dataset,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
) -> xr.Dataset:
    """
    Crop dataset to a specific bounding box (default: Tamil Nadu region).

    Args:
        ds: full xarray Dataset
        lat_min, lat_max, lon_min, lon_max: bounding box

    Returns:
        Cropped xarray Dataset
    """
    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def inspect_dataset(ds: xr.Dataset) -> None:
    """Print a quick summary of a dataset — dims, variables, time range."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Time range: {ds.time.values.min()} to {ds.time.values.max()}")
    print(f"Lat range: {float(ds.lat.min())} to {float(ds.lat.max())}")
    print(f"Lon range: {float(ds.lon.min())} to {float(ds.lon.max())}")
    print("=" * 60)


if __name__ == "__main__":
    print("This module expects Open-Meteo .nc files (see notebooks/download_open_meteo.py) in data/raw/")
    print("Example usage:")
    print("  ds = load_era5_data('data/raw/openmeteo_pressure_2022_07.nc')")
    print("  ds_region = subset_region(ds)")
    print("  inspect_dataset(ds_region)")