"""
Cloudburst-specific feature additions for the Weather Nowcasting MVP.

Adds two features on top of the six already computed by
calculate_all_features() in src/features.py:

1. moisture_flux_convergence = convergence * iwv
   Combines the existing low-level convergence and mid-tropospheric
   moisture proxy features. Extreme, localized rainfall is better
   indicated by moisture BEING FED INTO a converging column than by
   either signal alone.

2. humidity_500
   Specific humidity at 500 hPa. Your project's raw data only has 850 and
   500 hPa available (see features.py's module docstring), so 500 hPa is
   the natural "second vertical sample" to pair with the existing
   humidity_850 -- giving the model a read on moisture higher up in the
   profile, not just near the surface.

Both reuse your existing relative_humidity_to_specific_humidity() and
calculate_humidity_at_level() from src/features.py rather than
reimplementing the Bolton-formula conversion, so training and any live
inference path stay consistent with how humidity is already handled
everywhere else in the project.

Usage:
    from src.features import calculate_all_features
    from src.features_cloudburst import add_cloudburst_features

    feature_ds = calculate_all_features(ds)               # existing 6 features
    feature_ds = add_cloudburst_features(ds, feature_ds)   # + 2 more (8 total)
"""

from __future__ import annotations

import xarray as xr

from src.features import relative_humidity_to_specific_humidity, calculate_humidity_at_level


def add_cloudburst_features(ds: xr.Dataset, feature_ds: xr.Dataset) -> xr.Dataset:
    """
    Parameters
    ----------
    ds : xr.Dataset
        The raw, region-subset dataset with t/r/u/v -- i.e. exactly what
        you pass into calculate_all_features().
    feature_ds : xr.Dataset
        The output of calculate_all_features(ds). Must already contain
        'iwv' and 'convergence'.

    Returns
    -------
    xr.Dataset
        feature_ds with two additional variables appended:
        moisture_flux_convergence, humidity_500.
    """
    feature_ds = feature_ds.copy()

    feature_ds["moisture_flux_convergence"] = feature_ds["convergence"] * feature_ds["iwv"]

    q = relative_humidity_to_specific_humidity(ds["r"], ds["t"])
    feature_ds["humidity_500"] = calculate_humidity_at_level(q, level=500)

    return feature_ds
