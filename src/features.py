"""
Core atmospheric feature engineering for the weather nowcasting MVP.

IMPORTANT: this project's download_open_meteo.py (training) and
live_fetch.py (live) fetch only 2 pressure levels (850, 500 hPa) and
RELATIVE humidity ('r'), not specific humidity ('q').
This module derives specific humidity internally from relative humidity +
temperature (same Bolton-formula approach used for live data in
src/live_fetch.py, kept consistent between training and inference), and
uses 850mb (the lowest level available) wherever the original design used
1000mb ("surface") as a reference.

The lifted-index feature is explicitly a proxy, not a full
parcel-thermodynamics calculation. IWV is now a MID-TROPOSPHERIC MOISTURE
PROXY, not a true column-integrated water vapor value — see the note in
calculate_iwv() for why.

Usage:
    from src.features import calculate_all_features
    feature_ds = calculate_all_features(ds)
"""

import numpy as np
import xarray as xr

G = 9.81  # gravity, m/s^2


def relative_humidity_to_specific_humidity(
    relative_humidity: xr.DataArray, temperature_k: xr.DataArray, level_dim: str = "level"
) -> xr.DataArray:
    """
    Convert relative humidity (%) to specific humidity (kg/kg) using the
    Bolton (1980) approximation for saturation vapor pressure. This is the
    SAME formula used in src/live_fetch.py for live-data humidity
    conversion, kept consistent so training and live inference treat
    humidity identically.

    Args:
        relative_humidity: RH in % (0-100), dims include level_dim
        temperature_k: temperature in Kelvin, same dims as relative_humidity
        level_dim: name of the pressure-level dimension (used as pressure
                   in hPa for the conversion — assumes the level
                   COORDINATE VALUES are in hPa)

    Returns:
        Specific humidity in kg/kg
    """
    temp_c = temperature_k - 273.15
    pressure_hpa = temperature_k[level_dim]  # broadcasts against the level dim
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))  # saturation vapor pressure, hPa
    e = (relative_humidity / 100.0) * es
    q = 0.622 * e / (pressure_hpa - 0.378 * e)
    return q


def calculate_iwv(specific_humidity: xr.DataArray, level_dim: str = "level") -> xr.DataArray:
    """
    Moisture proxy via trapezoidal integration over the AVAILABLE pressure
    levels only.

    NOTE ON ACCURACY: true Integrated Water Vapor (IWV) integrates over the
    full atmospheric column (typically surface/1000mb up through ~300mb or
    higher). This project's data only has 850mb and 500mb available, so
    this integrates JUST that layer — it excludes near-surface moisture
    below 850mb, which in the tropics often holds a large share of total
    column moisture. Treat this as a "mid-tropospheric moisture proxy",
    not a scientifically accurate IWV value. Disclose this limitation if
    presenting IWV-based results.

    Args:
        specific_humidity: DataArray with dims (..., level, ...) in kg/kg
        level_dim: name of the pressure level dimension

    Returns:
        Moisture proxy in mm-equivalent, with the level dimension removed
    """
    levels = specific_humidity[level_dim].values
    q_sorted = specific_humidity.sortby(level_dim)
    p_sorted = np.sort(levels)

    dp = np.gradient(p_sorted) * 100  # hPa -> Pa
    dp_da = xr.DataArray(dp, coords={level_dim: q_sorted[level_dim]}, dims=[level_dim])

    iwv = (q_sorted * dp_da).sum(dim=level_dim) / G
    # kg/m^2 IS already mm of water (water density 1000 kg/m^3), no extra
    # unit conversion needed here.
    return iwv


def calculate_wind_shear(
    u: xr.DataArray, v: xr.DataArray, level_dim: str = "level",
    low_level: int = 850, high_level: int = 500
) -> xr.DataArray:
    """
    Bulk wind shear between two pressure levels.

    Defaults changed from the original 1000/500 to 850/500 — 1000mb is not
    among this project's downloaded levels, and with only 850/500
    available, using 850 explicitly (rather than relying on
    nearest-neighbor matching to silently resolve 1000 -> 850) makes the
    choice of levels an intentional, documented decision rather than an
    implicit side effect.

    Args:
        u, v: wind component DataArrays with a level dimension
        low_level, high_level: pressure levels (hPa) to compute shear between

    Returns:
        Wind shear magnitude in m/s
    """
    u_low = u.sel({level_dim: low_level}, method="nearest")
    u_high = u.sel({level_dim: high_level}, method="nearest")
    v_low = v.sel({level_dim: low_level}, method="nearest")
    v_high = v.sel({level_dim: high_level}, method="nearest")

    shear = np.sqrt((u_high - u_low) ** 2 + (v_high - v_low) ** 2)
    return shear


def calculate_lifted_index_simple(
    temperature: xr.DataArray, level_dim: str = "level", ref_level: int = 500
) -> xr.DataArray:
    """
    Simplified Lifted Index proxy: lowest-available-level-to-500mb
    temperature difference.

    Uses temperature[level_dim].max() to find the lowest (highest-pressure)
    available level automatically — this already adapts correctly to only
    having 850/500 available (max() picks 850), so no change needed here
    beyond documenting that behavior explicitly.

    NOTE: This is a simplified proxy, not the full thermodynamic LI
    calculation. Good enough for an MVP; replace with proper
    parcel-lifting math later (e.g. using MetPy) for production accuracy.

    Args:
        temperature: temperature DataArray with a level dimension (Kelvin)
        ref_level: upper level to compare against (hPa)

    Returns:
        Temperature difference proxy (K) — larger values suggest more instability
    """
    t_surface = temperature.sel({level_dim: temperature[level_dim].max()}, method="nearest")
    t_upper = temperature.sel({level_dim: ref_level}, method="nearest")
    return t_surface - t_upper


def calculate_humidity_at_level(
    specific_humidity: xr.DataArray, level_dim: str = "level", level: int = 850
) -> xr.DataArray:
    """Extract specific humidity at a single pressure level (default 850mb)."""
    return specific_humidity.sel({level_dim: level}, method="nearest")


def calculate_convergence(u: xr.DataArray, v: xr.DataArray, lat_dim="lat", lon_dim="lon") -> xr.DataArray:
    """
    Surface wind convergence: -(du/dx + dv/dy).
    Positive values indicate lifting (storm trigger mechanism).

    The input grid is geographic latitude/longitude, so convert degree
    spacing to metres before taking derivatives. This keeps the feature in
    approximately s^-1 units instead of using degree^-1 gradients.
    """
    earth_radius = 6_371_000.0
    lat_rad = np.deg2rad(u[lat_dim])

    du_dlon = u.differentiate(lon_dim)
    dv_dlat = v.differentiate(lat_dim)
    dx = earth_radius * np.cos(lat_rad) * np.pi / 180.0
    dy = earth_radius * np.pi / 180.0
    du_dx = du_dlon / dx
    dv_dy = dv_dlat / dy
    return -(du_dx + dv_dy)


def calculate_all_features(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the core MVP feature set from a raw atmospheric dataset
    (Open-Meteo for both training and live data, see
    notebooks/download_open_meteo.py and src/live_fetch.py).

    Expects ds to have variables: t (temperature, K), r (relative
    humidity, %), u, v (wind components) — each with dims
    (time, level, lat, lon). Specific humidity is derived internally from
    r and t; there is no 'q' variable in the raw download for this
    project (see relative_humidity_to_specific_humidity()).

    Returns:
        xarray Dataset with one variable per feature, dims (time, lat, lon)
    """
    features = xr.Dataset()

    q = relative_humidity_to_specific_humidity(ds["r"], ds["t"])

    features["iwv"] = calculate_iwv(q)
    features["wind_shear"] = calculate_wind_shear(ds["u"], ds["v"], low_level=850, high_level=500)
    features["lifted_index_proxy"] = calculate_lifted_index_simple(ds["t"], ref_level=500)
    features["humidity_850"] = calculate_humidity_at_level(q, level=850)
    features["convergence"] = calculate_convergence(
        ds["u"].sel(level=850, method="nearest"),
        ds["v"].sel(level=850, method="nearest"),
    )

    # Temporal trend features (rate of change over previous timestep)
    features["iwv_trend"] = features["iwv"].diff(dim="time")

    return features


if __name__ == "__main__":
    print("Feature engineering module — import and call calculate_all_features(ds)")
    print("Core MVP features: IWV (850-500mb moisture proxy), wind shear,")
    print("lifted index proxy, humidity@850mb, convergence, IWV trend")
    print("\nThese features derive specific humidity from relative humidity")
    print("internally, since this project's Open-Meteo download only includes 'r'.")