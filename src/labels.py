"""Rainfall-proxy target construction for the weather nowcasting MVP.

The current MVP does not have a direct thunderstorm-observation label. It therefore
uses extreme area-mean precipitation as a proxy for severe convective weather.
This is explicitly a rainfall proxy, not a claim of meteorological thunderstorm truth.
"""

import os
import sys
import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LEAD_TIME_HOURS, TRAIN_SPLIT

STORM_PERCENTILE = 85


def compute_area_mean_precip(precip_ds: xr.Dataset, precip_var: str = "tp") -> xr.DataArray:
    """Return spatially averaged precipitation in mm per source timestep."""
    precip_mm = precip_ds[precip_var] * 1000
    return precip_mm.mean(dim=["lat", "lon"])


def build_storm_labels(
    precip_ds: xr.Dataset,
    precip_var: str = "tp",
    storm_percentile: float = STORM_PERCENTILE,
    lead_time_hours: int = LEAD_TIME_HOURS,
    threshold_reference_fraction: float = TRAIN_SPLIT,
) -> xr.DataArray:
    """Build a forward-looking binary rainfall-proxy target.

    For each time T, the target is 1 when accumulated area-mean precipitation
    over T+1 ... T+lead_time_hours exceeds a percentile threshold. The threshold
    is estimated from the earliest training fraction only, preventing later
    validation/test periods from changing the target definition.
    """
    area_mean_mm = compute_area_mean_precip(precip_ds, precip_var)
    times = precip_ds.time.values
    if len(times) < 2:
        raise ValueError("At least two precipitation timesteps are required.")

    time_diffs = np.diff(times).astype("timedelta64[s]").astype(np.int64) / 3600.0
    step_hours = float(np.median(time_diffs))
    if step_hours <= 0:
        raise ValueError("Precipitation timestamps must be strictly increasing.")
    if not np.allclose(time_diffs, step_hours, atol=1e-6):
        raise ValueError("Precipitation data must have a regular time interval for this MVP.")

    steps_ahead = max(1, int(round(lead_time_hours / step_hours)))
    if not np.isclose(steps_ahead * step_hours, lead_time_hours):
        raise ValueError(
            f"Lead time {lead_time_hours}h is not an integer multiple of data interval {step_hours}h."
        )

    # Explicitly exclude T itself: target is rainfall in the NEXT N steps.
    forward_values = sum(area_mean_mm.shift(time=-k) for k in range(1, steps_ahead + 1))
    forward_values = forward_values.rename("future_rainfall")

    reference_count = max(1, int(len(times) * threshold_reference_fraction))
    reference_end_time = times[reference_count - 1]
    reference = forward_values.sel(time=slice(None, reference_end_time)).values
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        raise ValueError("No valid rainfall windows available for label threshold estimation.")

    threshold_mm = float(np.percentile(reference, storm_percentile))
    labels = xr.where(forward_values.notnull(), (forward_values >= threshold_mm).astype("float32"), np.nan)

    print(
        f"[labels] Forward target: T+1..T+{lead_time_hours}h; "
        f"threshold={threshold_mm:.3f} mm, estimated from first "
        f"{threshold_reference_fraction:.0%} of the timeline; "
        f"top {100 - storm_percentile:.0f}% = positive"
    )
    print(
        "[labels] IMPORTANT: this is a heavy-rainfall proxy label, not a direct "
        "thunderstorm observation label."
    )
    return labels


def compute_district_mean_precip(
    precip_ds: xr.Dataset, district_masks: np.ndarray, precip_var: str = "tp"
) -> np.ndarray:
    """
    Area-mean precipitation PER DISTRICT (instead of one state-wide mean).

    Args:
        precip_ds: dataset with dims (time, lat, lon)
        district_masks: (n_districts, n_lat, n_lon) normalized masks from
                         src.districts.build_district_masks(), built from
                         THIS dataset's own lat/lon coordinates
        precip_var: name of the precipitation variable

    Returns:
        np.ndarray shape (n_time, n_districts), mm per source timestep
    """
    precip_mm = (precip_ds[precip_var] * 1000).values  # (time, lat, lon)
    # einsum: for each district d, sum over (lat,lon) of precip * mask
    district_mean = np.einsum("tlm,dlm->td", precip_mm, district_masks)
    return district_mean


def build_district_storm_labels(
    precip_ds: xr.Dataset,
    district_masks: np.ndarray,
    precip_var: str = "tp",
    storm_percentile: float = STORM_PERCENTILE,
    lead_time_hours: int = LEAD_TIME_HOURS,
    threshold_reference_fraction: float = TRAIN_SPLIT,
):
    """
    Per-district version of build_storm_labels(): same forward-window,
    train-only-threshold methodology, but each district gets its OWN
    area-mean precipitation series and its OWN percentile threshold
    (a district that is climatologically wetter shouldn't be judged
    against a drier district's cutoff).

    Returns:
        (labels, thresholds_mm):
          labels: np.ndarray (n_time, n_districts) of 0/1/NaN
          thresholds_mm: list of per-district threshold values (mm), in
                         the same order as district_masks' first axis
    """
    times = precip_ds.time.values
    if len(times) < 2:
        raise ValueError("At least two precipitation timesteps are required.")

    time_diffs = np.diff(times).astype("timedelta64[s]").astype(np.int64) / 3600.0
    step_hours = float(np.median(time_diffs))
    if step_hours <= 0:
        raise ValueError("Precipitation timestamps must be strictly increasing.")
    if not np.allclose(time_diffs, step_hours, atol=1e-6):
        raise ValueError("Precipitation data must have a regular time interval for this MVP.")

    steps_ahead = max(1, int(round(lead_time_hours / step_hours)))
    if not np.isclose(steps_ahead * step_hours, lead_time_hours):
        raise ValueError(
            f"Lead time {lead_time_hours}h is not an integer multiple of data interval {step_hours}h."
        )

    district_mean = compute_district_mean_precip(precip_ds, district_masks, precip_var)  # (T, D)
    n_time, n_districts = district_mean.shape

    # Forward sum over T+1..T+steps_ahead, per district (same idea as the
    # single-region version, vectorized across the district axis).
    forward = np.full_like(district_mean, np.nan)
    for t in range(n_time):
        end = t + steps_ahead
        if end < n_time:
            forward[t] = district_mean[t + 1: end + 1].sum(axis=0)
        # else stays NaN -- not enough future data for this T

    reference_count = max(1, int(n_time * threshold_reference_fraction))
    reference = forward[:reference_count]  # (reference_count, D)

    labels = np.full_like(forward, np.nan)
    thresholds_mm = []
    for d in range(n_districts):
        ref_col = reference[:, d]
        ref_col = ref_col[np.isfinite(ref_col)]
        if len(ref_col) == 0:
            raise ValueError(f"No valid rainfall windows for district index {d} threshold estimation.")
        threshold_mm = float(np.percentile(ref_col, storm_percentile))
        thresholds_mm.append(threshold_mm)

        col = forward[:, d]
        valid = np.isfinite(col)
        labels[valid, d] = (col[valid] >= threshold_mm).astype("float32")

    print(
        f"[labels] Per-district forward target: T+1..T+{lead_time_hours}h; "
        f"{n_districts} districts; thresholds estimated from first "
        f"{threshold_reference_fraction:.0%} of the timeline; "
        f"top {100 - storm_percentile:.0f}% = positive (per district)"
    )
    print("[labels] IMPORTANT: this is a heavy-rainfall proxy label, not a direct "
          "thunderstorm observation label.")
    return labels, thresholds_mm


def inspect_precip_distribution(precip_ds: xr.Dataset, precip_var: str = "tp") -> None:
    """Print rainfall statistics for sanity checking."""
    precip_mm = precip_ds[precip_var] * 1000
    pixel_values = precip_mm.values.flatten()
    pixel_values = pixel_values[np.isfinite(pixel_values)]
    area_mean = compute_area_mean_precip(precip_ds, precip_var).values
    area_mean = area_mean[np.isfinite(area_mean)]

    print("=" * 60)
    print("PRECIPITATION DISTRIBUTION")
    print("=" * 60)
    print(f"Per-pixel (mm/timestep): min={pixel_values.min():.3f} max={pixel_values.max():.3f} mean={pixel_values.mean():.3f}")
    print(f"Area-mean  (mm/timestep): min={area_mean.min():.3f} max={area_mean.max():.3f} mean={area_mean.mean():.3f}")
    print("\nArea-mean percentiles:")
    for pct in [50, 75, 85, 90, 95, 99]:
        print(f"  {pct}th percentile: {np.percentile(area_mean, pct):.4f} mm")
    print("=" * 60)


if __name__ == "__main__":
    print("Rainfall-proxy labeling module.")
