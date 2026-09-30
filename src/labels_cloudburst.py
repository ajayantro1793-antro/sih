"""
Cloudburst label generation for the Weather Nowcasting MVP.

Reuses your existing build_storm_labels() (src/labels.py) as-is -- it's
already generic over the percentile threshold, uses a forward-looking
window (T+1..T+lead_time_hours), and estimates its threshold from only
the first `threshold_reference_fraction` of the timeline (TRAIN_SPLIT by
default) so validation/test periods never leak into the label
definition. There's no need to duplicate that logic; this module just
supplies a stricter default percentile for the rarer, more extreme event,
plus a matching wrapper around your build_district_storm_labels() for the
per-district path.

Usage:
    from src.labels_cloudburst import build_cloudburst_labels
    labels_da = build_cloudburst_labels(precip_ds)  # xr.DataArray, dims (time,)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from configs.config import LEAD_TIME_HOURS, TRAIN_SPLIT
from src.labels import build_storm_labels, build_district_storm_labels

# Thunderstorm uses the 85th percentile (~top 15%). Cloudbursts should be
# the rarer, more extreme tail of the SAME rainfall signal -- 97th
# percentile (~top 3%) by default. Tune this after inspecting your own
# precip distribution (src/labels.py's inspect_precip_distribution() is
# handy for that).
CLOUDBURST_PERCENTILE = 97


def build_cloudburst_labels(
    precip_ds: xr.Dataset,
    precip_var: str = "tp",
    cloudburst_percentile: float = CLOUDBURST_PERCENTILE,
    lead_time_hours: int = LEAD_TIME_HOURS,
    threshold_reference_fraction: float = TRAIN_SPLIT,
) -> xr.DataArray:
    """
    Thin wrapper around build_storm_labels() with a stricter default
    percentile. Same forward-looking window and same train-fraction-only
    threshold estimation -- only the percentile default changes. Prints
    the same "heavy-rainfall proxy label" caveat your existing function
    already prints, since it's the same underlying mechanism, just tuned
    for a rarer event.
    """
    return build_storm_labels(
        precip_ds,
        precip_var=precip_var,
        storm_percentile=cloudburst_percentile,
        lead_time_hours=lead_time_hours,
        threshold_reference_fraction=threshold_reference_fraction,
    )


def build_district_cloudburst_labels(
    precip_ds: xr.Dataset,
    district_masks: np.ndarray,
    precip_var: str = "tp",
    cloudburst_percentile: float = CLOUDBURST_PERCENTILE,
    lead_time_hours: int = LEAD_TIME_HOURS,
    threshold_reference_fraction: float = TRAIN_SPLIT,
):
    """
    Per-district cloudburst labels, mirroring build_district_storm_labels():
    each district gets its own area-mean precip series and its own
    percentile threshold, since a climatologically wetter district
    shouldn't be judged against a drier district's cutoff.

    Returns
    -------
    (labels, thresholds_mm) -- same shapes/semantics as
    build_district_storm_labels(): labels is (n_time, n_districts) of
    0/1/NaN, thresholds_mm is a list of per-district mm thresholds.
    """
    return build_district_storm_labels(
        precip_ds,
        district_masks,
        precip_var=precip_var,
        storm_percentile=cloudburst_percentile,
        lead_time_hours=lead_time_hours,
        threshold_reference_fraction=threshold_reference_fraction,
    )
