"""
Approximate district bounding boxes for Tamil Nadu, used to turn the
existing STATE-WIDE grid/model into PER-DISTRICT thunderstorm
predictions without changing how data is downloaded or grid features
are computed.

IMPORTANT LIMITATION: these are simple rectangular bounding-box
approximations for prototype purposes, NOT real administrative
boundaries (which are irregular polygons). Some districts are merged
or approximated for brevity, and district edges will be inaccurate
near boundaries. For a production system, replace DISTRICT_BOXES with
real polygon boundaries (e.g. an official/OSM Tamil Nadu district
shapefile) rasterized onto the model grid with a proper
point-in-polygon test instead of a lat/lon box test.

Every district box must fall inside the download region defined in
configs/config.py (LAT_MIN/LAT_MAX/LON_MIN/LON_MAX), used by both
notebooks/download_open_meteo.py (training) and src/live_fetch.py
(live), or its mask will be empty.
"""

import numpy as np

# name -> (lat_min, lat_max, lon_min, lon_max)
# Rough boxes covering Tamil Nadu's major districts. Extend/refine as needed.
DISTRICT_BOXES = {
    "Chennai":          (12.90, 13.25, 80.05, 80.35),
    "Vellore":          (12.70, 13.15, 78.85, 79.40),
    "Cuddalore":        (11.50, 11.95, 79.35, 79.90),
    "Salem":            (11.45, 11.85, 77.85, 78.35),
    "Erode":            (11.15, 11.55, 77.45, 77.95),
    "Coimbatore":       (10.75, 11.35, 76.75, 77.25),
    "Tiruchirappalli":  (10.55, 10.95, 78.50, 78.95),
    "Thanjavur":        (10.55, 10.95, 78.95, 79.35),
    "Nagapattinam":     (10.55, 10.95, 79.60, 80.05),
    "Madurai":          (9.70, 10.15, 77.85, 78.30),
    "Dindigul":         (10.15, 10.55, 77.70, 78.15),
    "Tirunelveli":      (8.55, 8.95, 77.50, 77.95),
    "Kanyakumari":      (8.00, 8.30, 77.25, 77.60),
}

DISTRICT_NAMES = list(DISTRICT_BOXES.keys())


def build_district_masks(lat_values: np.ndarray, lon_values: np.ndarray) -> np.ndarray:
    """
    Build a (n_districts, n_lat, n_lon) array of NORMALIZED averaging
    masks (each district's grid cells sum to 1.0) from DISTRICT_BOXES,
    matched against the actual model grid coordinates (lat/lon values
    that come from your training files, e.g. features_ds.lat.values).

    Raises rather than silently producing an empty mask if a district's
    box contains zero grid cells at this resolution/region -- that
    means either the box needs widening or the download region doesn't
    cover that district.
    """
    lat_values = np.asarray(lat_values)
    lon_values = np.asarray(lon_values)
    n_lat, n_lon = len(lat_values), len(lon_values)
    masks = np.zeros((len(DISTRICT_NAMES), n_lat, n_lon), dtype="float32")

    for d_idx, name in enumerate(DISTRICT_NAMES):
        lat_min, lat_max, lon_min, lon_max = DISTRICT_BOXES[name]
        lat_in = (lat_values >= lat_min) & (lat_values <= lat_max)
        lon_in = (lon_values >= lon_min) & (lon_values <= lon_max)
        box_mask = np.outer(lat_in, lon_in).astype("float32")
        n_cells = float(box_mask.sum())
        if n_cells == 0:
            raise ValueError(
                f"District '{name}' box {DISTRICT_BOXES[name]} contains no grid "
                f"cells in this dataset's lat/lon range "
                f"(lat {lat_values.min():.2f}-{lat_values.max():.2f}, "
                f"lon {lon_values.min():.2f}-{lon_values.max():.2f}). "
                f"Widen the box in DISTRICT_BOXES, use a finer download grid, "
                f"or drop this district."
            )
        masks[d_idx] = box_mask / n_cells

    return masks


if __name__ == "__main__":
    print(f"{len(DISTRICT_NAMES)} districts configured: {DISTRICT_NAMES}")
