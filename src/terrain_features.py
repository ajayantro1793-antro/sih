"""
Static terrain-derived features (elevation, slope, and a simplified
Topographic Wetness Index) for the weather nowcasting MVP, built from a
downloaded DEM GeoTIFF rather than fetched live -- terrain doesn't
change, so these are computed once and reused for both training and
live inference (see notebooks/01c_add_terrain_features.py).

WHY THIS ISN'T AS SIMPLE AS THE ATMOSPHERIC FEATURES
------------------------------------------------------
Slope and flow-accumulation/TWI are only physically meaningful when
computed on the DEM's NATIVE resolution (tens to hundreds of meters),
not on the model's coarse ~27km grid cells -- a "slope" between two
Open-Meteo grid points 27km apart would be a near-meaningless number
that throws away all the local terrain structure that actually drives
flash-flood-style risk. So the pipeline here is:

  1. Reproject the DEM from geographic (lat/lon degrees) to a metric
     UTM CRS -- slope and flow-direction algorithms assume square,
     meter-sized cells, which lat/lon degree cells are not (especially
     east-west, which shrinks with cos(latitude)).
  2. Downsample to a manageable WORKING resolution (default 200m) --
     a full-resolution 30m DEM over all of Tamil Nadu is hundreds of
     millions of pixels, too slow for flow-accumulation even with a
     fast library, let alone the pure-numpy fallback below. 200m is
     still >100x finer than the model's ~27km grid cells.
  3. Compute slope and D8 flow accumulation / TWI at that working
     resolution.
  4. Reproject back to WGS84 and AVERAGE-aggregate onto the model's
     coarse grid (every fine pixel inside a coarse cell is averaged
     into that cell's value).

CAVEATS (documented explicitly, matching this project's existing
convention for other proxy features like lifted_index_proxy):
  - TWI here is TWI = ln(specific_catchment_area / tan(slope)), the
    standard textbook formula, but specific_catchment_area is
    approximated from D8 flow accumulation (single steepest-descent
    flow direction per cell) rather than a multi-flow-direction
    algorithm. D8 is the standard GIS-software simplification, but it
    systematically underestimates divergent flow on convex slopes.
  - Averaging fine terrain metrics up to a ~27km grid cell tells the
    model "this general area drains fast/slow", not "there's a stream
    at this exact point".

DEPENDENCIES: rasterio (required), richdem (recommended -- fast C++
flow accumulation; falls back to a slow pure-numpy D8 implementation
if not installed, see _numpy_d8_flow_accumulation()'s docstring for
why that fallback needs a coarser --terrain-resolution-m).

    pip install rasterio richdem
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import rasterio.transform as rio_transform
from affine import Affine

TERRAIN_FEATURE_NAMES = ["elevation_m", "slope_degrees", "twi"]


def load_dem(filepath: str):
    """Load a DEM GeoTIFF. Returns (elevation, transform, crs)."""
    with rasterio.open(filepath) as src:
        elevation = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)
    return elevation, transform, crs


def _utm_crs_for(lon: float, lat: float) -> str:
    """Pick a reasonable UTM zone EPSG code for the DEM's rough location."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327  # EPSG:326xx northern, 327xx southern
    return f"EPSG:{hemisphere}{zone:02d}"


def reproject_to_utm(elevation: np.ndarray, transform, crs, dst_crs: str):
    """Reproject a DEM array from its native CRS to a metric UTM CRS."""
    bounds = rio_transform.array_bounds(elevation.shape[0], elevation.shape[1], transform)
    dst_transform, width, height = calculate_default_transform(
        crs, dst_crs, elevation.shape[1], elevation.shape[0], *bounds
    )
    dst = np.full((height, width), np.nan, dtype=np.float64)
    reproject(
        source=elevation, destination=dst,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst, dst_transform


def _resample_utm(elevation_utm: np.ndarray, transform, crs, target_cellsize_m: float):
    """Downsample (average) a UTM DEM to a coarser working resolution."""
    src_cellsize = abs(transform.a)
    if target_cellsize_m <= src_cellsize:
        return elevation_utm, transform

    scale = target_cellsize_m / src_cellsize
    new_h = max(1, int(elevation_utm.shape[0] / scale))
    new_w = max(1, int(elevation_utm.shape[1] / scale))
    dst_transform = Affine(target_cellsize_m, 0, transform.c, 0, -target_cellsize_m, transform.f)
    dst = np.full((new_h, new_w), np.nan, dtype=np.float64)
    reproject(
        source=np.nan_to_num(elevation_utm, nan=np.nanmean(elevation_utm)), destination=dst,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=crs,
        resampling=Resampling.average,
    )
    return dst, dst_transform


def compute_slope_degrees(elevation_utm: np.ndarray, cellsize_m: float) -> np.ndarray:
    """Slope in degrees from a UTM (meter-gridded) elevation array."""
    dzdy, dzdx = np.gradient(elevation_utm, cellsize_m)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    return np.degrees(slope_rad)


def _numpy_d8_flow_accumulation(elevation: np.ndarray) -> np.ndarray:
    """
    Fallback D8 flow accumulation (used only if richdem isn't installed):
    each cell drains entirely to its single steepest downhill neighbor;
    accumulation is computed by processing cells in descending elevation
    order so every upstream cell resolves before the cell it drains into.

    SLOW: this is a Python loop over every cell. Fine at a few hundred
    thousand cells (e.g. 200m resolution over a few hundred km); NOT
    fine at native 30m resolution over a whole state (hundreds of
    millions of cells). If this is taking too long, raise
    --terrain-resolution-m (e.g. to 500 or 1000) or install richdem.
    """
    h, w = elevation.shape
    filled = elevation.copy()
    filled[np.isnan(filled)] = np.nanmin(filled) - 1

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    flow_to = np.full((h, w, 2), -1, dtype=np.int32)

    for r in range(h):
        for c in range(w):
            best_drop, best_rc = -np.inf, None
            for dr, dc in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    dist = np.hypot(dr, dc)
                    drop = (filled[r, c] - filled[nr, nc]) / dist
                    if drop > best_drop:
                        best_drop, best_rc = drop, (nr, nc)
            if best_rc is not None and best_drop > 0:
                flow_to[r, c] = best_rc

    order = np.dstack(np.unravel_index(np.argsort(-filled, axis=None), filled.shape))[0]
    accum = np.ones((h, w), dtype=np.float64)
    for r, c in order:
        tr, tc = flow_to[r, c]
        if tr >= 0:
            accum[tr, tc] += accum[r, c]
    return accum


def compute_flow_accumulation_and_twi(elevation_utm: np.ndarray, cellsize_m: float, slope_degrees: np.ndarray):
    """D8 flow accumulation and a simplified Topographic Wetness Index."""
    try:
        import richdem as rd

        dem = rd.rdarray(np.nan_to_num(elevation_utm, nan=np.nanmin(elevation_utm)), no_data=-9999)
        dem.geotransform = (0, cellsize_m, 0, 0, 0, -cellsize_m)
        filled = rd.FillDepressions(dem, epsilon=True, in_place=False)
        flow_acc = np.array(rd.FlowAccumulation(filled, method="D8"))
    except ImportError:
        print("richdem not installed -- using the slower pure-numpy D8 fallback "
              "(see _numpy_d8_flow_accumulation()'s docstring). "
              "`pip install richdem` for a faster, more robust implementation.")
        flow_acc = _numpy_d8_flow_accumulation(elevation_utm)

    specific_catchment_area = flow_acc * cellsize_m  # area per unit contour width, approx.
    slope_rad = np.radians(np.clip(slope_degrees, 0.01, None))  # avoid tan(0) -> inf
    twi = np.log(specific_catchment_area / np.tan(slope_rad))
    return flow_acc, twi


def aggregate_to_model_grid(fine_array: np.ndarray, fine_transform, fine_crs, lat_values, lon_values) -> np.ndarray:
    """
    Average-aggregate a fine-resolution raster onto the model's coarse
    (lat_values x lon_values) grid -- every fine pixel inside a coarse
    cell is averaged into that cell's value.

    Assumes lat_values is ASCENDING (south -> north), matching this
    project's convention elsewhere (see safe_subset_region()).
    """
    lat_values = np.asarray(lat_values)
    lon_values = np.asarray(lon_values)
    h, w = len(lat_values), len(lon_values)

    dlat = float(np.mean(np.diff(lat_values))) if h > 1 else 0.25
    dlon = float(np.mean(np.diff(lon_values))) if w > 1 else 0.25
    west = lon_values.min() - dlon / 2
    east = lon_values.max() + dlon / 2
    south = lat_values.min() - dlat / 2
    north = lat_values.max() + dlat / 2

    dst_transform = from_bounds(west, south, east, north, w, h)
    dst = np.full((h, w), np.nan, dtype=np.float64)

    reproject(
        source=np.nan_to_num(fine_array, nan=np.nanmean(fine_array)), destination=dst,
        src_transform=fine_transform, src_crs=fine_crs,
        dst_transform=dst_transform, dst_crs="EPSG:4326",
        resampling=Resampling.average,
    )
    return dst[::-1, :]  # from_bounds fills north->south; flip to south->north


def build_terrain_feature_stack(
    dem_path: str, lat_values, lon_values, terrain_resolution_m: float = 200.0
) -> dict:
    """
    End-to-end: load a DEM GeoTIFF, compute elevation/slope/TWI at a
    working resolution, aggregate all three onto the model's grid.

    Returns
    -------
    dict[str, np.ndarray], each shape (len(lat_values), len(lon_values)):
        {"elevation_m": ..., "slope_degrees": ..., "twi": ...}
    """
    elevation, transform, crs = load_dem(dem_path)

    utm_crs = _utm_crs_for(float(np.mean(lon_values)), float(np.mean(lat_values)))
    elevation_utm, utm_transform = reproject_to_utm(elevation, transform, crs, utm_crs)
    elevation_utm, utm_transform = _resample_utm(elevation_utm, utm_transform, utm_crs, terrain_resolution_m)
    cellsize_m = abs(utm_transform.a)

    slope_degrees = compute_slope_degrees(elevation_utm, cellsize_m)
    _, twi = compute_flow_accumulation_and_twi(elevation_utm, cellsize_m, slope_degrees)

    return {
        "elevation_m": aggregate_to_model_grid(elevation_utm, utm_transform, utm_crs, lat_values, lon_values),
        "slope_degrees": aggregate_to_model_grid(slope_degrees, utm_transform, utm_crs, lat_values, lon_values),
        "twi": aggregate_to_model_grid(twi, utm_transform, utm_crs, lat_values, lon_values),
    }


if __name__ == "__main__":
    print("Terrain feature module -- import build_terrain_feature_stack(dem_path, lat_values, lon_values)")
    print(f"Produces: {TERRAIN_FEATURE_NAMES}")
