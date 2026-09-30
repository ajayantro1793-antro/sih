"""
DEPRECATED: ERA5 downloader for SIH Weather Nowcasting (CDS API).

This script is no longer part of the pipeline. The project now trains on
Open-Meteo's Historical Forecast API instead of ERA5, so that training
and live prediction (src/live_fetch.py) share a single data source with
identical variables/units -- see notebooks/download_open_meteo.py and
CHANGES.md. This file is kept only for reference / anyone who
specifically wants an ERA5-trained model; notebooks/01_build_dataset.py
no longer reads its output by default.

Downloads one month at a time so that:
- requests remain manageable
- failed months can be resumed
- existing files are skipped
- one year or multiple years can be downloaded easily

Data:
    Pressure levels:
        RH 850 hPa
        U/V wind 850 hPa
        U/V wind 500 hPa
        Temperature 850 hPa
        Geopotential 850 hPa

    Single level:
        Total precipitation

Region:
    Tamil Nadu + surrounding area
"""

from pathlib import Path
from datetime import datetime
import calendar

import cdsapi


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with one full year
START_YEAR = 2020
START_MONTH = 6

END_YEAR = 2021
END_MONTH = 5

# Tamil Nadu + surrounding region
NORTH = 13.5
WEST = 76.5
SOUTH = 8.0
EAST = 80.5

HOURS = [
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "11:00",
    "12:00",
    "13:00",
    "14:00",
    "15:00",
    "16:00",
    "17:00",
    "18:00",
    "19:00",
    "20:00",
    "21:00",
    "22:00",
    "23:00",
]

PRESSURE_LEVELS = [
    "850",
    "500",
]


# ============================================================
# CDS CLIENT
# ============================================================

client = cdsapi.Client()


# ============================================================
# MONTH RANGE
# ============================================================

def generate_months(
    start_year,
    start_month,
    end_year,
    end_month,
):

    year = start_year
    month = start_month

    while True:

        yield year, month

        if year == end_year and month == end_month:
            break

        month += 1

        if month > 12:
            month = 1
            year += 1


# ============================================================
# DOWNLOAD
# ============================================================

for year, month in generate_months(
    START_YEAR,
    START_MONTH,
    END_YEAR,
    END_MONTH,
):

    month_str = f"{month:02d}"

    days_in_month = calendar.monthrange(
        year,
        month
    )[1]

    days = [
        f"{day:02d}"
        for day in range(1, days_in_month + 1)
    ]

    print("\n")
    print("=" * 75)
    print(f"ERA5 DOWNLOAD: {year}-{month_str}")
    print("=" * 75)

    # --------------------------------------------------------
    # PRESSURE LEVELS
    # --------------------------------------------------------

    pressure_file = (
        OUTPUT_DIR /
        f"era5_pressure_{year}_{month_str}.nc"
    )

    if pressure_file.exists():

        print(
            f"[SKIP] Pressure file already exists: "
            f"{pressure_file.name}"
        )

    else:

        print(
            "[DOWNLOAD] Pressure-level ERA5..."
        )

        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",

                "variable": [
                    "relative_humidity",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "temperature",
                    "geopotential",
                ],

                "pressure_level": PRESSURE_LEVELS,

                "year": str(year),

                "month": month_str,

                "day": days,

                "time": HOURS,

                "area": [
                    NORTH,
                    WEST,
                    SOUTH,
                    EAST,
                ],

                "format": "netcdf",
            },

            str(pressure_file),
        )

        print(
            f"[DONE] {pressure_file.name}"
        )

    # --------------------------------------------------------
    # PRECIPITATION
    # --------------------------------------------------------

    precip_file = (
        OUTPUT_DIR /
        f"era5_precip_{year}_{month_str}.nc"
    )

    if precip_file.exists():

        print(
            f"[SKIP] Precipitation file already exists: "
            f"{precip_file.name}"
        )

    else:

        print(
            "[DOWNLOAD] Single-level precipitation..."
        )

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",

                "variable": [
                    "total_precipitation",
                ],

                "year": str(year),

                "month": month_str,

                "day": days,

                "time": HOURS,

                "area": [
                    NORTH,
                    WEST,
                    SOUTH,
                    EAST,
                ],

                "format": "netcdf",
            },

            str(precip_file),
        )

        print(
            f"[DONE] {precip_file.name}"
        )


print("\n")
print("=" * 75)
print("ALL ERA5 DOWNLOADS COMPLETE")
print("=" * 75)

print(
    f"Period: "
    f"{START_YEAR}-{START_MONTH:02d} "
    f"to "
    f"{END_YEAR}-{END_MONTH:02d}"
)

print(
    f"Region: "
    f"{NORTH}°N → {SOUTH}°N, "
    f"{WEST}°E → {EAST}°E"
)

print(
    f"Output: {OUTPUT_DIR.resolve()}"
)