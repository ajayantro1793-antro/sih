# Weather Nowcasting MVP — Severe Weather / Heavy-Rainfall Nowcasting

AI-driven severe-weather nowcasting prototype for SIH 2026.

## What this repository currently implements

- **Region:** Tamil Nadu bounding box (8.0–13.5°N, 76.5–80.5°E)
- **Data source:** Open-Meteo, for BOTH training and live prediction — training uses the Historical Forecast API (`historical-forecast-api.open-meteo.com`, archived operational forecast model output), live prediction uses the regular Forecast API (`api.open-meteo.com`). Same variable definitions/units in both, so there is a single, consistent atmospheric data source end-to-end (see CHANGES.md).
- **Resolution:** hourly, approximately 0.25° grid
- **Input:** 24-hour sequence of six engineered atmospheric features
- **Model:** TimeDistributed CNN + LSTM
- **Target:** next 3 hours of **extreme area-mean rainfall as a thunderstorm/severe-convection proxy**
- **Dashboard:** Streamlit historical test-set viewer + Open-Meteo live proof-of-concept

> **Important scientific limitation:** the current dataset does not contain direct thunderstorm observations. The binary target is a rainfall-based proxy. The system should therefore be described as a severe-weather/heavy-rainfall nowcasting prototype until direct thunderstorm labels are added.

## Methodology safeguards

The current pipeline is deliberately leakage-safe:

1. Raw features are saved without normalization.
2. Labels use rainfall from **T+1 through T+3**, excluding rainfall at T itself.
3. The rainfall percentile cutoff is estimated from the earliest training portion only.
4. Train/validation/test samples are separated chronologically.
5. A purge gap is used around split boundaries because sliding windows overlap heavily.
6. Feature normalization mean/std are fitted on **training samples only**.
7. The alert threshold is tuned on validation data and saved separately.
8. The exact test indices used for evaluation are saved and reused by Streamlit.

## Folder structure

```text
weather-nowcasting/
├── app.py
├── README.md
├── PROJECT_SUMMARY.txt
├── requirements.txt
├── configs/
│   ├── __init__.py
│   └── config.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── live_fetch.py
│   ├── live_log.py
│   ├── live_predict_core.py
│   ├── model.py
│   └── thresholds.py
├── notebooks/
│   ├── 00_quick_start_synthetic_test.py
│   ├── 01_build_dataset.py
│   ├── 02_train_model.py
│   ├── 03_live_predict.py
│   ├── download_open_meteo.py
│   └── download_era5.py   # deprecated, kept for reference only
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── outputs/
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

To download training data, no API key or registration is needed — Open-Meteo's Historical Forecast API is free and keyless:

```bash
python notebooks/download_open_meteo.py
```

(The old `notebooks/download_era5.py`, which required a CDS API credential, is deprecated and no longer part of the pipeline.)

## Run order

### 1. Synthetic sanity check

```bash
python notebooks/00_quick_start_synthetic_test.py
```

This checks that the CNN+LSTM can build, train, and evaluate. It does **not** validate meteorological performance.

### 2. Build the real dataset

```bash
python notebooks/01_build_dataset.py
```

This creates:

- `X.npy` — raw feature sequences
- `y.npy` — rainfall-proxy targets
- `feature_names.npy`
- `sample_times.npy`

### 3. Train and evaluate

```bash
python notebooks/02_train_model.py
```

This creates:

- `feature_mean.npy` / `feature_std.npy` — training-only normalization statistics
- `train_indices.npy`, `val_indices.npy`, `test_indices.npy`
- `alert_threshold.npy` — validation-tuned alert threshold
- `models/thunderstorm_mvp.h5`
- `models/model_metadata.json`
- `models/training_history.json`

The final test metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC and Brier score where applicable.

### 4. Historical dashboard

```bash
streamlit run app.py
```

The historical tab uses the exact test indices saved during training, so the dashboard and training evaluation refer to the same held-out samples.

### 5. Live proof of concept

```bash
python notebooks/03_live_predict.py
```

or use the Live tab in Streamlit.

## Current feature set

- Integrated water vapor (IWV)
- 1000–500 hPa bulk wind shear
- surface-to-500 hPa temperature-difference proxy for lifted index
- 850 hPa specific humidity
- surface wind convergence using geographic-distance correction
- IWV temporal trend

The lifted-index feature is explicitly a **proxy**, not a full parcel Lifted Index calculation.

## Live-data caveat

Training and live inference now both use Open-Meteo operational forecast data (Historical Forecast API for training, regular Forecast API for live), so there is a single consistent data source and no train/live model mismatch. Live predictions are nonetheless still an integration proof of concept — the historical record is short and district boundaries are rectangular approximations — not a validated operational forecast.

## Recommended next steps

1. Add multiple years of historical data and evaluate on a completely unseen year.
2. Add direct thunderstorm observations/lightning/radar-based labels where available.
3. Add CAPE, CIN, lapse rates, multi-level shear, vorticity, vertical velocity and moisture convergence.
4. Calibrate probabilities and inspect reliability diagrams.
5. Add a Tamil Nadu spatial risk map.
6. Validate the live pipeline against objective observations rather than manual annotations.
7. Only then expand toward INSAT ingestion, multi-hazard prediction and production alerting.
