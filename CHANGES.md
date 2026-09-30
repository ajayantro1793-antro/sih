# Corrections applied

## Critical methodology fixes
- Changed the target from a backward/ambiguous rolling window to an explicit forward window: T+1 through T+3.
- Label percentile threshold is estimated from the earliest training portion only.
- Removed dataset-wide normalization. Mean/std are now fitted on training samples only during training.
- Replaced randomized block splitting with chronological train/validation/test splits plus a purge gap to prevent overlapping sliding-window leakage.
- Streamlit now uses the exact test indices saved by the training script.
- Alert threshold is tuned on validation data and saved to `data/processed/alert_threshold.npy` rather than manually copied into source code.

## Evaluation improvements
- Added PR-AUC and Brier score in addition to accuracy, precision, recall, F1 and ROC-AUC.
- Saves `model_metadata.json` and `training_history.json` for reproducible reporting.
- Training now fails clearly when a split contains only one class instead of silently producing meaningless metrics.

## Scientific/documentation corrections
- Explicitly describes the current label as a heavy-rainfall proxy, not direct thunderstorm truth.
- Keeps `lifted_index_proxy` clearly named as a proxy rather than a true Lifted Index.
- Corrected convergence to account for geographic latitude/longitude spacing in metres.
- Synchronized README, project summary, configuration and live-prediction documentation with the current ERA5 + Tamil Nadu implementation.
- Added the missing `cdsapi` dependency required by the ERA5 downloader.

## Important limitation
The uploaded ZIP contains source code and documentation, but not the raw ERA5 dataset or a newly trained model. The corrected repository therefore cannot truthfully include a new benchmark such as AUC=0.88. Run the pipeline in this order after downloading data:

```text
python notebooks/download_era5.py
python notebooks/01_build_dataset.py
python notebooks/02_train_model.py
streamlit run app.py
```
