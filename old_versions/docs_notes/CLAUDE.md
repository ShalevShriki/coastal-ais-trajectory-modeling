# AIS Vessel Trajectory Prediction — Project Overview

## What this project does
Builds a deep learning pipeline for **long-horizon vessel trajectory prediction** using AIS (Automatic Identification System) maritime data. The pipeline downloads raw AIS data from NOAA, cleans and segments it by coastal region, builds sliding-window samples, and trains models to predict future vessel positions.

## Data pipeline (run in order)
1. **Download & clean** per day: `processing/INCREMENTAL_PROCESS <Coast>.py` (one per coast)
   - Downloads daily NOAA AIS `.csv.zst` files, filters by region, writes per-day parquet files
   - Tracks processed days in `data/processed/processed_days_<region>.txt`
2. **Finalize** (segment + window building): called automatically at end of incremental process, or via `--finalize-only`
   - Outputs: `data/processed/<Coast>/ais_<region>_long_horizon/coastal_segments.parquet`
   - Outputs: `data/processed/<Coast>/ais_<region>_long_horizon/model_ready_windows.parquet`
3. **Audit data**: `AIS_AUDIT.py --segments <path>` — produces CSVs and histograms in an `audit/` subfolder
4. **Build windows (memory-efficient)**: `build_mexican_4d_windows.py` — batch-processes Mexcany Beach 4d dataset

## Coastal regions
Configured in [coast_paths.py](coast_paths.py). Three coasts:
- **West Coast** — California + PNW bounding boxes
- **Mexcany Beach** — Mexican Pacific + Gulf of Mexico
- **Eastern coast** — US East Coast + Gulf

## Model training
Models live in `models/`:
- `LINEAR_REGRESSION.py`
- `RNN.py`
- `rebuild_windows.py`

## Key data files (not in git)
- `data/processed/<Coast>/ais_<region>_long_horizon/model_ready_windows.parquet` — main ML input
- `data/processed/<Coast>/ais_<region>_long_horizon/coastal_segments.parquet` — cleaned trajectories
- `data/processed/days/<region>/YYYY-MM-DD.parquet` — per-day cleaned files

## Window format
- History features (15 cols): `x_t000_<feature>` ... `x_tNNN_<feature>`
  - lat, lon, sog, cog_sin, cog_cos, heading_sin/cos, heading_missing, dt_sec, dlat, dlon, dsog, dcog, v_north_kmh, v_east_kmh
- Target: `y_t000_lat`, `y_t000_lon` ... (future positions)
- Default: 24h history → 12h future, resampled at 10 min (144 history steps, 72 future steps)

## Key utility modules
- [coast_paths.py](coast_paths.py) — `COAST_CONFIGS`, region bounds, path resolution helpers
- [window_data.py](window_data.py) — `load_windows`, `build_window_arrays`, `trajectory_splits`, `scale_history_features`
- [coast_frame.py](coast_frame.py) — (coastal frame utilities)
- [AIS_AUDIT.py](AIS_AUDIT.py) — data quality audit tool
- [EXPORT_COAST_DATA.py](EXPORT_COAST_DATA.py) — data export
- [VISUALIZE.py](VISUALIZE.py) — visualization

## Dependencies
`pandas`, `numpy`, `torch`, `scikit-learn`, `pyarrow`, `matplotlib`, `folium`, `contextily`, `pyproj`, `requests`, `zstandard`

Install: `pip install -r requirements.txt`

## Data source
NOAA AIS: `https://coast.noaa.gov/htdata/CMSP/AISDataHandler/<year>/ais-YYYY-MM-DD.csv.zst`
Also supports Danish AIS source (via `--source danish`).