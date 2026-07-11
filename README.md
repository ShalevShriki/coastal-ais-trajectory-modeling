# AIS Vessel Trajectory Prediction

Deep learning pipeline for **long-horizon vessel trajectory prediction** from AIS (Automatic Identification System) data. The project studies **how much temporal context** is needed for forecasting: fixed-context models, receding-horizon sliding windows, Transformers, and an **adaptive multi-scale autoregressive RNN** that learns soft weights over 9h / 12h / 18h / 24h history.

**Main task:** predict **12 hours** into the future (72 steps @ 10 min) from variable-length history, on **USA Combined** coasts with a **smart-motion** training filter.

See [project_research.md](project_research.md) for the full research plan and [docs/trajectory_loss.md](docs/trajectory_loss.md) for the training loss.

## Quick start

```bash
cd /path/to/project/proj/project
pip install -r requirements.txt
export PYTHONPATH=/path/to/project   # parent of proj/

# Flat LSTM — 24h history → 12h future
python models/RNN.py \
  --coast "USA Combined" \
  --input data/processed/combined_filtered_smart/train.parquet \
  --sample 400000 \
  --horizon-hours 12.0 \
  --run-tag my_run/flat_lstm

# Autoregressive LSTM — 12h history → 12h future
python models/RNN_AR.py \
  --coast "USA Combined" \
  --input data/processed/combined_filtered_smart/train.parquet \
  --sample 400000 \
  --history-hours 12 \
  --future-hours 12 \
  --horizon-hours 12 \
  --teacher-forcing 0.3 \
  --run-tag my_run/ar_12h
```

On the cluster, sbatch scripts source `_env.sh` with absolute paths and set `PYTHONPATH` to the project parent.

## Research experiments (`exp_final`)

Active suite aligned with [project_research.md](project_research.md). Data: `combined_filtered_smart`, 400k train subsample, trajectory split, seed 42.

| ID | Model | Config | Run tag |
|----|-------|--------|---------|
| Baseline | Kinematic SOG+COG | auto at eval | — |
| B1 | Flat LSTM | 24h → 12h | `exp_clean/B1_flat` ✓ |
| B2 | Transformer | 24h → 12h | `exp_clean/B2_transformer` ✓ |
| AR9–AR24 | RNN_AR + anchor | 9 / 12 / 18 / 24h → 12h | `exp_final/AR_*` |
| Slide | Receding horizon | 24h → 3h × 4 rollouts | `exp_final/sliding_3h` |
| Adaptive | Multi-scale AR | 9+12+18+24h gated → 12h | `exp_final/adaptive_multiscale` |

Submit the serial GPU chain:

```bash
bash scripts/exp_final/submit_all.sh
```

Logs: `LOG/exp_final_*.out`. Results:

```text
data/results/USA Combined/unknown/exp_final/<run_tag>/
data/results/USA Combined/unknown/exp_clean/B1_flat/   # flat + transformer baselines
```

**Note:** Residual-naive AR and old `exp_context` (6h-future) runs were dropped. Legacy v1/v2 `experiment1` results remain under `data/results/.../v1/` and `v2/` for reference only.

## Repository layout

```text
project/
├── processing/              # NOAA download, clean, segment, window build (per coast)
├── scripts/
│   ├── exp_final/           # Current research suite (sbatch + submit_all.sh)
│   ├── exp_clean/           # Flat + Transformer baselines (B1, B2)
│   ├── combine_datasets.py
│   ├── apply_training_filters.py
│   └── smart_motion_filter.sbatch
├── models/
│   ├── RNN.py               # Flat LSTM — direct multi-step prediction
│   ├── RNN_AR.py            # Autoregressive LSTM + anchor offsets
│   ├── RNN_AR_adaptive.py   # Multi-scale context gating (main contribution)
│   ├── RNN_recursive_1h.py  # Sliding-window chunk forecaster (1h or 3h chunks)
│   ├── transformers.py      # Transformer encoder
│   ├── training_utils.py    # TrajectoryLoss, curriculum, teacher forcing
│   └── plot_utils.py
├── docs/trajectory_loss.md
├── project_research.md
├── coast_paths.py
├── window_data.py           # Loads, splits, filters, variable history slicing
├── data/                    # NOT in git (parquets, checkpoints)
└── LOG/                     # Slurm logs
```

## Data pipeline (run in order)

### 1. Download and clean (per day, per coast)

```bash
python "processing/INCREMENTAL_PROCESS West Coast.py" --dataset-tag feb --start 2025-02-01 --end 2025-02-28
```

Same pattern for `Eastern coast` and `Mexcany Beach`.

### 2. Finalize (segments + windows)

```bash
python "processing/INCREMENTAL_PROCESS West Coast.py" --finalize-only --dataset-tag feb
```

Outputs per coast: `coastal_segments.parquet`, `model_ready_windows.parquet` (default **24h history → 12h future**).

### 3. Combine coasts

```bash
python scripts/combine_datasets.py --include-misplaced --tag feb
```

→ `data/processed/combined/{train,val,test}.parquet`

### 4. Filters

**Stationary filter** (history-only, removes near-stationary windows):

```bash
python scripts/apply_training_filters.py
```

→ `data/processed/combined_filtered/`

**Smart-motion filter** (keeps vessels with meaningful motion; ~36% of rows retained):

```bash
sbatch scripts/smart_motion_filter.sbatch
# or run the filter script directly — see filter_report.json in output dir
```

→ `data/processed/combined_filtered_smart/`

This is the **primary training dataset** for current experiments.

### 5. Audit (optional)

```bash
python AIS_AUDIT.py --segments data/processed/.../coastal_segments.parquet
```

## Window format

Each row is one supervised sample:

| Part | Columns | Default (full parquet) |
|------|---------|------------------------|
| History | `x_t{t:03d}_{feature}` | 144 steps × 15 features (24h) |
| Future | `y_t{t:03d}_lat`, `y_t{t:03d}_lon` | 72 steps (12h) |

History features: `lat`, `lon`, `sog`, `cog_sin`, `cog_cos`, `heading_sin`, `heading_cos`, `heading_missing`, `dt_sec`, `dlat`, `dlon`, `dsog`, `dcog`, `v_north_kmh`, `v_east_kmh`.

**Variable context:** `--history-hours` takes the **suffix** of the full history ending at the anchor (e.g. `--history-hours 9` uses the last 54 steps). `--future-hours` takes the **prefix** of the future (e.g. `--future-hours 12` uses 72 steps). Implemented in `window_data.resolve_window_hours()` and `build_window_arrays()`.

## Models

| Script | Architecture | Role |
|--------|--------------|------|
| `RNN.py` | Flat LSTM | One-shot 12h prediction (reference) |
| `RNN_AR.py` | Encoder–decoder AR LSTM | Fixed-context autoregressive forecast |
| `RNN_AR_adaptive.py` | Multi-encoder + softmax gating + AR decoder | Learns context weights α₉, α₁₂, α₁₈, α₂₄ |
| `RNN_recursive_1h.py` | Chunk displacement RNN | Train 24h→3h (or 1h); recursive rollout to 12h |
| `transformers.py` | Transformer encoder | Attention over 24h history |
| `LINEAR_REGRESSION.py` | Linear baseline | Simple baseline |

### Training (shared)

- **Loss:** `TrajectoryLoss` — Huber on anchor-offset deltas + geographic km term ([docs](docs/trajectory_loss.md))
- **Baselines at eval:** kinematic constant SOG+COG, naive last-step delta
- **AR training:** scheduled teacher forcing (0.3→0), horizon curriculum (6h→12h) unless `--no-curriculum`
- **Metrics:** FDE, ADE, nFDE, nADE, straight/maneuver buckets
- **Adaptive model:** saves `context_alpha_weights.json` per test sample for feature analysis

### CLI highlights

```text
--history-hours 9|12|18|24    # context length (RNN, RNN_AR, Transformer)
--future-hours 12             # prediction horizon for train/eval
--horizon-hours 12            # FDE evaluation step
--teacher-forcing 0.3           # AR models
--chunk-hours 3               # RNN_recursive_1h.py sliding window
--run-tag exp_final/AR_12h    # results subdirectory
```

### Output paths

```text
data/results/USA Combined/unknown/<run_tag>/<Model>/
  ├── *_metrics.json
  ├── *_training_history.png
  ├── *_scatter.png
  └── *_sample_trajectories.json   # or context_alpha_weights.json (adaptive)

data/models/USA Combined/unknown/<run_tag>/
  └── *.pt
```

Regenerate training plots from metrics:

```bash
python scripts/plot_training_history.py --metrics path/to/*_metrics.json
```

## Coastal regions

Configured in `coast_paths.py`:

| Coast | Regions |
|-------|---------|
| Eastern coast | US East Coast + Gulf |
| West Coast | California + PNW |
| Mexcany Beach | Mexican Pacific + Gulf |
| **USA Combined** | All USA coasts merged (`combined_filtered_smart`) |

## Evaluation

Primary metric: **12h FDE** (Haversine km at the final future step).

Also reported: full-path ADE, horizon-wise error, kinematic baseline, stratified buckets (straight / maneuver / anchored / other).

Example completed results (smart-motion, test set):

| Model | Median FDE @ 12h |
|-------|------------------|
| Kinematic baseline | ~106 km |
| Flat LSTM (`exp_clean/B1`) | ~18 km |
| Transformer (`exp_clean/B2`) | ~20 km |

## Visualization

```bash
python models/visualize_model_results.py --metrics path/to/*_metrics.json
python scripts/map_filtered_tracks.py --mode both
python scripts/plot_training_history.py --metrics path/to/*_metrics.json
```

## Slurm (cluster)

| Script | Purpose |
|--------|---------|
| `scripts/exp_final/submit_all.sh` | **Current** serial research suite |
| `scripts/exp_clean/submit_all.sh` | Flat LSTM + Transformer only |
| `scripts/smart_motion_filter.sbatch` | Build `combined_filtered_smart` |
| `scripts/feb_combine_filter.sbatch` | Combine + stationary filter |

**Important:** sbatch scripts must `source` `_env.sh` via **absolute path** (Slurm copies scripts to `/var/spool/slurmd/`).

Logs: `LOG/`. GPU: `part-ugproj`, QoS: `qos-ugproj`.

## Data source

```text
https://coast.noaa.gov/htdata/CMSP/AISDataHandler/<year>/ais-YYYY-MM-DD.csv.zst
```

Danish AIS: `--source danish` in processing scripts.

## What is not in git

- Raw/processed parquets (`data/processed/`)
- Model checkpoints (`data/models/`, `*.pt`)
- Slurm logs (`LOG/`)

**In git:** source, sbatch scripts, metrics JSON and plots under `data/results/`.

## Dependencies

```text
pandas numpy torch scikit-learn pyarrow matplotlib folium contextily pyproj requests zstandard
```

```bash
pip install -r requirements.txt
```

## License / course

Technion deep learning project (crml-prj10844). NOAA public AIS data.
